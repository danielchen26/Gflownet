using Distributions: Categorical
using StatsBase: sample, Weights
using NNlib: softmax
using Random
using Lux
using Optimisers

"""
    state_to_features(state::AbstractState)

Convert a state to a feature vector. Should be implemented by concrete types.
"""
function state_to_features end

"""
    reward(state::AbstractState)

Calculate the reward for a state. Should be implemented by concrete types.
Includes validation to ensure GFlowNet mathematical requirements.

Note: GFlowNets require all rewards to be positive for mathematical correctness.
"""
function reward end

# validate_reward is now in utils/validation.jl



# forward_transition_prob moved to transitions.jl

# backward_transition_prob moved to transitions.jl



"""
    safe_model_call(model, features, parameters, states)

Helper function to safely call a Lux model with proper feature formatting and validation.
Ensures features are properly shaped for Lux models and handles batch dimensions.
Includes comprehensive input validation to prevent numerical issues.
"""
function safe_model_call(model, features, parameters, states)
    # Comprehensive input validation
    validate_neural_network_input(features, "features")
    validate_model_parameters(parameters, "parameters")
    
    # Convert to Float32 to ensure type stability
    features = convert(Array{Float32}, features)
    
    # Reshape features to ensure they're a matrix with correct dimensions
    # Lux expects input in the format [features, batch]
    if features isa Vector
        features = reshape(features, :, 1)
    end
    
    # Additional safety: clamp extreme values to prevent overflow
    features = clamp.(features, Float32(-1e10), Float32(1e10))
    
    try
        # Use proper Lux API for model application
        outputs, new_states = model(features, parameters, states)
        
        # Validate outputs before returning
        validate_neural_network_output(outputs, "model output")
        
        # If outputs have batch dimension of 1, flatten to a vector
        if size(outputs, 2) == 1
            outputs = vec(outputs)
        end
        
        return outputs, new_states
    catch e
        @error "Neural network model call failed" error=e features_shape=size(features) features_stats=(min=minimum(features), max=maximum(features), mean=mean(features))
        rethrow(e)
    end
end

# validate_neural_network_input is now in utils/validation.jl

# validate_neural_network_output is now in utils/validation.jl

# validate_model_parameters and _validate_params_recursive are now in utils/validation.jl

"""
    sample_trajectory(model::GFlowNetModel; rng=nothing)

Sample a complete trajectory from the GFlowNet using learned stochastic policy.
"""
function sample_trajectory(model::GFlowNetModel; rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    states = [model.dag.initial_state]
    # Note: actions tracking removed as it's not used in this implementation
    
    current_state = model.dag.initial_state
    
    while current_state ∉ model.dag.terminal_states
        # Get next state probabilities
        features = state_to_features(current_state)
        
        # Use safe model call to get logits
        logits, _ = safe_model_call(
            model.forward_policy.model,
            features,
            model.parameters.forward,
            model.states.forward
        )
        
        # Get all possible next states
        next_states = get_next_states(model.dag, current_state)
        if isempty(next_states)
            break
        end
        
        # Map next states to their corresponding actions
        action_indices = Int[]
        for next_state in next_states
            if next_state.is_terminal  # Check terminal first
                push!(action_indices, 5)  # Terminate action
            elseif next_state.x > current_state.x  # Moving right
                push!(action_indices, 4)  # MoveRight action
            elseif next_state.x < current_state.x  # Moving left
                push!(action_indices, 3)  # MoveLeft action
            elseif next_state.y > current_state.y  # Moving up
                push!(action_indices, 1)  # MoveUp action
            elseif next_state.y < current_state.y  # Moving down
                push!(action_indices, 2)  # MoveDown action
            else
                # This shouldn't happen in a grid world
                error("Invalid state transition from $(current_state) to $(next_state)")
            end
        end
        
        # Ensure all action indices are valid (1-5)
        if any(idx -> idx < 1 || idx > 5, action_indices)
            error("Invalid action indices: $action_indices")
        end
        
        relevant_logits = logits[action_indices]
        
        # Add numerical stability and appropriate temperature scaling
        relevant_logits = clamp.(relevant_logits, Float32(-20.0), Float32(20.0))  # Prevent extreme values
        temperature = Float32(1.0)  # Start with no temperature scaling for debugging
        probs = softmax(relevant_logits ./ temperature)
        
        # Ensure probabilities are valid (no NaN/Inf)
        if any(isnan.(probs)) || any(isinf.(probs))
            # Fallback to uniform distribution if numerical issues
            probs = fill(Float32(1.0) / length(next_states), length(next_states))
        end
        
        # Sample according to learned policy (natural GFlowNet exploration)
        next_state_idx = sample(1:length(next_states), Weights(probs))
        next_state = next_states[next_state_idx]

        push!(states, next_state)
        current_state = next_state
    end

    return Trajectory(states)
end

