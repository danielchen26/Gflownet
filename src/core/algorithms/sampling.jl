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
FIXED: Now properly generic - works for any domain, not just grid worlds.
"""
function sample_trajectory(model::GFlowNetModel; rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    # FIXED: Use concrete type annotations for type stability
    states = AbstractState[model.dag.initial_state]
    current_state::AbstractState = model.dag.initial_state
    
    while !is_terminal_state(current_state)
        # Get state features using the interface function
        features = state_to_features(current_state)
        
        # Use safe model call to get action logits
        logits, _ = safe_model_call(
            model.forward_policy.model,
            features,
            model.parameters.forward,
            model.states.forward
        )
        
        # FIXED: Use proper interface functions instead of hardcoded logic
        # Get all applicable actions for current state
        applicable_actions = Vector{AbstractAction}()
        for action in model.dag.actions
            if is_applicable(action, current_state)
                push!(applicable_actions, action)
            end
        end
        
        if isempty(applicable_actions)
            break
        end
        
        # FIXED: Use action indices directly (1 to length(actions))
        # Assumes neural network outputs probabilities for all possible actions
        action_indices = Vector{Int}(1:length(applicable_actions))
        
        # Get relevant logits for applicable actions only
        if length(action_indices) <= length(logits)
            relevant_logits = logits[action_indices]
        else
            # Fallback: use available logits
            relevant_logits = logits[1:min(length(logits), length(action_indices))]
        end
        
        # Add numerical stability
        relevant_logits = clamp.(relevant_logits, Float32(-20.0), Float32(20.0))
        probs = softmax(relevant_logits)
        
        # Ensure probabilities are valid (no NaN/Inf)
        if any(isnan.(probs)) || any(isinf.(probs))
            # Fallback to uniform distribution if numerical issues
            probs = fill(Float32(1.0) / length(applicable_actions), length(applicable_actions))
        end
        
        # Sample action according to learned policy
        action_idx = sample(1:length(applicable_actions), Weights(probs))
        chosen_action = applicable_actions[action_idx]
        
        # Apply the chosen action to get next state
        next_state = apply_action(chosen_action, current_state)
        
        push!(states, next_state)
        current_state = next_state
    end

    return Trajectory(states)
end

