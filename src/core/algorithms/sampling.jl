using Distributions: Categorical
using StatsBase: sample, Weights
using NNlib: softmax
using Random
using Lux
using Optimisers
using Zygote

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
    # Comprehensive input validation (non-differentiable)
    Zygote.@ignore validate_neural_network_input(features, "features")
    Zygote.@ignore validate_model_parameters(parameters, "parameters")

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

        # Validate outputs before returning (non-differentiable)
        Zygote.@ignore validate_neural_network_output(outputs, "model output")

        # If outputs have batch dimension of 1, flatten to a vector
        if size(outputs, 2) == 1
            outputs = vec(outputs)
        end

        return outputs, new_states
    catch e
        @error "Neural network model call failed" error = e features_shape = size(features) features_stats = (min=minimum(features), max=maximum(features), mean=mean(features))
        rethrow(e)
    end
end

# validate_neural_network_input is now in utils/validation.jl

# validate_neural_network_output is now in utils/validation.jl

# validate_model_parameters and _validate_params_recursive are now in utils/validation.jl

"""
    sample_trajectory(model::GFlowNetModel; rng=nothing)

Sample a complete trajectory from the GFlowNet using learned stochastic policy.
Zygote-compatible version that avoids all mutations and uses functional programming.
"""
function sample_trajectory(model::GFlowNetModel; rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end

    # Pre-allocate trajectory to avoid mutations during sampling
    max_trajectory_length = 100  # Reasonable default

    return _sample_trajectory_recursive(
        model,
        model.dag.initial_state,
        AbstractState[],
        max_trajectory_length,
        rng
    )
end

"""
    _sample_trajectory_recursive(model, current_state, states_so_far, remaining_steps, rng)

Recursive helper for Zygote-compatible trajectory sampling.
Uses functional approach with no mutations.
"""
function _sample_trajectory_recursive(
    model::GFlowNetModel,
    current_state::AbstractState,
    states_so_far::Vector{AbstractState},
    remaining_steps::Int,
    rng
)
    # Base cases
    if is_terminal_state(current_state) || remaining_steps <= 0
        return Trajectory(vcat(states_so_far, [current_state]))
    end

    # Get applicable actions using filter (functional approach)
    applicable_actions = filter(
        action -> is_applicable(action, current_state),
        model.dag.actions
    )

    if isempty(applicable_actions)
        return Trajectory(vcat(states_so_far, [current_state]))
    end

    # Get state features
    features = state_to_features(current_state)

    # Get action probabilities
    logits, _ = safe_model_call(
        model.forward_policy.model,
        features,
        model.parameters.forward,
        model.states.forward
    )

    # Compute action probabilities safely
    n_actions = length(applicable_actions)
    relevant_logits = if n_actions <= length(logits)
        logits[1:n_actions]
    else
        # Pad with zeros if needed
        vcat(logits, zeros(Float32, n_actions - length(logits)))
    end

    # Numerical stability
    stable_logits = clamp.(relevant_logits, Float32(-20.0), Float32(20.0))
    probs = softmax(stable_logits)

    # Handle numerical issues
    if any(isnan.(probs)) || any(isinf.(probs)) || sum(probs) ≈ 0.0
        probs = fill(Float32(1.0) / n_actions, n_actions)
    end

    # Sample action
    action_idx = sample(rng, 1:n_actions, Weights(probs))
    chosen_action = applicable_actions[action_idx]

    # Apply action to get next state
    next_state = apply_action(chosen_action, current_state)

    # Recursive call with updated state list
    return _sample_trajectory_recursive(
        model,
        next_state,
        vcat(states_so_far, [current_state]),
        remaining_steps - 1,
        rng
    )
end
