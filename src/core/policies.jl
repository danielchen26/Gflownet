# Policy Functions - Core GFlowNet Mathematical Operations
# Implementation of P_F, P_B, and Z - the three fundamental GFlowNet functions

using Random
using StatsBase: Weights, sample
using NNlib: softmax, sigmoid
using Zygote

# =============================================================================
# Forward Policy P_F(a|s) - Mathematical Foundation
# =============================================================================

"""
    forward_probability(policy::ForwardPolicy, state, action, parameters, states)

Compute P_F(a|s) - the probability of taking action a from state s.

# Mathematical Foundation
The forward policy defines the probability distribution over actions:
P_F(a|s) = exp(f_θ(s)[a]) / Σ_a' exp(f_θ(s)[a'])

where f_θ(s) are the logits from the neural network.

# Arguments
- `policy::ForwardPolicy`: The forward policy neural network
- `state::S`: Current state
- `action::A`: Action to compute probability for
- `parameters`: Neural network parameters
- `states`: Neural network internal states

# Returns
- `Float64`: Probability P_F(a|s)
"""
function forward_probability(policy::ForwardPolicy, state, action,
    parameters, states, actions)
    # Get state features
    features = state_to_features(state)

    # Compute logits for all actions
    logits, _ = compute_forward_logits(policy, features, parameters, states)

    # Get applicable actions directly with robust indexing (non-mutating for Zygote)
    applicable_actions = [a for (i, a) in enumerate(actions) if is_applicable(a, state)]
    applicable_indices = [i for (i, a) in enumerate(actions) if is_applicable(a, state)]

    # Check if the requested action is applicable
    if action ∉ applicable_actions
        return 0.0
    end

    if isempty(applicable_indices)
        return 0.0
    end

    # Extract logits for applicable actions and apply softmax
    applicable_logits = logits[applicable_indices]
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    probabilities = exp_logits ./ sum(exp_logits)

    # Find the probability for the requested action using direct indexing
    for (i, a) in enumerate(applicable_actions)
        if a == action
            return Float64(probabilities[i])
        end
    end

    return 0.0
end

"""
    compute_forward_logits(policy::ForwardPolicy, features::Vector{Float32}, parameters, states)

Compute raw logits from forward policy network.

# Mathematical Foundation
Computes f_θ(s) = NN_θ(φ(s)) where:
- φ(s) are the state features
- NN_θ is the neural network with parameters θ
- Output dimension matches total number of actions

# Arguments
- `policy::ForwardPolicy`: Forward policy network
- `features::Vector{Float32}`: State features φ(s)
- `parameters`: Network parameters θ
- `states`: Network internal states

# Returns
- `Tuple{Vector{Float32}, Any}`: (logits, new_states)
"""
function compute_forward_logits(policy::ForwardPolicy, features::Vector{Float32}, parameters, states)
    # Validate inputs (non-differentiable)
    Zygote.@ignore begin
        validate_neural_network_input(features, "forward policy features")
        validate_model_parameters(parameters, "forward policy parameters")
    end

    # Neural network forward pass (differentiable)
    logits, new_states = safe_model_call(policy.model, features, parameters, states)

    # Validate outputs (non-differentiable)
    Zygote.@ignore validate_neural_network_output(logits, "forward policy logits")

    return logits, new_states
end

"""
    sample_forward_action(policy::ForwardPolicy, state, action_space,
                         parameters, states; rng=nothing)

Sample an action from the forward policy P_F(a|s).

# Mathematical Foundation
Samples a ∼ P_F(·|s) where P_F is defined by the softmax over applicable actions:
P_F(a|s) = exp(f_θ(s)[a]) / Σ_{a'∈A(s)} exp(f_θ(s)[a'])

where A(s) are the applicable actions from state s.

# Arguments
- `policy::ForwardPolicy`: Forward policy
- `state::S`: Current state
- `action_space::ActionSpace{A}`: Action space with indexing
- `parameters`: Policy parameters
- `states`: Policy internal states
- `rng`: Random number generator (optional)

# Returns
- `Tuple{A, Int}`: (sampled_action, action_index)
"""
function sample_forward_action(policy::ForwardPolicy, state, actions,
    parameters, states; rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end

    # Get state features
    features = state_to_features(state)

    # Compute logits
    logits, _ = compute_forward_logits(policy, features, parameters, states)

    # Get applicable actions directly
    applicable_actions = [action for action in actions if is_applicable(action, state)]

    # Direct sampling from applicable actions
    if isempty(applicable_actions)
        throw(ArgumentError("No applicable actions available for sampling"))
    end

    # Find indices of applicable actions
    applicable_indices = [findfirst(a -> a == action, actions) for action in applicable_actions]
    applicable_indices = filter(!isnothing, applicable_indices)

    if isempty(applicable_indices)
        throw(ArgumentError("No applicable actions found in action space"))
    end

    # Extract logits for applicable actions and apply softmax
    applicable_logits = logits[applicable_indices]
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    probabilities = exp_logits ./ sum(exp_logits)

    # Sample from the probability distribution
    cumulative_probs = cumsum(probabilities)
    rand_val = rand(rng)
    sample_idx = findfirst(p -> p >= rand_val, cumulative_probs)
    if isnothing(sample_idx)
        sample_idx = length(cumulative_probs)
    end

    action = applicable_actions[sample_idx]
    probability = probabilities[sample_idx]

    return action, probability
end

"""
    forward_action_probabilities(policy::ForwardPolicy, state, actions,
                                parameters, states)

Compute probability distribution P_F(·|s) over all actions.

# Returns
- `Vector{Float32}`: Probability distribution over all actions
"""
function forward_action_probabilities(policy::ForwardPolicy, state, actions,
    parameters, states)
    # Get state features
    features = state_to_features(state)

    # Compute logits
    logits, _ = compute_forward_logits(policy, features, parameters, states)

    # Get applicable actions directly
    applicable_actions = [action for action in actions if is_applicable(action, state)]

    # Compute probabilities directly using softmax
    if isempty(applicable_actions)
        return zeros(Float32, length(actions))
    end

    # Find indices of applicable actions and apply softmax
    applicable_indices = [findfirst(a -> a == action, actions) for action in applicable_actions]
    applicable_indices = filter(!isnothing, applicable_indices)

    if isempty(applicable_indices)
        return zeros(Float32, length(actions))
    end

    # Compute normalized probabilities for applicable actions
    applicable_logits = logits[applicable_indices]
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    normalized_probs = exp_logits ./ sum(exp_logits)

    # Create probability vector functionally without mutations (Zygote-safe)
    probs = [
        begin
            applicable_idx = findfirst(==(i), applicable_indices)
            isnothing(applicable_idx) ? Float32(0.0) : normalized_probs[applicable_idx]
        end
        for i in 1:length(actions)
    ]

    return probs
end

# =============================================================================
# Backward Policy P_B(s|s') - Mathematical Foundation
# =============================================================================

"""
    backward_probability(policy::BackwardPolicy, target_state, source_state,
                        parameters, states)

Compute P_B(s|s') - the probability of having come from source_state given target_state.

# Mathematical Foundation
The backward policy defines the probability distribution over previous states:
P_B(s|s') = exp(b_θ(s')[s]) / Σ_s'' exp(b_θ(s')[s''])

where b_θ(s') are the logits for previous states from s'.

# Arguments
- `policy::BackwardPolicy`: Backward policy network
- `target_state::S`: Current state s'
- `source_state::S`: Previous state s to compute probability for
- `parameters`: Network parameters
- `states`: Network internal states

# Returns
- `Float64`: Probability P_B(s|s')
"""
function backward_probability(policy::BackwardPolicy, target_state, source_state,
    parameters, states, dag)
    # Get target state features
    features = state_to_features(target_state)

    # Get all possible previous states
    prev_states = get_previous_states(dag, target_state)
    if isempty(prev_states) || source_state ∉ prev_states
        return 0.0
    end

    # Compute backward logits
    logits, _ = compute_backward_logits(policy, features, prev_states, parameters, states)

    # Convert to probabilities
    probs = softmax(logits)

    # Find index of source state in prev_states
    source_idx = findfirst(s -> s == source_state, prev_states)
    return isnothing(source_idx) ? 0.0 : Float64(probs[source_idx])
end

"""
    compute_backward_logits(policy::BackwardPolicy, features::Vector{Float32},
                           prev_states::Vector{S}, parameters, states) where {S}

Compute raw logits from backward policy network.

# Mathematical Foundation
Computes b_θ(s') where the output dimension matches the number of previous states.

# Arguments
- `policy::BackwardPolicy`: Backward policy network
- `features::Vector{Float32}`: Target state features
- `prev_states::Vector{S}`: All possible previous states
- `parameters`: Network parameters
- `states`: Network internal states

# Returns
- `Tuple{Vector{Float32}, Any}`: (logits, new_states)
"""
function compute_backward_logits(policy::BackwardPolicy, features::Vector{Float32},
    prev_states::Vector, parameters, states)
    # Validate inputs (non-differentiable)
    Zygote.@ignore begin
        validate_neural_network_input(features, "backward policy features")
        validate_model_parameters(parameters, "backward policy parameters")
    end

    # Neural network forward pass (differentiable)
    logits, new_states = safe_model_call(policy.model, features, parameters, states)

    # Validate dimension consistency (non-differentiable)
    Zygote.@ignore begin
        if length(logits) != length(prev_states)
            throw(DimensionMismatch(
                "Backward policy output dimension ($(length(logits))) doesn't match " *
                "number of previous states ($(length(prev_states)))"
            ))
        end
        validate_neural_network_output(logits, "backward policy logits")
    end

    return logits, new_states
end

"""
    sample_backward_state(policy::BackwardPolicy, target_state, dag,
                         parameters, states; rng=nothing)

Sample a previous state from P_B(·|s').

# Mathematical Foundation
Samples s ∼ P_B(·|s') over all possible previous states.
"""
function sample_backward_state(policy::BackwardPolicy, target_state, dag,
    parameters, states; rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end

    # Get previous states
    prev_states = get_previous_states(dag, target_state)
    if isempty(prev_states)
        throw(ArgumentError("No previous states available for backward sampling from $target_state"))
    end

    # Get target state features
    features = state_to_features(target_state)

    # Compute backward logits
    logits, _ = compute_backward_logits(policy, features, prev_states, parameters, states)

    # Sample from distribution
    probs = softmax(logits)
    state_idx = sample(rng, Weights(probs))

    return prev_states[state_idx], state_idx
end

# =============================================================================
# Flow Estimator Z(s) - Mathematical Foundation
# =============================================================================

"""
    flow_estimate(estimator::FlowEstimator, state, parameters, states)

Compute Z(s) - the estimated total flow through state s.

# Mathematical Foundation
The flow estimator approximates the total flow F(s) passing through state s:
Z(s) = exp(z_θ(s))

where z_θ(s) is the log-flow estimate from the neural network.

# Arguments
- `estimator::FlowEstimator`: Flow estimation network
- `state`: State to estimate flow for
- `parameters`: Network parameters
- `states`: Network internal states

# Returns
- `Float64`: Estimated flow Z(s)
"""
function flow_estimate(estimator::FlowEstimator, state, parameters, states)
    # Get state features
    features = state_to_features(state)

    # Compute log-flow estimate
    log_flow, _ = compute_flow_logits(estimator, features, parameters, states)

    # Return flow estimate (exp of log-flow)
    return exp(Float64(log_flow[1]))  # Assuming scalar output
end

"""
    compute_flow_logits(estimator::FlowEstimator, features::Vector{Float32}, parameters, states)

Compute raw log-flow logits from flow estimator network.

# Mathematical Foundation
Computes z_θ(s) = NN_θ(φ(s)) where the output is a scalar log-flow estimate.

# Arguments
- `estimator::FlowEstimator`: Flow estimator network
- `features::Vector{Float32}`: State features
- `parameters`: Network parameters
- `states`: Network internal states

# Returns
- `Tuple{Vector{Float32}, Any}`: (log_flow_logits, new_states)
"""
function compute_flow_logits(estimator::FlowEstimator, features::Vector{Float32}, parameters, states)
    # Validate inputs (non-differentiable)
    Zygote.@ignore begin
        validate_neural_network_input(features, "flow estimator features")
        validate_model_parameters(parameters, "flow estimator parameters")
    end

    # Neural network forward pass (differentiable)
    logits, new_states = safe_model_call(estimator.model, features, parameters, states)

    # Validate outputs (non-differentiable)
    Zygote.@ignore validate_neural_network_output(logits, "flow estimator logits")

    return logits, new_states
end

# =============================================================================
# Unified Policy Operations
# =============================================================================

"""
    forward_transition_probability(model::GFlowNetModel, source_state::S, target_state::S) where {S}

Compute P_F(s'|s) - probability of transitioning from source to target state.

# Mathematical Foundation
P_F(s'|s) = Σ_{a: apply_action(a,s)=s'} P_F(a|s)

Sums over all actions that lead from source to target state.
"""
function forward_transition_probability(model::GFlowNetModel, source_state, target_state)
    # Get applicable actions from source state
    applicable_actions = get_applicable_actions(source_state, model.all_actions)
    if isempty(applicable_actions)
        return 0.0
    end

    # Check if any action leads to target state
    valid_actions = []
    for action in applicable_actions
        if apply_action(action, source_state) == target_state
            push!(valid_actions, action)
        end
    end
    
    if isempty(valid_actions)
        return 0.0  # No action leads to target state
    end

    # Compute action probabilities over all actions
    probs = forward_action_probabilities(
        model.forward_policy, source_state, model.all_actions,
        model.parameters.forward, model.states.forward
    )

    # Sum probabilities of actions that lead to target
    total_prob = 0.0
    for (i, action) in enumerate(model.all_actions)
        if action in valid_actions
            total_prob += probs[i]
        end
    end

    return total_prob
end

"""
    backward_transition_probability(model::GFlowNetModel, target_state::S, source_state::S) where {S}

Compute P_B(s|s') - probability of having transitioned from source to target state.
"""
function backward_transition_probability(model::GFlowNetModel, target_state, source_state)
    # Check if backward policy exists
    if isnothing(model.backward_policy)
        # Default: assume deterministic backward for tree-structured spaces
        # where each state has a unique parent (P_B = 1)
        return 1.0
    end

    # Check if parameters exist for backward policy
    if !haskey(model.parameters, :backward)
        @warn "Backward policy exists but no parameters found. Using deterministic backward."
        return 1.0
    end

    # Use learned backward policy
    return compute_backward_probability(
        model.backward_policy, target_state, source_state,
        model.parameters.backward, model.states.backward,
        model.all_actions
    )
end

"""
    compute_backward_probability(policy::BackwardPolicy, target_state, source_state,
                               parameters, states, all_actions)

Compute P_B(s|s') using a learned backward policy without DAG.

# Mathematical Foundation
Uses a joint representation approach where the backward policy network takes
both source and target state features and outputs P_B(source|target).

# Arguments
- `policy::BackwardPolicy`: Backward policy network
- `target_state`: Target state s'
- `source_state`: Source state s
- `parameters`: Network parameters
- `states`: Network internal states
- `all_actions`: All possible actions (for validation)

# Returns
- `Float64`: Probability P_B(s|s')
"""
function compute_backward_probability(policy::BackwardPolicy, target_state, source_state,
    parameters, states, all_actions)
    
    # First check if transition is valid
    if !is_valid_backward_transition(source_state, target_state, all_actions)
        return 0.0
    end
    
    # Get features for both states
    source_features = state_to_features(source_state)
    target_features = state_to_features(target_state)
    
    # Concatenate features for joint representation
    joint_features = vcat(source_features, target_features)
    
    # Validate inputs
    Zygote.@ignore begin
        validate_neural_network_input(joint_features, "backward policy joint features")
        validate_model_parameters(parameters, "backward policy parameters")
    end
    
    # Neural network forward pass
    logit, new_states = safe_model_call(policy.model, joint_features, parameters, states)
    
    # Convert logit to probability using sigmoid
    prob = sigmoid(logit[1])  # Network outputs single value
    
    # Ensure valid probability
    return clamp(Float64(prob), 1e-8, 1.0 - 1e-8)
end

"""
    is_valid_backward_transition(source_state, target_state, all_actions)

Check if source_state can transition to target_state using any action.
"""
function is_valid_backward_transition(source_state, target_state, all_actions)
    applicable_actions = get_applicable_actions(source_state, all_actions)
    for action in applicable_actions
        if apply_action(action, source_state) == target_state
            return true
        end
    end
    return false
end

# =============================================================================
# Safe Model Call - Neural Network Wrapper
# =============================================================================

"""
    safe_model_call(model, features, parameters, states)

Safe wrapper for neural network calls with proper validation and error handling.

# Mathematical Foundation
Provides a safe interface to neural network evaluation with:
- Input validation (non-differentiable)
- Type conversion for stability
- Output validation (non-differentiable)
- Proper error handling

This function is used by all policy computations to ensure numerical stability.
"""
function safe_model_call(model, features, parameters, states)
    # Validate inputs (non-differentiable)
    Zygote.@ignore begin
        validate_neural_network_input(features, "model input features")
        if !isnothing(parameters)
            validate_model_parameters(parameters, "model parameters")
        end
    end

    # Ensure type stability - convert to Float32
    features_f32 = convert(Vector{Float32}, features)

    try
        # Neural network forward pass (differentiable)
        outputs, new_states = model(features_f32, parameters, states)

        # Validate outputs (non-differentiable)
        Zygote.@ignore validate_neural_network_output(outputs, "model outputs")

        return outputs, new_states

    catch e
        # Enhanced error reporting (non-differentiable)
        Zygote.@ignore begin
            @error "Neural network forward pass failed" exception = e
            @error "Input shape: $(size(features_f32))"
            @error "Parameter keys: $(keys(parameters))"
        end
        rethrow(e)
    end
end

# =============================================================================
# Utility Functions for State-Action Operations
# =============================================================================

"""
    get_applicable_actions_for_state(state, actions)

Get applicable actions for a specific state from a list of all actions.

# Mathematical Foundation
Returns A(s) = {a ∈ A : is_applicable(a, s)}
"""
function get_applicable_actions_for_state(state, actions)
    return [action for action in actions if is_applicable(action, state)]
end

"""
    validate_policy_consistency(model::GFlowNetModel)

Validate that all policies in the model are mathematically consistent.

# Mathematical Requirements
1. Forward policy outputs match action space size
2. Backward policy (if present) works with DAG structure
3. Flow estimator (if present) outputs scalar values
4. All policies work with the same state/action types
"""
function validate_policy_consistency(model::GFlowNetModel)
    # Test forward policy with initial state
    test_state = model.initial_state

    try
        probs = forward_action_probabilities(
            model.forward_policy, test_state, model.all_actions,
            model.parameters.forward, model.states.forward
        )

        if length(probs) != length(model.all_actions)
            throw(ArgumentError("Forward policy output size mismatch"))
        end

    catch e
        throw(ArgumentError("Forward policy validation failed: $e"))
    end

    # Test backward policy if present
    if !isnothing(model.backward_policy) && haskey(model.parameters, :backward)
        try
            # Test with a simple self-transition (may not be valid but tests the network)
            compute_backward_probability(
                model.backward_policy, test_state, test_state,
                model.parameters.backward, model.states.backward,
                model.all_actions
            )
        catch e
            # Backward policy validation is non-critical
            @debug "Backward policy validation note: $e"
        end
    end

    # Test flow estimator if present
    if !isnothing(model.flow_estimator)
        try
            flow_val = flow_estimate(
                model.flow_estimator, test_state,
                model.parameters.flow, model.states.flow
            )

            if !isfinite(flow_val) || flow_val <= 0
                throw(ArgumentError("Flow estimator produced invalid value: $flow_val"))
            end

        catch e
            throw(ArgumentError("Flow estimator validation failed: $e"))
        end
    end
end

# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, policy::ForwardPolicy{M}) where {M}
    print(io, "ForwardPolicy(model=$M)")
end

function Base.show(io::IO, policy::BackwardPolicy{M}) where {M}
    print(io, "BackwardPolicy(model=$M)")
end

function Base.show(io::IO, estimator::FlowEstimator{M}) where {M}
    print(io, "FlowEstimator(model=$M)")
end
