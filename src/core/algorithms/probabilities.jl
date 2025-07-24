# Core Probability Functions for GFlowNet
# This file implements the fundamental probability computations that connect
# policies to the mathematical framework of GFlowNets

using NNlib: softmax
using ComponentArrays
using Lux

# =============================================================================
# State Feature Extraction
# =============================================================================

"""
    state_to_features(state::AbstractState)

Convert a state to a feature vector that can be used as input to neural networks.
This function must be implemented for each concrete state type.

For SimpleState, we use the data vector directly.
"""
function state_to_features(state::SimpleState)
    return Float32.(state.data)
end

# Generic fallback that requires implementation for other state types
function state_to_features(state::AbstractState)
    error("state_to_features not implemented for state type $(typeof(state)). " *
          "Please implement this function for your specific state type.")
end

# =============================================================================
# Forward Transition Probabilities
# =============================================================================

"""
    forward_transition_prob(model::GFlowNetModel, source_state::AbstractState, target_state::AbstractState)

Compute the forward transition probability P_F(target_state | source_state) using the model's forward policy.

This is a core function that connects the policy network to the GFlowNet mathematical framework.
The probability is computed by:
1. Converting the source state to features
2. Computing action logits using the forward policy
3. Converting to probabilities via softmax
4. Selecting the probability for the action that leads to target_state

# Mathematical Foundation
P_F(s' | s) = softmax(f_θ(φ(s)))[a] where a is the action s.t. apply_action(a, s) = s'
"""
function forward_transition_prob(model::GFlowNetModel, source_state::AbstractState, target_state::AbstractState)
    # Convert source state to features
    features = state_to_features(source_state)
    
    # Get all possible actions from this state
    possible_actions = get_possible_actions(model.dag, source_state)
    
    if isempty(possible_actions)
        return 0.0  # No actions possible from this state
    end
    
    # Compute action logits using the forward policy
    # Extract forward parameters and states from the model
    forward_params = if isa(model.parameters, ComponentArray) && haskey(model.parameters, :forward)
        model.parameters.forward
    elseif isa(model.parameters, NamedTuple) && haskey(model.parameters, :forward)
        model.parameters.forward
    else
        error("Forward parameters not found in model.parameters")
    end

    forward_states = if haskey(model.states, :forward) && !isnothing(model.states.forward)
        model.states.forward
    else
        error("Forward states not found in model.states")
    end

    logits = compute_forward_logits(model.forward_policy, features, possible_actions, forward_params, forward_states)
    
    # Convert to probabilities
    probs = softmax(logits)
    
    # Find which action leads to the target state
    for (i, action) in enumerate(possible_actions)
        if is_applicable(action, source_state)
            next_state = apply_action(action, source_state)
            if next_state == target_state
                return probs[i]
            end
        end
    end
    
    return 0.0  # No action leads to target state
end

# =============================================================================
# Backward Transition Probabilities
# =============================================================================

"""
    backward_transition_prob(model::GFlowNetModel, source_state::AbstractState, target_state::AbstractState)

Compute the backward transition probability P_B(source_state | target_state) using the model's backward policy.

This function is only available if the model has a backward policy.

# Mathematical Foundation
P_B(s | s') = softmax(g_θ(φ(s')))[a] where a is the action s.t. apply_action(a, s) = s'
"""
function backward_transition_prob(model::GFlowNetModel, source_state::AbstractState, target_state::AbstractState)
    if isnothing(model.backward_policy)
        error("Backward transition probability requested but model has no backward policy")
    end
    
    # Convert target state to features
    features = state_to_features(target_state)
    
    # Get all possible previous states that could lead to target_state
    possible_prev_states = get_previous_states(model.dag, target_state)
    
    if isempty(possible_prev_states)
        return 0.0  # No previous states possible
    end
    
    # Compute backward logits using the backward policy
    # Extract backward parameters and states from the model
    backward_params = if isa(model.parameters, ComponentArray) && haskey(model.parameters, :backward)
        model.parameters.backward
    elseif isa(model.parameters, NamedTuple) && haskey(model.parameters, :backward)
        model.parameters.backward
    else
        error("Backward parameters not found in model.parameters")
    end

    backward_states = if haskey(model.states, :backward) && !isnothing(model.states.backward)
        model.states.backward
    else
        error("Backward states not found in model.states")
    end

    logits = compute_backward_logits(model.backward_policy, features, possible_prev_states, backward_params, backward_states)
    
    # Convert to probabilities
    probs = softmax(logits)
    
    # Find which previous state matches the source
    for (i, prev_state) in enumerate(possible_prev_states)
        if prev_state == source_state
            return probs[i]
        end
    end
    
    return 0.0  # Source state not in possible previous states
end

# =============================================================================
# Policy Network Interface Functions
# =============================================================================

"""
    compute_forward_logits(policy::ForwardPolicy, features::Vector{Float32}, actions::Vector, parameters, states)

Compute logits for forward actions given state features using Lux.jl neural network.
This function requires proper parameters and states from the model.
"""
function compute_forward_logits(policy::ForwardPolicy, features::Vector{Float32}, actions::Vector, parameters, states)
    n_actions = length(actions)
    if n_actions == 0
        return Float32[]
    end

    # Reshape features for neural network input
    input_features = reshape(features, :, 1)  # (feature_dim, batch_size)

    # Apply the neural network model with proper parameters and states
    raw_output, _ = Lux.apply(policy.model, input_features, parameters, states)

    # Extract the required number of action logits
    if size(raw_output, 1) >= n_actions
        return Float32.(raw_output[1:n_actions, 1])
    else
        # If the network doesn't output enough logits, pad with zeros
        padded = zeros(Float32, n_actions)
        padded[1:size(raw_output, 1)] = raw_output[:, 1]
        return padded
    end
end

"""
    compute_backward_logits(policy::BackwardPolicy, features::Vector{Float32}, prev_states::Vector, parameters, states)

Compute logits for backward transitions given state features using Lux.jl neural network.
"""
function compute_backward_logits(policy::BackwardPolicy, features::Vector{Float32}, prev_states::Vector, parameters, states)
    n_states = length(prev_states)
    if n_states == 0
        return Float32[]
    end

    # Reshape features for neural network input
    input_features = reshape(features, :, 1)

    # Apply the neural network model with proper parameters and states
    raw_output, _ = Lux.apply(policy.model, input_features, parameters, states)

    # Extract the required number of state logits
    if size(raw_output, 1) >= n_states
        return Float32.(raw_output[1:n_states, 1])
    else
        # If the network doesn't output enough logits, pad with zeros
        padded = zeros(Float32, n_states)
        padded[1:size(raw_output, 1)] = raw_output[:, 1]
        return padded
    end
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    get_possible_actions(dag::DirectedAcyclicGraph, state::AbstractState)

Get all possible actions from a given state in the DAG.
"""
function get_possible_actions(dag::DirectedAcyclicGraph, state::AbstractState)
    actions = typeof(dag.actions[1])[]  # Empty vector of correct action type

    for action in dag.actions
        if is_applicable(action, state)
            push!(actions, action)
        end
    end

    return actions
end

"""
    get_next_states(dag::DirectedAcyclicGraph, state::AbstractState)

Get all possible next states reachable from a given state.
"""
function get_next_states(dag::DirectedAcyclicGraph, state::AbstractState)
    next_states = typeof(dag.states[1])[]  # Empty vector of correct state type

    for action in dag.actions
        if is_applicable(action, state)
            next_state = apply_action(action, state)
            if !(next_state in next_states)  # Avoid duplicates
                push!(next_states, next_state)
            end
        end
    end

    return next_states
end

"""
    get_previous_states(dag::DirectedAcyclicGraph, state::AbstractState)

Get all possible previous states that could lead to a given state.
This is computed by checking all states in the DAG and seeing which ones
can reach the target state through some action.
"""
function get_previous_states(dag::DirectedAcyclicGraph, state::AbstractState)
    prev_states = typeof(dag.states[1])[]  # Empty vector of correct state type

    for prev_state in dag.states
        for action in dag.actions
            if is_applicable(action, prev_state)
                next_state = apply_action(action, prev_state)
                if next_state == state
                    if !(prev_state in prev_states)  # Avoid duplicates
                        push!(prev_states, prev_state)
                    end
                end
            end
        end
    end

    return prev_states
end
