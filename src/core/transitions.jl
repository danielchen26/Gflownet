# Core Transition Operations for GFlowNet
# This file implements fundamental operations connecting states, features, and flows

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
    features = Float32.(state.data)
    
    # Validate features before returning
    validate_state_features(features, "SimpleState features")
    
    return features
end

# =============================================================================
# Forward Transition Probabilities
# =============================================================================

"""
    forward_transition_prob(model::GFlowNetModel, source::AbstractState, target::AbstractState)

Compute the forward transition probability from source to target state.
"""
function forward_transition_prob(model::GFlowNetModel, source::AbstractState, target::AbstractState)
    # Use the forward policy to compute transition probability
    features = state_to_features(source)
    
    # Use safe model call helper
    logits, _ = safe_model_call(
        model.forward_policy.model,
        features,
        model.parameters.forward,
        model.states.forward
    )
    
    # Get all possible next states from source
    next_states = get_next_states(model.dag, source)
    if isempty(next_states) || target ∉ next_states
        return 0.0
    end
    
    # Apply softmax to get probabilities
    probs = softmax(logits)
    
    # Find index of target state and return probability
    target_index = findfirst(s -> s == target, next_states)
    return isnothing(target_index) ? 0.0 : Float64(probs[target_index])
end

"""
    backward_transition_prob(model::GFlowNetModel, target::AbstractState, source::AbstractState)

Compute the backward transition probability from target back to source state.
"""
function backward_transition_prob(model::GFlowNetModel, target::AbstractState, source::AbstractState)
    # Convert target state to features
    features = state_to_features(target)
    
    # Use safe model call helper
    logits, _ = safe_model_call(
        model.backward_policy.model,
        features,
        model.parameters.backward,
        model.states.backward
    )
    
    # Get all possible previous states to target
    prev_states = get_previous_states(model.dag, target)
    if isempty(prev_states) || source ∉ prev_states
        return 0.0
    end
    
    # Apply softmax to get probabilities
    probs = softmax(logits)
    
    # Find index of source state and return probability
    source_index = findfirst(s -> s == source, prev_states)
    return isnothing(source_index) ? 0.0 : Float64(probs[source_index])
end

# =============================================================================
# Flow Computations
# =============================================================================

"""
    flow(model::GFlowNetModel, state::AbstractState)

Compute the flow F(s) for a given state using the model's flow estimator.
Includes memoization to prevent infinite recursion and improve performance.

The flow represents the total amount of "probability mass" flowing through a state.
For terminal states, F(s) = R(s). For non-terminal states, flow is estimated
using the flow estimator network or computed recursively.

# Mathematical Foundation
- Terminal states: F(s) = R(s)
- Non-terminal states: F(s) = Σ_{s_prev} F(s_prev) * P_F(s_prev→s) (incoming flow)
- Or: F(s) = flow_estimator(φ(s)) (learned flow)

# Arguments
- `model`: GFlowNet model containing the flow estimator
- `state`: State to compute flow for

# Returns
- Flow value F(s) > 0
"""
function flow(model::GFlowNetModel, state::AbstractState)
    # Check if this is a terminal state
    if is_terminal_state(state)
        return reward(state)
    end
    
    # Use flow estimator if available
    if !isnothing(model.flow_estimator)
        return compute_flow_estimate(model, model.flow_estimator, state)
    else
        # Fallback: compute flow recursively with memoization
        return compute_recursive_flow_memoized(model, state)
    end
end

# Use thread-local cache instead of global cache to avoid thread safety issues
const FLOW_CACHE = Ref{Dict{Tuple{Any, Any}, Float64}}()

"""
    compute_recursive_flow_memoized(model::GFlowNetModel, state::AbstractState)

Memoized version of recursive flow computation to prevent infinite recursion
and improve performance for repeated computations.
"""
function compute_recursive_flow_memoized(model::GFlowNetModel, state::AbstractState)
    # Initialize thread-local cache if needed
    if !isassigned(FLOW_CACHE)
        FLOW_CACHE[] = Dict{Tuple{Any, Any}, Float64}()
    end
    
    # Create cache key
    cache_key = (objectid(model), objectid(state))
    
    # Check if already computed in thread-local cache
    if haskey(FLOW_CACHE[], cache_key)
        return FLOW_CACHE[][cache_key]
    end
    
    # Compute flow and cache result
    flow_value = compute_recursive_flow(model, state)
    FLOW_CACHE[][cache_key] = flow_value
    
    return flow_value
end

"""
    clear_flow_cache!()

Clear the flow computation cache. Should be called when model parameters change.
"""
function clear_flow_cache!()
    # Clear thread-local cache without locking
    if isassigned(FLOW_CACHE)
        empty!(FLOW_CACHE[])
    end
end

"""
    compute_flow_estimate(model::GFlowNetModel, flow_estimator::FlowEstimator, state::AbstractState)

Compute flow estimate using the flow estimator network.

This function converts the state to features and passes them through
the flow estimator to get F(s).
"""
function compute_flow_estimate(model::GFlowNetModel, flow_estimator::FlowEstimator, state::AbstractState)
    # Convert state to features
    features = state_to_features(state)

    # Extract flow parameters and states from the model
    flow_params = if isa(model.parameters, ComponentArray) && haskey(model.parameters, :flow)
        model.parameters.flow
    elseif isa(model.parameters, NamedTuple) && haskey(model.parameters, :flow)
        model.parameters.flow
    else
        error("Flow parameters not found in model.parameters")
    end

    flow_states = if haskey(model.states, :flow) && !isnothing(model.states.flow)
        model.states.flow
    else
        error("Flow states not found in model.states")
    end

    # Use the extracted parameters and states
    flow_value = compute_flow_logits(flow_estimator, features, flow_params, flow_states)

    # Ensure positive flow
    return exp(flow_value) + 1e-6
end

"""
    compute_recursive_flow(model::GFlowNetModel, state::AbstractState)

Compute flow recursively using proper GFlowNet flow conservation equations.

This implements the fundamental GFlowNet flow equations:
- Terminal states: F(s) = R(s) 
- Non-terminal states: F(s) = Σ_{s'} F(s') * P_B(s'→s)

# Arguments
- `model`: GFlowNet model containing DAG and policies
- `state`: State to compute flow for

# Returns
- Flow value computed using proper conservation equations
"""
function compute_recursive_flow(model::GFlowNetModel, state::AbstractState)
    # Check if state is terminal using proper interface
    if is_terminal_state(state)
        # Terminal states: F(s) = R(s)
        return reward(state)
    end
    
    # Check if this is the initial state
    if state == model.dag.initial_state
        # Initial state flow equals partition function
        if !isnothing(model.partition_function)
            return model.partition_function
        else
            # Estimate partition function if not available
            return estimate_partition_function(model)
        end
    end
    
    # Non-terminal states - use incoming flow computation
    # F(s) = Σ_{s_prev} F(s_prev) * P_F(s_prev→s) where s_prev are previous states
    total_flow = 0.0
    prev_states = get_previous_states(model.dag, state)
    
    if isempty(prev_states)
        # No incoming edges - this shouldn't happen unless it's the initial state
        @warn "Non-terminal, non-initial state with no incoming edges: $state"
        return 1e-6
    end
    
    for prev_state in prev_states
        # Recursively compute flow of previous state
        prev_flow = flow(model, prev_state)
        
        # Get forward probability P_F(s_prev→s)
        forward_prob = forward_transition_prob(model, prev_state, state)
        
        # Add contribution: F(s_prev) * P_F(s_prev→s)
        total_flow += prev_flow * forward_prob
    end
    
    return max(total_flow, 1e-6)  # Ensure positive flow
end

"""
    edge_flow(model::GFlowNetModel, source_state::AbstractState, target_state::AbstractState)

Compute the flow along a specific edge F(s → s').

This is computed as: F(s → s') = F(s) * P_F(s' | s)

# Arguments
- `model`: GFlowNet model
- `source_state`: Source state of the edge
- `target_state`: Target state of the edge

# Returns
- Edge flow value
"""
function edge_flow(model::GFlowNetModel, source_state::AbstractState, target_state::AbstractState)
    source_flow = flow(model, source_state)
    forward_prob = forward_transition_prob(model, source_state, target_state)
    return source_flow * forward_prob
end

# =============================================================================
# Policy Network Interface Functions
# =============================================================================

"""
    compute_forward_logits(policy::ForwardPolicy, features::Vector{Float32}, actions::Vector, parameters, states)

Compute logits for forward actions given state features using Lux.jl neural network.
"""
function compute_forward_logits(policy::ForwardPolicy, features::Vector{Float32}, actions::Vector, parameters, states)
    n_actions = length(actions)
    if n_actions == 0
        return Float32[]
    end

    # Reshape features for neural network input
    input_features = reshape(features, :, 1)  # (feature_dim, batch_size)

    # Use proper Lux API for model application
    raw_output, new_states = policy.model(input_features, parameters, states)

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

    # Use proper Lux API for model application
    raw_output, new_states = policy.model(input_features, parameters, states)

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

"""
    compute_flow_logits(flow_estimator::FlowEstimator, features::Vector{Float32}, parameters, states)

Compute log flow using the flow estimator model with Lux.jl neural network.
"""
function compute_flow_logits(flow_estimator::FlowEstimator, features::Vector{Float32}, parameters, states)
    # Reshape features for neural network input
    input_features = reshape(features, :, 1)  # (feature_dim, batch_size)

    # Use proper Lux API for model application
    raw_output, new_states = flow_estimator.model(input_features, parameters, states)

    # Extract scalar log flow value
    return Float32(raw_output[1, 1])
end 