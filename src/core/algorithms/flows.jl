# Flow Estimation for GFlowNets
# This file implements flow computation F(s) for states in the GFlowNet

using ComponentArrays
using Lux

"""
    flow(model::GFlowNetModel, state::AbstractState)

Compute the flow F(s) for a given state using the model's flow estimator.

The flow represents the total amount of "probability mass" flowing through a state.
For terminal states, F(s) = R(s). For non-terminal states, flow is estimated
using the flow estimator network or computed recursively.

# Mathematical Foundation
- Terminal states: F(s) = R(s)
- Non-terminal states: F(s) = Σ_{s'} F(s') * P_B(s|s') (backward flow)
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
        # Fallback: compute flow recursively using backward transitions
        return compute_recursive_flow(model, state)
    end
end

"""
    is_terminal_state(state::AbstractState)

Check if a state is terminal (no outgoing transitions possible).

For SimpleState, we consider states with data [-1] as terminal sink states,
and states where no actions are applicable as terminal.
"""
function is_terminal_state(state::AbstractState)
    if isa(state, SimpleState)
        # Terminal sink state
        if state.data == [-1]
            return true
        end
        
        # Check if any actions are applicable
        # For SimpleState, if sum >= 10 and only increment is possible, it's terminal
        # If sum <= 0 and only decrement is possible, it's terminal
        data_sum = sum(state.data)
        
        # Check if terminate action (value -1) is the only applicable action
        # This is a simplified check - in practice, you'd check against the DAG
        if data_sum >= 10
            return true  # Can only terminate
        end
        
        return false
    else
        # For other state types, this needs to be implemented
        error("is_terminal_state not implemented for state type $(typeof(state))")
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

    flow_value = compute_flow_logits(flow_estimator, features, flow_params, flow_states)

    # Ensure positive flow
    return exp(flow_value) + 1e-6
end

"""
    compute_flow_logits(flow_estimator::FlowEstimator, features::Vector{Float32}, parameters, states)

Compute log flow using the flow estimator model with Lux.jl neural network.
"""
function compute_flow_logits(flow_estimator::FlowEstimator, features::Vector{Float32}, parameters, states)
    # Reshape features for neural network input
    input_features = reshape(features, :, 1)  # (feature_dim, batch_size)

    # Apply the neural network model with proper parameters and states
    raw_output, _ = Lux.apply(flow_estimator.model, input_features, parameters, states)

    # Extract scalar log flow value
    return Float32(raw_output[1, 1])
end

"""
    compute_recursive_flow(model::GFlowNetModel, state::AbstractState)

Compute flow recursively using the flow conservation equation.

This is a fallback method when no flow estimator is available.
For SimpleState, we use a simple heuristic to avoid infinite recursion.
"""
function compute_recursive_flow(model::GFlowNetModel, state::AbstractState)
    # For SimpleState, use a simple heuristic based on the state data
    if isa(state, SimpleState)
        # Simple heuristic: flow decreases as we move away from initial state
        data_sum = sum(abs.(state.data))
        base_flow = 2.0  # Base flow value

        # Flow decreases exponentially with distance from origin
        flow_value = base_flow * exp(-data_sum * 0.1)

        return max(flow_value, 1e-6)
    else
        # For other state types, this needs proper implementation
        # For now, return a default value to avoid recursion
        return 1.0
    end
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
