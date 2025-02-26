using ..GFlowNet: GFlowNetModel, edge_flow, get_previous_states, get_next_states

"""
    flow_matching_loss(model::GFlowNetModel)

Compute the Flow Matching loss for the GFlowNet.

This loss ensures flow consistency at each state (excluding initial and terminal states).
The flow into each state must equal the flow out of that state.
"""
function flow_matching_loss(model::GFlowNetModel)
    total_loss = 0.0
    
    # Exclude initial and terminal states
    non_terminal_states = filter(s -> s != model.dag.initial_state && 
                                 s ∉ model.dag.terminal_states && 
                                 s != model.dag.terminal_sink,
                                 model.dag.states)
    
    for state in non_terminal_states
        # Sum of incoming flows
        incoming_flow = sum(edge_flow(model, prev, state) 
                           for prev in get_previous_states(model.dag, state))
        
        # Sum of outgoing flows
        outgoing_flow = sum(edge_flow(model, state, next) 
                           for next in get_next_states(model.dag, state))
        
        # Squared difference
        diff = incoming_flow - outgoing_flow
        total_loss += diff^2
    end
    
    return total_loss
end

"""
    flow_matching_loss_grad(model::GFlowNetModel)

Compute the gradient of the Flow Matching loss for optimization.
"""
function flow_matching_loss_grad(model::GFlowNetModel)
    # Implementation depends on the AD system being used
    # For Lux + Zygote, we would use something like:
    
    # Extract parameters
    ps = Lux.ComponentArray(model.forward_policy.model)
    if !isnothing(model.backward_policy)
        ps = vcat(ps, Lux.ComponentArray(model.backward_policy.model))
    end
    if !isnothing(model.flow_estimator)
        ps = vcat(ps, Lux.ComponentArray(model.flow_estimator.model))
    end
    
    # Compute gradient using Zygote
    return gradient(ps -> flow_matching_loss(model, ps), ps)[1]
end

"""
    flow_matching_loss(model::GFlowNetModel, ps)
    
Compute the Flow Matching loss with explicitly provided parameters.
This is useful for automatic differentiation.
"""
function flow_matching_loss(model::GFlowNetModel, ps)
    # Create a temporary model with updated parameters
    temp_model = deepcopy(model)
    
    # Update parameters
    # This would depend on how parameters are stored and updated
    # For Lux, something like:
    # temp_model.forward_policy.model = Lux.update_params(temp_model.forward_policy.model, ps)
    
    return flow_matching_loss(temp_model)
end 