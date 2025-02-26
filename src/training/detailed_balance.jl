using ..GFlowNet: GFlowNetModel, flow, forward_transition_prob, backward_transition_prob, get_next_states

"""
    detailed_balance_loss(model::GFlowNetModel)

Compute the Detailed Balance loss for the GFlowNet.

The detailed balance constraint requires that for each edge (s, s') in the graph,
the forward flow equals the backward flow:
    F(s) * P_F(s → s') = F(s') * P_B(s' → s)
where F is the flow function, P_F is the forward policy, and P_B is the backward policy.
"""
function detailed_balance_loss(model::GFlowNetModel)
    total_loss = 0.0
    
    # For each edge in the graph
    for source in model.dag.states
        for target in get_next_states(model.dag, source)
            # Skip edges to terminal sink
            if target == model.dag.terminal_sink
                continue
            end
            
            # Forward flow
            forward_flow = flow(model, source) * forward_transition_prob(model, source, target)
            
            # Backward flow
            backward_flow = flow(model, target) * backward_transition_prob(model, target, source)
            
            # Squared difference (can also use log ratio)
            diff = forward_flow - backward_flow
            total_loss += diff^2
        end
    end
    
    return total_loss
end

"""
    detailed_balance_loss_grad(model::GFlowNetModel)

Compute the gradient of the Detailed Balance loss for optimization.
"""
function detailed_balance_loss_grad(model::GFlowNetModel)
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
    return gradient(ps -> detailed_balance_loss(model, ps), ps)[1]
end

"""
    detailed_balance_loss(model::GFlowNetModel, ps)
    
Compute the Detailed Balance loss with explicitly provided parameters.
This is useful for automatic differentiation.
"""
function detailed_balance_loss(model::GFlowNetModel, ps)
    # Create a temporary model with updated parameters
    temp_model = deepcopy(model)
    
    # Update parameters
    # This would depend on how parameters are stored and updated
    # For Lux, something like:
    # temp_model.forward_policy.model = Lux.update_params(temp_model.forward_policy.model, ps)
    
    return detailed_balance_loss(temp_model)
end 