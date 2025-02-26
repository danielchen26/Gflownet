using ..GFlowNet: GFlowNetModel, AbstractGFlowNetObjective, FlowMatchingObjective, 
                 DetailedBalanceObjective, TrajectoryBalanceObjective, sample_trajectory

"""
    train!(model::GFlowNetModel, n_iterations::Int; 
           batch_size::Int=32, verbose::Bool=true)

Train a GFlowNet model for a specified number of iterations.
"""
function train!(model::GFlowNetModel, n_iterations::Int; 
                batch_size::Int=32, verbose::Bool=true)
    
    losses = []
    
    for iter in 1:n_iterations
        # Sample trajectories if needed
        trajectories = [sample_trajectory(model) for _ in 1:batch_size]
        
        # Compute total loss and gradient
        total_loss, total_grad = compute_loss_and_grad(model, trajectories)
        
        # Apply optimizer
        apply_optimizer!(model, total_grad)
        
        push!(losses, total_loss)
        
        # Re-estimate partition function periodically
        if iter % 10 == 0
            model.partition_function = estimate_partition_function(model)
        end
        
        if verbose && (iter % 10 == 0 || iter == 1)
            @info "Iteration $iter: Loss = $(total_loss)"
        end
    end
    
    return losses
end

"""
    compute_loss_and_grad(model::GFlowNetModel, trajectories)

Compute the total loss and gradient for all objectives.
"""
function compute_loss_and_grad(model::GFlowNetModel, trajectories)
    total_loss = 0.0
    total_grad = nothing
    
    for objective in model.objectives
        # Compute loss and gradient based on objective type
        if objective isa FlowMatchingObjective
            loss = flow_matching_loss(model) * objective.weight
            grad = flow_matching_loss_grad(model)
            
            total_loss += loss
            total_grad = update_gradient(total_grad, grad, objective.weight)
            
        elseif objective isa DetailedBalanceObjective
            loss = detailed_balance_loss(model) * objective.weight
            grad = detailed_balance_loss_grad(model)
            
            total_loss += loss
            total_grad = update_gradient(total_grad, grad, objective.weight)
            
        elseif objective isa TrajectoryBalanceObjective
            loss = trajectory_balance_loss(model, trajectories) * objective.weight
            grad = trajectory_balance_loss_grad(model, trajectories)
            
            total_loss += loss
            total_grad = update_gradient(total_grad, grad, objective.weight)
        end
    end
    
    return total_loss, total_grad
end

"""
    update_gradient(total_grad, grad, weight)

Update the total gradient with a new weighted gradient.
"""
function update_gradient(total_grad, grad, weight)
    if isnothing(total_grad)
        return grad .* weight
    else
        return total_grad .+ grad .* weight
    end
end

"""
    apply_optimizer!(model::GFlowNetModel, grad)

Apply the optimizer to update model parameters using the gradient.
"""
function apply_optimizer!(model::GFlowNetModel, grad)
    # This depends on the specific optimizer and parameter structure
    # For Lux + Optimisers.jl, something like:
    
    # Update forward policy
    model.forward_policy.model, model.optimizer.forward = 
        Optimisers.update(model.optimizer.forward, model.forward_policy.model, grad.forward)
    
    # Update backward policy if it exists
    if !isnothing(model.backward_policy)
        model.backward_policy.model, model.optimizer.backward = 
            Optimisers.update(model.optimizer.backward, model.backward_policy.model, grad.backward)
    end
    
    # Update flow estimator if it exists
    if !isnothing(model.flow_estimator)
        model.flow_estimator.model, model.optimizer.flow = 
            Optimisers.update(model.optimizer.flow, model.flow_estimator.model, grad.flow)
    end
end

# Import training objectives
include("flow_matching.jl")
include("detailed_balance.jl")
include("trajectory_balance.jl") 