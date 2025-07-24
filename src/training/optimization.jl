# Optimization and Gradient Computation for GFlowNet Training
# This file contains functions for computing gradients and updating model parameters

using ..GFlowNet: GFlowNetModel, Trajectory
using Zygote
using Optimisers

# Import objective functions for gradient computation
using ..GFlowNet: trajectory_balance_loss, trajectory_balance_loss_grad, general_trajectory_balance_loss_grad
using ..GFlowNet: sub_trajectory_balance_loss_grad, flow_consistency_loss_grad
using ..GFlowNet: FlowConsistencyMode, STATE_LEVEL

"""
    update_model_parameters!(model::GFlowNetModel, config::TrainingConfig, trajectories::Vector{<:Trajectory})

Update model parameters based on gradients computed for the given trajectories.

# Arguments
- `model`: GFlowNet model to update
- `config`: Training configuration
- `trajectories`: Batch of trajectories for gradient computation
"""
function update_model_parameters!(model::GFlowNetModel, config::TrainingConfig, trajectories::Vector{<:Trajectory})
    # Compute gradients based on objective
    if config.objective == TRAJECTORY_BALANCE
        grad = trajectory_balance_loss_grad(model, trajectories)
    elseif config.objective == GENERAL_TRAJECTORY_BALANCE
        grad = general_trajectory_balance_loss_grad(model, trajectories)
    elseif config.objective in [SUB_TRAJECTORY_BALANCE, HIERARCHICAL_SUB_TB, ADAPTIVE_SUB_TB]
        grad = sub_trajectory_balance_loss_grad(model, trajectories)
    elseif config.objective == FLOW_CONSISTENCY
        mode = get(config.sub_trajectory_config, :flow_consistency_mode, STATE_LEVEL)
        grad = flow_consistency_loss_grad(model, trajectories; mode=mode)
    else
        error("Unknown training objective: $(config.objective)")
    end
    
    # Apply gradients to update parameters
    apply_gradients!(model, grad, config.learning_rate)
end

"""
    apply_gradients!(model::GFlowNetModel, gradients, learning_rate::Float64)

Apply gradients to update model parameters.

# Arguments
- `model`: GFlowNet model to update
- `gradients`: Computed gradients
- `learning_rate`: Learning rate for parameter updates
"""
function apply_gradients!(model::GFlowNetModel, gradients, learning_rate::Float64)
    # Handle case where gradients is Nothing
    if isnothing(gradients)
        println("⚠️  Gradients are Nothing, skipping parameter update")
        return
    end
    
    println("🔄 Applying gradients with learning rate: $learning_rate")
    
    # Create new optimizer and parameter tuples
    new_optimizer = model.optimizer
    new_parameters = model.parameters
    
    # Update forward policy
    if !isnothing(model.forward_policy) && haskey(gradients, :forward) && !isnothing(gradients.forward)
        result = Optimisers.update(model.optimizer.forward, model.parameters.forward, gradients.forward)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (forward = result[1],))
            new_parameters = merge(new_parameters, (forward = result[2],))
        end
    end
    
    # Update backward policy if it exists
    if !isnothing(model.backward_policy) && haskey(gradients, :backward) && !isnothing(gradients.backward)
        result = Optimisers.update(model.optimizer.backward, model.parameters.backward, gradients.backward)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (backward = result[1],))
            new_parameters = merge(new_parameters, (backward = result[2],))
        end
    end
    
    # Update flow estimator if it exists
    if !isnothing(model.flow_estimator) && haskey(gradients, :flow) && !isnothing(gradients.flow)
        result = Optimisers.update(model.optimizer.flow, model.parameters.flow, gradients.flow)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (flow = result[1],))
            new_parameters = merge(new_parameters, (flow = result[2],))
        end
    end
    
    # Set the new optimizer and parameters
    model.optimizer = new_optimizer
    model.parameters = new_parameters
end

"""
    compute_gradient_norm(gradients)

Compute the L2 norm of gradients for monitoring training stability.

# Arguments
- `gradients`: Computed gradients (typically a NamedTuple)

# Returns
- L2 norm of all gradients combined
"""
function compute_gradient_norm(gradients)
    total_norm_squared = 0.0
    
    for (name, grad) in pairs(gradients)
        if !isnothing(grad)
            # Handle different gradient structures
            if grad isa AbstractArray
                total_norm_squared += sum(abs2, grad)
            elseif grad isa NamedTuple
                # Recursively compute norm for nested structures
                total_norm_squared += compute_gradient_norm(grad)
            else
                # For scalar gradients
                total_norm_squared += abs2(grad)
            end
        end
    end
    
    return sqrt(total_norm_squared)
end

"""
    clip_gradients!(gradients, max_norm::Float64)

Clip gradients to prevent exploding gradients during training.

# Arguments
- `gradients`: Gradients to clip (modified in-place)
- `max_norm`: Maximum allowed gradient norm

# Returns
- Actual gradient norm before clipping
"""
function clip_gradients!(gradients, max_norm::Float64)
    current_norm = compute_gradient_norm(gradients)
    
    if current_norm > max_norm
        scale_factor = max_norm / current_norm
        
        # Scale all gradients
        for (name, grad) in pairs(gradients)
            if !isnothing(grad) && grad isa AbstractArray
                grad .*= scale_factor
            end
        end
    end
    
    return current_norm
end

"""
    setup_optimizers(model::GFlowNetModel, learning_rate::Float64=0.001; optimizer_type=:adam)

Setup optimizers for all components of a GFlowNet model.

# Arguments
- `model`: GFlowNet model
- `learning_rate`: Learning rate for optimizers
- `optimizer_type`: Type of optimizer (:adam, :sgd, :rmsprop)

# Returns
- NamedTuple of optimizers for each component
"""
function setup_optimizers(model::GFlowNetModel, learning_rate::Float64=0.001; optimizer_type=:adam)
    # Choose optimizer
    if optimizer_type == :adam
        opt_fn = () -> Optimisers.Adam(learning_rate)
    elseif optimizer_type == :sgd
        opt_fn = () -> Optimisers.Descent(learning_rate)
    elseif optimizer_type == :rmsprop
        opt_fn = () -> Optimisers.RMSProp(learning_rate)
    else
        error("Unknown optimizer type: $optimizer_type")
    end
    
    # Setup optimizers for each component
    optimizers = []
    
    # Forward policy optimizer
    if !isnothing(model.forward_policy)
        push!(optimizers, :forward => opt_fn())
    end
    
    # Backward policy optimizer
    if !isnothing(model.backward_policy)
        push!(optimizers, :backward => opt_fn())
    end
    
    # Flow estimator optimizer
    if !isnothing(model.flow_estimator)
        push!(optimizers, :flow => opt_fn())
    end
    
    return NamedTuple(optimizers)
end

# =============================================================================
# Legacy Functions for Backward Compatibility
# =============================================================================

"""
    compute_loss_and_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Legacy function for backward compatibility with examples.
Use modern train_gflownet() interface for new code.
"""
function compute_loss_and_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    # Use the modern trajectory balance implementation
    loss_fn = ps -> trajectory_balance_loss(model, trajectories)
    
    # Use Zygote to compute gradients  
    loss, grad = Zygote.withgradient(loss_fn, model.parameters)
    
    return loss, grad[1]
end

"""
    apply_optimizer!(model::GFlowNetModel, grad)

Legacy function for backward compatibility with examples.
Use modern train_gflownet() interface for new code.
"""
function apply_optimizer!(model::GFlowNetModel, grad)
    # Create new optimizer and parameter tuples
    new_optimizer = model.optimizer
    new_parameters = model.parameters
    
    # Update forward policy
    if !isnothing(model.forward_policy) && haskey(grad, :forward) && !isnothing(grad.forward)
        result = Optimisers.update(model.optimizer.forward, model.parameters.forward, grad.forward)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (forward = result[1],))
            new_parameters = merge(new_parameters, (forward = result[2],))
        end
    end
    
    # Update backward policy if it exists
    if !isnothing(model.backward_policy) && haskey(grad, :backward) && !isnothing(grad.backward)
        result = Optimisers.update(model.optimizer.backward, model.parameters.backward, grad.backward)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (backward = result[1],))
            new_parameters = merge(new_parameters, (backward = result[2],))
        end
    end
    
    # Update flow estimator if it exists
    if !isnothing(model.flow_estimator) && haskey(grad, :flow) && !isnothing(grad.flow)
        result = Optimisers.update(model.optimizer.flow, model.parameters.flow, grad.flow)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (flow = result[1],))
            new_parameters = merge(new_parameters, (flow = result[2],))
        end
    end
    
    # Set the new optimizer and parameters
    model.optimizer = new_optimizer
    model.parameters = new_parameters
    
    return model
end

"""
    create_adam_optimizer(learning_rate::Float64=0.001)

Create an Adam optimizer with the specified learning rate.
Legacy function for examples.
"""
function create_adam_optimizer(learning_rate::Float64=0.001)
    return Optimisers.Adam(learning_rate)
end

"""
    update_parameters_with_gradients!(parameters, gradients, optimizer_state, learning_rate::Float64)

Update parameters using gradients and optimizer state.
Legacy function for examples.
"""
function update_parameters_with_gradients!(parameters, gradients, optimizer_state, learning_rate::Float64)
    result = Optimisers.update(optimizer_state, parameters, gradients)
    
    if !isnothing(result) && length(result) == 2
        return result[1], result[2]  # new_optimizer_state, new_parameters
    else
        return optimizer_state, parameters  # No update
    end
end 