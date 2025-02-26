using ..GFlowNet: GFlowNetModel, AbstractState, Trajectory, reward, estimate_partition_function, forward_transition_prob
using Zygote
using LinearAlgebra
using Optimisers

"""
    compute_loss_and_grad(model::GFlowNetModel, trajectories::Vector{Trajectory})

Compute the total loss and gradients for GFlowNet training.
This implementation works with the example files and properly handles Lux models.
"""
function compute_loss_and_grad(model::GFlowNetModel, trajectories::Vector{Trajectory})
    # For the simple trajectory balance loss used in examples
    loss_fn = ps -> trajectory_balance_loss(model, trajectories, ps)
    
    # Use Zygote to compute gradients
    loss, grad = Zygote.withgradient(loss_fn, model.parameters)
    
    return loss, grad[1]
end

"""
    trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{Trajectory}, ps=nothing)

Compute the trajectory balance loss for a batch of trajectories.
Specialized for the examples in the codebase.
"""
function trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{Trajectory}, ps=nothing)
    # If parameters are provided, create a new model with those parameters
    working_model = ps === nothing ? model : GFlowNetModel(
        model.dag,
        model.forward_policy,
        model.backward_policy,
        model.flow_estimator,
        model.partition_function,
        model.objectives,
        model.optimizer,
        ps,  # Use provided parameters
        model.states
    )
    
    total_loss = 0.0
    n_trajectories = length(trajectories)
    
    for trajectory in trajectories
        # Last state in trajectory
        final_state = trajectory.states[end]
        
        # Product of forward probabilities along the trajectory
        forward_prob_product = 1.0
        for i in 1:(length(trajectory.states)-1)
            source = trajectory.states[i]
            target = trajectory.states[i+1]
            prob = forward_transition_prob(working_model, source, target)
            # Prevent numerical issues
            prob = max(prob, 1e-10)
            forward_prob_product *= prob
        end
        
        # Compute the reward of the final state
        final_reward = reward(final_state)
        # Ensure reward is positive
        final_reward = max(final_reward, 1e-10)
        
        # Compute Z (partition function)
        Z = isnothing(working_model.partition_function) ? 
            estimate_partition_function(working_model) : working_model.partition_function
        Z = max(Z, 1e-10)
        
        # Compute the ratio (should be 1 for perfect balance)
        ratio = (Z * forward_prob_product) / final_reward
        
        # Squared log error
        log_ratio = log(ratio)
        total_loss += log_ratio^2
    end
    
    return total_loss / n_trajectories
end

"""
    apply_optimizer!(model::GFlowNetModel, grad)

Apply the optimizer to update model parameters based on gradients.
Creates new NamedTuples for the optimizer state and parameters to avoid mutating immutable structures.
"""
function apply_optimizer!(model::GFlowNetModel, grad)
    # Create new optimizer and parameter tuples
    new_optimizer = model.optimizer
    new_parameters = model.parameters
    
    # Update forward policy
    if !isnothing(model.forward_policy) && !isnothing(grad.forward)
        # Get updated optimizer state and parameters
        result = Optimisers.update(model.optimizer.forward, model.parameters.forward, grad.forward)
        
        # Create new named tuples with updated values
        if !isnothing(result)
            # Update just the forward components
            new_optimizer = merge(new_optimizer, (forward = result[1],))
            new_parameters = merge(new_parameters, (forward = result[2],))
        end
    end
    
    # Update backward policy if it exists
    if !isnothing(model.backward_policy) && !isnothing(grad.backward)
        # Get updated optimizer state and parameters
        result = Optimisers.update(model.optimizer.backward, model.parameters.backward, grad.backward)
        
        # Create new named tuples with updated values
        if !isnothing(result)
            # Update just the backward components
            new_optimizer = merge(new_optimizer, (backward = result[1],))
            new_parameters = merge(new_parameters, (backward = result[2],))
        end
    end
    
    # Update flow estimator if it exists
    if !isnothing(model.flow_estimator) && !isnothing(grad.flow)
        # Get updated optimizer state and parameters
        result = Optimisers.update(model.optimizer.flow, model.parameters.flow, grad.flow)
        
        # Create new named tuples with updated values
        if !isnothing(result)
            # Update just the flow components
            new_optimizer = merge(new_optimizer, (flow = result[1],))
            new_parameters = merge(new_parameters, (flow = result[2],))
        end
    end
    
    # Set the new optimizer and parameters
    model.optimizer = new_optimizer
    model.parameters = new_parameters
    
    return model
end 