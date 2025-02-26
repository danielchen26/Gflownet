using ..GFlowNet: GFlowNetModel, Trajectory, forward_transition_prob, reward

"""
    trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{Trajectory})

Compute the Trajectory Balance loss for the GFlowNet.

The trajectory balance objective directly relates the probability of a complete trajectory τ
to the reward of the terminal state:
    P_F(τ) = R(s_τ) / Z
where Z is the partition function, P_F is the product of forward transition probabilities,
and R is the reward function.
"""
function trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    total_loss = 0.0
    n_trajectories = length(trajectories)
    
    for trajectory in trajectories
        # Last state in trajectory (before sink)
        final_state = trajectory.states[end]
        
        # Product of forward probabilities along the trajectory
        forward_prob_product = 1.0
        for i in 1:(length(trajectory.states)-1)
            source = trajectory.states[i]
            target = trajectory.states[i+1]
            prob = forward_transition_prob(model, source, target)
            forward_prob_product *= prob
        end
        
        # Compute the reward of the final state
        final_reward = reward(final_state)
        
        # Compute Z (partition function)
        Z = isnothing(model.partition_function) ? 
            estimate_partition_function(model) : model.partition_function
        
        # Compute the ratio (should be 1 for perfect balance)
        ratio = (Z * forward_prob_product) / final_reward
        
        # Squared log error (can be more numerically stable than direct squared error)
        log_ratio = log(ratio)
        total_loss += log_ratio^2
    end
    
    return total_loss / n_trajectories
end

"""
    trajectory_balance_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Compute the gradient of the Trajectory Balance loss for optimization.
"""
function trajectory_balance_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
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
    return gradient(ps -> trajectory_balance_loss(model, trajectories, ps), ps)[1]
end

"""
    trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory}, ps)
    
Compute the Trajectory Balance loss with explicitly provided parameters.
This is useful for automatic differentiation.
"""
function trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory}, ps)
    # Create a temporary model with updated parameters
    temp_model = deepcopy(model)
    
    # Update parameters
    # This would depend on how parameters are stored and updated
    # For Lux, something like:
    # temp_model.forward_policy.model = Lux.update_params(temp_model.forward_policy.model, ps)
    
    return trajectory_balance_loss(temp_model, trajectories)
end 