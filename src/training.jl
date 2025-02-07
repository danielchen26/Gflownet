using Lux, Optimisers, ComponentArrays

function trajectory_balance_loss(forward::ForwardPolicy, backward::BackwardPolicy, traj::Trajectory)
    total_loss = 0f0
    logZ = log(forward.logZ)
    
    for (s, a, s_next, log_pf) in zip(traj.states[1:end-1], traj.actions, traj.states[2:end], traj.log_probs)
        # Forward pass through backward policy
        pb, _ = backward.model(s_next, backward.ps, backward.st)
        log_pb = log(pb[a])
        
        reward = traj.rewards[findfirst(==(s_next), traj.states)]
        loss = (logZ + sum(log_pf) - log(reward) - sum(log_pb))^2
        total_loss += loss
    end
    
    return total_loss / length(traj.actions)
end

function train!(forward::ForwardPolicy, backward::BackwardPolicy, env::DiscreteEnvironment, opt; epochs=100)
    # Combine parameters
    all_ps = ComponentArray(; forward=forward.ps, backward=backward.ps)
    
    # Setup optimizer
    opt_state = Optimisers.setup(opt, all_ps)
    
    for epoch in 1:epochs
        traj = generate_trajectory(forward, env)
        
        # Compute loss and gradients
        loss, grad = Lux.value_and_gradient(all_ps) do ps
            fwd_ps = ComponentArray(ps.forward)
            bwd_ps = ComponentArray(ps.backward)
            trajectory_balance_loss(
                ForwardPolicy(forward.model, forward.logZ, fwd_ps, forward.st),
                BackwardPolicy(backward.model, bwd_ps, backward.st),
                traj
            )
        end
        
        # Update parameters
        opt_state, all_ps = Optimisers.update(opt_state, all_ps, grad)
        
        # Update forward policy
        forward = ForwardPolicy(
            forward.model, 
            forward.logZ,
            ComponentArray(all_ps.forward),
            forward.st
        )
    end
    
    return forward, backward
end 