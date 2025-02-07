struct Trajectory
    states::Vector{Vector{Float32}}
    actions::Vector{Int}
    rewards::Vector{Float32}
    log_probs::Vector{Float32}
end

function generate_trajectory(forward::ForwardPolicy, env::DiscreteEnvironment; max_steps=100)
    states = Vector{Vector{Float32}}()
    actions = Int[]
    rewards = Float32[]
    log_probs = Float32[]
    
    state = initial_state(env)
    push!(states, state)
    
    for _ in 1:max_steps
        env.is_terminal(state) && break
        
        # Forward pass through policy
        probs, _ = forward.model(state, forward.ps, forward.st)
        action = sample(Weights(probs))
        next_state = env.transition_fn(state, action)
        reward = env.reward_fn(state, action, next_state)
        
        push!(actions, action)
        push!(rewards, reward)
        push!(log_probs, log(probs[action]))
        push!(states, next_state)
        
        state = next_state
    end
    
    Trajectory(states, actions, rewards, log_probs)
end