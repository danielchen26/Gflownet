abstract type AbstractEnvironment end

struct DiscreteEnvironment <: AbstractEnvironment
    state_space::Vector
    action_space::Vector
    transition_fn::Function
    reward_fn::Function
    is_terminal::Function
end

function initial_state(env::DiscreteEnvironment)
    return env.state_space[1]
end 