module DiscreteEnvironments

export DiscreteEnv

struct DiscreteEnv
    action_space::Vector{Int}
    state_dim::Int
end

function get_legal_actions(env::DiscreteEnv, state)
    return env.action_space
end

end
