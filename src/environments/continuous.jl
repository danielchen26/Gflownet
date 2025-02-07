module ContinuousEnvironments

using Distributions

export ContinuousEnv

"""
    ContinuousEnv(action_bounds, state_dim)

Continuous action space environment for GFlowNets.

# Fields
- `action_bounds::Matrix{Float32}`: [low, high] bounds for each action dimension
- `state_dim::Int`: Dimensionality of state vectors
"""
struct ContinuousEnv
    action_bounds::Matrix{Float32}
    state_dim::Int
end

function get_action_distribution(env::ContinuousEnv, state)
    Uniform.(env.action_bounds[1,:], env.action_bounds[2,:])
end

end
