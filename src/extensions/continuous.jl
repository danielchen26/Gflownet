using ..GFlowNet: AbstractState, AbstractAction

"""
    ContinuousState <: AbstractState

Abstract type for continuous state representations in GFlowNets.
"""
abstract type ContinuousState <: AbstractState end

"""
    ContinuousAction <: AbstractAction

Abstract type for continuous actions in GFlowNets.
"""
abstract type ContinuousAction <: AbstractAction end

"""
    Gaussian distribution for continuous transitions.
"""
struct GaussianPolicy{M}
    mean_network::M
    log_std_network::M
end

"""
    create_gaussian_policy(input_dim::Int, hidden_dim::Int, action_dim::Int, rng=nothing)

Create a Gaussian policy for continuous actions.
"""
function create_gaussian_policy(input_dim::Int, hidden_dim::Int, action_dim::Int, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    # Create networks for mean and log standard deviation
    mean_network = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => action_dim)
    )
    
    log_std_network = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => action_dim)
    )
    
    # Initialize parameters
    mean_ps, mean_st = Lux.setup(rng, mean_network)
    log_std_ps, log_std_st = Lux.setup(rng, log_std_network)
    
    return GaussianPolicy(mean_network, log_std_network), 
           (mean_ps, log_std_ps), 
           (mean_st, log_std_st)
end

"""
    gaussian_log_prob(mean, log_std, action)

Calculate the log probability of an action under a Gaussian distribution.
"""
function gaussian_log_prob(mean, log_std, action)
    variance = exp.(2 .* log_std)
    log_prob = -0.5 * log.(2π * variance) - 0.5 * ((action - mean).^2) ./ variance
    return sum(log_prob)
end

"""
    sample_continuous_action(policy::GaussianPolicy, state::ContinuousState, ps, st, rng=nothing)

Sample a continuous action from a Gaussian policy.
"""
function sample_continuous_action(policy::GaussianPolicy, state::ContinuousState, ps, st, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    mean_ps, log_std_ps = ps
    mean_st, log_std_st = st
    
    # Get state features
    features = state_to_features(state)
    
    # Compute mean and log standard deviation
    mean, new_mean_st = policy.mean_network(features, mean_ps, mean_st)
    log_std, new_log_std_st = policy.log_std_network(features, log_std_ps, log_std_st)
    
    # Clip log_std for numerical stability
    log_std = clamp.(log_std, -20, 2)
    
    # Sample from Gaussian distribution
    std = exp.(log_std)
    action = mean + std .* randn(rng, size(mean))
    
    # Compute log probability of the sampled action
    log_prob = gaussian_log_prob(mean, log_std, action)
    
    return action, log_prob, (new_mean_st, new_log_std_st)
end

"""
    continuous_action_log_prob(policy::GaussianPolicy, state::ContinuousState, action, ps, st)

Calculate the log probability of a continuous action under the policy.
"""
function continuous_action_log_prob(policy::GaussianPolicy, state::ContinuousState, action, ps, st)
    mean_ps, log_std_ps = ps
    mean_st, log_std_st = st
    
    # Get state features
    features = state_to_features(state)
    
    # Compute mean and log standard deviation
    mean, new_mean_st = policy.mean_network(features, mean_ps, mean_st)
    log_std, new_log_std_st = policy.log_std_network(features, log_std_ps, log_std_st)
    
    # Clip log_std for numerical stability
    log_std = clamp.(log_std, -20, 2)
    
    # Compute log probability
    log_prob = gaussian_log_prob(mean, log_std, action)
    
    return log_prob, (new_mean_st, new_log_std_st)
end 