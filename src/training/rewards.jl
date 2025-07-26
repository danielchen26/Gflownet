"""
    Reward Framework for GFlowNet

This module provides a general framework for defining rewards in GFlowNet.
It includes abstract types, contexts, and utility functions for computing rewards.
"""

"""
    RewardFunction

Abstract type for all reward functions. Concrete implementations should define
how to compute rewards based on state, trajectory, and environment data.
"""
abstract type RewardFunction end

"""
    RewardContext

Abstract type for reward contexts. Contexts provide all the data needed
to compute rewards, including state, trajectory, and environment data.
"""
abstract type RewardContext end

"""
    StandardContext <: RewardContext

Standard implementation of a reward context that provides state, trajectory, and environment data.

# Fields
- `state`: The current state (typically a terminal state)
- `trajectory`: The full trajectory that led to the state (can be nothing)
- `env_data`: Dictionary containing environmental data
"""
struct StandardContext <: RewardContext
    state::Any
    trajectory::Any
    env_data::Dict{Symbol,Any}
end

"""
    compute_reward(reward_fn::RewardFunction, context::RewardContext)

Compute a reward value for a given reward function and context.

# Arguments
- `reward_fn`: The reward function to use
- `context`: The context containing state and environment data

# Returns
- Reward value
"""
function compute_reward(reward_fn::RewardFunction, context::RewardContext)
    error("compute_reward not implemented for $(typeof(reward_fn))")
end

"""
    ensure_positive(reward_fn::RewardFunction, raw_reward)

Ensure that a reward value is positive, as required by GFlowNet.

# Arguments
- `reward_fn`: The reward function used (may contain custom logic)
- `raw_reward`: The raw reward value to make positive

# Returns
- Positive reward value
"""
function ensure_positive(reward_fn::RewardFunction, raw_reward)
    # Default implementation adds a small constant and takes absolute value
    return max(0.01, raw_reward)
end

"""
    FunctionalReward <: RewardFunction

Generic reward function that uses a user-provided function to compute rewards.

# Fields
- `reward_fn`: Function that takes a context and returns a reward
"""
struct FunctionalReward <: RewardFunction
    reward_fn::Function  # Takes context, returns reward
end

"""
    compute_reward(reward_fn::FunctionalReward, context::RewardContext)

Compute reward using the function provided in the FunctionalReward.

# Arguments
- `reward_fn`: The functional reward
- `context`: The context containing state and environment data

# Returns
- Reward value
"""
function compute_reward(reward_fn::FunctionalReward, context::RewardContext)
    return reward_fn.reward_fn(context)
end

"""
    ValueMinusCostReward <: RewardFunction

Reward function that computes value minus cost.

# Fields
- `value_fn`: Function that computes value from context
- `cost_fn`: Function that computes cost from context
- `value_weight`: Weight for value component
- `cost_weight`: Weight for cost component
"""
struct ValueMinusCostReward <: RewardFunction
    value_fn::Function
    cost_fn::Function
    value_weight::Float64
    cost_weight::Float64
end

"""
    ValueMinusCostReward(value_fn::Function, cost_fn::Function)

Constructor with default weights of 1.0.
"""
function ValueMinusCostReward(value_fn::Function, cost_fn::Function)
    return ValueMinusCostReward(value_fn, cost_fn, 1.0, 1.0)
end

"""
    compute_reward(reward_fn::ValueMinusCostReward, context::RewardContext)

Compute value minus cost reward.

# Arguments
- `reward_fn`: The value-minus-cost reward function
- `context`: The context containing state and environment data

# Returns
- Value minus cost
"""
function compute_reward(reward_fn::ValueMinusCostReward, context::RewardContext)
    value = reward_fn.value_fn(context) * reward_fn.value_weight
    cost = reward_fn.cost_fn(context) * reward_fn.cost_weight
    return value - cost
end

"""
    reward(state::AbstractState, env_data=Dict())

Compute the reward for a given state.

This function implements domain-agnostic reward computation that works with
different state types. For SimpleState, it uses a mathematical function
based on the state data.

# Arguments
- `state`: The state to compute reward for
- `env_data`: Optional environment data dictionary

# Returns
- Positive reward value (always > 0 for valid terminal states)
"""
function reward(state::AbstractState, env_data=Dict())
    if haskey(env_data, :reward_function) && env_data[:reward_function] isa RewardFunction
        # Use custom reward function if provided
        context = StandardContext(state, nothing, env_data)
        raw_reward = compute_reward(env_data[:reward_function], context)
        return ensure_positive(env_data[:reward_function], raw_reward)
    else
        # Use default reward computation based on state type
        return _compute_default_reward(state)
    end
end

# SimpleState-specific reward computation moved to test/test_utilities.jl
# Domain-specific reward implementations are in src/applications/

# Generic fallback for other state types
function _compute_default_reward(state::AbstractState)
    error("Default reward computation not implemented for state type $(typeof(state)). " *
          "Please provide a custom reward function in env_data or implement _compute_default_reward for this type.")
end

"""
    _domain_specific_reward(state, env_data)

Placeholder for domain-specific reward implementation.
Override this in your application code.
"""
function _domain_specific_reward(state, env_data)
    return 1.0  # Placeholder
end
