using ..GFlowNet: GFlowNetModel, sample_trajectory, reward, flow
using Statistics
using Random
using Zygote
using Optimisers

"""
    AbstractPartitionFunctionEstimator

Abstract type for different partition function estimation strategies.
"""
abstract type AbstractPartitionFunctionEstimator end

"""
    SimplePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator

Simple estimator that sums all terminal state rewards.
This is the current default implementation.
"""
struct SimplePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator end

"""
    LearnablePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator

Learnable partition function as a parameter that's updated via gradient descent.

# Fields
- `log_Z`: Learnable log partition function parameter
- `optimizer`: Optimizer for the log_Z parameter
"""
mutable struct LearnablePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator
    log_Z::Float64
    optimizer::Any
end

"""
    SamplingPartitionFunctionEstimator <: AbstractPartitionFunctionEstimator

Sampling-based estimator that estimates Z by sampling from the current policy.

# Fields
- `n_samples`: Number of samples to use for estimation
- `history_length`: Number of past estimates to keep for smoothing
- `smoothing_factor`: Exponential smoothing factor
"""
mutable struct SamplingPartitionFunctionEstimator <: AbstractPartitionFunctionEstimator
    n_samples::Int
    history_length::Int
    smoothing_factor::Float64
    estimate_history::Vector{Float64}
end

"""
    AdaptivePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator

Adaptive estimator that switches between different methods based on training progress.

# Fields
- `simple_estimator`: Simple sum-based estimator
- `sampling_estimator`: Sampling-based estimator
- `learnable_estimator`: Learnable parameter estimator
- `method`: Current estimation method (:simple, :sampling, :learnable)
- `switch_thresholds`: Thresholds for switching methods
"""
mutable struct AdaptivePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator
    simple_estimator::SimplePartitionFunctionEstimator
    sampling_estimator::SamplingPartitionFunctionEstimator
    learnable_estimator::Union{Nothing, LearnablePartitionFunctionEstimator}
    method::Symbol
    switch_thresholds::Dict{Symbol, Float64}
    training_iteration::Int
end

# ============================================================================
# Simple Partition Function Estimation (Current Default)
# ============================================================================

"""
    estimate_partition_function(estimator::SimplePartitionFunctionEstimator, model::GFlowNetModel)

Estimate partition function by summing all terminal state rewards.
"""
function estimate_partition_function(estimator::SimplePartitionFunctionEstimator, model::GFlowNetModel)
    total = 0.0
    for state in model.dag.terminal_states
        total += reward(state)
    end
    return total
end

# ============================================================================
# Learnable Partition Function
# ============================================================================

"""
    LearnablePartitionFunctionEstimator(initial_log_Z::Float64=0.0; lr::Float64=0.01)

Create a learnable partition function estimator.
"""
function LearnablePartitionFunctionEstimator(initial_log_Z::Float64=0.0; lr::Float64=0.01)
    return LearnablePartitionFunctionEstimator(initial_log_Z, Optimisers.Adam(lr))
end

"""
    estimate_partition_function(estimator::LearnablePartitionFunctionEstimator, model::GFlowNetModel)

Get the current estimate from the learnable parameter.
"""
function estimate_partition_function(estimator::LearnablePartitionFunctionEstimator, model::GFlowNetModel)
    return exp(estimator.log_Z)
end

"""
    update_learnable_partition_function!(estimator::LearnablePartitionFunctionEstimator, 
                                       model::GFlowNetModel, trajectories::Vector{Trajectory})

Update the learnable partition function parameter using gradient descent.
"""
function update_learnable_partition_function!(estimator::LearnablePartitionFunctionEstimator, 
                                             model::GFlowNetModel, trajectories)
    # Compute gradient of trajectory balance loss w.r.t. log_Z
    loss_fn = log_Z -> begin
        temp_Z = exp(log_Z)
        total_loss = 0.0
        
        for trajectory in trajectories
            final_state = trajectory.states[end]
            
            # Product of forward probabilities
            forward_prob_product = 1.0
            for i in 1:(length(trajectory.states)-1)
                source = trajectory.states[i]
                target = trajectory.states[i+1]
                prob = forward_transition_prob(model, source, target)
                forward_prob_product *= prob
            end
            
            final_reward = reward(final_state)
            ratio = (temp_Z * forward_prob_product) / final_reward
            log_ratio = log(ratio)
            total_loss += log_ratio^2
        end
        
        return total_loss / length(trajectories)
    end
    
    # Compute gradient and update
    loss, grad = Zygote.withgradient(loss_fn, estimator.log_Z)
    
    # Apply optimizer update
    opt_state, new_log_Z = Optimisers.update(estimator.optimizer, estimator.log_Z, grad[1])
    estimator.optimizer = opt_state
    estimator.log_Z = new_log_Z
    
    return loss
end

# ============================================================================
# Sampling-Based Partition Function Estimation
# ============================================================================

"""
    SamplingPartitionFunctionEstimator(n_samples::Int=100; history_length::Int=10, 
                                      smoothing_factor::Float64=0.1)

Create a sampling-based partition function estimator.
"""
function SamplingPartitionFunctionEstimator(n_samples::Int=100; history_length::Int=10, 
                                           smoothing_factor::Float64=0.1)
    return SamplingPartitionFunctionEstimator(n_samples, history_length, smoothing_factor, Float64[])
end

"""
    estimate_partition_function(estimator::SamplingPartitionFunctionEstimator, model::GFlowNetModel)

Estimate partition function by sampling from the current policy.
"""
function estimate_partition_function(estimator::SamplingPartitionFunctionEstimator, model::GFlowNetModel)
    # Sample trajectories and collect terminal rewards
    rewards = Float64[]
    
    for _ in 1:estimator.n_samples
        try
            trajectory = sample_trajectory(model)
            final_state = trajectory.states[end]
            push!(rewards, reward(final_state))
        catch e
            # Handle sampling failures gracefully
            @warn "Sampling failed during partition function estimation: $e"
        end
    end
    
    if isempty(rewards)
        # Fallback to simple estimation
        return estimate_partition_function(SimplePartitionFunctionEstimator(), model)
    end
    
    # Estimate Z as the average reward times the number of terminal states
    current_estimate = mean(rewards) * length(model.dag.terminal_states)
    
    # Apply exponential smoothing with history
    if !isempty(estimator.estimate_history)
        smoothed_estimate = (1 - estimator.smoothing_factor) * estimator.estimate_history[end] + 
                           estimator.smoothing_factor * current_estimate
    else
        smoothed_estimate = current_estimate
    end
    
    # Update history
    push!(estimator.estimate_history, smoothed_estimate)
    if length(estimator.estimate_history) > estimator.history_length
        popfirst!(estimator.estimate_history)
    end
    
    return smoothed_estimate
end

# ============================================================================
# Adaptive Partition Function Estimation
# ============================================================================

"""
    AdaptivePartitionFunctionEstimator(; initial_method::Symbol=:simple)

Create an adaptive partition function estimator that switches methods based on training progress.
"""
function AdaptivePartitionFunctionEstimator(; initial_method::Symbol=:simple)
    simple_est = SimplePartitionFunctionEstimator()
    sampling_est = SamplingPartitionFunctionEstimator(50; history_length=5, smoothing_factor=0.2)
    
    switch_thresholds = Dict(
        :simple_to_sampling => 100,     # Switch to sampling after 100 iterations
        :sampling_to_learnable => 500,  # Switch to learnable after 500 iterations
        :variance_threshold => 0.1       # Switch if estimate variance is too high
    )
    
    return AdaptivePartitionFunctionEstimator(
        simple_est, sampling_est, nothing, initial_method, switch_thresholds, 0
    )
end

"""
    estimate_partition_function(estimator::AdaptivePartitionFunctionEstimator, model::GFlowNetModel)

Estimate partition function using the current method, with automatic method switching.
"""
function estimate_partition_function(estimator::AdaptivePartitionFunctionEstimator, model::GFlowNetModel)
    estimator.training_iteration += 1
    
    # Check if we should switch methods
    should_switch_method!(estimator, model)
    
    # Estimate using current method
    if estimator.method == :simple
        return estimate_partition_function(estimator.simple_estimator, model)
    elseif estimator.method == :sampling
        return estimate_partition_function(estimator.sampling_estimator, model)
    elseif estimator.method == :learnable && !isnothing(estimator.learnable_estimator)
        return estimate_partition_function(estimator.learnable_estimator, model)
    else
        # Fallback to simple
        return estimate_partition_function(estimator.simple_estimator, model)
    end
end

"""
    should_switch_method!(estimator::AdaptivePartitionFunctionEstimator, model::GFlowNetModel)

Check if the estimation method should be switched and perform the switch if needed.
"""
function should_switch_method!(estimator::AdaptivePartitionFunctionEstimator, model::GFlowNetModel)
    current_iter = estimator.training_iteration
    
    # Switch from simple to sampling
    if estimator.method == :simple && current_iter >= estimator.switch_thresholds[:simple_to_sampling]
        estimator.method = :sampling
        @info "Switched partition function estimation to sampling method at iteration $current_iter"
    end
    
    # Switch from sampling to learnable
    if estimator.method == :sampling && current_iter >= estimator.switch_thresholds[:sampling_to_learnable]
        # Initialize learnable estimator if not already done
        if isnothing(estimator.learnable_estimator)
            current_estimate = estimate_partition_function(estimator.sampling_estimator, model)
            initial_log_Z = log(max(current_estimate, 1e-10))
            estimator.learnable_estimator = LearnablePartitionFunctionEstimator(initial_log_Z)
        end
        
        estimator.method = :learnable
        @info "Switched partition function estimation to learnable method at iteration $current_iter"
    end
    
    # Check variance-based switching
    if estimator.method == :sampling && length(estimator.sampling_estimator.estimate_history) >= 3
        recent_estimates = estimator.sampling_estimator.estimate_history[end-2:end]
        if std(recent_estimates) / mean(recent_estimates) > estimator.switch_thresholds[:variance_threshold]
            @info "High variance detected in sampling estimates, considering method switch"
        end
    end
end

"""
    update_adaptive_partition_function!(estimator::AdaptivePartitionFunctionEstimator, 
                                       model::GFlowNetModel, trajectories)

Update the adaptive partition function estimator.
"""
function update_adaptive_partition_function!(estimator::AdaptivePartitionFunctionEstimator, 
                                            model::GFlowNetModel, trajectories)
    if estimator.method == :learnable && !isnothing(estimator.learnable_estimator)
        return update_learnable_partition_function!(estimator.learnable_estimator, model, trajectories)
    end
    return 0.0  # No update needed for non-learnable methods
end

# ============================================================================
# Integration with Existing Training Loop
# ============================================================================

"""
    estimate_partition_function(model::GFlowNetModel)

Wrapper function that maintains backward compatibility with existing code.
Uses the model's partition function estimator if available, otherwise falls back to simple estimation.
"""
function estimate_partition_function(model::GFlowNetModel)
    # Check if model has a partition function estimator
    if hasfield(typeof(model), :partition_estimator) && !isnothing(model.partition_estimator)
        return estimate_partition_function(model.partition_estimator, model)
    else
        # Fallback to current simple method
        return estimate_partition_function(SimplePartitionFunctionEstimator(), model)
    end
end

"""
    update_partition_function!(model::GFlowNetModel, trajectories=nothing)

Update the partition function estimate. This should be called during training.
"""
function update_partition_function!(model::GFlowNetModel, trajectories=nothing)
    if hasfield(typeof(model), :partition_estimator) && !isnothing(model.partition_estimator)
        estimator = model.partition_estimator
        
        # Update based on estimator type
        if estimator isa LearnablePartitionFunctionEstimator && !isnothing(trajectories)
            update_learnable_partition_function!(estimator, model, trajectories)
        elseif estimator isa AdaptivePartitionFunctionEstimator && !isnothing(trajectories)
            update_adaptive_partition_function!(estimator, model, trajectories)
        end
        
        # Update the model's partition function value
        model.partition_function = estimate_partition_function(estimator, model)
    else
        # Fallback to current method
        model.partition_function = estimate_partition_function(SimplePartitionFunctionEstimator(), model)
    end
end 