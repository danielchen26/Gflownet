# Main GFlowNet Training Interface
# This file contains the high-level training functions and training loop

using ..GFlowNet: GFlowNetModel, sample_trajectory, estimate_partition_function, update_partition_function!, reward
using Statistics
using Random
using Dates

# Import algorithm functions
using ..GFlowNet: trajectory_balance_loss, general_trajectory_balance_loss
using ..GFlowNet: sub_trajectory_balance_loss, hierarchical_sub_trajectory_balance_loss, adaptive_sub_trajectory_balance_loss
using ..GFlowNet: flow_consistency_loss, FlowConsistencyMode, STATE_LEVEL
using ..GFlowNet: SimplePartitionFunctionEstimator, LearnablePartitionFunctionEstimator, 
                  SamplingPartitionFunctionEstimator, AdaptivePartitionFunctionEstimator

# Import config and optimization
# config.jl is already included in main module
# optimization.jl is already included in main module

# Simple logging function for training metrics
function log_to_file!(logger, iteration, loss, reward_mean, reward_std, start_time)
    if !isnothing(logger) && !isnothing(logger.log_file)
        open(logger.log_file, "a") do io
            elapsed_time = time() - start_time
            println(io, "$(now()),$iteration,$loss,$reward_mean,$reward_std,,$elapsed_time")
        end
    end
end

"""
    setup_partition_function_estimator(method::PartitionFunctionMethod, model::GFlowNetModel)

Create and initialize a partition function estimator based on the specified method.
"""
function setup_partition_function_estimator(method::PartitionFunctionMethod, model::GFlowNetModel)
    if method == SIMPLE_ESTIMATION
        return SimplePartitionFunctionEstimator()
    elseif method == LEARNABLE_PARAMETER
        # Initialize with a reasonable starting value
        initial_estimate = estimate_partition_function(SimplePartitionFunctionEstimator(), model)
        initial_log_Z = log(max(initial_estimate, 1e-10))
        return LearnablePartitionFunctionEstimator(initial_log_Z)
    elseif method == SAMPLING_BASED
        return SamplingPartitionFunctionEstimator(100; history_length=10, smoothing_factor=0.1)
    elseif method == ADAPTIVE_ESTIMATION
        return AdaptivePartitionFunctionEstimator()
    else
        error("Unknown partition function method: $method")
    end
end

"""
    compute_loss(config::TrainingConfig, model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Compute the loss according to the specified training objective.
"""
function compute_loss(config::TrainingConfig, model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    if config.objective == TRAJECTORY_BALANCE
        return trajectory_balance_loss(model, trajectories)
    elseif config.objective == GENERAL_TRAJECTORY_BALANCE
        return general_trajectory_balance_loss(model, trajectories)
    elseif config.objective == SUB_TRAJECTORY_BALANCE
        return sub_trajectory_balance_loss(model, trajectories; 
                                          min_length=config.sub_trajectory_config[:min_length],
                                          max_length=config.sub_trajectory_config[:max_length],
                                          n_subtrajectories=config.sub_trajectory_config[:n_subtrajectories])
    elseif config.objective == HIERARCHICAL_SUB_TB
        return hierarchical_sub_trajectory_balance_loss(model, trajectories;
                                                       scales=config.sub_trajectory_config[:scales])
    elseif config.objective == ADAPTIVE_SUB_TB
        return adaptive_sub_trajectory_balance_loss(model, trajectories;
                                                   difficulty_threshold=config.sub_trajectory_config[:difficulty_threshold])
    elseif config.objective == FLOW_CONSISTENCY
        # Use mode from config, default to STATE_LEVEL
        mode = get(config.sub_trajectory_config, :flow_consistency_mode, STATE_LEVEL)
        return flow_consistency_loss(model, trajectories; mode=mode)
    else
        error("Unknown training objective: $(config.objective)")
    end
end

"""
    train_gflownet(model::GFlowNetModel, config::TrainingConfig; 
                   validation_data=nothing, verbose=true)

Train a GFlowNet model using the specified configuration.

This is the main training function that supports all available training objectives
and partition function estimation methods.

# Arguments
- `model`: GFlowNet model to train
- `config`: Training configuration
- `validation_data`: Optional validation trajectories for early stopping
- `verbose`: Whether to print training progress

# Returns
- `training_history`: Dictionary containing loss history and other metrics
"""
function train_gflownet(model::GFlowNetModel, config::TrainingConfig; 
                       validation_data=nothing, verbose=true, logger=nothing)
    
    # Validate configuration
    validate_training_config(config)
    
    # Setup partition function estimator
    if hasfield(typeof(model), :partition_estimator)
        if isnothing(model.partition_estimator)
            model.partition_estimator = setup_partition_function_estimator(config.partition_function_method, model)
        end
    else
        @warn "Model does not support partition function estimators. Using simple estimation."
    end
    
    # Training history
    history = Dict(
        :losses => Float64[],
        :partition_function_estimates => Float64[],
        :validation_losses => Float64[],
        :best_loss => Inf,
        :patience_counter => 0,
        :config => config,
        :start_time => time()
    )
    
    if verbose
        println("Starting GFlowNet training...")
        println("  Objective: $(config.objective)")
        println("  Partition Function Method: $(config.partition_function_method)")
        println("  Batch Size: $(config.batch_size)")
        println("  Learning Rate: $(config.learning_rate)")
        println("  Iterations: $(config.n_iterations)")
        println()
    end
    
    for iteration in 1:config.n_iterations
        if verbose
            println("\n📊 Iteration $iteration/$(config.n_iterations)")
            println("   📈 Sampling $(config.batch_size) trajectories...")
        end
        
        # Sample trajectories for this batch
        trajectories = [sample_trajectory(model) for _ in 1:config.batch_size]
        
        if verbose
            avg_traj_length = mean(length(traj.states) for traj in trajectories)
            println("   ✅ Sampled trajectories (avg length: $(round(avg_traj_length, digits=2)))")
            println("   🧮 Computing loss...")
        end
        
        # Compute loss
        current_loss = compute_loss(config, model, trajectories)
        # FIXED: Use vcat instead of push! to avoid Zygote mutation error
        history[:losses] = vcat(history[:losses], [current_loss])
        
        # Compute trajectory rewards for logging
        rewards = [reward(traj.states[end]) for traj in trajectories]
        reward_mean = mean(rewards)
        reward_std = std(rewards)
        
        # Log metrics if logger is provided
        if !isnothing(logger) && iteration % 10 == 0  # Log every 10 iterations
            log_to_file!(logger, iteration, current_loss, reward_mean, reward_std, history[:start_time])
        end
        
        if verbose
            println("   📉 Loss: $(round(current_loss, digits=6))")
            println("   🎯 Reward: μ=$(round(reward_mean, digits=3)), σ=$(round(reward_std, digits=3))")
        end
        
        # Update partition function if needed
        if iteration % config.partition_update_frequency == 0
            update_partition_function!(model, trajectories)
            current_Z = estimate_partition_function(model)
            # FIXED: Use vcat instead of push! to avoid Zygote mutation error
            history[:partition_function_estimates] = vcat(history[:partition_function_estimates], [current_Z])
            
            if verbose && iteration % (config.validation_frequency * 2) == 0
                println("🔄 Iteration $iteration: Loss = $(round(current_loss, digits=6)), Z = $(round(current_Z, digits=6))")
            end
        end
        
        # Validation
        if !isnothing(validation_data) && iteration % config.validation_frequency == 0
            val_loss = compute_loss(config, model, validation_data)
            # FIXED: Use vcat instead of push! to avoid Zygote mutation error
            history[:validation_losses] = vcat(history[:validation_losses], [val_loss])
            
            # Early stopping check
            if val_loss < history[:best_loss]
                history[:best_loss] = val_loss
                history[:patience_counter] = 0
            else
                history[:patience_counter] += 1
            end
            
            if history[:patience_counter] >= config.early_stopping_patience
                if verbose
                    println("Early stopping at iteration $iteration (validation loss: $(round(val_loss, digits=6)))")
                end
                break
            end
        end
        
        # Compute gradients and update model
        if verbose
            println("   🔄 Computing gradients and updating parameters...")
        end
        update_model_parameters!(model, config, trajectories)
        
        # Update learnable partition function if applicable
        if hasfield(typeof(model), :partition_estimator) && 
           !isnothing(model.partition_estimator) &&
           model.partition_estimator isa Union{LearnablePartitionFunctionEstimator, AdaptivePartitionFunctionEstimator}
            update_partition_function!(model, trajectories)
        end
    end
    
    # Finalize training history
    history[:end_time] = time()
    history[:total_time] = history[:end_time] - history[:start_time]
    
    if verbose
        final_loss = history[:losses][end]
        final_Z = estimate_partition_function(model)
        println("Training completed!")
        println("  Final Loss: $(round(final_loss, digits=6))")
        println("  Final Z Estimate: $(round(final_Z, digits=6))")
        println("  Total Iterations: $(length(history[:losses]))")
        println("  Total Time: $(round(history[:total_time], digits=2)) seconds")
    end
    
    return history
end

"""
    train_gflownet_simple(model::GFlowNetModel, n_iterations::Int=1000; 
                         batch_size::Int=32, verbose::Bool=true)

Simplified training function for backward compatibility.
Uses trajectory balance with simple partition function estimation.
"""
function train_gflownet_simple(model::GFlowNetModel, n_iterations::Int=1000; 
                              batch_size::Int=32, verbose::Bool=true)
    config = TrainingConfig(n_iterations=n_iterations, batch_size=batch_size)
    return train_gflownet(model, config; verbose=verbose)
end

"""
    evaluate_model(model::GFlowNetModel, test_data::Vector, config::TrainingConfig)

Evaluate a trained model on test data.

# Arguments
- `model`: Trained GFlowNet model
- `test_data`: Test trajectories
- `config`: Training configuration used

# Returns
- Dictionary with evaluation metrics
"""
function evaluate_model(model::GFlowNetModel, test_data::Vector, config::TrainingConfig)
    test_loss = compute_loss(config, model, test_data)
    
    # Compute additional metrics
    n_trajectories = length(test_data)
    avg_trajectory_length = mean(length(traj.states) for traj in test_data)
    
    # Sample some trajectories to evaluate sampling quality
    n_samples = min(100, n_trajectories)
    sampled_trajectories = [sample_trajectory(model) for _ in 1:n_samples]
    avg_sampled_length = mean(length(traj.states) for traj in sampled_trajectories)
    
    return Dict(
        :test_loss => test_loss,
        :n_test_trajectories => n_trajectories,
        :avg_test_trajectory_length => avg_trajectory_length,
        :avg_sampled_trajectory_length => avg_sampled_length,
        :partition_function_estimate => estimate_partition_function(model)
    )
end

"""
    save_training_checkpoint(model::GFlowNetModel, history::Dict, filepath::String)

Save a training checkpoint including model and training history.

# Arguments
- `model`: GFlowNet model to save
- `history`: Training history
- `filepath`: Path to save checkpoint
"""
function save_training_checkpoint(model::GFlowNetModel, history::Dict, filepath::String)
    checkpoint = Dict(
        :model_state => model,  # In practice, would serialize only necessary parts
        :training_history => history,
        :timestamp => time(),
        :julia_version => VERSION
    )
    
    # In practice, would use JLD2 or similar for serialization
    @warn "Checkpoint saving not fully implemented - would save to $filepath"
    
    return checkpoint
end

"""
    load_training_checkpoint(filepath::String)

Load a training checkpoint.

# Arguments
- `filepath`: Path to checkpoint file

# Returns
- Tuple of (model, history)
"""
function load_training_checkpoint(filepath::String)
    @warn "Checkpoint loading not fully implemented - would load from $filepath"
    
    # In practice, would deserialize from file
    return nothing, nothing
end 