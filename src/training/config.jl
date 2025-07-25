# Training Configuration and Enums
# This file defines the configuration interface for GFlowNet training



using ..GFlowNet: TrainingObjective, PartitionFunctionMethod, TrainingConfig

# TrainingConfig constructor is now defined in types.jl to avoid method overwriting

"""
    validate_training_config(config::TrainingConfig)

Validate a training configuration and throw informative errors if invalid.

# Arguments
- `config`: Training configuration to validate

# Throws
- `ArgumentError` if configuration is invalid
"""
function validate_training_config(config::TrainingConfig)
    # Validate batch size
    if config.batch_size <= 0
        throw(ArgumentError("Batch size must be positive, got $(config.batch_size)"))
    end
    
    # Validate learning rate
    if config.learning_rate <= 0.0
        throw(ArgumentError("Learning rate must be positive, got $(config.learning_rate)"))
    end
    
    # Validate iterations
    if config.n_iterations <= 0
        throw(ArgumentError("Number of iterations must be positive, got $(config.n_iterations)"))
    end
    
    # Validate frequencies
    if config.partition_update_frequency <= 0
        throw(ArgumentError("Partition update frequency must be positive, got $(config.partition_update_frequency)"))
    end
    
    if config.validation_frequency <= 0
        throw(ArgumentError("Validation frequency must be positive, got $(config.validation_frequency)"))
    end
    
    if config.early_stopping_patience <= 0
        throw(ArgumentError("Early stopping patience must be positive, got $(config.early_stopping_patience)"))
    end
    
    # Validate sub-trajectory configuration
    sub_config = config.sub_trajectory_config
    
    if haskey(sub_config, :min_length) && sub_config[:min_length] < 1
        throw(ArgumentError("Minimum sub-trajectory length must be at least 1"))
    end
    
    if haskey(sub_config, :max_length) && !isnothing(sub_config[:max_length]) && sub_config[:max_length] < sub_config[:min_length]
        throw(ArgumentError("Maximum sub-trajectory length must be >= minimum length"))
    end
    
    if haskey(sub_config, :n_subtrajectories) && sub_config[:n_subtrajectories] < 1
        throw(ArgumentError("Number of sub-trajectories must be at least 1"))
    end
    
    if haskey(sub_config, :difficulty_threshold) && (sub_config[:difficulty_threshold] < 0.0 || sub_config[:difficulty_threshold] > 1.0)
        throw(ArgumentError("Difficulty threshold must be between 0.0 and 1.0"))
    end
    
    if haskey(sub_config, :max_grad_norm) && sub_config[:max_grad_norm] < 0.0
        throw(ArgumentError("Maximum gradient norm must be non-negative"))
    end
    
    return true
end

"""
    get_config_summary(config::TrainingConfig)

Get a human-readable summary of the training configuration.

# Arguments
- `config`: Training configuration

# Returns
- String summary of the configuration
"""
function get_config_summary(config::TrainingConfig)
    summary = """
    Training Configuration Summary:
    ==============================
    Objective: $(config.objective)
    Partition Function Method: $(config.partition_function_method)
    Batch Size: $(config.batch_size)
    Learning Rate: $(config.learning_rate)
    Iterations: $(config.n_iterations)
    Partition Update Frequency: $(config.partition_update_frequency)
    Validation Frequency: $(config.validation_frequency)
    Early Stopping Patience: $(config.early_stopping_patience)
    
    Sub-trajectory Configuration:
    """
    
    for (key, value) in config.sub_trajectory_config
        summary *= "    $key: $value\n"
    end
    
    return summary
end 