# Training Configuration and Enums
# This file defines the configuration interface for GFlowNet training

"""
    TrainingObjective

Enum for different training objectives available in GFlowNet.
"""
@enum TrainingObjective begin
    TRAJECTORY_BALANCE
    GENERAL_TRAJECTORY_BALANCE  # Full TB with P_B term
    SUB_TRAJECTORY_BALANCE     # Sub-trajectory balance
    HIERARCHICAL_SUB_TB        # Hierarchical sub-trajectory balance
    ADAPTIVE_SUB_TB           # Adaptive sub-trajectory balance
    FLOW_CONSISTENCY          # Unified flow consistency (edge/state/mixed levels)
end

"""
    PartitionFunctionMethod

Enum for different partition function estimation methods.
"""
@enum PartitionFunctionMethod begin
    SIMPLE_ESTIMATION     # Current default: sum of terminal rewards
    LEARNABLE_PARAMETER   # New: Learnable log(Z) parameter
    SAMPLING_BASED       # New: Sampling-based estimation
    ADAPTIVE_ESTIMATION  # New: Adaptive method switching
end

"""
    TrainingConfig

Configuration for GFlowNet training with all available options.

# Fields
- `objective`: Training objective to use
- `partition_function_method`: Method for estimating/learning Z
- `batch_size`: Number of trajectories per batch
- `learning_rate`: Learning rate for optimization
- `n_iterations`: Number of training iterations
- `partition_update_frequency`: How often to update partition function estimate
- `validation_frequency`: How often to run validation
- `early_stopping_patience`: Number of iterations without improvement before stopping
- `sub_trajectory_config`: Configuration for sub-trajectory methods
"""
struct TrainingConfig
    objective::TrainingObjective
    partition_function_method::PartitionFunctionMethod
    batch_size::Int
    learning_rate::Float64
    n_iterations::Int
    partition_update_frequency::Int
    validation_frequency::Int
    early_stopping_patience::Int
    sub_trajectory_config::Dict{Symbol, Any}
end

"""
    TrainingConfig(; objective=TRAJECTORY_BALANCE, partition_function_method=SIMPLE_ESTIMATION, 
                   batch_size=32, learning_rate=0.001, n_iterations=1000, 
                   partition_update_frequency=10, validation_frequency=50,
                   early_stopping_patience=100, sub_trajectory_config=Dict())

Create a training configuration with sensible defaults.
"""
function TrainingConfig(; objective=TRAJECTORY_BALANCE, partition_function_method=SIMPLE_ESTIMATION, 
                       batch_size=32, learning_rate=0.001, n_iterations=1000, 
                       partition_update_frequency=10, validation_frequency=50,
                       early_stopping_patience=100, sub_trajectory_config=Dict())
        
    # Default sub-trajectory configuration
    default_sub_config = Dict(
        :min_length => 2,
        :max_length => nothing,
        :n_subtrajectories => 5,
        :scales => [2, 4, 8],
        :difficulty_threshold => 0.1,
        :flow_consistency_mode => :STATE_LEVEL  # Default flow consistency mode (will be imported later)
    )
    
    merged_sub_config = merge(default_sub_config, sub_trajectory_config)
    
    return TrainingConfig(objective, partition_function_method, batch_size, learning_rate, 
                         n_iterations, partition_update_frequency, validation_frequency,
                         early_stopping_patience, merged_sub_config)
end

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