# Partition Function API

## Overview

This page documents the partition function functionality in GFlowNet.jl, including the new learnable partition function feature.

## Types

### PartitionFunctionMethod

```julia
@enum PartitionFunctionMethod begin
    SIMPLE_ESTIMATION       # Z = 1 (fixed)
    LEARNABLE_ESTIMATION   # Z is learned as parameter
end
```

An enumeration specifying how the partition function Z should be handled during training.

## Configuration

### TrainingConfig Fields

```julia
struct TrainingConfig
    # ... other fields ...
    partition_function_method::PartitionFunctionMethod
    partition_update_frequency::Int
    # ... other fields ...
end
```

- `partition_function_method`: How to handle Z (default: `SIMPLE_ESTIMATION`)
- `partition_update_frequency`: How often to log Z value during training (default: 10)

## Model Fields

### GFlowNetModel

```julia
mutable struct GFlowNetModel
    # ... other fields ...
    log_partition_function::Union{Nothing,Float64}
    # ... other fields ...
end
```

- `log_partition_function`: The log of the partition function Z
  - `nothing` when using `SIMPLE_ESTIMATION`
  - `Float64` value when using `LEARNABLE_ESTIMATION`

## Functions

### Creating Models with Learnable Z

```julia
create_grid_world_gflownet(
    grid_size::Int;
    hidden_dim::Int = 64,
    n_hidden_layers::Int = 2,
    partition_function_method::PartitionFunctionMethod = SIMPLE_ESTIMATION,
    kwargs...
) -> GFlowNetModel
```

Creates a grid world GFlowNet with optional learnable partition function.

#### Arguments
- `grid_size::Int`: Size of the grid (grid_size × grid_size)
- `hidden_dim::Int = 64`: Hidden layer dimension
- `n_hidden_layers::Int = 2`: Number of hidden layers
- `partition_function_method::PartitionFunctionMethod = SIMPLE_ESTIMATION`: How to handle Z

#### Example
```julia
# With fixed Z = 1
model_fixed = create_grid_world_gflownet(
    grid_size = 5,
    partition_function_method = SIMPLE_ESTIMATION
)

# With learnable Z
model_learnable = create_grid_world_gflownet(
    grid_size = 5,
    partition_function_method = LEARNABLE_ESTIMATION
)
```

### Training with Learnable Z

```julia
train_gflownet(
    model::GFlowNetModel, 
    config::TrainingConfig;
    verbose::Bool = false,
    validation_data = nothing,
    callback = nothing
) -> TrainingHistory
```

Trains a GFlowNet model, optionally learning the partition function.

#### With LEARNABLE_ESTIMATION
- Includes log_Z in the parameter array
- Updates log_Z through gradient descent
- Synchronizes model.log_partition_function after each update
- Tracks Z evolution in history.partition_function_estimates

### Accessing the Partition Function

```julia
# During/after training with LEARNABLE_ESTIMATION
if model.log_partition_function !== nothing
    Z = exp(model.log_partition_function)
    log_Z = model.log_partition_function
    
    # Also available in parameters
    log_Z_param = model.parameters.log_Z
end
```

### Loss Computation with Learnable Z

The trajectory balance loss is computed as:

```julia
# When partition_function_method == LEARNABLE_ESTIMATION
log_Z = params.log_Z  # Trainable parameter
trajectory_balance_error = log_Z + log_P_F - log_R

# When partition_function_method == SIMPLE_ESTIMATION  
log_Z = 0.0  # Fixed (Z = 1)
trajectory_balance_error = log_P_F - log_R
```

## Parameter Structure

When using `LEARNABLE_ESTIMATION`, the parameter structure includes:

```julia
ComponentArray(
    forward_policy = NamedTuple(
        layers = [...],  # Neural network layers
    ),
    log_Z = 0.0  # Initial log partition function
)
```

The `log_Z` parameter:
- Initialized to 0.0 (Z = 1)
- Updated via gradient descent
- Accessible as `model.parameters.log_Z`

## Validation Functions

```julia
validate_Z_learning(
    model::GFlowNetModel,
    true_Z::Float64;
    tolerance::Float64 = 0.01
) -> Bool
```

Validates that the learned partition function is close to the true value.

```julia
validate_Z_convergence(
    history::Dict{Symbol, Vector};
    window::Int = 100,
    threshold::Float64 = 0.001
) -> Bool
```

Checks if the partition function has converged during training.

## Best Practices

### When to Use LEARNABLE_ESTIMATION

```julia
# Recommended for complex environments
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    partition_function_method = LEARNABLE_ESTIMATION,
    n_iterations = 2000,      # More iterations often needed
    batch_size = 128,         # Larger batches help
    learning_rate = 0.001     # Slightly lower LR
)
```

### Monitoring Z During Training

```julia
# The partition function evolution is automatically tracked
history = train_gflownet(model, config; verbose=true)

# Access Z history
Z_evolution = history.partition_function_estimates

# Plot convergence
using Plots
plot(Z_evolution, label="Z", xlabel="Iteration", ylabel="Partition Function")
```

### Hyperparameter Guidelines

| Parameter | SIMPLE_ESTIMATION | LEARNABLE_ESTIMATION |
|-----------|------------------|---------------------|
| batch_size | 32-64 | 64-128 |
| learning_rate | 0.01 | 0.001-0.005 |
| n_iterations | 500-1000 | 1000-2000 |

## Implementation Notes

1. **Numerical Stability**: We learn log(Z) rather than Z directly
2. **Gradient Flow**: log_Z receives gradients like any other parameter
3. **Initialization**: log_Z starts at 0.0 (equivalent to Z = 1)
4. **Synchronization**: Model field updated after each optimization step

## Future Extensions

The learnable partition function is designed to support future multi-start GFlowNets:

```julia
# Current: Single Z for single initial state
log_Z::Float64

# Future: Multiple Z values for multiple initial states
log_Z_values::Vector{Float64}  # One per initial state
P(s₀ⁱ) ∝ exp(log_Z[i])  # Initial state distribution
```

## See Also

- [Learnable Partition Function Guide](../guide/learnable_partition_function.md)
- [Partition Function Theory](../theory/partition_function.md)
- [Training Configuration](training.md#trainingconfig)