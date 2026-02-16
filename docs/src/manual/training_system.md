# Modern Training System

## TrainingConfig Structure

The modern GFlowNet.jl framework uses a configuration-based training system that encapsulates all training parameters:

```julia
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
```

## Core Training Function

The main entry point for training is the `train_gflownet` function:

```julia
history = train_gflownet(model, config; verbose=true, validation_data=nothing)
```

### Returns
Training history with:
- `:losses` - Training loss over iterations
- `:partition_function_estimates` - Z evolution
- `:mean_rewards` - Average rewards (if available)
- `:max_rewards` - Maximum rewards (if available)

## Automatic Configuration

GFlowNet.jl provides domain-specific optimal configurations:

```julia
# Simple deterministic domains
config = create_modern_training_config(:grid_world)        

# Sequential decision making
config = create_modern_training_config(:active_learning)   

# Non-deterministic paths
config = create_modern_training_config(:feature_acquisition) 

# Graph structure learning
config = create_modern_training_config(:causal_discovery)  

# Multi-scale structures
config = create_modern_training_config(:molecular_design) 
```

## Training Parameters

### Core Parameters
- `objective`: Which training objective to use (see [objectives.md](objectives.md))
- `batch_size`: Number of trajectories per iteration (typically 16-64)
- `learning_rate`: Optimizer learning rate (typically 0.001-0.01)
- `n_iterations`: Total training iterations

### Advanced Parameters
- `partition_function_method`: How to handle Z - `SIMPLE_ESTIMATION` (Z=1) or `LEARNABLE_ESTIMATION` (learn Z)
- `validation_frequency`: How often to compute validation metrics
- `early_stopping_patience`: Iterations without improvement before stopping
- `sub_trajectory_config`: Configuration for sub-trajectory objectives

## Training Workflow

### 1. Model Creation
```julia
model = create_gflownet(
    initial_state,
    all_actions;
    state_dim = state_dim,
    hidden_dim = 64,
    include_backward = false  # Enable for full trajectory balance
)
```

### 2. Configuration
```julia
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 1000,
    batch_size = 32,
    learning_rate = 0.01
)
```

### 3. Training
```julia
history = train_gflownet(model, config; verbose=true)
```

### 4. Evaluation
```julia
# Sample trajectories
trajectories = [sample_trajectory(model) for _ in 1:100]

# Analyze results
mean_reward = mean([reward(t.states[end]) for t in trajectories])
unique_terminals = unique([t.states[end] for t in trajectories])
```

## Optimization Methods

The framework supports various optimizers:

```julia
@enum OptimizationMethod begin
    ADAM      # Default, good for most cases
    RMSPROP   # Alternative for some domains
    SGD       # Basic stochastic gradient descent
    ADAMW     # Adam with weight decay
end
```

## Monitoring Training

### Verbose Output
When `verbose=true`, training provides regular updates:

```
🚀 Starting GFlowNet training...
   Configuration:
     - Objective: TRAJECTORY_BALANCE
     - Iterations: 1000
     - Batch size: 32
     - Learning rate: 0.01
   Iteration 100:
     - Loss: 2.3451
     - Avg Loss (5): 2.4123
     - Gradient norm: 1.2345
     - Time: 0.23s
```

### Training History
Access detailed metrics after training:

```julia
# Plot training loss
plot(history.losses, label="Training Loss")

# Check convergence
final_loss = history.losses[end]
converged = final_loss < 0.1
```

## Performance Tips

### Batch Size Selection
- Larger batches (32-64): More stable gradients, better GPU utilization
- Smaller batches (8-16): Faster iterations, more exploration

### Learning Rate Tuning
- Start with 0.01 for trajectory balance
- Use 0.001 for more complex objectives
- Consider learning rate scheduling for long training

### Early Stopping
```julia
config = TrainingConfig(
    # ... other params ...
    early_stopping_patience = 50,
    validation_frequency = 10
)
```

### Memory Management
- Monitor GPU memory with large state spaces
- Reduce batch size if out of memory
- Use gradient checkpointing for very deep models

## Troubleshooting

### High Loss Values
- Check reward scaling (should be positive)
- Verify state features are normalized
- Ensure actions are properly defined

### Slow Convergence
- Increase learning rate carefully
- Check if backward policy is needed
- Verify reward signal is informative

### Unstable Training
- Reduce learning rate
- Enable gradient clipping
- Check for numerical issues in rewards

## Advanced Usage

### Custom Training Loop
For more control, implement a custom training loop:

```julia
for iter in 1:n_iterations
    # Sample trajectories
    trajectories = [sample_trajectory(model) for _ in 1:batch_size]
    
    # Compute loss
    loss = trajectory_balance_loss(model, trajectories)
    
    # Update parameters
    # ... gradient computation and optimization ...
end
```

### Multi-GPU Training
Currently not implemented, but planned for future releases.

### Checkpointing
Save and restore training state:

```julia
# Save model
save_model(model, "checkpoint.jld2")

# Load model
model = load_model("checkpoint.jld2")
```

## See Also
- [Training Objectives](objectives.md) - Detailed objective descriptions
- [API Reference](../api/training.md) - Complete API documentation
- [Examples](../guide/examples.md) - Working training examples