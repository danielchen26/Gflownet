# Training

This page documents the training procedures and objectives in GFlowNet.jl.

## Core Training Functions

* `train!(model, trajectories, iterations)`: Main training function
* `compute_loss_and_grad(model, trajectories)`: Computes loss and gradient
* `apply_optimizer!(model, gradient)`: Applies gradient updates

## Training Objectives

GFlowNet.jl implements three main training objectives:

### Flow Matching

The Flow Matching objective ensures that the flow into each non-terminal state equals the flow out of it:

$$\mathcal{L}_{\text{FM}}(s) = \left( F(s) - \sum_{s' \in \text{Parents}(s)} F(s', s) \right)^2$$

This objective directly enforces flow conservation within the state space.

### Detailed Balance

The Detailed Balance objective ensures that the product of forward and backward probabilities is consistent with the flow ratio:

$$\mathcal{L}_{\text{DB}}(s, s') = \left( \log \frac{F(s')}{F(s)} - \log \frac{P_F(s'|s)}{P_B(s|s')} \right)^2$$

This local objective can provide more stable training in some cases.

### Trajectory Balance

The Trajectory Balance objective enforces consistency of flow along complete trajectories:

$$\mathcal{L}_{\text{TB}}(\tau) = \left( \log R(s_T) - \log F(s_0) - \sum_{t=0}^{T-1} \log P_F(s_{t+1}|s_t) \right)^2$$

Where $\tau = (s_0, s_1, \ldots, s_T)$ is a complete trajectory.

## Training Loop

A typical training loop in GFlowNet.jl involves:

1. Sampling trajectories from the current policy
2. Computing losses using one or more objectives
3. Computing gradients and updating parameters
4. Periodically evaluating the model's performance

## Code Example

```julia
function train_gflownet()
    # Create a GFlowNet model
    model = create_model(...)
    
    # Training parameters
    batch_size = 32
    iterations = 10000
    
    # Training loop
    for iter in 1:iterations
        # Sample trajectories
        trajectories = [sample_trajectory(model) for _ in 1:batch_size]
        
        # Train on the batch
        loss = train!(model, trajectories)
        
        # Logging and evaluation
        if iter % 100 == 0
            println("Iteration $iter: Loss = $loss")
            
            # Evaluate model
            terminal_states = [trajectory.states[end] for trajectory in trajectories]
            rewards = [reward(state) for state in terminal_states]
            println("Mean reward: $(mean(rewards))")
            println("Max reward: $(maximum(rewards))")
        end
    end
    
    return model
end

# Advanced example with multiple objectives
function train_with_multiple_objectives()
    # Create model with multiple objectives
    model = GFlowNetModel(
        # ... other parameters ...
        objectives = [
            FlowMatchingObjective(weight=0.5),
            DetailedBalanceObjective(weight=0.3),
            TrajectoryBalanceObjective(weight=0.2)
        ]
    )
    
    # Custom training loop
    for iter in 1:10000
        trajectories = sample_batch(model, 32)
        
        # Compute losses for each objective
        fm_loss = flow_matching_loss(model, trajectories)
        db_loss = detailed_balance_loss(model, trajectories)
        tb_loss = trajectory_balance_loss(model, trajectories)
        
        # Compute weighted loss
        total_loss = 0.5*fm_loss + 0.3*db_loss + 0.2*tb_loss
        
        # Compute gradients and update
        grads = compute_gradient(model, total_loss)
        apply_optimizer!(model, grads)
    end
    
    return model
end
```

## Advanced Training Techniques

### Multiple Objectives

You can combine multiple training objectives with different weights:

```julia
model = GFlowNetModel(
    # ...other parameters...
    objectives = [
        TrajectoryBalanceObjective(0.8),
        FlowMatchingObjective(0.2)
    ],
    # ...other parameters...
)
```

### Custom Training Loops

For more control, you can implement your own training loop:

```julia
for iter in 1:n_iterations
    # Sample batch of trajectories
    trajectories = [sample_trajectory(model) for _ in 1:batch_size]
    
    # Compute loss and gradients
    loss, grads = compute_loss_and_grad(model, trajectories)
    
    # Update parameters
    apply_optimizer!(model, grads)
    
    # Logging
    if iter % log_interval == 0
        println("Iteration $iter: Loss = $loss")
    end
end
```

### Hyperparameter Tuning

Important hyperparameters to tune include:
- Learning rate
- Batch size
- Model architecture
- Balance of multiple objectives (if used)
- Exploration parameters
