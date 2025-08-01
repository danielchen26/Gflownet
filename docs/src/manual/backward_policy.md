# Backward Policy Implementation

## Overview

GFlowNet.jl now supports full trajectory balance with learned backward policies, enabling more accurate credit assignment without requiring explicit DAG construction.

## Creating Models with Backward Policy

### Basic Usage

Enable backward policy during model creation:

```julia
# Enable backward policy
model = create_gflownet(
    initial_state,
    all_actions;
    state_dim = state_dim,
    hidden_dim = 64,
    include_backward = true  # Enable backward policy
)
```

### Domain-Specific Example

```julia
# Grid world with backward policy
model = create_grid_world_gflownet(
    grid_size = 5,
    hidden_dim = 64,
    include_backward = true
)
```

## How It Works

The backward policy uses a **joint state representation** approach:

1. Takes both source and target state features as input
2. Outputs P_B(source|target) directly
3. No need to enumerate all possible parent states

### Architecture

```julia
# Input: concatenated features
input_features = [source_state_features; target_state_features]

# Output: single probability
P_B(source|target) = sigmoid(network(input_features))
```

## Implementation Details

### Network Architecture

The backward policy network follows the same pattern as the forward policy:

```julia
function create_backward_policy(state_dim::Int, hidden_dim::Int, rng)
    input_dim = 2 * state_dim  # Concatenated source + target features
    
    backward_net = Chain(
        Dense(input_dim => hidden_dim, tanh),
        Dense(hidden_dim => hidden_dim, tanh),
        Dense(hidden_dim => 1)  # Single probability logit
    )
    
    ps, st = Lux.setup(rng, backward_net)
    return BackwardPolicy(backward_net), ps, st
end
```

### Computing Backward Probabilities

```julia
function compute_backward_probability(policy, target_state, source_state, 
                                    params, states, actions)
    # Check if transition is valid
    if !is_valid_backward_transition(source_state, target_state, actions)
        return 0.0
    end
    
    # Get features for both states
    source_features = state_to_features(source_state)
    target_features = state_to_features(target_state)
    
    # Concatenate features for joint representation
    joint_features = vcat(source_features, target_features)
    
    # Compute probability
    logit, _ = policy.model(joint_features, params, states)
    return sigmoid(logit[1])
end
```

## Trajectory Balance with Backward Policy

### Mathematical Formulation

The full trajectory balance formula:
```
log Z(s₀) + Σ log P_F(s_{i+1}|s_i) - log R(s_T) - Σ log P_B(s_i|s_{i+1}) = 0
```

### Loss Computation

```julia
function trajectory_balance_loss(model, trajectory)
    # Forward probability term
    log_P_F = compute_forward_log_prob(model, trajectory)
    
    # Reward term
    log_R = log(reward(trajectory.states[end]))
    
    # Backward probability term (if enabled)
    log_P_B = 0.0
    if !isnothing(model.backward_policy)
        log_P_B = compute_backward_log_prob(model, trajectory)
    end
    
    # Trajectory balance loss (assuming Z = 1)
    loss = (log_P_F - log_R - log_P_B)²
    return loss
end
```

## When to Use Backward Policy

### Recommended For

1. **Non-deterministic environments**
   - Multiple paths to reach the same state
   - Stochastic transitions

2. **Complex credit assignment**
   - Long trajectories with delayed rewards
   - When early actions critically affect outcomes

3. **Better exploration**
   - Backward policy helps distribute credit more accurately
   - Can discover alternative paths more effectively

### Optional For

1. **Simple deterministic paths**
   - Basic grid worlds with unique paths
   - When computational efficiency is critical

2. **Short trajectories**
   - When credit assignment is straightforward
   - Minimal benefit from backward policy

## Performance Considerations

### Training Time
- **Without backward**: Baseline speed
- **With backward**: ~1.5-2x slower
  - Additional network forward pass
  - Larger parameter space

### Memory Usage
- Additional parameters: `2 * state_dim * hidden_dim`
- Negligible for most applications

### Convergence
- Often better final performance
- May require more iterations initially
- More stable credit assignment

## Example Comparison

### Without Backward Policy
```julia
model_simple = create_grid_world_gflownet(
    grid_size = 5,
    include_backward = false
)

# Training uses only forward probabilities
# Loss: (log P_F(τ) - log R(s_T))²
```

### With Backward Policy
```julia
model_full = create_grid_world_gflownet(
    grid_size = 5,
    include_backward = true
)

# Training uses both forward and backward
# Loss: (log P_F(τ) - log R(s_T) - log P_B(τ))²
```

## Verification Example

```julia
# Sample a trajectory
traj = sample_trajectory(model_with_backward)

# Check probabilities
s1, s2 = traj.states[1], traj.states[2]

# Forward probability
p_forward = forward_transition_probability(model, s1, s2)
println("P_F($s2|$s1) = $p_forward")

# Backward probability
p_backward = backward_transition_probability(model, s2, s1)
println("P_B($s1|$s2) = $p_backward")

# In a well-trained model, these should satisfy
# detailed balance approximately
```

## Advanced Usage

### Custom Backward Policy Architecture

```julia
function create_custom_backward_policy(state_dim, hidden_dim, rng)
    # Example: asymmetric architecture
    backward_net = Chain(
        Dense(2 * state_dim => 2 * hidden_dim, relu),
        Dense(2 * hidden_dim => hidden_dim, relu),
        Dropout(0.1),
        Dense(hidden_dim => 1)
    )
    
    ps, st = Lux.setup(rng, backward_net)
    return BackwardPolicy(backward_net), ps, st
end
```

### Separate Learning Rates

```julia
# Different learning rates for forward/backward
optimizer = Optimisers.Adam(1e-3)
backward_optimizer = Optimisers.Adam(5e-4)
```

## Limitations and Future Work

### Current Limitations
1. Joint representation may be inefficient for very high-dimensional states
2. No explicit normalization guarantee for P_B
3. Requires valid transition checking

### Future Improvements
1. Attention-based architectures for state pairs
2. Normalizing flow backends
3. Conditional normalization strategies

## Troubleshooting

### High Backward Loss
- Check if transitions are valid
- Verify state features are consistent
- May need different architecture

### Slow Convergence
- Try different learning rates for backward policy
- Increase hidden dimension
- Check reward scaling

### Memory Issues
- Reduce batch size
- Use smaller hidden dimensions
- Consider gradient checkpointing

## See Also
- [Training Objectives](objectives.md) - Using backward policy with different objectives
- [Architecture Analysis](../internals/architecture.md) - Implementation details
- [Mathematical Background](../theory/mathematical_background.md) - Theoretical foundations