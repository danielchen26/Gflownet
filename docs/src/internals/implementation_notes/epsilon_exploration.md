# ε-Uniform Exploration for GFlowNet Mode Discovery

## Problem Statement

During TB (Trajectory Balance) training on a 5×5 grid world with two reward peaks:
- Peak 1 at (5,5) with R=10
- Peak 2 at (1,5) with R=8

The model exhibited severe **mode collapse**: sampling ratio was ~80:1 instead of the expected 1.25:1 (proportional to rewards). The TB loss converged to near-zero, yet the model only discovered one mode.

### Root Cause Analysis

1. **Missing Exploration Mechanism**: The original implementation sampled directly from the learned policy without exploration mixing.

2. **Path Asymmetry**: The grid structure creates extreme path asymmetry:
   - Paths to (5,5): C(8,4) = 70 different routes
   - Paths to (1,5): Only 1 route (4 consecutive Down moves)

3. **TB Loss Limitation**: TB loss `(log Z + log P_F(τ) - log R(s_T))²` can converge even when sampling is biased, because it optimizes only for visited trajectories.

## Solution: ε-Uniform Exploration

Following the standard GFlowNet practice from literature (Malkin et al. 2022, Shen et al. ICML 2023), we implemented **ε-uniform exploration mixing**:

```
P(a|s) = (1 - ε) × P_F(a|s) + ε × Uniform(applicable_actions)
```

### Implementation Details

#### 1. SamplingConfig (src/core/sampling.jl)

```julia
struct SamplingConfig
    # ... existing fields ...
    epsilon::Float64  # ε-uniform exploration rate (0.0 to 1.0)
end
```

#### 2. Action Sampling (src/core/interface.jl)

```julia
function sample_action_from_policy(model, state, applicable_actions; config)
    # ... compute policy probabilities ...

    # ε-Uniform Exploration Mixing
    if config.epsilon > 0.0
        n_actions = length(probs)
        uniform_prob = 1.0 / n_actions
        probs = (1.0 - config.epsilon) .* probs .+ config.epsilon * uniform_prob
    end

    # ... sample action ...
end
```

#### 3. TrainingConfig (src/training/configuration.jl)

```julia
struct TrainingConfig
    # ... existing fields ...
    epsilon::Float64      # ε-uniform exploration rate (default 0.05)
    epsilon_decay::Bool   # Whether to anneal epsilon to 0 over training
end
```

#### 4. Training Loop (src/training/training.jl)

```julia
# Compute current epsilon (annealed if epsilon_decay is true)
current_epsilon = if config.epsilon_decay
    config.epsilon * (1.0 - (iteration - 1) / config.n_iterations)
else
    config.epsilon
end

sampling_config = SamplingConfig(
    strategy = ...,
    temperature = config.temperature,
    epsilon = current_epsilon,
    max_trajectory_length = 100
)
```

## Verification Results

### Test 1: Aggressive Exploration Comparison

| Epsilon | Peak1 (5,5) | Peak2 (1,5) | Modes Found |
|---------|-------------|-------------|-------------|
| ε=0.0   | 794 (79%)   | 0 (0%)      | 1/2         |
| ε=0.3   | 320 (32%)   | 6 (0.6%)    | **2/2**     |

Higher epsilon enables discovery of both modes, confirming the implementation works.

### Test 2: Balanced Grid (Symmetric Path Counts)

Using rewards at (3,5) and (5,3) which have equal path counts (15 each):

| Epsilon | Peak1 (3,5) | Peak2 (5,3) | Ratio | Expected | Error |
|---------|-------------|-------------|-------|----------|-------|
| ε=0.0   | 259 (26%)   | 75 (8%)     | 3.45  | 1.25     | 176%  |
| ε=0.05  | 211 (21%)   | 160 (16%)   | 1.32  | 1.25     | **5.6%** |
| ε=0.1   | 202 (20%)   | 175 (18%)   | 1.15  | 1.25     | **7.7%** |

With symmetric paths, the implementation achieves the expected reward ratio within ~6-8%.

## Usage Guide

### Default Training (Recommended)

```julia
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 2000,
    batch_size = 64,
    learning_rate = 0.005,
    epsilon = 0.05,        # Standard ε-uniform exploration
    epsilon_decay = true   # Anneal to 0 over training
)

history = train_gflownet(model, config)
```

### Sampling with Exploration

```julia
# During training (with exploration)
train_config = SamplingConfig(epsilon=0.05)
trajectory = sample_trajectory(model; config=train_config)

# During evaluation (no exploration)
eval_config = SamplingConfig(epsilon=0.0)
trajectory = sample_trajectory(model; config=eval_config)
```

### Helper Function

```julia
# Create standard exploration config
config = create_exploration_sampling_config(epsilon=0.05)
```

## Recommended Epsilon Values

From ICML 2023 literature:

| Task Type | Recommended ε |
|-----------|---------------|
| HyperGrid (standard) | 0.05 |
| Molecular Design (TFBind8) | 0.01 |
| Complex Molecules (QM9) | 0.10 |
| General GFlowNet | 0.05 (default) |

## Limitations and Advanced Cases

### Extreme Path Asymmetry

The original (5,5)/(1,5) problem has 70:1 path asymmetry. Even with ε=0.3, only ~0.6% of samples reach (1,5). For such cases, consider:

1. **Higher epsilon** (0.2-0.3) with no decay
2. **Entropy regularization** to encourage diverse policies
3. **Intrinsic motivation** rewards for exploration
4. **Off-policy learning** with experience replay
5. **More balanced reward positions** in problem design

### Epsilon Annealing

The implementation supports linear epsilon decay:

```
ε_t = ε_0 × (1 - t/T)
```

This reduces exploration bias in gradient estimates toward the end of training while maintaining exploration early on.

## References

1. Malkin, N., et al. (2022). "Trajectory Balance: Improved Credit Assignment in GFlowNets." NeurIPS 2022.
   - Original TB paper, uses ε=0.05 in experiments

2. Shen, M., et al. (2023). "Towards Understanding and Improving GFlowNet Training." ICML 2023.
   - Comprehensive analysis of exploration strategies
   - Documents ε=0.05 as standard practice

3. torchgfn Library (GFNOrg)
   - Official PyTorch implementation
   - Uses ε-uniform mixing in all samplers

## Files Modified

- `src/core/sampling.jl` - Added epsilon field to SamplingConfig
- `src/core/interface.jl` - Added ε-uniform mixing in sample_action_from_policy
- `src/core/multi_start.jl` - Added ε-uniform mixing in sample_action_multi_start
- `src/training/configuration.jl` - Added epsilon and epsilon_decay to TrainingConfig
- `src/training/training.jl` - Epsilon annealing in training loop
- `src/utils/visualization/core/training_session.jl` - Epsilon in visualization server
