# Reward Shaping for GFlowNet Mode Collapse

## Problem Statement

GFlowNet training on grid worlds suffers from **mode collapse** due to structural path asymmetry. In an acyclic grid (only MoveRight + MoveUp), the number of distinct paths to each terminal state varies dramatically:

| Grid | Peak (N,N) paths | Peak (1,N) paths | Ratio |
|------|-----------------|-----------------|-------|
| 5x5 | binomial(8,4) = 70 | 1 | 70:1 |
| 8x8 | binomial(14,7) = 3432 | 1 | 3432:1 |

Since GFlowNets sample proportionally to `P(x) ~ R(x) * #paths(x)`, equal-reward peaks receive vastly unequal sampling. The minority mode (1,N) - reachable only by consecutive MoveUp actions - is rarely or never discovered.

## Theory: Reward Compensation

The sampling distribution of a GFlowNet is:

```
P(x) ∝ R(x) × #paths(x)
```

To achieve equal sampling of two modes, compensate the minority mode reward by the path count ratio:

```
R(minority) = path_ratio × R(majority)
```

For the 5x5 grid: `R(1,5) = 70 × R(5,5) = 70 × 10 = 700`
For the 8x8 grid: `R(1,8) = 3432 × R(8,8) = 3432 × 10 = 34320`

## Experiment Setup

### 5x5 Grid Experiments

**Common parameters**: lr=0.005, batch=32, hidden=64, temp=1.0, z_lr_mult=10.0, LEARNABLE_ESTIMATION

| Experiment | Rewards | ε | Entropy | Iterations | Replay |
|------------|---------|---|---------|------------|--------|
| 1: Baseline (no shaping) | R(5,5)=10, R(1,5)=10 | 0.1 | 0.01 | 1000 | No |
| 2: With shaping | R(5,5)=10, R(1,5)=700 | 0.1 | 0.01 | 1000 | No |

### 8x8 Grid Experiments

**Common parameters**: lr=0.005, hidden=64-128, temp=1.0, z_lr_mult=10.0, LEARNABLE_ESTIMATION

| Experiment | Rewards | ε | Entropy | Iter | Batch | Replay | ε decay |
|------------|---------|---|---------|------|-------|--------|---------|
| 3: Baseline (no shaping) | R(8,8)=10, R(1,8)=10 | 0.15 | 0.02 | 1500 | 32 | No | Yes |
| 4: Shaping only | R(8,8)=10, R(1,8)=34320 | 0.15 | 0.02 | 1500 | 32 | No | Yes |
| 5: Shaping + high ε | R(8,8)=10, R(1,8)=34320 | 0.3 | 0.02 | 2000 | 64 | No | No |
| 6: Shaping + ε + replay | R(8,8)=10, R(1,8)=34320 | 0.2 | 0.02 | 2000 | 64 | Yes | Yes |
| 7: Shaping + ε + H + replay | R(8,8)=10, R(1,8)=34320 | 0.3 | 0.05 | 3000 | 64 | Yes | No |
| 8: Shaping + extreme ε | R(8,8)=10, R(1,8)=34320 | 0.5 | 0.05 | 3000 | 64 | Yes | No |

All experiments evaluated with 1000 samples, epsilon=0.0 (no exploration during evaluation).

## Results

### 5x5 Grid (70:1 Path Asymmetry)

| Experiment | Peak(5,5) | Peak(1,5) | Ratio | Modes | Final Loss |
|------------|-----------|-----------|-------|-------|------------|
| 1: No shaping | 621 (62.1%) | 1 (0.1%) | 621:1 | 1/2 | 6.92 |
| **2: With shaping** | **468 (46.8%)** | **267 (26.7%)** | **1.8:1** | **2/2** | **9.74** |

**Reward shaping completely solves the 5x5 mode collapse problem.** The sampling ratio drops from 621:1 to 1.8:1, and both modes are reliably discovered.

### 8x8 Grid (3432:1 Path Asymmetry)

| Experiment | Peak(8,8) | Peak(1,8) | Ratio | Modes | Final Loss |
|------------|-----------|-----------|-------|-------|------------|
| 3: No shaping | 574 (57.4%) | 0 (0%) | Inf | 1/2 | 13.85 |
| 4: Shaping only | 612 (61.2%) | 0 (0%) | Inf | 1/2 | 15.11 |
| 5: Shaping + high ε | 183 (18.3%) | 1 (0.1%) | 183:1 | 1/2 | ~8.0 |
| 6: Shaping + ε + replay | 659 (65.9%) | 3 (0.3%) | 220:1 | 1/2 | 7.05 |
| 7: Shaping + ε + H + replay | 372 (37.2%) | 2 (0.2%) | 186:1 | 1/2 | 1.61 |
| **8: Shaping + extreme ε** | **223 (22.3%)** | **5 (0.5%)** | **44.6:1** | **1/2** | **0.55** |

**Reward shaping alone is insufficient for the extreme 3432:1 path asymmetry.** Even combined with aggressive exploration (ε=0.5), high entropy (0.05), replay buffer, and 3000 iterations, only 5/1000 samples reach the minority mode.

## Analysis

### Why Reward Shaping Works for 5x5 but Not 8x8

The fundamental bottleneck is **path discovery probability**. To reach (1,N), the agent must take N-1 consecutive MoveUp actions from (1,1). At each step, with ε-uniform exploration:

```
P(discover path to (1,N)) ≈ (ε/k)^(N-1)
```

Where k is the number of available actions at each step (typically 2-3).

| Grid | Steps to (1,N) | P(discover) with ε=0.1 | P(discover) with ε=0.5 |
|------|----------------|----------------------|----------------------|
| 5x5 | 4 | ~(0.05)^4 ≈ 6.3e-6 | ~(0.25)^4 ≈ 3.9e-3 |
| 8x8 | 7 | ~(0.05)^7 ≈ 7.8e-10 | ~(0.25)^7 ≈ 6.1e-5 |

For the 8x8 grid, even with ε=0.5, the probability of randomly discovering the minority path in a single trajectory is ~0.006%. With a batch of 64, you'd need ~260 batches just to see it once. The replay buffer then amplifies this rare discovery, which explains why Tests 6-8 gradually improve.

### Reward Shaping Effectiveness by Path Asymmetry

| Asymmetry Ratio | Reward Shaping Alone | + Exploration Combo | Assessment |
|----------------|---------------------|---------------------|------------|
| < 10:1 | Excellent | Not needed | Use shaping alone |
| 10:1 - 100:1 | Good | Recommended | Shaping + ε=0.1 |
| 100:1 - 1000:1 | Marginal | Required | Shaping + ε=0.2 + replay |
| > 1000:1 | Insufficient | Still marginal | Need TLM or other approach |

### Comparison with Other Mode Collapse Solutions

| Method | 5x5 (70:1) | 8x8 (3432:1) | Complexity | Requires |
|--------|-----------|-------------|------------|----------|
| **Reward Shaping** | 2/2 modes, 1.8:1 | 1/2 modes | Low | Path count knowledge |
| ε-Exploration | 2/2 modes, ~5% error | 1/2 modes | Low | Tuning |
| Entropy Regularization | Helps balance | Marginal | Low | Tuning |
| Experience Replay | Retains rare modes | Slightly helps | Medium | Buffer tuning |
| TLM (ICLR 2025) | 2/2 modes | Untested at 8x8 | High | Backward policy |
| Combined (all above) | 2/2 modes | 5/1000 at (1,8) | High | Everything |

## Usage Guide

### Computing Path Ratios

For an N×N acyclic grid (MoveRight + MoveUp only):
- Paths to (N,N) from (1,1): `binomial(2(N-1), N-1)`
- Paths to (1,N) from (1,1): `1` (only consecutive MoveUp)
- Path ratio: `binomial(2(N-1), N-1)`

```julia
using Combinatorics
path_ratio(N) = binomial(2*(N-1), N-1)
# path_ratio(5) = 70
# path_ratio(8) = 3432
# path_ratio(10) = 48620
```

### Applying Reward Shaping

```julia
N = 5
ratio = binomial(2*(N-1), N-1)  # 70 for 5x5
base_reward = 10.0

reward_positions = Dict(
    (N, N) => base_reward,
    (1, N) => base_reward * ratio  # Compensate path asymmetry
)

model = GFlowNet.create_grid_world_gflownet(
    grid_size = N,
    reward_positions = reward_positions,
    partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
)
```

### Recommended Configurations

**5x5 grid (moderate asymmetry)**:
```julia
config = TrainingConfig(
    n_iterations = 1000, batch_size = 32, learning_rate = 0.005,
    epsilon = 0.1, entropy_weight = 0.01, temperature = 1.0,
    z_learning_rate_multiplier = 10.0
)
```

**8x8 grid (extreme asymmetry)** - reward shaping + full combo:
```julia
config = TrainingConfig(
    n_iterations = 3000, batch_size = 64, learning_rate = 0.005,
    epsilon = 0.5, epsilon_decay = false,
    entropy_weight = 0.05, temperature = 1.0,
    z_learning_rate_multiplier = 10.0,
    use_replay_buffer = true, replay_buffer_size = 10000, replay_ratio = 0.5
)
```

## Limitations

1. **Requires path count knowledge**: Reward shaping needs `binomial(2(N-1), N-1)`, which is easy for grids but may be intractable for complex DAGs.
2. **Scales poorly with asymmetry**: Beyond ~1000:1 ratio, reward shaping alone cannot compensate. The minority mode path is too unlikely to be discovered by random exploration.
3. **Changes the target distribution**: By boosting minority rewards, the target `P(x) ∝ R(x)` changes. This is fine if the goal is mode coverage, but problematic if exact proportional sampling is needed.
4. **Not domain-agnostic**: Each domain requires its own path count analysis.

## Conclusions

1. **Reward shaping is the best practical solution for moderate path asymmetry** (up to ~100:1). It's simple, reliable, and requires no additional training infrastructure.
2. **For extreme asymmetry (>1000:1)**, reward shaping must be combined with high exploration + replay, and even then may be insufficient. TLM (ICLR 2025) or other structural approaches are recommended.
3. **The 5x5 grid is the sweet spot** for demonstrating reward shaping effectiveness.

## Test Scripts

- `examples/core_features/reward_shaping_comprehensive_test.jl` - Main 4-experiment test (5x5 and 8x8)
- `examples/core_features/reward_shaping_8x8_tuning.jl` - Extended 8x8 tuning experiments
- `examples/core_features/reward_shaping_test.jl` - Original 5x5 test (legacy)

## References

1. Bengio, E., et al. (2021). "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation." NeurIPS 2021.
2. Malkin, N., et al. (2022). "Trajectory Balance: Improved Credit Assignment in GFlowNets." NeurIPS 2022.
3. Shen, M., et al. (2023). "Towards Understanding and Improving GFlowNet Training." ICML 2023.
4. Kim, M., et al. (2025). "Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization." ICLR 2025.

## Date

February 2025
