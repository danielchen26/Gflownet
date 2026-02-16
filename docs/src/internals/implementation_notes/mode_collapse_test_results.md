# Mode Collapse Test Results

Confirmed test results on 5×5 grid with 70:1 path asymmetry.
Run date: February 2025.

## Test Setup

- **Grid**: 5×5, start at (1,1), only MoveRight/MoveUp allowed
- **Peaks**: (5,5) R=10 and (1,5) R=10 (equal rewards)
- **Path Asymmetry**: Peak(5,5) has 70 paths (binomial(8,4)), Peak(1,5) has 1 path
- **Expected**: With equal rewards, GFlowNet samples proportional to R(x) × #paths(x)
- **Training**: 1000 iterations, batch=32, TB objective, lr=0.005
- **Evaluation**: 1000 samples with ε=0 (pure policy)

## Individual Technique Results

| Technique | Peak(5,5) | Peak(1,5) | Both Modes? | Notes |
|-----------|-----------|-----------|-------------|-------|
| TB baseline (no features) | 802 | 0 | No | Complete mode collapse |
| TB + ε=0.3, entropy=0.1 | 729-730 | 0-1 | No | Exploration alone insufficient |
| TB + replay buffer alone | 582 | 5 | No | Better but not enough |
| TB + ALL features (ε+entropy+replay+Z×10) | ~490 | ~8 | Barely | Still mostly collapsed |
| **TB + replay + reward shaping (R=700)** | **456** | **303** | **Yes** | **Both modes discovered** |

## Reward Shaping Details

Test file: `examples/core_features/reward_shaping_test.jl`

Configuration:
- Peak(5,5): R=10, 70 paths → R×paths = 700
- Peak(1,5): R=700, 1 path → R×paths = 700 (balanced!)
- TB + replay buffer + ε=0.1 + entropy=0.01

Result: Peak(5,5)=456, Peak(1,5)=303, Ratio=1.5:1, Modes=2/2

## Key Findings

1. **Reward shaping is essential** for overcoming structural path asymmetry
2. **Replay buffer helps** retain discovered modes but cannot overcome 70:1 asymmetry alone
3. **TLM (backward policy training)** is 2× slower per iteration with no clear advantage when reward shaping is active
4. **ε-exploration + entropy** help exploration but cannot fix structural asymmetry

## Recommended Default Configuration

For visualization system (TB + replay + reward shaping):

```
objective: TRAJECTORY_BALANCE
temperature: 1.0
epsilon: 0.15 (with decay)
entropy_weight: 0.02
z_learning_rate_multiplier: 10.0
use_replay_buffer: true
replay_buffer_size: 10000
replay_ratio: 0.5
reward_shaping: true (auto-compensate path asymmetry)
```

## Path Count Formula

For a grid world with start at (1,1), only Right/Up moves:
- To reach position (x, y): need (x-1) Rights + (y-1) Ups
- Number of distinct paths: `binomial(x + y - 2, x - 1)`

Examples:
- 5×5 grid: (5,5)=70 paths, (1,5)=1 path → 70:1 ratio
- 8×8 grid: (8,8)=3432 paths, (1,8)=1 path → 3432:1 ratio

## Reward Shaping Formula

For each peak at position (x,y) with raw reward R:
```
path_count = binomial(x + y - 2, x - 1)
max_paths = maximum path count among all peaks
adjusted_R = R × (max_paths / path_count)
```

This compensates for structural path asymmetry so the GFlowNet can
discover all modes with proportional sampling.
