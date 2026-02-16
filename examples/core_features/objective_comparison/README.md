# DETAILED_BALANCE vs TRAJECTORY_BALANCE Comparison

This example demonstrates the differences between two core GFlowNet training objectives.

## Overview

GFlowNets support multiple training objectives that enforce flow conservation in different ways:

1. **TRAJECTORY_BALANCE (TB)**: Uses complete trajectory probabilities
   - Loss: `(log Z + log P_F(τ) - log R(s_T))²`
   - Can work without backward policy
   - Simpler to implement

2. **DETAILED_BALANCE (DB)**: Enforces local balance constraints
   - Loss: `(log P_F(s→s') + log F(s) - log P_B(s'→s) - log F(s'))²`
   - Requires backward policy
   - Better credit assignment for complex problems

## Running the Example

```bash
cd examples/core_features/objective_comparison
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. objective_comparison.jl
```

## What This Example Shows

1. **Training Dynamics**: How losses evolve for each objective
2. **State Exploration**: Visitation patterns differ between objectives
3. **Backward Policy**: DB learns reverse transition probabilities
4. **Performance Metrics**: Comparing rewards, diversity, and efficiency

## Key Results

The example generates:
- `results/objective_comparison.png`: Visual comparison
- `results/objective_comparison_metrics.csv`: Quantitative metrics

## Typical Observations

1. **Convergence**: DB often converges to lower loss values
2. **Exploration**: DB may explore more uniformly due to backward policy
3. **Credit Assignment**: DB better handles sparse rewards
4. **Computational Cost**: DB requires more computation (backward policy)

## When to Use Each Objective

### Use TRAJECTORY_BALANCE when:
- Problem has dense rewards
- Trajectories are relatively short
- Simplicity is important
- No need for reverse transitions

### Use DETAILED_BALANCE when:
- Problem has sparse or delayed rewards
- Need better credit assignment
- Reverse transitions are meaningful
- Local constraints are important

## Mathematical Details

### Trajectory Balance
Enforces global conservation over entire trajectories:
```
Z · P_F(τ) = R(s_T)
```

### Detailed Balance
Enforces local conservation at each edge:
```
P_F(s→s') · F(s) = P_B(s'→s) · F(s')
```

Where:
- `P_F(s→s')`: Forward transition probability
- `P_B(s'→s)`: Backward transition probability
- `F(s)`: Flow through state s

## Implementation Notes

The backward policy uses joint state representation:
```julia
features = [state_features(s); state_features(s')]
P_B(s|s') = σ(NN(features))
```

This allows learning complex dependencies between states.