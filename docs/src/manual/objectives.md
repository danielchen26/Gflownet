# Training Objectives

This page describes all available training objectives in GFlowNet.jl and when to use each one.

## Overview

GFlowNet.jl supports multiple training objectives, each suited for different types of problems:

| Objective | Best For | Key Feature |
|-----------|----------|-------------|
| Trajectory Balance | Simple deterministic paths | Standard, well-tested |
| Sub-Trajectory Balance | Long sequences needing credit assignment | Better gradient flow |
| Hierarchical Sub-TB | Multi-scale structures | Scale-aware learning |
| Adaptive Sub-TB | Sparse rewards | Intelligent sampling |
| Flow Consistency | Local structure learning | Unifies DB + FM |

## Trajectory Balance (Standard)

```julia
objective = TRAJECTORY_BALANCE
```

### Use For
- Simple deterministic paths where each state has unique parent
- Grid worlds, simple sequential construction
- When you want the most tested, stable objective

### Mathematical Formulation
Without backward policy: 
$$L = (\log P_F(\tau) - \log R(s_T))^2$$

With backward policy:
$$L = (\log P_F(\tau) - \log R(s_T) - \log P_B(\tau))^2$$

### Example
```julia
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 1000,
    batch_size = 32
)
```

### Notes
- Assumes $Z = 1$ by default (valid for fixed initial state)
- Now supports learnable $Z$ with `partition_function_method=LEARNABLE_ESTIMATION`
- Supports optional backward policy with `include_backward=true`
- Most efficient for simple domains

## General Trajectory Balance (Not Yet Implemented)

```julia
objective = GENERAL_TRAJECTORY_BALANCE  # Currently not available
```

### Planned Features
- Non-deterministic paths with multiple ways to reach states
- Full backward policy support
- Better for graph generation, causal discovery

### Workaround
Use `TRAJECTORY_BALANCE` with `include_backward=true` for similar functionality.

## Sub-Trajectory Balance (Credit Assignment)

```julia
objective = SUB_TRAJECTORY_BALANCE
sub_trajectory_config = Dict(
    :min_length => 2,
    :max_length => 5,
    :n_subtrajectories => 3
)
```

### Use For
- Long sequences where credit assignment is difficult
- Sequential decision making (active learning, experiment design)
- When full trajectories have sparse rewards

### How It Works
- Samples sub-trajectories of varying lengths
- Applies trajectory balance to each sub-trajectory
- Improves gradient flow to early actions

### Configuration Parameters
- `min_length`: Minimum sub-trajectory length
- `max_length`: Maximum sub-trajectory length  
- `n_subtrajectories`: Number to sample per full trajectory

### Example
```julia
config = TrainingConfig(
    objective = SUB_TRAJECTORY_BALANCE,
    sub_trajectory_config = Dict(
        :min_length => 3,
        :max_length => 10,
        :n_subtrajectories => 5
    ),
    n_iterations = 1000
)
```

## Hierarchical Sub-Trajectory Balance (Multi-Scale)

```julia
objective = HIERARCHICAL_SUB_TB
sub_trajectory_config = Dict(
    :scales => [2, 4, 8, 16],
    :n_subtrajectories => 5
)
```

### Use For
- Multi-scale structures with hierarchical components
- Molecular design with functional groups
- Any domain with natural hierarchies

### How It Works
- Samples sub-trajectories at multiple scales
- Learns patterns at different granularities
- Balances local and global objectives

### Configuration Parameters
- `scales`: List of sub-trajectory lengths to use
- `n_subtrajectories`: Number per scale

### Example
```julia
# For molecule generation
config = TrainingConfig(
    objective = HIERARCHICAL_SUB_TB,
    sub_trajectory_config = Dict(
        :scales => [3, 6, 12, 24],  # Atom, group, fragment, molecule
        :n_subtrajectories => 4
    )
)
```

## Adaptive Sub-Trajectory Balance (Intelligent)

```julia
objective = ADAPTIVE_SUB_TB
sub_trajectory_config = Dict(
    :difficulty_threshold => 0.05,
    :n_subtrajectories => 8
)
```

### Use For
- Sparse important decisions with varying difficulty
- Feature acquisition with different costs
- Complex decision making with rare critical choices

### How It Works
- Identifies difficult decision points
- Focuses sub-trajectory sampling on hard regions
- Adapts to the learning progress

### Configuration Parameters
- `difficulty_threshold`: Threshold for identifying hard decisions
- `n_subtrajectories`: Base number of sub-trajectories

### Example
```julia
# For feature selection
config = TrainingConfig(
    objective = ADAPTIVE_SUB_TB,
    sub_trajectory_config = Dict(
        :difficulty_threshold => 0.1,
        :n_subtrajectories => 10
    )
)
```

## Flow Consistency (Local Balance)

```julia
objective = FLOW_CONSISTENCY
flow_mode = EDGE_LEVEL  # or STATE_LEVEL or MIXED_LEVEL
```

### Use For
- Local structure learning
- When flow conservation is critical
- Combining benefits of detailed balance and flow matching

### Mathematical Breakthrough
Unifies two classical approaches:

Edge-Level (Detailed Balance):
$$F(s) \cdot P_F(s \to s') = F(s') \cdot P_B(s' \to s)$$

State-Level (Flow Matching):
$$\sum \text{incoming flow} = \sum \text{outgoing flow}$$

### Modes
- `EDGE_LEVEL`: Focus on individual transitions
- `STATE_LEVEL`: Focus on state conservation  
- `MIXED_LEVEL`: Combine both constraints

### Note
Currently not fully implemented due to missing flow functions.

## Choosing the Right Objective

### Decision Tree

1. **Is your domain simple with unique paths?**
   - Yes → Use `TRAJECTORY_BALANCE`
   - No → Continue to 2

2. **Do you have long sequences with sparse rewards?**
   - Yes → Use `SUB_TRAJECTORY_BALANCE`
   - No → Continue to 3

3. **Does your domain have natural hierarchies?**
   - Yes → Use `HIERARCHICAL_SUB_TB`
   - No → Continue to 4

4. **Do you have rare but critical decisions?**
   - Yes → Use `ADAPTIVE_SUB_TB`
   - No → Use `TRAJECTORY_BALANCE` with backward policy

### Performance Considerations

| Objective | Training Speed | Memory Usage | Convergence |
|-----------|---------------|--------------|-------------|
| Trajectory Balance | Fast | Low | Good |
| Sub-Trajectory | Medium | Medium | Better |
| Hierarchical | Slow | High | Best for hierarchical |
| Adaptive | Medium | Medium | Best for sparse |
| Flow Consistency | Slow | High | Theoretical best |

## Implementation Status

### Fully Implemented
- ✅ Trajectory Balance (with optional backward policy)
- ✅ Sub-Trajectory Balance variants

### Partially Implemented  
- ⚠️ Flow Consistency (missing flow functions)

### Not Implemented
- ❌ General Trajectory Balance
- ❌ Detailed Balance (requires flow functions)
- ❌ Flow Matching (requires flow functions)

## Examples by Domain

### Grid World
```julia
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 100,
    batch_size = 16
)
```

### Molecular Design
```julia
config = TrainingConfig(
    objective = HIERARCHICAL_SUB_TB,
    sub_trajectory_config = Dict(
        :scales => [5, 10, 20, 40],
        :n_subtrajectories => 5
    ),
    n_iterations = 5000
)
```

### Active Learning
```julia
config = TrainingConfig(
    objective = SUB_TRAJECTORY_BALANCE,
    sub_trajectory_config = Dict(
        :min_length => 2,
        :max_length => 10,
        :n_subtrajectories => 4
    ),
    n_iterations = 2000
)
```

## Advanced Usage

### Combining Objectives
Future versions will support:
```julia
config = TrainingConfig(
    objective = COMBINED_OBJECTIVES,
    trajectory_weight = 0.5,
    sub_trajectory_weight = 0.3,
    flow_weight = 0.2
)
```

### Custom Objectives
See [Developer Guide](developer_guide.md) for implementing custom objectives.

## See Also
- [Training System](training_system.md) - How to configure training
- [Mathematical Background](../theory/training_objectives.md) - Theoretical details
- [Examples](../guide/examples.md) - Working examples