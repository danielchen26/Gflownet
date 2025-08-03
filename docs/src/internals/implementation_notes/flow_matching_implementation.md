# Flow Matching Implementation Details

This document describes the implementation of the FLOW_MATCHING training objective in GFlowNet.jl.

## Overview

Flow Matching directly optimizes the flow conservation equation by training a neural network to estimate flow values F(s) for each state.

## Mathematical Foundation

The objective minimizes:
```
L_FM(s) = (Z(s) - F(s))²
```

where:
- Z(s): Neural network's flow estimate
- F(s): True flow computed recursively
  - F(s) = R(s) for terminal states
  - F(s) = Σ_{s'} P_F(s'|s) * F(s') for non-terminal states

## Implementation Components

### 1. Flow Matching Loss Function
**Location**: `src/core/balance.jl`

```julia
function flow_matching_loss(model::GFlowNetModel, state)::Float64
    # Get neural network estimate
    estimated_flow = flow_estimate(model.flow_estimator, state, ...)
    
    # Compute expected flow recursively
    expected_flow = Σ P_F(s'|s) * F(s')
    
    # Return squared difference
    return (estimated_flow - expected_flow)^2
end
```

### 2. Integration with Training System
**Location**: `src/core/interface.jl`

The `compute_trajectory_loss` function handles FLOW_MATCHING:
- Extracts non-terminal states from trajectories
- Computes flow matching loss for each state
- Wraps recursive flow computation in `Zygote.@ignore`

### 3. Key Design Decisions

#### Flow Computation During Training
Flows are computed recursively but treated as constants during gradient computation:
```julia
expected_flow = Zygote.@ignore begin
    # Recursive flow computation
    # Not differentiated through
end
```

This prevents:
- Infinite recursion in gradients
- Computational explosion
- Numerical instability

#### State Sampling
- States are extracted from sampled trajectories
- Duplicates are removed to avoid bias
- Terminal states are skipped (conservation trivially satisfied)

## Usage Example

```julia
# Create model with flow estimator
model = create_grid_world_gflownet(
    grid_size = 5,
    hidden_dim = 64
)

# Configure training
config = TrainingConfig(
    objective = FLOW_MATCHING,
    n_iterations = 1000,
    batch_size = 32
)

# Train
history = train_gflownet(model, config)

# Access flow estimates
state = GridState(2, 3, false)
flow_value = flow_estimate(
    model.flow_estimator, state,
    model.parameters.flow, model.states.flow
)
```

## Advantages

1. **Explicit Flow Values**: Neural network provides direct F(s) estimates
2. **No Backward Policy**: Only requires forward policy
3. **Analysis Tool**: Flow values useful for debugging and visualization
4. **Theoretical Grounding**: Directly enforces conservation equation

## Limitations

1. **Computational Cost**: Recursive flow computation can be expensive
2. **Convergence Speed**: May be slower than trajectory-based methods
3. **Early Training**: Flow estimates may be poor initially

## Testing

Comprehensive tests are provided in:
- `test/objectives/flow_matching/test_flow_matching.jl`
- `test/objectives/flow_matching/test_flow_matching_comprehensive.jl`

Tests verify:
- Mathematical properties (non-negative loss, conservation)
- Convergence to correct flows
- Integration with training system
- Comparison with other objectives

## Future Improvements

1. **Batch Flow Computation**: Vectorize flow calculations
2. **Importance Sampling**: Focus on states with high flow
3. **Hybrid Objectives**: Combine with TB/DB for faster convergence
4. **Flow Visualization**: Tools for flow analysis