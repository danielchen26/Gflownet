# Flow Computation API

This page documents the flow computation functions in GFlowNet.jl.

## Overview

Flow computation is fundamental to GFlowNets, implementing the flow conservation equation:

$$F(s) = \sum_{s'} P_F(s'|s) \cdot F(s')$$

## Main Functions

### `flow`
```julia
flow(model::GFlowNetModel, state::AbstractState; 
     method::FlowComputationMethod=RECURSIVE_FLOW)::Float64
```

Unified interface for computing flow F(s) through a state.

**Arguments:**
- `model`: The GFlowNet model
- `state`: State to compute flow for
- `method`: Computation method (RECURSIVE_FLOW, DIRECT_FLOW, or MIXED_FLOW)

**Returns:** Flow value F(s)

**Example:**
```julia
# Compute flow using recursive method (default)
flow_value = flow(model, state)

# Use different methods
flow_recursive = flow(model, state; method=RECURSIVE_FLOW)
flow_mixed = flow(model, state; method=MIXED_FLOW)
```

### `partition_function`
```julia
partition_function(model::GFlowNetModel)::Float64
```

Computes the partition function Z = F(s₀).

**Arguments:**
- `model`: The GFlowNet model

**Returns:** Partition function Z

**Example:**
```julia
Z = partition_function(model)
println("Partition function: $Z")
```

### `edge_flow`
```julia
edge_flow(model::GFlowNetModel, source_state::AbstractState, 
          target_state::AbstractState)::Float64
```

Computes flow along an edge: F(s→s') = P_F(s'|s) * F(s).

**Arguments:**
- `model`: The GFlowNet model
- `source_state`: Source state s
- `target_state`: Target state s'

**Returns:** Edge flow F(s→s')

**Example:**
```julia
# Compute flow through specific transition
flow_edge = edge_flow(model, state1, state2)
```

## Flow Computation Methods

### `FlowComputationMethod`

Enumeration of available flow computation methods:

- `RECURSIVE_FLOW`: Uses recursive flow conservation equation
- `DIRECT_FLOW`: Uses flow estimator network (requires flow_estimator)
- `MIXED_FLOW`: Combines both methods for validation

## Utility Functions

### `clear_flow_cache!`
```julia
clear_flow_cache!()
```

Clears the global flow cache. Should be called when model parameters change.

### `validate_flow_conservation`
```julia
validate_flow_conservation(model::GFlowNetModel, state::AbstractState; 
                          tolerance::Float64=1e-6)::Bool
```

Validates that flow conservation holds for a state.

**Arguments:**
- `model`: The GFlowNet model
- `state`: State to validate
- `tolerance`: Numerical tolerance

**Returns:** true if conservation holds within tolerance

### `flow_analysis`
```julia
flow_analysis(model::GFlowNetModel, state::AbstractState)
```

Comprehensive flow analysis for debugging.

**Returns:** Named tuple with:
- `flow_value`: F(s)
- `is_terminal`: Whether state is terminal
- `next_states`: List of next states
- `transition_probs`: P_F(s'|s) for each next state
- `next_flows`: F(s') for each next state
- `conservation_check`: Whether flow conservation holds

## Implementation Details

### Recursive Flow Computation

For non-terminal states:
```julia
F(s) = Σ_{s'} P_F(s'|s) * F(s')
```

For terminal states:
```julia
F(s) = R(s)
```

### Memoization

Flow values are cached to avoid redundant computation:
- Cache key includes parameter hash to handle updates
- Automatic invalidation on parameter changes
- Significant performance improvement for deep state spaces

### On-Demand Computation

The implementation uses on-demand computation:
- No explicit DAG construction required
- Computes next states using `get_applicable_actions()` and `apply_action()`
- Fully compatible with the modern GFlowNet.jl architecture

## Example Usage

```julia
using GFlowNet

# Create model
model = create_grid_world_gflownet(grid_size=5)

# Train model
config = TrainingConfig(objective=TRAJECTORY_BALANCE, n_iterations=1000)
train_gflownet(model, config)

# Compute partition function
Z = partition_function(model)
println("Partition function: $Z")

# Analyze flow for a specific state
state = model.initial_state
analysis = flow_analysis(model, state)
println("Flow through initial state: $(analysis.flow_value)")
println("Conservation satisfied: $(analysis.conservation_check)")

# Check edge flows
for (next_state, prob, flow) in zip(analysis.next_states, 
                                   analysis.transition_probs, 
                                   analysis.next_flows)
    edge_flow_val = prob * analysis.flow_value
    println("Edge flow to $next_state: $edge_flow_val")
end
```

## Mathematical Properties

1. **Flow Conservation**: Well-trained models satisfy F(s) = Σ P_F(s'|s) * F(s')
2. **Boundary Condition**: F(s) = R(s) for terminal states
3. **Non-negativity**: F(s) ≥ 0 for all states
4. **Partition Function**: Z = F(s₀) represents total flow from initial state

## Performance Considerations

- Memoization dramatically improves performance for repeated queries
- Deep state spaces may require significant memory for caching
- Use `clear_flow_cache!()` if memory becomes an issue
- RECURSIVE_FLOW is exact but can be slow for very large state spaces
- DIRECT_FLOW (when available) provides fast approximation

## See Also

- [Training Objectives](../guide/training_objectives.md) - How flow functions are used in training
- [Core Types](core_types.md) - State and action interfaces  
- [Policies](policies.md) - Forward and backward policy functions
- [Mathematical Background](../guide/mathematical_background.md) - Theoretical foundations
- [Flow Consistency Theory](../theory/flow_consistency.md) - Mathematical theory behind flow conservation
- [Architecture Overview](../internals/architecture.md) - Implementation details