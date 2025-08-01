# GFlowNet.jl Architecture Overview

## Executive Summary (Updated: August 2025)

GFlowNet.jl implements Generative Flow Networks using a modern Julia architecture with:
- **ComponentArrays + Lux.jl** for neural networks
- **Zygote.jl** for automatic differentiation
- **On-demand computation** without explicit DAG construction
- **Optional backward policy** for improved credit assignment

## Current Architecture

### Core Components

1. **State and Action System**
   - Abstract types: `AbstractState`, `AbstractAction`
   - States must have `is_terminal::Bool` field
   - Actions are domain-specific implementations
   - On-demand action discovery via `get_applicable_actions()`

2. **Neural Network Policies**
   - `ForwardPolicy`: P_F(a|s) - action selection
   - `BackwardPolicy`: P_B(s|s') - optional, for credit assignment
   - `FlowEstimator`: Z(s) - currently returns 1.0 (fixed initial state assumption)

3. **Training System**
   - Uses `train_gflownet(model, config)`
   - Supports trajectory sampling and batch training
   - Gradient computation via Zygote.jl
   - Parameter updates via Optimisers.jl

### Model Structure

```julia
struct GFlowNetModel
    initial_state::AbstractState
    all_actions::Vector{<:AbstractAction}
    forward_policy::ForwardPolicy
    flow_estimator::FlowEstimator
    backward_policy::Union{BackwardPolicy, Nothing}  # Optional
    parameters::ComponentArray
    states::NamedTuple
    optimizer::Any
end
```

## Training Objectives Implementation Status

### Currently Working
- **TRAJECTORY_BALANCE**: ✅ Fully implemented
- **Flow Computation**: ✅ Implemented (recursive flow, memoization, edge flows)
  - Simple version: Forward policy only (assumes P_B uniform)
  - Full version: With backward policy for better credit assignment
  - Loss: (log P_F(τ) + log P_B(τ) - log R(s_T))²

### Defined but Not Implemented
- **DETAILED_BALANCE**: ❌ Placeholder only
- **FLOW_MATCHING**: ❌ Placeholder only  
- **SUB_TRAJECTORY_BALANCE**: ❌ Placeholder only
- **COMBINED_OBJECTIVES**: ❌ Placeholder only

**Important**: The training loop currently ignores `config.objective` and always uses trajectory balance.

## Key Design Decisions

### 1. On-Demand Computation
- No pre-computed DAG or state enumeration
- States discovered during sampling
- Actions filtered by `is_applicable(action, state)`
- Transitions computed via `apply_action(action, state)`

### 2. Partition Function Z = 1
- Valid assumption for fixed initial state (current case)
- All examples start from single s₀
- Would need modification for multiple initial states
- See [Partition Function Analysis](../theory/partition_function.md)

### 3. Pure Functional State Transitions
- No in-place mutations (Zygote compatibility)
- Always return new state instances
- Example:
  ```julia
  function apply_action(action::MoveRight, state::GridState)
      new_x = state.x + 1
      return GridState(new_x, state.y, false)  # New instance
  end
  ```

### 4. Backward Policy Implementation
- Optional component (can be `nothing`)
- When included, enables full trajectory balance
- Uses joint forward-backward training
- Created with `include_backward=true` flag

## API Design

### High-Level Interface
```julia
# Create domain-specific GFlowNet
model = create_grid_world_gflownet(
    grid_size = 5,
    hidden_dim = 64,
    learning_rate = 0.01,
    include_backward = false  # Optional backward policy
)

# Configure training
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,  # Only this works currently
    n_iterations = 1000,
    batch_size = 32
)

# Train
history = train_gflownet(model, config)

# Sample
trajectories = [sample_trajectory(model) for _ in 1:100]
```

### Required Domain Interface
Every domain must implement:
1. `state_to_features(state)::Vector{Float32}`
2. `is_terminal_state(state)::Bool`
3. `reward(state)::Float64` (must be positive)
4. `is_applicable(action, state)::Bool`
5. `apply_action(action, state)::State` (pure function)

## Flow Computation Implementation

### Overview
Flow computation is now fully implemented using on-demand computation without requiring explicit DAG construction.

### Key Components

1. **Recursive Flow Computation**
   ```julia
   F(s) = Σ_{s'} P_F(s'|s) * F(s')  # Non-terminal states
   F(s) = R(s)                       # Terminal states
   ```

2. **Memoization System**
   - Global cache for computed flow values
   - Cache invalidation on parameter changes
   - Significant performance improvement for deep state spaces

3. **Partition Function**
   - `Z = F(s₀)` - Total flow from initial state
   - No longer hardcoded to 1.0
   - Properly computed using recursive flow

4. **Edge Flow**
   - `F(s→s') = P_F(s'|s) * F(s)`
   - Useful for analyzing flow distribution

### Implementation Details
- **Location**: `src/core/flows.jl`
- **API Documentation**: [Flow Computation API](../api/flow_computation.md)
- **Zygote Compatible**: No mutations, pure functional
- **On-Demand**: Computes flows as needed, no pre-computation
- **Efficient**: Memoization prevents redundant calculations

## Current Limitations

1. **Training Objectives**: Only trajectory balance works; detailed balance and flow matching require backward policy
2. **Multiple Initial States**: Not yet supported (requires per-state Z computation)
3. **Flow Estimator Network**: Not implemented (DIRECT_FLOW method unavailable)
4. **GPU Acceleration**: Limited to neural network operations

## Performance Characteristics

- **Memory**: O(batch_size × trajectory_length) during training
- **Computation**: Dominated by neural network forward passes
- **Scaling**: Handles large action spaces via on-demand filtering
- **AD Compatibility**: Full Zygote support with careful state handling

## Future Improvements

1. Implement missing training objectives
2. Add proper flow computation for Z ≠ 1 cases
3. Support multiple initial states
4. GPU-accelerated trajectory sampling
5. Distributed training support

## Conclusion

GFlowNet.jl provides a clean, modern implementation of GFlowNets with:
- Working trajectory balance training
- Optional backward policy support
- Extensible domain interface
- Production-ready for single initial state problems

The architecture prioritizes simplicity and correctness over supporting every theoretical GFlowNet variant.