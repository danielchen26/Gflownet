# GFlowNet.jl Architecture Overview

## Executive Summary (Updated: January 2025)

GFlowNet.jl implements Generative Flow Networks using a modern Julia architecture with:
- **ComponentArrays + Lux.jl** for neural networks
- **Zygote.jl** for automatic differentiation
- **On-demand computation** without explicit DAG construction
- **Full backward policy support** with joint state representation
- **Complete flow computation** with memoization and caching
- **Multiple training objectives** including TRAJECTORY_BALANCE and DETAILED_BALANCE

## Current Architecture

### Core Components

1. **State and Action System**
   - Abstract types: `AbstractState`, `AbstractAction`
   - States must have `is_terminal::Bool` field
   - Actions are domain-specific implementations
   - On-demand action discovery via `get_applicable_actions()`

2. **Neural Network Policies**
   - `ForwardPolicy`: P_F(a|s) - action selection
   - `BackwardPolicy`: P_B(s|s') - fully implemented with joint state representation
   - `FlowEstimator`: Z(s) - neural network for flow estimation
   - `Learnable Z`: Optional learnable partition function parameter

3. **Training System** (Reorganized January 2025)
   - Uses `train_gflownet(model, config)`
   - Modular organization in `src/training/` directory
   - Supports trajectory sampling and batch training
   - Gradient computation via Zygote.jl
   - Parameter updates via Optimisers.jl

### Model Structure

```julia
struct GFlowNetModel
    initial_state::AbstractState
    all_actions::Vector{<:AbstractAction}
    forward_policy::ForwardPolicy
    backward_policy::Union{BackwardPolicy, Nothing}  # Optional
    flow_estimator::FlowEstimator
    log_partition_function::Union{Float64, Nothing}  # For LEARNABLE_ESTIMATION
    parameters::ComponentArray
    optimizer::Any
    states::NamedTuple
end
```

## Training Objectives Implementation Status

### Fully Implemented
- **TRAJECTORY_BALANCE**: ✅ Complete with optional backward policy
  - Loss: (log Z + log P_F(τ) - log R(s_T))²
  - Supports learnable partition function Z
  
- **DETAILED_BALANCE**: ✅ Fully implemented (January 2025)
  - Requires backward policy
  - Loss: (log P_F(s→s') + log F(s) - log P_B(s'→s) - log F(s'))²
  - Uses joint state representation for P_B(s|s')
  
- **Flow Computation**: ✅ Complete implementation
  - Recursive flow: F(s) = Σ P_F(s'|s) F(s')
  - Memoization with Zygote-compatible caching
  - Edge flows: F(s→s') = P_F(s'|s) F(s)
  - Partition function: Z = F(s₀)

- **FLOW_MATCHING**: ✅ Fully implemented (January 2025)
  - Loss: (Z(s) - F(s))² where Z(s) is neural network estimate
  - Uses flow estimator network for Z(s)
  - Compatible with memoized flow computation

### Ready to Implement
- **SUB_TRAJECTORY_BALANCE**: Can be implemented with current infrastructure

### Advanced Features
- **Multi-start support**: ✅ Implemented with per-initial-state Z values
  - `MultiStartGFlowNetModel` type
  - Per-initial-state partition functions
  - Initial state sampling based on learned Z values
- **COMBINED_OBJECTIVES**: Requires design decisions

## Key Design Decisions

### 1. On-Demand Computation
- No pre-computed DAG or state enumeration
- States discovered during sampling
- Actions filtered by `is_applicable(action, state)`
- Transitions computed via `apply_action(action, state)`

### 2. Partition Function Handling
- **SIMPLE_ESTIMATION**: Z = 1 (default, valid for fixed initial state)
- **LEARNABLE_ESTIMATION**: Z as trainable parameter (implemented)
- **SAMPLING_ESTIMATION**: Monte Carlo estimation (planned)
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
- Required for DETAILED_BALANCE objective
- Uses joint state representation: P_B(s|s') = σ(NN([features(s), features(s')]))
- Created with `include_backward=true` flag
- Enables better credit assignment

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
    objective = TRAJECTORY_BALANCE,  # or DETAILED_BALANCE
    n_iterations = 1000,
    batch_size = 32,
    partition_function_method = LEARNABLE_ESTIMATION  # Optional
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

## Training Module Organization

The training infrastructure has been reorganized for better maintainability:
- `training/configuration.jl` - Training types and configuration
- `training/objectives.jl` - Training objective definitions  
- `training/training.jl` - Main training loop
- `training/losses.jl` - Loss computation functions
- `training/utils.jl` - Training utilities
- `training/multi_start_training.jl` - Multi-start specific training

The `core/interface.jl` now contains only model creation and sampling functions.

## Current Limitations

1. **Training Objectives**: SUB_TRAJECTORY_BALANCE not yet implemented (FLOW_MATCHING now complete)
2. **Multiple Initial States**: ✅ NOW SUPPORTED with multi-start GFlowNets
3. **GPU Acceleration**: Limited to neural network operations, trajectory sampling is CPU-only
4. **Continuous State Spaces**: Experimental support only

## Performance Characteristics

- **Memory**: O(batch_size × trajectory_length) during training
- **Computation**: Dominated by neural network forward passes
- **Scaling**: Handles large action spaces via on-demand filtering
- **AD Compatibility**: Full Zygote support with careful state handling

## Future Improvements

1. Implement SUB_TRAJECTORY_BALANCE objective
2. Support multiple initial states with per-state Z values
3. GPU-accelerated trajectory sampling
4. Continuous state space support
5. Distributed training support

## Conclusion

GFlowNet.jl provides a clean, modern implementation of GFlowNets with:
- Multiple training objectives (TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING)
- Complete flow computation infrastructure
- Full backward policy support with joint representation
- Learnable partition function option
- Extensible domain interface
- Production-ready for single initial state problems

The architecture balances theoretical completeness with practical usability, providing both simple defaults and advanced features when needed.