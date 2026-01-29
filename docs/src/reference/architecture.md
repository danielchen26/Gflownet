# GFlowNet.jl Architecture

This document describes the architectural organization of the GFlowNet.jl codebase.

## Overview

GFlowNet.jl follows a clean, modular architecture with clear separation of concerns:

```
GFlowNet.jl/
├── src/
│   ├── GFlowNet.jl           # Main module with exports
│   ├── core/                 # Core mathematical engine
│   ├── training/             # Training infrastructure
│   ├── applications/         # Domain implementations
│   └── utils/                # Utilities and helpers
├── examples/                 # Usage examples
├── test/                     # Comprehensive test suite
└── docs/                     # Documentation
```

## Core Module Structure

### Core Mathematical Engine (`src/core/`)

The core module contains the mathematical foundations of GFlowNets.

**Files** (flat structure, no subdirectories):
- `types.jl` - Abstract types and core data structures
- `graphs.jl` - DAG (Directed Acyclic Graph) operations with on-demand construction
- `policies.jl` - Forward policy (P_F), backward policy (P_B), and partition function (Z)
- `flows.jl` - Flow computation with memoization
- `balance.jl` - Balance loss functions (trajectory balance, detailed balance)
- `sampling.jl` - Trajectory sampling algorithms
- `interface.jl` - High-level model creation and sampling API
- `multi_start.jl` - Multi-start GFlowNets with per-initial-state partition functions

**Key Characteristics**:
- Pure mathematical operations
- Zygote-compatible (automatic differentiation friendly)
- No side effects in differentiable functions
- Efficient caching with proper invalidation

### Training Infrastructure (`src/training/`)

Separate module for training-related functionality.

**Files**:
- `configuration.jl` - `TrainingConfig` type and validation
- `training.jl` - Main training loop (`train_gflownet`, `train_step!`)
- `objectives.jl` - Training objective enumeration and configuration
- `losses.jl` - Loss computation functions
- `utils.jl` - Training utilities and helpers
- `multi_start_training.jl` - Training for multi-start GFlowNets

**Design Principle**: Clean separation between model creation (core/) and model training (training/).

### Applications (`src/applications/`)

Domain-specific implementations following the interface pattern.

**Files**:
- `grid_world.jl` - Grid world domain (canonical example)
- Additional domains as they're developed

**Each application provides**:
1. State and action type definitions
2. Required interface implementations
3. Domain-specific convenience functions (e.g., `create_grid_world_gflownet`)
4. Reward function

### Utilities (`src/utils/`)

Helper functions and tools.

**Subdirectories**:
- `validation/` - Mathematical property validation
- `visualization/` - Web-based visualization system
  - `web/` - React + Three.js frontend
  - `api/` - Julia REST API backend
- `logging/` - Training logging utilities

## Architectural Principles

### 1. Composition Over Inheritance

GFlowNet uses composition for domain-specific data:

```julia
# Abstract base types
abstract type AbstractState end
abstract type AbstractAction end

# Concrete domain types
struct GridState <: AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

# Interface functions dispatch on concrete types
function state_to_features(state::GridState)
    return Float32[state.x, state.y, Float32(state.is_terminal)]
end
```

### 2. Type Safety

Leverage Julia's type system for correctness:
- Parametric types for DAG: `DirectedAcyclicGraph{S<:AbstractState, A<:AbstractAction}`
- `ComponentArray` for gradient-compatible parameters
- Concrete types in struct fields
- Type-stable function returns

### 3. Interface Segregation

Clear separation between:
- **Core algorithms** (mathematical operations, Zygote-compatible)
- **Training infrastructure** (loops, optimization, configuration)
- **Applications** (domain-specific logic, interface implementations)

### 4. Automatic Differentiation Compatibility

**CRITICAL**: Core functions must be Zygote-compatible:
- No in-place mutations (`+=`, `push!`, etc.)
- Pure functional transformations
- Careful cache management with `Zygote.@ignore`

### 5. On-Demand DAG Construction

State space graphs are built lazily during sampling, not pre-computed:
- Memory efficient for large state spaces
- Caching for frequently-visited states
- Cache invalidation when parameters change

## Critical Interfaces

Every GFlowNet domain **must** implement these 5 functions:

```julia
# 1. Convert state to neural network features
function state_to_features(state::YourState)
    return Float32[...]  # Must return Vector{Float32}
end

# 2. Check if action is applicable
function is_applicable(action::YourAction, state::YourState)
    return Bool  # true if action can be applied
end

# 3. Apply action to get new state (MUST be pure functional!)
function apply_action(action::YourAction, state::YourState)
    return YourState(...)  # Return new state, don't mutate input
end

# 4. Check if state is terminal
function is_terminal_state(state::YourState)
    return state.is_terminal  # Bool
end

# 5. Compute reward (MUST be positive for terminals!)
function reward(state::YourState)
    !state.is_terminal && return 0.0f0
    return Float32(positive_value)  # R(s_T) > 0 required
end
```

## Parameter Management

### Parameter Structure

Uses `ComponentArray` for automatic differentiation compatibility:

```julia
parameters = ComponentArray(
    forward = forward_policy_params,
    backward = backward_policy_params,  # Optional, for DETAILED_BALANCE
    flow = flow_estimator_params,       # Optional, for FLOW_MATCHING
    Z = partition_function_params        # Optional, for LEARNABLE_ESTIMATION
)
```

### Neural Network State

Separate from parameters, tracks network internal state:

```julia
states = (
    forward = forward_policy_state,
    backward = backward_policy_state,
    flow = flow_estimator_state
)
```

## Training Objective Architecture

### Available Objectives

1. **TRAJECTORY_BALANCE** (default)
   - Enforces: `Z · P_F(τ) = R(s_T)`
   - No special requirements

2. **DETAILED_BALANCE**
   - Enforces: `P_F(s'|s) · F(s) = P_B(s|s') · F(s')`
   - Requires: Backward policy (`include_backward=true`)

3. **SUB_TRAJECTORY_BALANCE**
   - O(T²) learning signals from sub-trajectories
   - Configurable sub-trajectory length

4. **FLOW_MATCHING**
   - Minimizes: `(Z(s) - F(s))²`
   - Requires: Flow estimator (`include_flow_estimator=true`)

5. **DIRECT_FLOW_OBJECTIVE**
   - Direct neural network flow estimation
   - Requires: Flow estimator (`include_flow_estimator=true`)

6. **COMBINED_OBJECTIVES**
   - Weighted combination of multiple objectives

### Partition Function Methods

1. **SIMPLE_ESTIMATION**: Z = 1 (fixed constant)
2. **LEARNABLE_ESTIMATION**: Z learned as trainable parameter (⭐ recommended)
3. **SAMPLING_ESTIMATION**: Sample-based estimation
4. **ADAPTIVE_ESTIMATION**: Adaptive approach

## High-Level API

Users interact through high-level functions, **never manually defining networks**:

```julia
# Model creation
model = create_gflownet(
    initial_state, actions;
    state_dim, hidden_dim, learning_rate,
    include_backward, include_flow_estimator
)

# Training
config = TrainingConfig(objective, n_iterations, batch_size, ...)
history = train_gflownet(model, config)

# Sampling
trajectory = sample_trajectory(model)
```

## Data Flow

```
User Input (domain types)
    ↓
Interface Functions (state_to_features, apply_action, etc.)
    ↓
Core Algorithms (sampling, flow computation, losses)
    ↓
Neural Policies (forward, backward, flow estimator)
    ↓
Training (gradient computation, parameter updates)
    ↓
Trained Model
```

## File Dependencies

**Core module dependencies**:
- `types.jl` → base types (no dependencies)
- `graphs.jl` → depends on types
- `policies.jl` → depends on types
- `flows.jl` → depends on graphs, policies
- `balance.jl` → depends on flows
- `sampling.jl` → depends on policies, graphs
- `interface.jl` → depends on all core modules

**Training module dependencies**:
- `configuration.jl` → depends on core/types
- `objectives.jl` → depends on configuration
- `losses.jl` → depends on core modules
- `training.jl` → depends on all training modules

**Import structure**:
- Training modules explicitly import from core via `using GFlowNet: function_name`
- Applications import both core and training functions as needed
- Examples import high-level API only

## Extension Points

To extend GFlowNet.jl:

1. **New Domain**: Implement the 5 required interface functions
2. **New Objective**: Add to objectives.jl, implement loss function in losses.jl
3. **New Partition Method**: Extend partition function estimation in policies.jl
4. **Custom Policy**: Extend policy creation in interface.jl

## Best Practices

1. **Never manually define neural networks** - use `create_gflownet()`
2. **Keep differentiable functions pure** - no mutations
3. **Use high-level API** - `train_gflownet()`, not manual loops
4. **Follow type discipline** - concrete types, proper parametrics
5. **Validate interfaces** - ensure all 5 functions implemented
6. **Test Zygote compatibility** - gradient tests for new features

## References

- [Project Structure](project_structure.md) - File organization
- [Core Concepts](core_concepts.md) - Mathematical foundations
- [Module Structure](module_structure.md) - Detailed module breakdown
