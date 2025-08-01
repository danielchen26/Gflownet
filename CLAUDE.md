# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Principles

- Remember forever, do not write Co-Authored-By: Claude <noreply@anthropic.com> or anything related to AI or Claude in the future and now
- Remember never show Co-Authored-By: Claude <noreply@anthropic.com> Commit FLOW_MATCHING implementation in the end

## Repository Overview

GFlowNet.jl is a production-ready Julia implementation of Generative Flow Networks (GFlowNets) - a breakthrough class of generative models that learn to sample diverse, high-quality objects proportionally to a reward function. The codebase follows modern ML package design with clean separation of concerns and comprehensive tooling.

## Recently Completed Features (January 2025)

### Newly Implemented
1. **DETAILED_BALANCE Training Objective** ✅
   - Full implementation with joint backward policy representation
   - Zygote-compatible gradient computation
   - Comprehensive test coverage

2. **Complete Flow Computation** ✅
   - Recursive flow calculation with memoization
   - Edge flows and partition function computation
   - Proper caching with Zygote compatibility

3. **Learnable Partition Function** ✅
   - LEARNABLE_ESTIMATION option for trainable Z
   - Improves exploration and theoretical correctness

4. **Backward Policy with Joint Representation** ✅
   - P_B(s|s') using concatenated state features
   - Required for DETAILED_BALANCE
   - Enables better credit assignment

## Development Guidance

- Remember that example folders are associated with different domains and for the examples related to the core development, you should put them into the core features sub folder
- Remember you should put the test in the relevent folder not just scatter it in the test folder, we already have a Hierarchical folders

## Common Development Commands

### Package Management & Testing
```bash
# Activate the project environment
julia --project=.

# Install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run specific test category
julia --project=. test/core/detailed_balance/test_detailed_balance.jl

# Run examples
cd examples/grid_world && julia --project=. grid_world.jl
```

### Development Workflow
```bash
# Precompile the package
julia --project=. -e "using Pkg; Pkg.precompile()"

# Enter REPL with package loaded
julia --project=. -e "using GFlowNet"

# Build documentation
julia --project=docs docs/make.jl

# Format code (if JuliaFormatter available)
julia --project=. -e "using JuliaFormatter; format(\"src\")"
```

## High-Level Architecture

The package follows a modular architecture with clear separation of concerns:

### Core Mathematical Engine (`src/core/`)
- **types.jl**: Fundamental abstract types (AbstractState, AbstractAction, GFlowNetModel)
- **graphs.jl**: DAG operations and state space analysis
- **policies.jl**: Forward policy P_F, backward policy P_B
- **flows.jl**: Flow conservation and computation (fully implemented)
- **balance.jl**: Training objectives (TB, DB implemented, FM ready)
- **sampling.jl**: Trajectory generation algorithms
- **objectives.jl**: Loss computation for training
- **interface.jl**: High-level API functions

### Training Infrastructure (`src/training/`)
- **configuration.jl**: TrainingConfig type with validation
- Supports objectives: TRAJECTORY_BALANCE, DETAILED_BALANCE
- FLOW_MATCHING ready to implement

### Domain Applications (`src/applications/`)
Each implements the required GFlowNet interface:
- **grid_world.jl**: Navigation tasks (flagship example)
- **molecular_design.jl**: Chemical synthesis
- **supply_chain_optimization.jl**: Business logistics

## Critical Implementation Rules

### Zygote/AD Compatibility (MOST IMPORTANT)
The codebase uses Zygote for automatic differentiation. **NO IN-PLACE MUTATIONS** in differentiable functions:

```julia
# ❌ WRONG - Breaks Zygote
state.position += action.delta  # Mutation!

# ✅ CORRECT - Pure functional
new_position = state.position + action.delta
return GridWorldState(new_position, state.is_terminal)
```

All validation functions must be wrapped with `Zygote.@ignore` to prevent inclusion in computational graph.

### Required Interface for New Domains
Every domain must implement these 5 functions:
1. `state_to_features(::YourState)::Vector{Float32}` - Neural network input
2. `is_terminal_state(::YourState)::Bool` - Check if terminal
3. `reward(::YourState)::Float64` - Terminal state rewards (must be positive)
4. `is_applicable(::YourAction, ::YourState)::Bool` - Action validity
5. `apply_action(::YourAction, ::YourState)::YourState` - State transitions (pure)

Plus equality and hashing for states/actions.

## What Works Now

### Training Objectives
- **TRAJECTORY_BALANCE** ✅ - With optional backward policy and learnable Z
- **DETAILED_BALANCE** ✅ - Requires backward policy, uses flow computation

### Core Features
- **Flow Computation** ✅ - Recursive with memoization
- **Backward Policy** ✅ - Joint state representation
- **Learnable Z** ✅ - Via LEARNABLE_ESTIMATION

### Next Ready to Implement
- **FLOW_MATCHING** - All prerequisites available
- **Multi-start GFlowNets** - Flow functions enable this

## Common Development Patterns

### Creating a New Domain
1. Define state and action types in `src/applications/your_domain.jl`
2. Implement the 5 required interface functions
3. Create `create_your_domain_gflownet()` high-level function
4. Add tests in `test/applications/your_domain/`
5. Create example in `examples/your_domain/`

### Training with Different Objectives
```julia
# Trajectory Balance (default)
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 1000,
    batch_size = 32
)

# Detailed Balance (requires backward policy)
model = create_grid_world_gflownet(include_backward = true)
config = TrainingConfig(
    objective = DETAILED_BALANCE,
    n_iterations = 1000,
    batch_size = 32
)

# With learnable Z
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    partition_function_method = LEARNABLE_ESTIMATION,
    n_iterations = 1000
)
```

## Test Organization

Tests are organized hierarchically:
- `test/core/` - Core functionality tests
- `test/core/detailed_balance/` - Detailed balance specific tests
- `test/objectives/` - Training objective tests
- `test/applications/` - Domain-specific tests
- `test/debugging/` - Diagnostic tests (not run by default)

## Key Development Principles

- **On-demand computation**: No explicit DAG construction
- **Type safety**: Consistent Float32 usage, concrete types
- **Pure functions**: No mutations for AD compatibility
- **Positive rewards**: Mathematical requirement for GFlowNets
- **Clean interfaces**: Use high-level `create_*_gflownet()` functions

## For Comprehensive Development Guidelines

See `COMPREHENSIVE_GFLOWNET_RULES_UPDATED.md` for:
- Complete code examples with correct/incorrect patterns
- Full testing frameworks
- Performance optimization strategies
- Debugging guides
- Generic development checklists