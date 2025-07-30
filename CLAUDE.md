# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

GFlowNet.jl is a production-ready Julia implementation of Generative Flow Networks (GFlowNets) - a breakthrough class of generative models that learn to sample diverse, high-quality objects proportionally to a reward function. The codebase follows modern ML package design with clean separation of concerns and comprehensive tooling.

## Common Development Commands

### Package Management & Testing
```bash
# Activate the project environment
julia --project=.

# Install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run a specific test file
julia --project=. test/test_core_functions.jl

# Run examples
cd examples/grid_world && julia --project=../.. grid_world.jl
```

### Development Workflow
```bash
# Precompile the package
julia --project=. -e "using Pkg; Pkg.precompile()"

# Enter REPL with package loaded
julia --project=. -e "using GFlowNet"

# Build documentation
julia --project=docs docs/make.jl
```

## High-Level Architecture

The package follows a modular architecture with clear separation of concerns:

### Core Mathematical Engine (`src/core/`)
- **types.jl**: Fundamental abstract types (AbstractState, AbstractAction, GFlowNetModel)
- **graphs.jl**: DAG operations and state space analysis
- **policies.jl**: Forward policy P_F, backward policy P_B, and flow estimator Z
- **flows.jl**: Flow conservation and mathematical guarantees
- **balance.jl**: Training objectives (TB, DB, FM)
- **sampling.jl**: Trajectory generation algorithms
- **objectives.jl**: Loss computation for training
- **interface.jl**: High-level API functions like `create_gflownet()`

### Training Infrastructure (`src/training/`)
- **configuration.jl**: TrainingConfig type with validation, supports objectives (TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING)

### Professional Tooling (`src/utils/`)
- **validation.jl**: Input validation with `@ignore` for Zygote compatibility
- **logging.jl**: Training progress monitoring
- **visualization.jl**: Publication-quality plotting system
- **report.jl**: HTML report generation and CSV export
- **utils.jl**: General utilities

### Domain Applications (`src/applications/`)
Each implements the required GFlowNet interface:
- **grid_world.jl**: Navigation tasks (flagship example)
- **molecular_design.jl**: Chemical synthesis
- **causal_discovery.jl**: DAG structure learning
- **active_learning.jl**: Experiment selection
- **supply_chain_optimization.jl**: Business logistics

### Extensions (`src/extensions/`)
- **continuous.jl**: Continuous state spaces
- **non_acyclic.jl**: Non-DAG structures
- **information.jl**: Information-theoretic objectives

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

### Type System
- All neural network features must be `Float32` for consistency
- States must inherit from `AbstractState` with `is_terminal::Bool` field
- Actions must inherit from `AbstractAction`
- Use concrete types, avoid type instability

## Common Development Patterns

### Creating a New Domain
1. Define state and action types in `src/applications/your_domain.jl`
2. Implement the 5 required interface functions
3. Create `create_your_domain_gflownet()` high-level function
4. Add tests in `test/` following existing patterns
5. Create example in `examples/your_domain/`

### Training Workflow Pattern
```julia
# Create model
model = create_grid_world_gflownet(grid_size=5, ...)

# Configure training
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    n_iterations=100,
    batch_size=32
)

# Train
history = train_gflownet(model, config; verbose=true)

# Sample and evaluate
trajectories = [sample_trajectory(model) for _ in 1:100]
```

### Important Gotchas
- Always ensure rewards are positive (use `max(reward, 1e-8)`)
- Never mutate states - always create new instances
- Use `Float32` consistently for neural network compatibility
- Wrap all validation with `Zygote.@ignore`
- Test AD compatibility with gradient checks

## Detailed Development Guide

For comprehensive development guidelines including:
- Complete code examples with correct/incorrect patterns
- Full testing frameworks and integration examples
- Performance optimization strategies
- Debugging guides and troubleshooting
- Generic development checklists
- Success indicators and red flags

**See `COMPREHENSIVE_GFLOWNET_RULES_UPDATED.md` - the complete development rulebook for this project.**

## Key Development Principles

From the comprehensive guide:
- **On-demand computation**: No explicit DAG construction, everything computed when needed
- **Type safety**: Consistent Float32 usage, concrete types
- **Pure functions**: No mutations for AD compatibility
- **Positive rewards**: Mathematical requirement for GFlowNets
- **Clean interfaces**: Use high-level `create_*_gflownet()` functions

## Testing Philosophy
- Unit tests for core mathematical functions
- Integration tests for each domain
- AD compatibility tests with Zygote
- Performance benchmarks for large problems