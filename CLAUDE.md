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

# Run examples (using example-specific Project.toml)
cd examples/grid_world && julia --project=. grid_world.jl
cd examples/supply_chain_optimization && julia --project=. ultimate_connected_gflownet.jl
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

### Testing
```bash
# Run the complete test suite
julia --project=. -e "using Pkg; Pkg.test()"

# Run individual test files
julia --project=. test/test_utilities.jl          # Test utilities and helpers
julia --project=. test/test_neural_networks.jl    # Neural network components
julia --project=. test/test_core_interface.jl     # Core GFlowNet interface
julia --project=. test/test_grid_world.jl         # Grid world application
julia --project=. test/test_training.jl           # Training infrastructure
julia --project=. test/test_supply_chain.jl       # Supply chain application

# Test via examples
cd examples/grid_world && julia --project=. grid_world.jl
cd examples/supply_chain_optimization && julia --project=. ultimate_connected_gflownet.jl
```

The test suite has been reorganized and updated to work with the current API. Old tests using deprecated APIs (DirectedAcyclicGraph, SimpleState, etc.) have been archived in `test/archive/old_tests/`.

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

## Critical Known Issues (January 2025)

### 1. Missing Core Functions
The following functions are called throughout the codebase but **never implemented**:
- `get_next_states()` - Used in flows.jl, balance.jl, policies.jl (10+ locations)
- `get_previous_states()` - Used in backward policy functions
- `get_root_state()` - Used in partition_function()

**Workaround**: Use `get_applicable_actions()` + `compute_next_state()` instead:
```julia
# Replace this (broken):
next_states = get_next_states(model.dag, state)

# With this (working):
applicable_actions = get_applicable_actions(state, model.all_actions)
next_states = [compute_next_state(action, state) for action in applicable_actions]
```

### 2. Broken Core Features
Due to missing functions, these core features don't work:
- `flow()`, `compute_recursive_flow()` - Core flow computations fail
- `partition_function()` - Calls non-existent get_root_state()
- `detailed_balance_loss()` - Requires missing backward_policy field
- `flow_matching_loss()` - Uses non-existent DAG functions
- `validate_flow_conservation()` - Uses non-existent DAG functions

### 3. API Issues
- Model doesn't have `backward_policy` field (but functions expect it)
- Optimizer name bug: Use `RMSProp` not `RMSprop` in configuration.jl:383
- Several exported functions don't exist (see test/README.md for full list)

### 4. Architecture Inconsistency
The codebase is transitioning between two approaches:
- **Old**: Explicit DAG with `get_next_states()`
- **New**: On-demand computation with `get_applicable_actions()`

Many core mathematical functions still use the old approach while infrastructure uses the new approach, causing runtime failures.

## Why Examples Still Work (Critical Understanding)

Despite all these broken functions, the examples work perfectly because:

### The Working Path
```julia
# This is what examples use - it works perfectly
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,  # ✅ Key to success
    n_iterations=100,
    batch_size=32
)
history = train_gflownet(model, config)
```

### What Makes It Work
1. **Trajectory Balance is Self-Contained**: 
   - Loss = (log P_F(τ) - log R(s_T))²
   - Only needs forward policy and rewards
   - No flow computation required

2. **On-Demand Computation Succeeds**:
   - `get_applicable_actions()` replaces `get_next_states()`
   - `apply_action()` handles state transitions
   - No explicit DAG needed

3. **Sampling is Simple**:
   - Just forward policy sampling
   - No backward policy needed
   - Pure state-to-state transitions

### What Would Break
```julia
# These would fail immediately:
config = TrainingConfig(objective=DETAILED_BALANCE)  # ❌ No backward_policy
config = TrainingConfig(objective=FLOW_MATCHING)     # ❌ No get_next_states()
Z = partition_function(model)                        # ❌ No get_root_state()
flow_val = flow(model, state)                        # ❌ No get_next_states()
```

### Key Takeaway
**Use `TRAJECTORY_BALANCE` and everything works**. The core training has been modernized, but advanced mathematical features haven't been migrated yet. See `test/README.md` for complete analysis.

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

## Project Structure Notes
- For each example folder, the Julia project should already run with an example-specific Project.toml
  - This ensures dependencies and configurations are tailored to each specific example
  - Allows for isolated and reproducible example environments