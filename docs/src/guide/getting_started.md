# Getting Started

This guide will help you get started with **modern** GFlowNet.jl.

## Installation

To install GFlowNet.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/GFlowNet.jl")
```

## Modern Example: Grid World

A simple example using the **modern training interface** with grid world:

```julia
using GFlowNet

# Define state and action types
struct GridState <: GFlowNet.AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

# Define actions
abstract type GridAction <: GFlowNet.AbstractAction end
struct MoveRightAction <: GridAction end
struct MoveLeftAction <: GridAction end
struct TerminateAction <: GridAction end

# Implement interface functions
function GFlowNet.is_applicable(action::MoveRightAction, state::GridState)
    !state.is_terminal && state.x < 5
end

function GFlowNet.apply_action(action::MoveRightAction, state::GridState)
    GridState(state.x + 1, state.y, false)
end

function GFlowNet.reward(state::GridState)
    state.is_terminal && state.x == 5 && state.y == 5 ? 10.0 : 0.0
end

# Create model (see examples/grid_world/ for complete implementation)
model = create_grid_world_model()

# Modern training configuration
config = GFlowNet.TrainingConfig(
    objective=GFlowNet.TRAJECTORY_BALANCE,
    partition_function_method=GFlowNet.SIMPLE_ESTIMATION,
    batch_size=32,
    learning_rate=0.001,
    n_iterations=1000
)

# Train using modern interface
training_history = GFlowNet.train_gflownet(model, config; verbose=true)

# Sample trajectories
trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:10]
```

## Advanced Features

The modern interface provides access to cutting-edge techniques:

```julia
# Sub-trajectory balance for better credit assignment
config = GFlowNet.TrainingConfig(
    objective=GFlowNet.SUB_TRAJECTORY_BALANCE,
    sub_trajectory_config=Dict(:min_length => 2, :max_length => 5)
)

# Adaptive Z estimation for complex spaces
config = GFlowNet.TrainingConfig(
    partition_function_method=GFlowNet.ADAPTIVE_ESTIMATION
)

# General trajectory balance for non-deterministic environments
config = GFlowNet.TrainingConfig(
    objective=GFlowNet.GENERAL_TRAJECTORY_BALANCE
)
```

## Next Steps

- Learn about the [core concepts](core_concepts.md) of GFlowNets
- Explore the [mathematical background](mathematical_background.md)
- Try different [training objectives](training_objectives.md)
- Check out [applications](../applications/causal_discovery.md) such as causal discovery
