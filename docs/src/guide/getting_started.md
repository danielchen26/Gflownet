# Getting Started

This guide will help you get started with GFlowNet.jl.

## Installation

To install GFlowNet.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/GFlowNet.jl")
```

## Basic Example: Grid World

A simple example using a grid world environment:

```julia
using GFlowNet

# Create a 5x5 grid world environment
env = create_grid_world_environment(5, 5)

# Define a reward function (reaching the bottom-right corner)
function reward_fn(state)
    if state.position == (5, 5)
        return 1.0
    else
        return 0.1
    end
end

# Set the reward function
set_reward_function!(env, reward_fn)

# Create policies
forward_policy = create_forward_policy(env)
flow_estimator = create_flow_estimator(env)

# Train the GFlowNet
train!(forward_policy, flow_estimator, env, epochs=1000)

# Generate samples
samples = sample(forward_policy, env, num_samples=10)

# Visualize trajectories
visualize_trajectories(samples)
```

## Next Steps

- Learn about the [core concepts](core_concepts.md) of GFlowNets
- Explore the [mathematical background](mathematical_background.md)
- Try different [training objectives](training_objectives.md)
- Check out [applications](../applications/causal_discovery.md) such as causal discovery
