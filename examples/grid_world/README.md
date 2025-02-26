# Grid World Example

This example demonstrates a simple grid world navigation task using GFlowNets. 

## Features
- Illustrates state and action representations for a 2D grid environment
- Implements rewards at specific grid positions
- Visualizes GFlowNet training progress and sampled trajectories
- Shows how to use feature vectors with neural networks

## Running the Example
From the project root directory:
```julia
julia examples/grid_world/grid_world.jl
```

## Output Files
The example generates several visualization files:
- `grid_world_loss.png`: Training loss curve
- `grid_world_trajectories.png`: Visualization of sampled trajectories
- `grid_world_rewards.png`: Distribution of rewards 