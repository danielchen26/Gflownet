# GFlowNet.jl

A Julia implementation of Generative Flow Networks (GFlowNets).

## Introduction

GFlowNets are a novel class of generative models designed to sample from complex probability distributions proportionally to an energy function or reward function. They are particularly well-suited for tasks where:

1. The goal is to generate diverse samples from a complex distribution
2. The samples are constructed through a sequential, step-by-step process
3. The reward/energy can only be evaluated at the end of the construction process

This package provides a flexible framework for implementing and training GFlowNets in Julia.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/GFlowNet.jl")
```

## Quick Start

```julia
using GFlowNet

# Create a simple grid world GFlowNet
model = create_grid_world_gflownet(
    grid_size = 5,
    hidden_dim = 64,
    partition_function_method = LEARNABLE_ESTIMATION  # NEW: Learn Z for better exploration
)

# Configure training
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    partition_function_method = LEARNABLE_ESTIMATION,  # NEW: Enable Z learning
    n_iterations = 1000,
    batch_size = 32,
    learning_rate = 0.001
)

# Train the model
history = train_gflownet(model, config; verbose=true)

# Sample from the trained model
trajectories = [sample_trajectory(model) for _ in 1:10]

# Analyze results
for (i, traj) in enumerate(trajectories[1:3])
    terminal_state = traj.states[end]
    println("Trajectory $i: reward = $(reward(terminal_state))")
end
```

## Contents

- [Getting Started](guide/getting_started.md): Installation and first steps
- [Core Concepts](guide/core_concepts.md): Overview of key components
- [Mathematical Background](guide/mathematical_background.md): Theoretical foundations
- [Training Objectives](guide/training_objectives.md): Different training objectives
- [Learnable Partition Function](guide/learnable_partition_function.md): Advanced Z learning feature
- [Applications](applications/causal_discovery.md): Various use cases for GFlowNets

## Key Features

- **Multiple Training Objectives**: Implementations of Flow Matching, Detailed Balance, and Trajectory Balance
- **Learnable Partition Function**: Option to learn Z as a trainable parameter for better exploration
- **Flexible Architecture**: Customizable state and action spaces for diverse applications
- **Domain Applications**: Support for molecular design, causal discovery, and active learning
- **Extensions**: Support for continuous state spaces, non-acyclic graphs, and information theory
- **Visualization Utilities**: Tools for visualizing graphs, trajectories, and distributions

## Getting Started

To get started with GFlowNet.jl, check out the Getting Started guide in the Guide section, which provides installation instructions and a basic example.

## Reference Documentation

For detailed information about the API, see the API Reference section.

## Examples

For examples of how to use GFlowNet.jl for various applications, see the Examples section.

## Contributing

Contributions to GFlowNet.jl are welcome! See the Contributing guide for information on how to get involved.

## References

1. Bengio, E., Jain, M., Korablyov, M., Precup, D., & Bengio, Y. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. Advances in Neural Information Processing Systems, 34.

2. Bengio, Y., Deleu, T., Lahlou, S., Hu, E.J., Tiwari, M., & Bengio, E. (2021). GFlowNet Foundations. arXiv:2111.09266.

3. Malkin, N., Jain, M., Bengio, E., Sun, C., & Bengio, Y. (2022). Trajectory balance: Improved credit assignment in GFlowNets. NeurIPS 2022. 