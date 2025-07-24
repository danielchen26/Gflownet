# GFlowNet.jl

*A clean, modern implementation of Generative Flow Networks in Julia*

[![Julia](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What are Generative Flow Networks (GFlowNets)?

**GFlowNets** are a novel class of generative models that learn to sample diverse, high-quality objects proportionally to a given reward function. Unlike traditional methods that converge to single optimal solutions, GFlowNets discover and sample from the entire distribution of high-reward candidates.

### Core Concept

Imagine you want to design molecules with specific properties. Traditional optimization finds one "best" molecule, but GFlowNets find **many diverse molecules** that are all good, sampling each with probability proportional to its reward. This is crucial for:

- **Drug discovery**: Need diverse molecular candidates, not just one
- **Experimental design**: Want varied experiments that are all informative  
- **Causal discovery**: Explore multiple plausible causal structures
- **Feature selection**: Find different feature combinations that work well

### Mathematical Foundation

GFlowNets model the generative process as a **flow network** where:
- **States** represent partial objects (e.g., partially built molecules)
- **Actions** extend objects (e.g., add atoms)
- **Flows** represent transition probabilities
- **Terminal rewards** define the target distribution

The key insight: **flow conservation** ensures that the probability of generating an object equals its normalized reward.

### Why GFlowNets vs Alternatives?

| Method | Single Solution | Diverse Solutions | Amortized Sampling | Exploration |
|--------|-----------------|-------------------|-------------------|-------------|
| **Gradient Descent** | ✅ | ❌ | ❌ | ❌ |
| **MCMC** | ❌ | ✅ | ❌ | ⚠️ Local |
| **Evolutionary** | ❌ | ⚠️ Limited | ❌ | ⚠️ Slow |
| **GFlowNets** | ❌ | ✅ | ✅ | ✅ Global |

## Package Architecture

This implementation follows modern ML package design principles with clean separation of concerns:

```
src/
├── core/                       # Core abstractions and algorithms
│   ├── types.jl               # Fundamental types and structures  
│   ├── dag.jl                 # Directed acyclic graph operations
│   ├── algorithms/            # Core algorithmic components
│   │   ├── objectives.jl      # Training objectives (TB, Sub-TB, Flow Consistency)
│   │   ├── sampling.jl        # Trajectory sampling and flow networks
│   │   └── partition.jl       # Partition function estimation methods
│   └── policies/              # Neural network policies
│       ├── base.jl           # Policy abstractions and utilities
│       ├── forward.jl        # Forward policy implementations  
│       ├── backward.jl       # Backward policy implementations
│       └── flow.jl           # Flow estimator implementations
├── training/                  # High-level training interface
│   ├── config.jl             # Training configuration and validation
│   ├── trainer.jl            # Main training loop and interface
│   └── optimization.jl       # Gradient computation and optimization
├── applications/              # Domain-specific implementations
│   ├── active_learning.jl    # Experiment selection
│   ├── causal_discovery.jl   # DAG structure learning
│   └── molecular_design.jl   # Sequential molecule construction
├── extensions/                # Advanced extensions
│   ├── continuous.jl         # Continuous state spaces
│   ├── information.jl        # Information-theoretic objectives
│   └── non_acyclic.jl        # Non-DAG graph structures
└── utils/                     # Utilities and helpers
    ├── rewards.jl            # Flexible reward framework
    ├── logging.jl            # Training progress logging
    ├── visualization.jl      # Plotting and visualization
    └── utils.jl              # General utility functions
```

### Design Principles

- **Clear Separation**: Core algorithms, training interface, and applications are logically separated
- **Modern Interface**: Configuration-based training with robust error handling
- **Extensible**: Easy to add new objectives, policies, and applications
- **Backward Compatible**: Legacy interfaces preserved while promoting modern usage

## Workflow

### 1. Installation

Since this package is under active development, install it in development mode:

```julia
using Pkg
# Clone and install in development mode
Pkg.develop(url="https://github.com/yourusername/GFlowNet.jl.git")

# Or if you've already cloned locally:
# Pkg.develop(path="path/to/GFlowNet.jl")
```

### 2. Basic Workflow

```julia
using GFlowNet

# Step 1: Define your environment (state space, actions, rewards)
env = create_environment(...)  # domain-specific

# Step 2: Configure training
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,           # Training objective
    partition_function_method = SIMPLE_ESTIMATION,  # Z estimation  
    batch_size = 32,
    n_iterations = 1000,
    learning_rate = 0.001
)

# Step 3: Train the GFlowNet
model, history = train_gflownet(env, config; verbose=true)

# Step 4: Sample diverse solutions
samples = [sample_trajectory(model) for _ in 1:100]
```

### 3. Training Objectives

Choose the right objective for your problem:

| **Objective** | **Use Case** | **Description** |
|---------------|-------------|----------------|
| `TRAJECTORY_BALANCE` | Standard problems | Basic GFlowNet training |
| `SUB_TRAJECTORY_BALANCE` | Long sequences | Better credit assignment on sub-trajectories |
| `ADAPTIVE_SUB_TB` | Sparse rewards | Focus on difficult sub-trajectories |
| `FLOW_CONSISTENCY` | Local structure | Unified detailed balance + flow matching |

### 4. Partition Function Estimation

The partition function Z normalizes the distribution:

| **Method** | **When to Use** | **Computational Cost** |
|------------|-----------------|----------------------|
| `SIMPLE_ESTIMATION` | Small, enumerable spaces | Low |
| `LEARNABLE_PARAMETER` | Medium complexity | Medium |
| `SAMPLING_BASED` | Dynamic spaces | Medium |
| `ADAPTIVE_ESTIMATION` | Unknown characteristics | Auto-adaptive |

### 5. Advanced Configuration

```julia
# For complex scenarios
config = TrainingConfig(
    objective = ADAPTIVE_SUB_TB,
    partition_function_method = ADAPTIVE_ESTIMATION,
    batch_size = 64,
    n_iterations = 5000,
    learning_rate = 0.001,
    sub_trajectory_config = Dict(
        :difficulty_threshold => 0.1,
        :n_subtrajectories => 5,
        :adaptive_depth => true
    ),
    validation_freq = 100,
    early_stopping = true
)
```

## Examples

### 🎯 Grid World (Beginner)

Navigate a grid to collect rewards, demonstrating basic GFlowNet concepts:

```julia
cd("examples/grid_world")
julia grid_world.jl
```

**What you'll learn:**
- Basic GFlowNet setup and training
- Trajectory sampling and visualization
- Different training objectives comparison

### 🧪 Active Learning (Intermediate)

Intelligent experiment selection for scientific discovery:

```julia
cd("examples/active_learning")  
julia active_learning.jl
```

**What you'll learn:**
- Sequential decision making with GFlowNets
- Sub-trajectory balance for better credit assignment
- Dynamic environment with changing values

### 🔬 Feature Acquisition (Advanced)

Strategic feature selection with budget constraints:

```julia
cd("examples/feature_acquisition")
julia main.jl
```

**What you'll learn:**
- Multi-experiment frameworks
- Advanced training objectives
- Comprehensive benchmarking and visualization
- Hybrid single/multi-experiment design

### 📊 Causal Discovery (Research)

Learning causal structures from observational data:

```julia
cd("examples/causal_discovery")
julia causal_discovery.jl
```

**What you'll learn:**
- DAG structure learning
- Non-deterministic environments
- Statistical evaluation metrics

### ⚛️ Molecular Design (Domain-Specific)

Sequential construction of molecular graphs:

```julia
cd("examples/molecule_design")
julia molecule_example.jl
```

**What you'll learn:**
- Constraint handling in generation
- Domain-specific state representations
- Chemical validity and optimization

## Quick Start Example

The best way to understand GFlowNets is through a working example. Here's the core workflow:

### 1. Define Your Domain

```julia
using GFlowNet

# Define state and action types for your domain
struct GridState <: GFlowNet.AbstractState
    x::Int  # x-coordinate
    y::Int  # y-coordinate
    is_terminal::Bool
end

abstract type GridAction <: GFlowNet.AbstractAction end
struct MoveRightAction <: GridAction end
struct TerminateAction <: GridAction end

# Implement required interface functions
function GFlowNet.is_applicable(action::MoveRightAction, state::GridState)
    !state.is_terminal && state.x < 5
end

function GFlowNet.apply_action(action::MoveRightAction, state::GridState)
    GridState(state.x + 1, state.y, false)
end

function GFlowNet.state_to_features(state::GridState)
    # Convert state to neural network input
    features = Float32[state.x, state.y, state.is_terminal ? 1.0 : 0.0]
    return features
end

function GFlowNet.reward(state::GridState)
    # Define reward function
    if !state.is_terminal
        return 0.0
    end
    return state.x == 5 && state.y == 5 ? 10.0 : 0.1
end
```

### 2. Create Model and Train

```julia
# Create GFlowNet model (see examples/grid_world/ for complete setup)
initial_state = GridState(1, 1, false)
terminal_states = [GridState(x, y, true) for x in 1:5, y in 1:5]
actions = [MoveRightAction(), TerminateAction()]

# Build DAG and neural networks (full example in examples/)
model = create_grid_world_model(initial_state, terminal_states, actions)

# Configure training
config = GFlowNet.TrainingConfig(
    objective = GFlowNet.TRAJECTORY_BALANCE,
    partition_function_method = GFlowNet.SIMPLE_ESTIMATION,
    batch_size = 32,
    n_iterations = 1000,
    learning_rate = 0.001
)

# Train the model
training_history = GFlowNet.train_gflownet(model, config; verbose=true)

# Sample diverse solutions
trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:10]
```

### 3. Complete Working Example

For a complete, runnable example:

```julia
cd("examples/grid_world")
julia grid_world.jl
```

This demonstrates the full workflow including DAG creation, neural network setup, training, and visualization.

## Testing

```julia
# Test package installation
using Pkg; Pkg.test("GFlowNet")

# Test specific examples
cd("examples/grid_world") && julia grid_world.jl
cd("examples/feature_acquisition") && julia main.jl

# Quick functionality check  
julia -e 'using GFlowNet; println("✅ Package loaded successfully!")'
```

## Documentation

- **[Core Concepts](docs/src/guide/core_concepts.md)**: Understanding GFlowNet fundamentals
- **[Training Objectives](docs/src/guide/training_objectives.md)**: Mathematical foundations
- **[API Reference](docs/api/)**: Complete function documentation
- **[Examples Guide](examples/README.md)**: Detailed example walkthroughs

## Contributing

1. **Fork** and create feature branch: `git checkout -b feature/new-capability`
2. **Follow architecture**: Add functionality in appropriate `core/`, `training/`, or `applications/` folders
3. **Use modern interface**: Implement using `TrainingConfig` patterns
4. **Add tests and documentation**
5. **Submit pull request**

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

- **GFlowNet Foundations**: Bengio, Y., et al. (2021). "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation." *NeurIPS 2021*.
- **Trajectory Balance**: Malkin, N., et al. (2022). "Trajectory balance: Improved credit assignment in GFlowNets." *NeurIPS 2022*.
- **Sub-Trajectory Balance**: Pan, L., et al. (2023). "Better Training of GFlowNets with Local Credit and Incomplete Trajectories." *ICML 2023*.

---

*Clean, modern, and production-ready Generative Flow Networks in Julia* 🚀

