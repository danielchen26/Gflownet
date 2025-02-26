# GFlowNet - Generative Flow Networks in Julia

This repository contains a Julia implementation of Generative Flow Networks (GFlowNets), a class of generative models for generating compositional objects by learning a stochastic policy for sequential construction.

## Project Structure

- `src/`: Core GFlowNet implementation
- `examples/`: Example applications in different domains
  - `grid_world/`: Simple grid navigation example
  - `molecule_design/`: Molecular design application
  - `causal_discovery/`: Discovering causal structures
  - `active_learning/`: Experimental design and active learning

## Running Examples

All examples should be run from the project root directory. For example:

```bash
julia examples/grid_world/grid_world.jl
```

Each example directory contains a README with more details about the specific implementation.

## Dependencies

This project uses several Julia packages:
- Lux.jl for neural networks
- Plots.jl for visualization
- StatsBase.jl for statistical utilities
- And others specified in the Project.toml file

## Getting Started

1. Clone this repository
2. Start Julia in the project directory 
3. Activate the project environment:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```
4. Run one of the examples as described above

## Example Applications

### Grid World Navigation

A simple grid world example that demonstrates the basic concepts of GFlowNets in a 2D grid environment.

### Molecular Design

Shows how GFlowNets can be used for molecular design tasks by sequentially building molecular structures.

### Causal Discovery

Demonstrates how GFlowNets can discover causal structures (directed acyclic graphs) that explain observed data.

### Active Learning

Uses GFlowNets for experimental design and selecting informative experiments.

# GFlowNet - Generative Flow Networks in Julia

This repository contains an implementation of Generative Flow Networks (GFlowNets) in Julia. GFlowNets are a novel approach to generative modeling, particularly useful for generating samples from complex, high-dimensional distributions.

## Overview

GFlowNets model the process of generating objects as a sequential decision-making process, creating a flow network over a structured state space. They are particularly well-suited for tasks where:

- The state space is compositional and structured
- The reward function is non-differentiable or only defined on complete objects
- Diversity in generated samples is important

This implementation provides a flexible framework for defining and training GFlowNets for various domains, such as molecule design, causal discovery, and active learning.

## Code Structure

The codebase is organized as follows:

```
src/
├── GFlowNet.jl                 # Main module file
├── types.jl                    # Core type definitions
├── directed_acyclic_graph.jl   # DAG construction and utilities
├── flow_networks.jl            # Flow computation and related operations
├── policies/                   # Policy implementations
│   ├── forward_policy.jl       # Forward policy (state → action)
│   └── backward_policy.jl      # Backward policy (state → previous state)
├── training/                   # Training objectives
│   ├── flow_matching.jl        # Flow matching objective
│   ├── detailed_balance.jl     # Detailed balance objective
│   └── trajectory_balance.jl   # Trajectory balance objective
├── applications/               # Domain-specific implementations
│   ├── molecular_design.jl     # Molecule generation
│   ├── causal_discovery.jl     # Causal structure discovery
│   └── active_learning.jl      # Active learning strategies
├── extensions/                 # Advanced extensions
│   ├── continuous.jl           # Support for continuous state spaces
│   ├── information.jl          # Information-theoretic extensions
│   └── non_acyclic.jl          # Support for non-acyclic state spaces
└── utils/                      # Utility functions
    └── utils.jl                # General utilities and helpers
```

## Getting Started

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/GFlowNet.jl.git
   cd GFlowNet.jl
   ```

2. Activate the Julia environment and install dependencies:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

### Basic Usage

Here's a simple example of how to use GFlowNets for molecular design:

```julia
using GFlowNet
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Create a GFlowNet model for molecular design
model = create_molecular_design_model()

# Create an initial state
initial_state = create_initial_molecule_state()

# Get available actions
available_actions = filter(a -> is_applicable(a, initial_state), model.dag.actions)

# Apply an action
action = available_actions[1]
new_state = apply_action(action, initial_state)

# Visualize the state
visualize_molecule(new_state)
```

For more detailed examples, check the `examples/` directory.

## Type Design and Composition

This implementation uses a composition-based approach for defining domain-specific types, which offers several advantages over traditional inheritance:

1. **Abstract base types**:
   - `AbstractState`: Base for all state representations
   - `AbstractAction`: Base for all action representations

2. **Domain-specific data structures**:
   - `MoleculeData`: For molecular design, contains atoms and bonds
   - `DAGData`: For causal discovery, contains graph structure
   - `ExperimentData`: For active learning, contains experiment information

3. **Concrete types through composition**:
   - For example, `MoleculeState <: AbstractState` contains a `MoleculeData` field
   - This approach separates data storage from behavior
   - Allows domain-specific implementations without complex inheritance hierarchies

4. **Parametric types for type safety**:
   - `DirectedAcyclicGraph{S,A}` is parameterized by specific state and action types
   - This ensures type safety while allowing specialized implementations

### Benefits of Composition Over Inheritance

1. **Modularity**: Domain-specific logic stays within its module
2. **Flexibility**: Easy to add new domains without changing core components
3. **Type Stability**: Julia compiler can better optimize with concrete types
4. **Simplicity**: Avoids deep inheritance hierarchies that are harder to understand

### Example: Molecular Design Implementation

```julia
# Define data structure
struct MoleculeData
    atoms::Vector{Symbol}
    bonds::Vector{Tuple{Int, Int, Int}}
end

# Define state through composition
struct MoleculeState <: AbstractState
    data::MoleculeData
    complete::Bool
end

# Define domain-specific actions
struct AddAtomAction <: AbstractAction
    atom_type::Symbol
    position::Tuple{Float64, Float64, Float64}
end

# Implement required interface
function apply_action(action::AddAtomAction, state::MoleculeState)
    # Create a new state with the atom added
    new_atoms = copy(state.data.atoms)
    push!(new_atoms, action.atom_type)
    
    return MoleculeState(
        MoleculeData(new_atoms, copy(state.data.bonds)),
        false
    )
end
```

This design allows each domain to define its own data structures and behavior while leveraging the common GFlowNet algorithms.

## Training a GFlowNet

GFlowNets can be trained using various objectives:

```julia
# Create a model with a trajectory balance objective
model = GFlowNetModel(
    dag,
    forward_policy,
    nothing,  # No backward policy needed for trajectory balance
    nothing,  # No flow estimator needed
    nothing,  # No partition function needed
    [TrajectoryBalanceObjective(1.0)],
    optimizer
)

# Train the model
for epoch in 1:num_epochs
    loss = train!(model, batch_size=32)
    println("Epoch $epoch: loss = $loss")
end
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Bengio, Y., Deleu, T., Rahaman, N., Ke, R., Lachapelle, S., Bilaniuk, O., ... & Zhang, S. (2021). A flow-based algorithm for generative flow networks. Advances in Neural Information Processing Systems, 34.

2. Bengio, Y., Gupta, P., Matthieu, M., Nikolentzos, G., & Romoff, J. (2021). Flow network based generative models for non-iterative diverse candidate generation. Advances in Neural Information Processing Systems, 34.

3. Jain, M., Bengio, Y., & Ozair, S. (2022). GFlowNet Foundations. arXiv preprint arXiv:2211.14302.

# Additional Examples

The package includes several example applications to demonstrate GFlowNet usage:

## Grid World Navigation

A simple grid world example demonstrating the basic concepts of GFlowNets. The agent learns to navigate
a 2D grid world to find high-reward locations.

Run the example:
```bash
julia examples/grid_world.jl
```

## Causal Discovery

An example showing how GFlowNets can be used for causal discovery, finding directed acyclic graphs
that best explain observed data.

Run the example:
```bash
julia examples/causal_discovery.jl
```

## Active Learning

An example demonstrating how GFlowNets can be used for active learning and experimental design,
selecting a diverse and informative set of experiments.

Run the example:
```bash
julia examples/active_learning.jl
```

## Applications

The package includes several more specialized applications:

1. **Molecular Design**: Generate molecular structures with desired properties
2. **Causal Discovery**: Discover causal structures from data
3. **Active Learning**: Select informative experiments

## Extensions

Additional extensions to the base GFlowNet framework:

1. **Continuous State Spaces**: Support for continuous state and action spaces
2. **Non-Acyclic GFlowNets**: Support for flow networks with cycles
3. **Information-Based Methods**: Entropy estimation and mutual information

