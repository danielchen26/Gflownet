# Core Types in GFlowNet.jl

This page documents the core types and data structures used in the GFlowNet.jl package.

## State and Action Types

GFlowNet defines abstract types for states and actions:

* `AbstractState`: The base type for all state representations
* `AbstractAction`: The base type for all actions that can be applied to states

These abstract types serve as the foundation for domain-specific implementations.

## Trajectory

The `Trajectory` type represents a path through the state space, defined by:

* A sequence of states
* A sequence of actions that transition between those states
* Associated rewards and other metadata

## Directed Acyclic Graph

The `DirectedAcyclicGraph` structure represents the state space as a graph, with:

* Nodes representing states
* Edges representing possible transitions between states
* Metadata for both nodes and edges

## Domain-Specific Data Types

GFlowNet.jl uses a composition pattern for domain-specific data:

* `MoleculeData`: Holds data specific to molecular design tasks
* `DAGData`: Contains data for causal discovery tasks
* `ExperimentData`: Stores data for active learning and experimental design

## Policy Types

Policy types define how actions are selected:

* `AbstractPolicy`: Base type for all policies
* `ForwardPolicy`: Determines forward transitions in the state space
* `BackwardPolicy`: Determines backward transitions for training

## Flow Estimator

The `FlowEstimator` type models flow values through the state space, typically using neural networks.

## Training Objectives

GFlowNet.jl implements several training objectives:

* `FlowMatchingObjective`: Matches predicted flows with target flows
* `DetailedBalanceObjective`: Ensures detailed balance in state transitions
* `TrajectoryBalanceObjective`: Balances flows along trajectories

## GFlowNet Model

The `GFlowNetModel` integrates all components:

* Forward and backward policies
* Flow estimators
* Training objectives
* State and action spaces

## Code Example

Here's how you might define a custom state type:

```julia
struct MyCustomState <: AbstractState
    values::Vector{Float64}
    terminal::Bool
end

# Define methods for the new state type
is_terminal(state::MyCustomState) = state.terminal
state_to_features(state::MyCustomState) = state.values
``` 