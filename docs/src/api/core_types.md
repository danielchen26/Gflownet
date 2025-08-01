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

## GFlowNet Model Architecture

GFlowNet.jl uses an on-demand computation architecture rather than explicit graph construction:

* States are generated dynamically during sampling
* Transitions are computed through action application
* No explicit state space enumeration required

## Domain-Specific Data Types

GFlowNet.jl uses a composition pattern for domain-specific data:

* `MoleculeData`: Holds data specific to molecular design tasks
* `DAGData`: Contains data for causal discovery tasks
* `ExperimentData`: Stores data for active learning and experimental design

## Policy Types

Policy types define how actions are selected:

* `AbstractPolicy`: Base type for all policies
* `ForwardPolicy`: Neural network that determines action probabilities during sampling

## Flow Estimator

The `FlowEstimator` type is experimental and not currently used in the working implementation. Flow computation is handled implicitly through the Trajectory Balance objective.

## Training Objectives

GFlowNet.jl implements several training objectives:

* `TrajectoryBalanceObjective`: Primary objective, balances flows along trajectories
* `FlowMatchingObjective`: Experimental, requires flow computation (not currently working)
* `DetailedBalanceObjective`: Experimental, requires backward policy (not currently working)

## GFlowNet Model

The `GFlowNetModel` integrates all components:

* Forward policy (neural network)
* Training objectives (primarily Trajectory Balance)
* Domain-specific state and action interfaces
* On-demand state space computation

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