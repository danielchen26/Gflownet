module GFlowNet

# Import required packages
using Graphs
using Distributions
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Plots
using Random
using Statistics
using StatsBase
using Zygote

# Re-export
export AbstractState, AbstractAction
export DirectedAcyclicGraph, create_dag
export GFlowNetModel, FlowMatchingObjective, DetailedBalanceObjective, TrajectoryBalanceObjective
export ForwardPolicy, BackwardPolicy, FlowEstimator, Trajectory
export flow, edge_flow, reward, sample_trajectory, state_to_features
export forward_transition_prob, backward_transition_prob
export is_applicable, apply_action, get_next_states, get_previous_states
export compute_loss_and_grad, apply_optimizer!, train!

# Include base type definitions
include("types.jl")

# Include DAG construction
include("directed_acyclic_graph.jl")

# Include flow network functionality
include("flow_networks.jl")

# Include policy definitions
include("policies/forward_policy.jl")
include("policies/backward_policy.jl")
include("policies/flow_estimator.jl")

# Include training methods
include("training/training.jl")

# Include extensions
include("extensions/continuous.jl")
include("extensions/non_acyclic.jl")
include("extensions/information.jl")

# Include applications
include("applications/molecular_design.jl")
include("applications/causal_discovery.jl")
include("applications/active_learning.jl")

# Include utilities
include("utils/utils.jl")

end # module
