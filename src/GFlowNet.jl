module GFlowNet

# External dependencies
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
using Dates
using GraphRecipes

# Type definitions first
include("types.jl")

# Core functionality
include("directed_acyclic_graph.jl")
include("flow_networks.jl")

# Policies and training
include("policies/forward_policy.jl")
include("policies/backward_policy.jl")
include("training/flow_matching.jl")
include("training/detailed_balance.jl")
include("training/trajectory_balance.jl")

# Applications
include("applications/molecular_design.jl")
include("applications/causal_discovery.jl")
include("applications/active_learning.jl")

# Extensions
include("extensions/continuous.jl")
include("extensions/information.jl")
include("extensions/non_acyclic.jl")

# Utilities (last, since they depend on other components)
include("utils/utils.jl")

# Export public API
export AbstractState, AbstractAction
export DirectedAcyclicGraph, Trajectory
export MoleculeData, DAGData, ExperimentData
export GFlowNetModel, FlowEstimator
export ForwardPolicy, BackwardPolicy
export FlowMatchingObjective, DetailedBalanceObjective, TrajectoryBalanceObjective
export create_dag, flow, edge_flow, state_to_features, reward
export forward_transition_prob, backward_transition_prob
export sample_trajectory, train!, compute_loss_and_grad
export is_applicable, apply_action, apply_optimizer!
export get_next_states, get_previous_states
export get_incoming_edges, get_outgoing_edges
export MoleculeState
export AddAtomAction, AddBondAction, TerminateMoleculeAction
export create_initial_molecule_state, create_molecular_design_model
export visualize_molecule

end # module
