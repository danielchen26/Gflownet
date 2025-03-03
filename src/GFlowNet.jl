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

# Add this line to disable precompilation until issues are resolved
__precompile__(false)

# Type definitions first
include("types.jl")

# Core functionality
include("directed_acyclic_graph.jl")
include("rewards.jl")  # Add our new rewards module
include("flow_networks.jl")

# Policies and training
include("policies/forward_policy.jl")
include("policies/backward_policy.jl")
include("policies/flow_estimator.jl")

# Include our custom implementation first
include("training/compute_loss_and_grad.jl")

# Then include other training files
include("training/flow_matching.jl")
include("training/detailed_balance.jl")
include("training/trajectory_balance.jl")

# Training controller
include("training/training.jl")

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

# Make sure to export apply_optimizer! which comes from utils
export apply_optimizer!

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
export is_applicable, apply_action
export get_next_states, get_previous_states
export get_incoming_edges, get_outgoing_edges
export MoleculeState
export AddAtomAction, AddBondAction, TerminateMoleculeAction
export create_initial_molecule_state, create_molecular_design_model
export visualize_molecule
export create_flow_estimator, create_forward_policy, estimate_partition_function

# Export reward framework
export RewardFunction, RewardContext, StandardContext
export FunctionalReward, ValueMinusCostReward
export compute_reward, ensure_positive

# Re-export utilities from the GFlowNetUtils module
export GFlowNetLogger, log_iteration!, get_metric
export visualize_training_progress, visualize_reward_distribution
export visualize_causal_graph, visualize_experiment_selection

end # module
