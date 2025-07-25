module GFlowNet

# Enable precompilation for better performance

# External dependencies
using Graphs
using Distributions
using LinearAlgebra
using Lux
using ComponentArrays
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


# =============================================================================
# Core Components (New Structure)
# =============================================================================

# Core types and data structures
include("core/types.jl")
include("core/dag.jl")
include("core/transitions.jl")

# Core algorithms
include("core/algorithms/sampling.jl")
include("core/algorithms/objectives.jl")
include("core/algorithms/partition.jl")

# Policy implementations
include("policies/base.jl")
include("policies/forward.jl")
include("policies/backward.jl")

# =============================================================================
# Training Infrastructure
# =============================================================================

# Training interface (loads config, rewards, and optimization internally)
include("training/rewards.jl")
include("training/config.jl")
include("training/optimization.jl")
include("training/trainer.jl")

# =============================================================================
# Applications and Extensions
# =============================================================================

# Extensions
include("extensions/continuous.jl")
include("extensions/information.jl")
include("extensions/non_acyclic.jl")

# Applications (load before visualization to define concrete types)
include("applications/active_learning.jl")
include("applications/causal_discovery.jl")
include("applications/molecular_design.jl")

# =============================================================================
# Utilities
# =============================================================================

# Utility functions and helpers
include("utils/validation.jl")
include("utils/logging.jl")
include("utils/visualization.jl")
# Note: utils.jl creates a submodule, so we need to import its exports

# =============================================================================
# Exports - Core Types and Data Structures
# =============================================================================

export State, Action, Trajectory, TrajectorySet
export DAG, TerminalSink, create_dag, add_state!, add_action!
export is_terminal, get_parents, get_children, get_actions
export DirectedAcyclicGraph, SimpleState, SimpleAction

# =============================================================================
# Exports - GFlowNet Model and Components
# =============================================================================

export GFlowNetModel, ForwardPolicy, BackwardPolicy, FlowEstimator
export to_component_array, create_gflownet_model_safe
export validate_reward, validate_state_features, validate_neural_network_input, validate_neural_network_output, validate_numerical_array, validate_model_parameters
export forward_transition_prob, backward_transition_prob, flow, sample_trajectory, clear_flow_cache!
export state_to_features, edge_flow, is_terminal_state
export get_next_states, get_previous_states, get_possible_actions

# =============================================================================
# Exports - Training Interface
# =============================================================================

# Core training functions
export train_gflownet, train_gflownet_simple
export TrainingConfig, TrainingObjective, PartitionFunctionMethod

# Training objectives
export TRAJECTORY_BALANCE, GENERAL_TRAJECTORY_BALANCE, SUB_TRAJECTORY_BALANCE
export HIERARCHICAL_SUB_TB, ADAPTIVE_SUB_TB, FLOW_CONSISTENCY

# Partition function methods
export SIMPLE_ESTIMATION, LEARNABLE_PARAMETER, SAMPLING_BASED, ADAPTIVE_ESTIMATION
export SimplePartitionFunctionEstimator, LearnablePartitionFunctionEstimator
export SamplingPartitionFunctionEstimator, AdaptivePartitionFunctionEstimator

# =============================================================================
# Exports - Training Objectives and Loss Functions
# =============================================================================

# Unified flow consistency functionality
export flow_consistency_loss, FlowConsistencyMode, EDGE_LEVEL, STATE_LEVEL, MIXED_LEVEL
export estimate_partition_function, update_partition_function!

# All loss functions
export trajectory_balance_loss, general_trajectory_balance_loss
export sub_trajectory_balance_loss, hierarchical_sub_trajectory_balance_loss
export adaptive_sub_trajectory_balance_loss

# Legacy compatibility for examples only
export compute_loss_and_grad, apply_optimizer!

# =============================================================================
# Exports - Policy Functions
# =============================================================================

# Policy creation and manipulation
export create_forward_policy, create_backward_policy, create_flow_estimator
export forward_transition_logits, backward_transition_logits
export forward_action_probabilities, backward_action_probabilities
export sample_action, sample_prev_state, estimate_flow, estimate_edge_flow

# Policy utilities
export state_to_features, normalize_probabilities, sample_from_probabilities
export clamp_probabilities, validate_policy_output
export PolicyError, PolicyMetrics, increment_policy_metric!

# =============================================================================
# Exports - Optimization and Training Utilities
# =============================================================================

export setup_optimizers, compute_gradient_norm, clip_gradients!
export validate_training_config, get_config_summary
export evaluate_model, save_training_checkpoint, load_training_checkpoint

# =============================================================================
# Exports - Reward Functions
# =============================================================================

export reward, RewardFunction, RewardContext, StandardContext
export FunctionalReward, ValueMinusCostReward
export compute_reward, ensure_positive

# =============================================================================
# Exports - Utilities
# =============================================================================

# Export logging utilities
export GFlowNetLogger, log_metric!, log_iteration!, get_metric, get_last_metric, reset!, save_metrics
export summarize_performance, time_execution, benchmark_sampling
export log_info, log_warning, log_error
export visualize_dag, plot_training_progress, save_trajectory_plot

# =============================================================================
# Exports - Applications
# =============================================================================

export ActiveLearningEnvironment, CausalDiscoveryEnvironment, MolecularDesignEnvironment
export setup_active_learning, setup_causal_discovery, setup_molecular_design
export MoleculeState, DAGState, ExperimentState  # Export concrete state types

# Application-specific actions
export SelectExperimentAction, TerminateExperimentAction
export AddEdgeAction, RemoveEdgeAction, TerminateDAGAction
export AddAtomAction, AddBondAction, TerminateMoleculeAction

# =============================================================================
# Exports - Extensions
# =============================================================================

export ContinuousGFlowNet, InformationGFlowNet, NonAcyclicGFlowNet
export continuous_action_space, information_objective, handle_cycles
export ContinuousState, ContinuousAction, GaussianPolicy
export CyclicFlowNetwork, create_cyclic_network

end # module GFlowNet
