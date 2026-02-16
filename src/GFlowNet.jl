# GFlowNet.jl - Generative Flow Networks
# Mathematical foundations for probabilistic generative modeling via flow conservation

module GFlowNet

# =============================================================================
# External Dependencies
# =============================================================================

using Random
using Statistics
using LinearAlgebra
using Dates

# Neural Networks and Optimization
using Lux
using Optimisers
using Zygote
using ComponentArrays
using NNlib

# Graph Operations
using Graphs

# Utilities
using StatsBase

# =============================================================================
# Core Mathematical Foundations
# =============================================================================

# Abstract types and interface contracts
include("core/types.jl")

# Load utilities module early as it's needed by other modules
include("utils/utils.jl")

# Graph theory and DAG operations
include("core/graphs.jl")

# Policy functions: P_F, P_B, Z
include("core/policies.jl")

# Flow conservation and computation
include("core/flows.jl")

# Balance conditions: TB, DB, FM
include("core/balance.jl")

# Trajectory sampling algorithms
include("core/sampling.jl")

# =============================================================================
# Training Configuration - Must come first
# =============================================================================

include("training/configuration.jl")

# =============================================================================
# High-level Interface
# =============================================================================

# High-level interface functions (model creation, sampling)
include("core/interface.jl")

# Multi-start GFlowNets core types
include("core/multi_start.jl")

# =============================================================================
# Training Infrastructure
# =============================================================================
include("training/replay_buffer.jl")  # Experience replay for off-policy learning
include("training/objectives.jl")
include("training/utils.jl")
include("training/losses.jl")
include("training/training.jl")
include("training/multi_start_training.jl")

# =============================================================================
# Utilities and Validation
# =============================================================================

# utils.jl already included earlier
include("utils/validation.jl")
include("utils/logging.jl")
include("utils/visualization.jl")
include("utils/report.jl")

# =============================================================================
# Applications and Extensions
# =============================================================================

include("applications/molecular_design.jl")
include("applications/causal_discovery.jl")
include("applications/active_learning.jl")
# include("applications/supply_chain_optimization.jl")  # Removed in core-fixes branch
include("applications/grid_world.jl")

include("extensions/continuous.jl")
include("extensions/non_acyclic.jl")
include("extensions/information.jl")

# =============================================================================
# Core Mathematical Types - Foundation Layer
# =============================================================================

# Abstract base types
export AbstractState, AbstractAction, AbstractPolicy

# Concrete mathematical types
export ForwardPolicy, BackwardPolicy, FlowEstimator
export Trajectory, GFlowNetModel

# =============================================================================
# Graph Operations - Structural Layer
# =============================================================================

# On-demand DAG operations
export get_applicable_actions, compute_next_state, is_valid_transition
export explore_state_space, count_reachable_states, analyze_state_space

# Legacy compatibility
export get_possible_actions

# =============================================================================
# Policy Functions - Core GFlowNet Mathematics
# =============================================================================

# Forward policy P_F(a|s)
export forward_probability, forward_action_probabilities, sample_forward_action
export compute_forward_logits

# Backward policy P_B(s|s')
export compute_backward_probability, is_valid_backward_transition

# Flow estimator Z(s)
export flow_estimate, compute_flow_logits

# Unified policy operations
export forward_transition_probability, backward_transition_probability
export safe_model_call, validate_policy_consistency

# =============================================================================
# Flow Conservation - Mathematical Core
# =============================================================================

# Flow computation methods
export flow, compute_recursive_flow, compute_flow_estimate
export FlowComputationMethod, RECURSIVE_FLOW, DIRECT_FLOW, MIXED_FLOW

# Flow analysis and validation
export validate_flow_conservation, validate_flow_consistency
export flow_analysis, partition_function, edge_flow

# Flow caching
export clear_flow_cache!, flow_computation_benchmark

# =============================================================================
# Balance Conditions - Training Mathematics
# =============================================================================

# Balance condition types
export BalanceCondition, TRAJECTORY_BALANCE_CONDITION, DETAILED_BALANCE_CONDITION, FLOW_MATCHING_CONDITION
export TrajectoryBalanceVariant, STANDARD_TB, GEOMETRIC_MEAN_TB

# Loss computation
export trajectory_balance_loss, sub_trajectory_balance_loss, sub_trajectory_balance_loss_batch
export detailed_balance_loss  # Now implemented!
export flow_matching_loss, flow_matching_loss_batch  # Now implemented!
# export flow_matching_loss  # Not fully implemented
export compute_balance_loss, validate_balance_conditions

# Balance utilities
export balance_condition_requirements, check_balance_condition_compatibility

# =============================================================================
# Trajectory Sampling - Inference Engine
# =============================================================================

# Sampling strategies and configuration
export SamplingStrategy, STOCHASTIC_SAMPLING, GREEDY_SAMPLING, TEMPERATURE_SAMPLING
export SamplingConfig

# Trajectory sampling
export sample_trajectory, sample_trajectory_batch, sample_backward_trajectory
export sample_backward_trajectories_from_terminals, find_parent_for_action
export sample_action_with_strategy



# Trajectory analysis
export is_valid_trajectory  # Only this function actually exists
export benchmark_sampling, get_trajectory_summary

# =============================================================================
# Training Objectives - Optimization Mathematics
# =============================================================================

# Training objective types
export ObjectiveConfig

# Objective computation
export trajectory_balance_objective
# export detailed_balance_objective, flow_matching_objective  # Not fully implemented
export combined_objective, compute_training_objective

# Regularization and analysis
export parameter_regularization_loss, policy_entropy_loss
export analyze_objective_components

# Gradient utilities
export compute_gradients, clip_gradients!

# =============================================================================
# Training Configuration and Infrastructure
# =============================================================================

# Training configuration
export TrainingConfig, TrainingState, TrainingMetrics, TrainingHistory
export TrainingObjective, TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING, SUB_TRAJECTORY_BALANCE, DIRECT_FLOW_OBJECTIVE, COMBINED_OBJECTIVES, TRAJECTORY_LIKELIHOOD_MAXIMIZATION
export PartitionFunctionMethod, OptimizationMethod
export SIMPLE_ESTIMATION, SAMPLING_ESTIMATION, LEARNABLE_ESTIMATION, ADAPTIVE_ESTIMATION
export ADAM, RMSPROP, SGD, ADAMW

# Configuration utilities
export validate_training_config, get_objective_requirements, estimate_training_time
export create_optimizer, create_default_config, create_fast_config, create_robust_config

# Training execution
export train_gflownet

# Sampling and trajectory generation
export sample_trajectory, sample_trajectory_batch

# =============================================================================
# State and Action Interface - Domain Integration
# =============================================================================

# Required interface methods (must be implemented by domains)
export state_to_features, is_terminal_state, reward
export is_applicable, apply_action

# Interface validation
export validate_state_interface, validate_action_interface

# =============================================================================
# Validation and Utilities
# =============================================================================

# Numerical validation
export validate_numerical_array, validate_neural_network_input, validate_neural_network_output
export validate_model_parameters, validate_reward
export validate_state_features, validate_policy_output, validate_state_for_policy

# Logging and monitoring
export setup_logging!, log_training_progress!, log_validation_results!
export create_training_logger, close_training_logger!

# Visualization and reporting
export plot_training_progress, plot_dag_structure, plot_trajectory_analysis
export generate_training_report, save_training_artifacts

# Z learning validation (LEARNABLE_ESTIMATION)
export validate_z_learning, validate_z_gradients, monitor_z_learning
export validate_z_mathematical_properties, compute_trajectory_log_probability

# Backward policy validation
export validate_backward_policy_normalization, validate_backward_policy_consistency
export monitor_backward_policy_learning

# =============================================================================
# Applications - Domain Implementations
# =============================================================================

# Molecular design application
export MolecularState, MolecularAction, create_molecular_gflownet
export molecular_reward, molecular_features

# Causal discovery application
export CausalState, CausalAction, create_causal_gflownet
export causal_reward, causal_features

# Active learning application
export ActiveLearningState, ActiveLearningAction, create_active_learning_gflownet
export active_learning_reward, active_learning_features

# Supply chain application - Removed in core-fixes branch
# export SupplyChainState, SupplyChainAction, SupplyChainNode, SupplyChainConnection, SupplyChainNetwork
# export AddConnectionAction, AddNodeAction, TerminateSupplyChainAction
# export NodeType, SUPPLIER, WAREHOUSE, CUSTOMER
# export create_supply_chain_gflownet, create_initial_supply_chain_state, create_supply_chain_actions

# Supply chain optimization application - Removed in core-fixes branch
# export SupplyChainState, SupplyChainAction, SupplyChainNetwork
# export Drug, Facility, PatientRegion, TransportRoute
# export DrugType, ONCOLOGY, VACCINES, GENERICS, BIOLOGICS
# export FacilityType, MANUFACTURING, DISTRIBUTION, DEPOT
# export StorageType, AMBIENT, COLD, FROZEN
# export ProduceAction, ShipAction, ServeAction, NextMonthAction, FinishPlanningAction

# Grid world application
export GridState, GridAction, MoveRight, MoveUp, MoveLeft, MoveDown, Terminate
export create_grid_world_gflownet, create_simple_grid_world, analyze_grid_world_results

# =============================================================================
# Extensions - Advanced Features
# =============================================================================

# Continuous state spaces
export ContinuousState, ContinuousGFlowNet
export continuous_sampling, continuous_flow_estimation

# Information-theoretic extensions
export mutual_information_reward, entropy_regularized_sampling
export information_bottleneck_objective

# Non-acyclic extensions (experimental)
export NonAcyclicGFlowNet, cycle_breaking_sampling

# =============================================================================
# Model Creation - High-Level Interface
# =============================================================================

# High-level model creation (following the rules for clean interface)
export create_forward_policy, create_backward_policy, create_flow_estimator
export create_gflownet, to_component_array

# Multi-start GFlowNets
export MultiStartGFlowNetModel, create_multi_start_gflownet
export sample_initial_state, get_initial_state_distribution

# =============================================================================
# Legacy Compatibility and Aliases
# =============================================================================

# Maintain some legacy names for backward compatibility
const get_possible_actions = get_applicable_actions

# =============================================================================
# Module-Level Documentation
# =============================================================================

"""
    GFlowNet

A Julia package for Generative Flow Networks (GFlowNets).

# Mathematical Foundation

GFlowNets are a class of probabilistic generative models that learn to sample from
unnormalized probability distributions by enforcing flow conservation:

**Flow Conservation Equation:**
```
F(s) = Σ_{s'} P_F(s'|s) * F(s')
```

where F(s) is the flow through state s, and P_F(s'|s) is the forward policy.

# Core Components

1. **State Space S**: The space of all possible states
2. **Action Space A**: The space of all possible actions
3. **Forward Policy P_F**: Probability of taking action a from state s
4. **Backward Policy P_B**: Probability of transitioning from s' back to s
5. **Flow Function F**: Amount of flow passing through each state
6. **Reward Function R**: Reward associated with terminal states

# Training Objectives

- **Trajectory Balance (TB)**: ∏P_F(s'|s) * Z(s₀) = R(s_T)
- **Detailed Balance (DB)**: P_F(s'|s) * F(s) = P_B(s|s') * F(s')
- **Flow Matching (FM)**: F(s) = Σ_{s'} P_F(s'|s) * F(s')

# Usage Example

```julia
using GFlowNet

# Define domain (states and actions)
initial_state = MyState(...)
actions = [MyAction(...), ...]

# Create DAG
config = DAGBuilderConfig(max_states=1000, exploration_strategy=:bfs)
dag = create_dag_with_exploration(initial_state, actions, config)

# Create model
model = create_gflownet_model_safe(dag, input_dim, hidden_dim, n_actions)

# Train
training_config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    n_iterations=1000,
    batch_size=32,
    learning_rate=0.01
)

# Sample trajectories
trajectory = sample_trajectory(model)
```

# Mathematical Guarantees

- **Flow Conservation**: Well-trained models satisfy flow conservation equations
- **Convergence**: Training objectives converge to true reward distribution
- **Diversity**: Sampling produces diverse trajectories proportional to rewards
- **Efficiency**: Amortized sampling without MCMC mixing time

For detailed mathematical foundations, see the individual module documentation.
"""
GFlowNet

end # module GFlowNet
