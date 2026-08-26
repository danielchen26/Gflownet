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
include("training/checkpoint.jl")  # Versioned checkpoint save/load (Prerequisite D)

# =============================================================================
# Utilities and Validation
# =============================================================================

# utils.jl already included earlier
include("utils/validation.jl")
include("utils/logging.jl")
include("utils/visualization.jl")
include("utils/report.jl")

# =============================================================================
# SMILES Representations (CAFE-GFN)
# =============================================================================

include("representations/smiles/tokenizer.jl")
include("representations/smiles/smiles_state.jl")
include("representations/smiles/smiles_policy.jl")

# =============================================================================
# Advanced Training (CAFE-GFN)
# =============================================================================

include("training/pretraining.jl")
include("training/smiles_replay_buffer.jl")
include("training/molecular_frontier_buffer.jl")
include("training/frontier_sampling.jl")
include("training/hierarchical_controller_dataset.jl")
include("training/hierarchical_controller_models.jl")
include("training/hierarchical_controller_training.jl")
include("training/edit_operators.jl")
include("training/edit_trajectory_buffer.jl")
include("training/finetuning.jl")
include("training/genetic_operations.jl")
include("training/scaffold_aware.jl")
include("training/boosting.jl")
include("data/zinc_loader.jl")
include("inference/qgfn.jl")

# =============================================================================
# GPU Acceleration (Metal)
# =============================================================================
include("training/metal_accel.jl")

# =============================================================================
# Applications and Extensions
# =============================================================================

include("applications/smiles_gflownet.jl")
include("applications/hierarchical_edit_gflownet.jl")
include("training/parent_controller_dataset.jl")
include("training/parent_controller_models.jl")
include("training/parent_controller_training.jl")
include("training/operator_controller_dataset.jl")
include("training/operator_controller_models.jl")
include("training/operator_controller_training.jl")
include("training/option_value_dataset.jl")
include("training/option_value_models.jl")
include("training/option_value_training.jl")
include("training/edit_tb_dataset.jl")
include("training/edit_tb_model.jl")
include("training/edit_tb_loss.jl")
include("training/edit_tb_training.jl")
include("training/option_flow_dataset.jl")
include("training/option_flow_model.jl")
include("training/option_flow_loss.jl")
include("training/option_flow_training.jl")
include("training/option_flow_real_catalog.jl")
include("applications/molecular_design.jl")
include("applications/causal_discovery.jl")
include("applications/active_learning.jl")
include("applications/supply_chain_optimization.jl")
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
export TrainingObjective, TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING, SUB_TRAJECTORY_BALANCE, DIRECT_FLOW_OBJECTIVE, COMBINED_OBJECTIVES, TRAJECTORY_LIKELIHOOD_MAXIMIZATION, MULTI_OBJECTIVE_TB, SHIFTED_COSH_TB
export PartitionFunctionMethod
export SIMPLE_ESTIMATION, SAMPLING_ESTIMATION, LEARNABLE_ESTIMATION, ADAPTIVE_ESTIMATION

# Configuration utilities
export validate_training_config, get_objective_requirements, estimate_training_time
export create_default_config, create_fast_config, create_robust_config

# Training execution
export train_gflownet

# Checkpoint versioning (Prerequisite D)
export ModelCheckpoint, save_checkpoint, load_checkpoint

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

# Molecular design application (atom-level, from molecular_design.jl)
export MoleculeState, MoleculeData, AddAtomAction, AddBondAction, TerminateMoleculeAction
# Fragment-based molecular types (MolState, FragmentAction, etc.) are in
# molecular_generation.jl, loaded by the visualization server — not this module.
# Reaction GFlowNet factory (Gap 4, from interface.jl)
export create_reaction_gflownet

# Causal discovery application
export CausalState, CausalAction, create_causal_gflownet
export causal_reward, causal_features

# Active learning application
export ActiveLearningState, ActiveLearningAction, create_active_learning_gflownet
export active_learning_reward, active_learning_features

# Supply chain application
export SupplyChainState, SupplyChainAction, SupplyChainNode, SupplyChainConnection, SupplyChainNetwork
export AddConnectionAction, AddNodeAction, TerminateSupplyChainAction
export NodeType, SUPPLIER, WAREHOUSE, CUSTOMER
export create_supply_chain_gflownet, create_initial_supply_chain_state, create_supply_chain_actions

# Supply chain optimization application
export SupplyChainState, SupplyChainAction, SupplyChainNetwork
export Drug, Facility, PatientRegion, TransportRoute
export DrugType, ONCOLOGY, VACCINES, GENERICS, BIOLOGICS
export FacilityType, MANUFACTURING, DISTRIBUTION, DEPOT
export StorageType, AMBIENT, COLD, FROZEN
export ProduceAction, ShipAction, ServeAction, NextMonthAction, FinishPlanningAction

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
# CAFE-GFN: SMILES GFlowNet (ICLR 2025)
# =============================================================================

# SMILES tokenizer
export SMILESVocabulary, tokenize_smiles, encode, decode
export pad_sequence, batch_encode, PAD_TOKEN, START_TOKEN, END_TOKEN
export has_token, get_or_add_token!

# SMILES state and actions
export SMILESState, SMILESTokenAction, create_initial_smiles_state, create_smiles_actions
export state_to_smiles, smiles_to_state

# GRU policy
export SMILESPolicyModel, create_smiles_policy, create_smiles_gru_layers
export compute_log_probs_teacher_forced, sample_smiles_autoregressive, sample_smiles_batch
export compute_mle_loss_batched
export has_term_head, convert_to_term_head_params

# Loss functions
export apply_tb_loss, compute_kl_regularization_loss, compute_kl_weight

# Pretraining
export PretrainingConfig, PretrainingHistory, pretrain_smiles_gflownet
export compute_mle_loss, compute_tb_pretrain_loss

# Fine-tuning
export FinetuningConfig, FinetuningHistory, finetune_smiles_gflownet
export compute_tb_finetune_loss, compute_kl_smiles_loss, compute_rwmle_loss, compute_replay_loss

# SMILES Replay Buffer
export SMILESReplayEntry, SMILESReplayBuffer
export add_to_replay!, add_batch_to_replay!, sample_replay, sample_replay_with_delta
export update_deltas!, get_top_molecules, replay_stats

# Molecular Frontier Buffer
export MolecularFrontierEntry, MolecularFrontierBuffer
export canonicalize_smiles_identity, add_to_frontier!, update_frontier_delta!, sample_frontier, frontier_topk, frontier_stats, frontier_source_summary

# Frontier Sampling / Hierarchical Search
export FrontierSnapshotEntry, FrontierSnapshot, BasinSummary, ScoredBasinCandidate, ScoredParentCandidate
export compute_frontier_snapshot_id, create_frontier_snapshot, summarize_basins, basin_score, candidate_basins, sample_scored_basin, sample_basin
export parent_score, candidate_parents, sample_scored_parent, sample_parent

# Basin Controller Dataset / Models / Training
export BasinDecisionCandidate, BasinDecisionLog, BasinAttemptOutcomeSummary, BasinDecisionRecord, BasinControllerDataset
export build_basin_decision_candidates, frontier_feature_vector, basin_candidate_feature_vector
export summarize_basin_attempt_outcomes, compute_basin_target, audit_basin_dataset_coverage
export extract_basin_controller_dataset, split_basin_controller_dataset, basin_controller_dataset_stats
export LearnedBasinController, MLPBasinController, create_learned_basin_controller, create_mlp_basin_controller
export basin_candidate_score, score_basin_candidates, select_basin
export save_learned_basin_controller, load_learned_basin_controller
export BasinControllerTrainingConfig, train_basin_controller, evaluate_basin_controller, compare_basin_regressors

# Parent Controller Dataset / Models / Training
export ParentAttemptOutcomeSummary, ParentDecisionRecord, ParentControllerDataset
export summarize_parent_attempt_outcomes, compute_parent_target, parent_context_feature_vector, parent_candidate_feature_vector
export audit_parent_dataset_coverage, extract_parent_controller_dataset, split_parent_controller_dataset, parent_controller_dataset_stats
export HeuristicTopParentController, LearnedParentController, MLPParentController, AnchoredParentController
export create_learned_parent_controller, create_mlp_parent_controller, create_anchored_parent_controller
export parent_candidate_score, score_parent_candidates, select_parent
export save_learned_parent_controller, load_learned_parent_controller
export ParentControllerTrainingConfig, train_parent_controller, evaluate_parent_controller, compare_parent_regressors

# Operator Controller Dataset / Models / Training
export OperatorAttemptOutcomeSummary, OperatorDecisionRecord, OperatorControllerDataset
export summarize_operator_attempt_outcomes, compute_operator_target, operator_eligibility_feature_vector, operator_context_feature_vector, operator_candidate_feature_vector
export audit_operator_dataset_coverage, extract_operator_controller_dataset, split_operator_controller_dataset, operator_controller_dataset_stats
export LearnedOperatorEligibilityModel, HeuristicTopOperatorController, LearnedOperatorController, AnchoredOperatorController, EligibilityGatedOperatorController
export create_learned_operator_eligibility_model, create_learned_operator_controller, create_anchored_operator_controller, create_gated_operator_controller
export operator_eligibility_score, operator_candidate_score, score_operator_candidates, select_operator, operator_selection_metadata
export save_learned_operator_controller, load_learned_operator_controller
export OperatorControllerTrainingConfig, filter_operator_controller_dataset, train_operator_eligibility_model, evaluate_operator_eligibility_model, train_operator_controller, evaluate_operator_controller

# Option Value Dataset / Models / Training
export OptionValueRecord, OptionValueDataset
export option_value_feature_vector, option_override_feature_vector, extract_option_value_dataset, split_option_value_dataset, option_value_dataset_stats
export LearnedOptionValueModel, CalibratedOrdinalOptionPolicy, create_learned_option_value_model, create_calibrated_ordinal_option_policy
export option_value_score, option_override_confidence
export save_learned_option_value_model, load_learned_option_value_model, save_calibrated_ordinal_option_policy, load_calibrated_ordinal_option_policy
export OptionValueTrainingConfig, OptionCalibrationConfig, train_option_value_model, train_option_override_confidence_model, train_calibrated_ordinal_option_policy, evaluate_option_value_model

# Edit Operators
export AbstractEditOperator, MutateOperator, CrossoverOperator, AddFragmentOperator
export ReplaceFragmentOperator, DeleteFragmentOperator, TerminateOperator
export EditProposal, trusted_edit_operators, experimental_fragment_operators
export available_edit_operators, propose_edit, propose_edit_with_diagnostics, choose_partner, unique_child_proposals

# Edit Trajectory Buffer
export EditTrajectoryEntry, EditTrajectoryBuffer
export add_edit_trajectory!, sample_edit_trajectories, edit_trajectory_stats

# Genetic Operations & Augmentation
export augment_smiles_rdkit, create_augment_fn
export smiles_crossover_rdkit, smiles_mutate_rdkit, smiles_mutate_tokens
export ScaffoldFilter, get_scaffold, should_add_molecule, register_molecule!
export scaffold_diversity_stats, generate_genetic_molecules

# QGFN
export QFunctionNetwork, create_q_function, compute_q_values
export apply_q_masking, QTrainingBuffer, train_q_function!, compute_p_quantile
export add_q_transition!, sample_q_batch
export fill_transitions_with_reward!, collect_and_fill_q_buffer!

# Boosting
export BoostedGFlowNet, BoostedModelCheckpoint, n_rounds, ensemble_Z
export add_boosting_round!, should_continue_boosting, sample_from_ensemble
export cached_oracle_call, compute_residual_reward, get_ensemble_stats
export run_boosting_round!, run_boosted_training

# Data loading
export load_zinc_smiles, prepare_zinc_dataset, create_batch_iterator, load_pretrained_checkpoint

# Factory
export SMILESGFlowNetConfig, create_smiles_gflownet, create_smiles_training_config

# Hierarchical Edit Framework
export HierarchicalEditConfig, FrontierCommitRecord, HierarchicalEditStep, HierarchicalEditEpisode
export HierarchicalEditDecisionLog, HierarchicalEditProposalLog, ParentDecisionCandidate, ParentDecisionLog, OperatorDecisionCandidate, OperatorDecisionLog, HierarchicalEditDiagnosticsBuffer
export add_decision_log!, add_proposal_log!, add_basin_log!, add_parent_log!, add_operator_log!, decision_log_stats, proposal_log_stats
export choose_basin, choose_parent, choose_operator, choose_operator_action, frontier_quality_summary, compute_frontier_utility_delta
export run_hierarchical_edit_episode!, probe_parent_interventions, probe_coupled_hierarchy_options, probe_frontier_allocation_opportunities
export extract_option_subtrajectory_records, compare_option_value_surfaces
export FrontierAllocationRegionRecord, FrontierAllocationSnapshotRecord, FrontierAllocationDataset
export FrontierAllocationLinearModel, SelectiveFrontierAllocator
export extract_frontier_allocation_dataset, frontier_allocation_dataset_stats
export frontier_allocation_override_score, frontier_allocation_region_score
export evaluate_selective_frontier_allocator, train_selective_frontier_allocator
export OpportunityStateRecord, OpportunityStateDataset, OpportunityStateDetector
export extract_opportunity_state_dataset, opportunity_state_dataset_stats, opportunity_state_score
export evaluate_opportunity_state_detector, evaluate_opportunity_state_conditional_oracle
export opportunity_state_threshold_stability, evaluate_opportunity_state_repeatability
export select_sparse_positive_operating_point, evaluate_sparse_positive_operating_point, evaluate_sparse_positive_operating_points
export OpportunityRepairAuditRecord, OpportunityRepairAuditDataset
export extract_opportunity_repair_audit_dataset, opportunity_repair_audit_dataset_stats
export evaluate_opportunity_repair_binary_probe, evaluate_opportunity_repair_ordinal_probe
export evaluate_opportunity_representation_semantics_repair
export extract_intervention_geometry_atlas, intervention_geometry_atlas_stats, compare_intervention_geometry_atlas
export train_opportunity_state_detector

# Edit-TB: Factored Within-HE Edit Policy Pilot
export EditTBBasinChoice, EditTBParentChoice, EditTBOperatorChoice, EditTBStep, EditTBTrajectory
export EditTBConfig, EditTBDataset
export build_edit_tb_frontier_features, build_edit_tb_basin_candidate_features
export build_edit_tb_parent_candidate_features, build_edit_tb_operator_candidate_features
export compute_edit_tb_terminal_reward, load_edit_tb_dataset
export create_edit_policy, init_edit_policy, count_edit_policy_params
export compute_edit_rwmle_loss, compute_edit_tb_style_loss
export train_edit_policy!, choose_with_edit_policy
export EditTBPolicyController, set_edit_tb_task!
export save_edit_policy, load_edit_policy

# Option-Flow v0 POC
export OptionFlowCandidate, OptionFlowCatalog, OptionFlowMLPConfig, OptionFlowTrainingConfig
export normalize_option_utilities, make_option_flow_catalog, validate_option_flow_catalog
export option_flow_state_dim, option_flow_option_dim, option_flow_input_dim, option_flow_input_matrix, option_flow_utilities
export grouped_split_option_flow_catalogs, synthetic_option_flow_catalogs, option_flow_catalog_stats
export create_option_flow_mlp, init_option_flow_params, option_flow_logits, option_flow_log_probs, option_flow_probs, option_flow_param_count
export catalog_cross_entropy_loss, mean_catalog_cross_entropy_loss, uniform_catalog_cross_entropy, catalog_kl_to_target
export option_entropy, flow_residual_diagnostics, top_utility_mass, uniform_top_utility_mass, rank_correlation, option_utility_rank_correlation
export evaluate_option_flow_model, train_option_flow_model
export OptionFlowRealEncoding, discover_he_summary_files, load_he_summary_rows, option_flow_real_artifact_audit
export build_option_flow_real_encoding, build_summary_proxy_catalogs, filter_real_headline_catalogs
export infer_option_flow_source_family, option_flow_proxy_group_key, option_flow_real_state_features, option_flow_real_option_features
export option_flow_real_utility, catalog_has_option_feature_variation, evaluate_real_option_flow_model
export typed_path_feature_vector, augment_rows_with_typed_path_features
export evaluate_metadata_prior_baseline, uniform_policy_metrics, real_catalog_task_breakdown, summarize_real_catalog_collection, e1_summary_proxy_gate

# GPU Acceleration (Metal)
export init_metal_accel!, metal_available, to_metal, from_metal

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
