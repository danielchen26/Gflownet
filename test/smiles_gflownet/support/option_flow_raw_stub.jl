# Lightweight GFlowNet raw-artifact deserialization stubs for Option-Flow E2.
#
# This file is intentionally used only by test/runner scripts that need to read
# he_raw_diagnostics.jls / he_raw_trajectory.jls without loading the full
# GFlowNet molecular stack (PythonCall/RDKit/CondaPkg). It registers a minimal
# module for the serialized package id and defines only the structs required by
# historical HE artifacts.

using UUIDs

module GFlowNet
struct EditTrajectoryEntry
    basin_scaffold::String
    parent_smiles::String
    parent_reward::Float64
    operator::Symbol
    child_smiles::String
    child_reward::Float64
    reward_delta::Float64
    step_index::Int
    terminated::Bool
    metadata::Dict{String,Any}
end

struct BasinDecisionCandidate
    scaffold::String
    count::Int
    best_reward::Float64
    mean_reward::Float64
    mean_novelty::Float64
    mean_delta::Float64
    heuristic_score::Float64
    target_match::Bool
end

struct BasinDecisionLog
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    budget_remaining::Int
    created_at_step::Int
    frontier_size::Int
    frontier_top1::Float64
    frontier_top10_mean::Float64
    frontier_scaffold_count::Int
    candidate_basins::Vector{BasinDecisionCandidate}
    chosen_index::Int
    chosen_basin_scaffold::String
    chosen_basin_score::Float64
end

struct HierarchicalEditDecisionLog
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    basin_scaffold::String
    basin_score::Float64
    parent_smiles::String
    parent_reward::Float64
    operator::Symbol
    candidate_count::Int
    chosen_child_smiles::String
    child_reward::Float64
    reward_delta::Float64
    terminated::Bool
    parent_scaffold::String
    child_scaffold::String
    family_transition_type::String
    enters_topk::Bool
    delta_top1::Float64
    delta_top10_mean::Float64
    family_novelty_bonus::Float64
    frontier_utility_delta::Float64
    commit_applied::Bool
end

struct HierarchicalEditProposalLog
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    basin_scaffold::String
    basin_score::Float64
    parent_smiles::String
    parent_reward::Float64
    parent_scaffold::String
    operator::Symbol
    partner_smiles::Union{Nothing,String}
    raw_candidate_count::Int
    duplicate_candidate_count::Int
    empty_child_count::Int
    self_child_count::Int
    cached_child_count::Int
    unique_valid_count::Int
    same_family_count::Int
    cross_family_count::Int
    no_scaffold_count::Int
    chosen_child_smiles::String
    chosen_reward::Float64
    chosen_reward_delta::Float64
    reward_q25::Float64
    reward_q50::Float64
    reward_q75::Float64
    reward_max::Float64
    empty_after_filter::Bool
end

struct ParentDecisionCandidate
    smiles::String
    scaffold::String
    reward::Float64
    novelty_score::Float64
    tb_delta_abs::Float64
    source::String
    heuristic_score::Float64
    visit_count::Int
    basin_match::Bool
    target_match::Bool
end

struct ParentDecisionLog
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    basin_scaffold::String
    basin_score::Float64
    candidate_parents::Vector{ParentDecisionCandidate}
    chosen_index::Int
    chosen_parent_smiles::String
    chosen_parent_score::Float64
    heuristic_top_index::Int
    learned_top_index::Int
    heuristic_margin::Float64
    learned_margin::Float64
    learned_advantage_vs_heuristic::Float64
    heuristic_entropy::Float64
    learned_entropy::Float64
    override_applied::Bool
    abstained_to_heuristic::Bool
    selection_reason::String
end

struct OperatorDecisionCandidate
    operator::Symbol
    heuristic_score::Float64
    total_count::Int
    positive_delta_count::Int
    exploration_bonus::Float64
    structural_bias::Float64
end

struct OperatorDecisionLog
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    basin_scaffold::String
    basin_score::Float64
    parent_smiles::String
    parent_reward::Float64
    parent_scaffold::String
    parent_novelty_score::Float64
    parent_tb_delta_abs::Float64
    parent_source::String
    candidate_operators::Vector{OperatorDecisionCandidate}
    chosen_index::Int
    chosen_operator::Symbol
    chosen_heuristic_score::Float64
    predicted_eligible::Bool
    eligibility_score::Float64
    acted_on::Bool
    preserved_to_heuristic::Bool
    heuristic_top_index::Int
    learned_top_index::Int
    heuristic_margin::Float64
    learned_margin::Float64
    learned_advantage_vs_heuristic::Float64
    heuristic_entropy::Float64
    learned_entropy::Float64
    override_applied::Bool
    abstained_to_heuristic::Bool
    selection_reason::String
end
end

const OPTION_FLOW_RAW_GFLOWNET_PKGID = Base.PkgId(UUID("2d7ca041-c8ad-46e9-a25d-4e8f55c0c8f5"), "GFlowNet")
Base.loaded_modules[OPTION_FLOW_RAW_GFLOWNET_PKGID] = Main.GFlowNet
