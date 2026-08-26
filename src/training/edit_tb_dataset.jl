# ============================================================================
# Edit-TB Dataset — Leak-Free Data Pipeline
# ============================================================================
#
# Factored within-HE edit policy pilot: basin + parent + operator decisions.
# All features are decision-time observables only. No post-decision leakage.
#
# Field names verified against actual artifact structs (2026-03-17):
#   BasinDecisionLog.candidate_basins  (not .candidates)
#   ParentDecisionLog.candidate_parents (not .candidates)
#   OperatorDecisionLog.candidate_operators (not .candidates)
#   ParentDecisionCandidate.reward (direct, not .entry.reward)
#   ParentDecisionCandidate.heuristic_score (not .score)

using Serialization
using Statistics
using Random

# ============================================================================
# Data Structures
# ============================================================================

struct EditTBBasinChoice
    frontier_features::Vector{Float32}    # 8-d: from frontier state at this step
    candidate_features::Matrix{Float32}   # (5 or 6) × n: from BasinDecisionLog.candidate_basins
    chosen_index::Int
    n_candidates::Int
end

struct EditTBParentChoice
    basin_features::Vector{Float32}       # 4-d: from chosen basin context
    candidate_features::Matrix{Float32}   # (4 or 5) × n: from ParentDecisionLog.candidate_parents
    chosen_index::Int
    n_candidates::Int
end

struct EditTBOperatorChoice
    parent_features::Vector{Float32}      # 6-d: parent + step context
    candidate_features::Matrix{Float32}   # (3 or 4) × n: from OperatorDecisionLog.candidate_operators
    chosen_index::Int
    n_candidates::Int
end

struct EditTBStep
    basin::EditTBBasinChoice
    parent::EditTBParentChoice
    operator::EditTBOperatorChoice
end

struct EditTBTrajectory
    task_name::String
    run_id::String
    episode_id::String
    steps::Vector{EditTBStep}
    terminal_reward::Float64              # R(τ), configurable
    phase::Symbol                         # :warmup or :interleaved
end

struct EditTBConfig
    # Loss selection
    objective::Symbol                     # :rwmle (default) or :tb
    reward_mode::Symbol                   # :top10_delta (PRE-REGISTERED headline)

    # Shared hyperparameters
    n_epochs::Int
    learning_rate::Float64
    beta::Float64                         # reward exponent
    gradient_clip::Float64
    batch_size::Int
    val_fraction::Float64

    # TB-specific (only used when objective == :tb)
    lr_z::Float64
    threshold::Float64                    # cosh threshold
    log_z_grad_clip::Float64

    # Online stabilizer
    kl_to_heuristic::Float64
    heuristic_interpolation::Float64      # λ in (1-λ)·heuristic + λ·learned
    entropy_floor::Float64

    # Ablation controls
    include_heuristic_scores::Bool        # default: true; false for ablation

    # Final-theory candidate
    task_conditioning::Bool               # prepend task one-hot features to every decision context
end

function EditTBConfig(;
    objective=:rwmle, reward_mode=:top10_delta,
    n_epochs=50, learning_rate=1e-3, beta=4.0,
    gradient_clip=1.0, batch_size=32, val_fraction=0.15,
    lr_z=1e-2, threshold=2.0, log_z_grad_clip=5.0,
    kl_to_heuristic=0.1, heuristic_interpolation=0.5,
    entropy_floor=0.1, include_heuristic_scores=true,
    task_conditioning=false
)
    EditTBConfig(objective, reward_mode, n_epochs, learning_rate,
        beta, gradient_clip, batch_size, val_fraction,
        lr_z, threshold, log_z_grad_clip,
        kl_to_heuristic, heuristic_interpolation, entropy_floor,
        include_heuristic_scores, task_conditioning)
end

struct EditTBDataset
    trajectories::Vector{EditTBTrajectory}
    task_names::Vector{String}
    stats::Dict{String, Any}
end

"""Return a stable one-hot feature vector for the given task name."""
function _edit_tb_task_features(task_name::String, task_names::Vector{String}; enabled::Bool=false)::Vector{Float32}
    (!enabled || isempty(task_names)) && return Float32[]
    feats = zeros(Float32, length(task_names))
    idx = findfirst(==(task_name), task_names)
    idx === nothing || (feats[idx] = 1.0f0)
    return feats
end

"""Number of task-conditioning features implied by this dataset/config pair."""
function _edit_tb_task_feature_dim(task_names::Vector{String}; enabled::Bool=false)::Int
    return enabled ? length(task_names) : 0
end

# ============================================================================
# Feature Engineering — LEAK-FREE (decision-time observables only)
# ============================================================================

"""
Build 8-d frontier features from a BasinDecisionLog.
All fields are observable at basin-choice decision time.
"""
function build_edit_tb_frontier_features(basin_log)::Vector{Float32}
    Float32[
        basin_log.frontier_top1,                                                # 1
        basin_log.frontier_top10_mean,                                          # 2
        basin_log.frontier_size / 256.0f0,                                      # 3
        basin_log.frontier_scaffold_count / max(basin_log.frontier_size, 1),    # 4
        basin_log.budget_remaining / 3000.0f0,                                  # 5
        basin_log.frontier_top1 - basin_log.frontier_top10_mean,                # 6
        basin_log.frontier_scaffold_count / 20.0f0,                             # 7
        basin_log.step_index / 3.0f0                                            # 8
    ]
end

"""
Build basin candidate features from a BasinDecisionCandidate.
Actual fields: scaffold, count, best_reward, mean_reward, mean_novelty, mean_delta, heuristic_score, target_match
"""
function build_edit_tb_basin_candidate_features(
    candidate;
    include_heuristic::Bool=true
)::Vector{Float32}
    feats = Float32[
        candidate.count / 20.0f0,         # 1. basin size
        candidate.best_reward,             # 2. best in basin
        candidate.mean_reward,             # 3. mean in basin
        candidate.mean_novelty,            # 4. novelty
        candidate.mean_delta,              # 5. mean delta (historical)
    ]
    if include_heuristic
        push!(feats, Float32(candidate.heuristic_score))  # 6. heuristic score
    end
    return feats
end

"""
Build parent candidate features from a ParentDecisionCandidate.
Actual fields: smiles, scaffold, reward, novelty_score, tb_delta_abs, source, heuristic_score, visit_count, basin_match, target_match
"""
function build_edit_tb_parent_candidate_features(
    candidate;
    include_heuristic::Bool=true
)::Vector{Float32}
    feats = Float32[
        candidate.reward,                           # 1. parent reward
        candidate.visit_count / 10.0f0,             # 2. visit count
        candidate.basin_match ? 1.0f0 : 0.0f0,     # 3. basin match
        candidate.target_match ? 1.0f0 : 0.0f0,    # 4. target match
    ]
    if include_heuristic
        push!(feats, Float32(candidate.heuristic_score))  # 5. heuristic score
    end
    return feats
end

"""
Build operator candidate features from an OperatorDecisionCandidate.
Actual fields: operator, heuristic_score, total_count, positive_delta_count, exploration_bonus, structural_bias
"""
function build_edit_tb_operator_candidate_features(
    candidate;
    include_heuristic::Bool=true
)::Vector{Float32}
    feats = Float32[
        candidate.total_count / 50.0f0,                                         # 1. historical count
        candidate.positive_delta_count / max(candidate.total_count, 1),         # 2. success ratio
        candidate.exploration_bonus,                                             # 3. exploration bonus
    ]
    if include_heuristic
        push!(feats, Float32(candidate.heuristic_score))  # 4. heuristic score
    end
    return feats
end

# ============================================================================
# Terminal Reward Computation
# ============================================================================

"""
Compute terminal reward for an episode from before/after frontier summaries.
Pre-registered headline: top10_delta. Others are secondary ablations only.
"""
function compute_edit_tb_terminal_reward(ep_summary; mode::Symbol=:top10_delta, epsilon::Float64=0.001)
    fb = ep_summary["frontier_before_summary"]
    fa = ep_summary["frontier_after_summary"]

    utility = if mode == :top10_delta
        fa["top10_mean"] - fb["top10_mean"]
    elseif mode == :top1_delta
        fa["top1"] - fb["top1"]
    elseif mode == :composite
        dt10 = fa["top10_mean"] - fb["top10_mean"]
        dt1 = fa["top1"] - fb["top1"]
        ds = (get(fa, "n_scaffolds", get(fa, "scaffold_count", 0)) -
              get(fb, "n_scaffolds", get(fb, "scaffold_count", 0))) / 10.0
        0.5 * dt10 + 0.3 * dt1 + 0.2 * max(ds, 0.0)
    else
        error("Unknown reward mode: $mode")
    end

    return max(epsilon, Float64(utility))
end

# ============================================================================
# Helper Functions
# ============================================================================

"""Index a vector of log objects by episode_id."""
function _edit_tb_index_logs_by_episode(logs::Vector)::Dict{String, Vector}
    result = Dict{String, Vector}()
    for log in logs
        eid = log.episode_id
        push!(get!(result, eid, []), log)
    end
    return result
end

"""Find the log matching a specific step_index and attempt_index."""
function _edit_tb_find_log_for_step(logs::Vector, step_index::Int, attempt_index::Int)
    for log in logs
        if log.step_index == step_index && log.attempt_index == attempt_index
            return log
        end
    end
    return nothing
end

"""Build a feature matrix from a vector of candidates using a feature function."""
function _edit_tb_build_candidate_matrix(candidates::Vector, feature_fn::Function)::Matrix{Float32}
    isempty(candidates) && return Matrix{Float32}(undef, 0, 0)
    feats = [feature_fn(c) for c in candidates]
    d = length(feats[1])
    mat = Matrix{Float32}(undef, d, length(candidates))
    for (i, f) in enumerate(feats)
        mat[:, i] = f
    end
    return mat
end

"""Extract basin context features from the chosen basin in a BasinDecisionLog."""
function _edit_tb_extract_basin_features(basin_log)::Vector{Float32}
    chosen = basin_log.candidate_basins[basin_log.chosen_index]
    Float32[
        chosen.count / 20.0f0,
        chosen.best_reward,
        chosen.mean_reward,
        chosen.mean_novelty
    ]
end

# ============================================================================
# Dataset Loading
# ============================================================================

"""
Load the edit-TB dataset from truth sprint artifacts.

Joins trajectory entries with decision-log diagnostics and episode summaries.
Uses corrected field names verified against actual artifact structs.
"""
function load_edit_tb_dataset(
    artifact_root::String;
    config::EditTBConfig=EditTBConfig()
)::EditTBDataset
    trajectories = EditTBTrajectory[]
    include_h = config.include_heuristic_scores

    shard_dirs = String[]
    for name in readdir(artifact_root)
        full = joinpath(artifact_root, name)
        isdir(full) && contains(name, "tb_he_full_locked") && push!(shard_dirs, full)
    end

    for shard_dir in shard_dirs
        he_dir = joinpath(shard_dir, "he_artifacts", "tb_he_full_locked")
        isdir(he_dir) || continue

        for task_name in readdir(he_dir)
            task_dir = joinpath(he_dir, task_name)
            isdir(task_dir) || continue

            for run_name in readdir(task_dir)
                run_dir = joinpath(task_dir, run_name)
                isdir(run_dir) || continue

                traj_file = joinpath(run_dir, "he_raw_trajectory.jls")
                diag_file = joinpath(run_dir, "he_raw_diagnostics.jls")
                ep_file = joinpath(run_dir, "he_episode_summary.jls")
                (isfile(traj_file) && isfile(diag_file) && isfile(ep_file)) || continue

                entries = deserialize(traj_file)
                diagnostics = deserialize(diag_file)
                episodes = deserialize(ep_file)

                # Group trajectory entries by episode_id
                ep_groups = Dict{String, Vector}()
                for e in entries
                    eid = get(e.metadata, "episode_id", "unknown")
                    push!(get!(ep_groups, eid, []), e)
                end

                # Index diagnostics by episode_id
                # diagnostics is a Dict{String, Vector}
                basin_logs = get(diagnostics, "basin_logs", [])
                parent_logs = get(diagnostics, "parent_logs", [])
                operator_logs = get(diagnostics, "operator_logs", [])

                basin_by_ep = _edit_tb_index_logs_by_episode(basin_logs)
                parent_by_ep = _edit_tb_index_logs_by_episode(parent_logs)
                operator_by_ep = _edit_tb_index_logs_by_episode(operator_logs)

                run_id = "$(task_name)__$(run_name)"

                for (eid, ep_entries) in ep_groups
                    # Find matching episode summary
                    ep_idx = findfirst(ep -> ep["episode_id"] == eid, episodes)
                    ep_idx === nothing && continue
                    ep_summary = episodes[ep_idx]

                    terminal_reward = compute_edit_tb_terminal_reward(
                        ep_summary; mode=config.reward_mode
                    )

                    ep_basin_logs = get(basin_by_ep, eid, [])
                    ep_parent_logs = get(parent_by_ep, eid, [])
                    ep_operator_logs = get(operator_by_ep, eid, [])

                    sort!(ep_entries, by=e -> e.step_index)
                    steps = EditTBStep[]

                    for entry in ep_entries
                        ai = get(entry.metadata, "attempt_index", 1)

                        basin_log = _edit_tb_find_log_for_step(ep_basin_logs, entry.step_index, ai)
                        parent_log = _edit_tb_find_log_for_step(ep_parent_logs, entry.step_index, ai)
                        operator_log = _edit_tb_find_log_for_step(ep_operator_logs, entry.step_index, ai)

                        # Skip steps without complete decision logs
                        (basin_log === nothing || parent_log === nothing || operator_log === nothing) && continue

                        # Frontier features from basin_log (captures CURRENT frontier state)
                        frontier_feats = build_edit_tb_frontier_features(basin_log)

                        # Basin choice
                        basin_choice = EditTBBasinChoice(
                            frontier_feats,
                            _edit_tb_build_candidate_matrix(
                                basin_log.candidate_basins,
                                c -> build_edit_tb_basin_candidate_features(c; include_heuristic=include_h)
                            ),
                            basin_log.chosen_index,
                            length(basin_log.candidate_basins)
                        )

                        # Parent choice — field is candidate_parents (not candidates)
                        parent_choice = EditTBParentChoice(
                            _edit_tb_extract_basin_features(basin_log),
                            _edit_tb_build_candidate_matrix(
                                parent_log.candidate_parents,
                                c -> build_edit_tb_parent_candidate_features(c; include_heuristic=include_h)
                            ),
                            parent_log.chosen_index,
                            length(parent_log.candidate_parents)
                        )

                        # Operator choice — field is candidate_operators (not candidates)
                        parent_ctx = Float32[
                            entry.parent_reward,
                            entry.parent_reward / max(Float32(basin_log.frontier_top1), 0.01f0),
                            Float32(entry.step_index) / 3.0f0,
                            Float32(ai) / 3.0f0,
                            basin_log.frontier_size / 256.0f0,
                            basin_log.budget_remaining / 3000.0f0
                        ]

                        operator_choice = EditTBOperatorChoice(
                            parent_ctx,
                            _edit_tb_build_candidate_matrix(
                                operator_log.candidate_operators,
                                c -> build_edit_tb_operator_candidate_features(c; include_heuristic=include_h)
                            ),
                            operator_log.chosen_index,
                            length(operator_log.candidate_operators)
                        )

                        push!(steps, EditTBStep(basin_choice, parent_choice, operator_choice))
                    end

                    isempty(steps) && continue

                    phase_sym = Symbol(get(ep_summary, "phase", "unknown"))

                    push!(trajectories, EditTBTrajectory(
                        task_name, run_id, eid, steps, terminal_reward, phase_sym
                    ))
                end
            end
        end
    end

    task_names = sort(unique(t.task_name for t in trajectories))
    stats = Dict{String, Any}(
        "n_trajectories" => length(trajectories),
        "n_steps" => sum(length(t.steps) for t in trajectories; init=0),
        "tasks" => task_names,
        "runs" => sort(unique(t.run_id for t in trajectories)),
        "mean_terminal_reward" => isempty(trajectories) ? 0.0 :
            mean(t.terminal_reward for t in trajectories),
        "mean_steps_per_episode" => isempty(trajectories) ? 0.0 :
            mean(length(t.steps) for t in trajectories),
        "reward_mode" => config.reward_mode,
        "include_heuristic_scores" => config.include_heuristic_scores,
        "task_conditioning" => config.task_conditioning,
        "task_feature_dim" => _edit_tb_task_feature_dim(task_names; enabled=config.task_conditioning),
    )

    return EditTBDataset(trajectories, task_names, stats)
end
