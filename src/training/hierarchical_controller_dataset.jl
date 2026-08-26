# Hierarchical Controller Dataset Utilities
#
# Batch 1A.1 scope:
# - capture deterministic basin candidate sets
# - join basin choice logs with truthful attempt-level proposal + decision outcomes
# - audit dataset coverage and drop reasons
# - produce a richer offline dataset for basin-only controller learning

using Random
using Statistics

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

struct BasinAttemptOutcomeSummary
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    basin_scaffold::String
    has_proposal_log::Bool
    has_decision_log::Bool
    raw_candidate_count::Int
    unique_valid_count::Int
    empty_after_filter::Bool
    chosen_reward_delta::Float64
    reward_q75::Float64
    reward_max::Float64
    proposal_positive::Bool
    decision_count::Int
    max_reward_delta::Float64
    max_frontier_utility_delta::Float64
    enters_topk::Bool
    outcome_class::String
end

struct BasinDecisionRecord
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    frontier_features::Vector{Float32}
    candidate_features::Vector{Vector{Float32}}
    candidate_scaffolds::Vector{String}
    heuristic_scores::Vector{Float32}
    chosen_index::Int
    chosen_reward_delta::Float32
    chosen_frontier_utility_delta::Float32
    chosen_enters_topk::Bool
    target_value::Float32
    success_label::Bool
    has_proposal_log::Bool
    has_decision_log::Bool
    empty_after_filter::Bool
    unique_valid_count::Int
    outcome_class::String
end

mutable struct BasinControllerDataset
    records::Vector{BasinDecisionRecord}
end

BasinControllerDataset() = BasinControllerDataset(BasinDecisionRecord[])
Base.length(dataset::BasinControllerDataset) = length(dataset.records)
Base.isempty(dataset::BasinControllerDataset) = isempty(dataset.records)

function build_basin_decision_candidates(snapshot::FrontierSnapshot,
                                         candidates::Vector{ScoredBasinCandidate})
    return BasinDecisionCandidate[
        BasinDecisionCandidate(
            item.basin.scaffold,
            item.basin.count,
            item.basin.best_reward,
            item.basin.mean_reward,
            item.basin.mean_novelty,
            item.basin.mean_delta,
            item.score,
            !isnothing(snapshot.target_scaffold) && item.basin.scaffold == snapshot.target_scaffold,
        ) for item in candidates
    ]
end

function _frontier_top1(snapshot::FrontierSnapshot)
    isempty(snapshot.entries) && return 0.0
    return maximum(e.reward for e in snapshot.entries)
end

function _frontier_top10_mean(snapshot::FrontierSnapshot)
    isempty(snapshot.entries) && return 0.0
    rewards = sort([e.reward for e in snapshot.entries], rev=true)
    return mean(rewards[1:min(10, length(rewards))])
end

function _frontier_scaffold_count(snapshot::FrontierSnapshot)
    return length(Set(e.scaffold for e in snapshot.entries if !isempty(e.scaffold)))
end

function frontier_feature_vector(frontier_size::Int,
                                 frontier_top1::Float64,
                                 frontier_top10_mean::Float64,
                                 frontier_scaffold_count::Int,
                                 budget_remaining::Int,
                                 step_index::Int,
                                 candidate_count::Int)
    return Float32[
        frontier_size,
        frontier_top1,
        frontier_top10_mean,
        frontier_scaffold_count,
        budget_remaining,
        step_index,
        candidate_count,
    ]
end

function frontier_feature_vector(snapshot::FrontierSnapshot, candidate_count::Int; step_index::Int=0)
    return frontier_feature_vector(
        length(snapshot.entries),
        _frontier_top1(snapshot),
        _frontier_top10_mean(snapshot),
        _frontier_scaffold_count(snapshot),
        snapshot.budget_remaining,
        step_index,
        candidate_count,
    )
end


function _heuristic_context_features(scores::AbstractVector{<:Real}, idx::Int)
    isempty(scores) && return Float32[1.0, 0.0, 0.0, 0.0, 0.0]
    vals = Float64.(scores)
    order = sortperm(vals, rev=true)
    rank = something(findfirst(==(idx), order), 1)
    top_score = vals[order[1]]
    next_score = rank < length(order) ? vals[order[rank + 1]] : vals[order[end]]
    median_score = sort(vals)[cld(length(vals), 2)]
    current = vals[idx]
    rank_fraction = length(vals) == 1 ? 0.0 : (rank - 1) / (length(vals) - 1)
    return Float32[
        rank,
        rank_fraction,
        current - top_score,
        current - next_score,
        top_score - median_score,
    ]
end

function basin_candidate_feature_vector(log::BasinDecisionLog,
                                        candidate::BasinDecisionCandidate;
                                        all_candidates::Vector{BasinDecisionCandidate}=log.candidate_basins,
                                        candidate_index::Int=1,
                                        feature_mode::Symbol=:basic)
    frontier = frontier_feature_vector(
        log.frontier_size,
        log.frontier_top1,
        log.frontier_top10_mean,
        log.frontier_scaffold_count,
        log.budget_remaining,
        log.step_index,
        length(log.candidate_basins),
    )
    base = Float32[
        candidate.count,
        candidate.best_reward,
        candidate.mean_reward,
        candidate.mean_novelty,
        candidate.mean_delta,
        candidate.heuristic_score,
        candidate.target_match ? 1.0f0 : 0.0f0,
    ]
    if feature_mode == :basic
        return vcat(frontier, base)
    end

    heuristic_scores = Float64[c.heuristic_score for c in all_candidates]
    extra = Float32[
        candidate.count / max(log.frontier_size, 1),
        candidate.best_reward - candidate.mean_reward,
        log.frontier_top1 - candidate.best_reward,
        log.frontier_top10_mean - candidate.mean_reward,
    ]
    return vcat(frontier, base, extra, _heuristic_context_features(heuristic_scores, candidate_index))
end

function basin_candidate_feature_vector(snapshot::FrontierSnapshot,
                                        candidate::ScoredBasinCandidate;
                                        step_index::Int=0,
                                        candidate_count::Int=0,
                                        all_candidates::Vector{ScoredBasinCandidate}=ScoredBasinCandidate[candidate],
                                        candidate_index::Int=1,
                                        feature_mode::Symbol=:basic)
    count_local = candidate_count > 0 ? candidate_count : max(length(all_candidates), 1)
    frontier = frontier_feature_vector(snapshot, count_local; step_index=step_index)
    target_match = !isnothing(snapshot.target_scaffold) && candidate.basin.scaffold == snapshot.target_scaffold
    base = Float32[
        candidate.basin.count,
        candidate.basin.best_reward,
        candidate.basin.mean_reward,
        candidate.basin.mean_novelty,
        candidate.basin.mean_delta,
        candidate.score,
        target_match ? 1.0f0 : 0.0f0,
    ]
    if feature_mode == :basic
        return vcat(frontier, base)
    end

    heuristic_scores = Float64[c.score for c in all_candidates]
    frontier_top1 = _frontier_top1(snapshot)
    frontier_top10_mean = _frontier_top10_mean(snapshot)
    extra = Float32[
        candidate.basin.count / max(length(snapshot.entries), 1),
        candidate.basin.best_reward - candidate.basin.mean_reward,
        frontier_top1 - candidate.basin.best_reward,
        frontier_top10_mean - candidate.basin.mean_reward,
    ]
    return vcat(frontier, base, extra, _heuristic_context_features(heuristic_scores, candidate_index))
end

_attempt_key(snapshot_id::UInt64,
             episode_id::String,
             task_name::String,
             step_index::Int,
             attempt_index::Int,
             basin_scaffold::String) =
    (snapshot_id, episode_id, task_name, step_index, attempt_index, basin_scaffold)

function _attempt_key(log::BasinDecisionLog)
    return _attempt_key(log.snapshot_id, log.episode_id, log.task_name, log.step_index, log.attempt_index, log.chosen_basin_scaffold)
end

function _classify_basin_outcome(has_proposal_log::Bool,
                                 has_decision_log::Bool,
                                 empty_after_filter::Bool,
                                 unique_valid_count::Int,
                                 reward_max::Float64,
                                 chosen_reward_delta::Float64,
                                 max_reward_delta::Float64,
                                 max_frontier_utility_delta::Float64,
                                 enters_topk::Bool)
    if !has_proposal_log
        return "missing_proposal"
    elseif empty_after_filter || unique_valid_count == 0
        return "degenerate"
    elseif enters_topk || max_frontier_utility_delta > 0 || max_reward_delta > 0.05
        return "productive"
    elseif has_decision_log || reward_max > 0 || chosen_reward_delta > 0
        return "weak_productive"
    else
        return "null"
    end
end

function summarize_basin_attempt_outcomes(proposal_logs,
                                          decision_logs)
    proposal_summary = Dict{Tuple{UInt64,String,String,Int,Int,String}, Dict{String,Any}}()
    for log in proposal_logs
        key = _attempt_key(
            getfield(log, :snapshot_id),
            getfield(log, :episode_id),
            getfield(log, :task_name),
            getfield(log, :step_index),
            getfield(log, :attempt_index),
            getfield(log, :basin_scaffold),
        )
        current = get!(proposal_summary, key, Dict{String,Any}(
            "has_proposal_log" => false,
            "raw_candidate_count" => 0,
            "unique_valid_count" => 0,
            "empty_after_filter" => false,
            "chosen_reward_delta" => 0.0,
            "reward_q75" => 0.0,
            "reward_max" => 0.0,
            "proposal_positive" => false,
        ))
        current["has_proposal_log"] = true
        current["raw_candidate_count"] = max(Int(current["raw_candidate_count"]), Int(getfield(log, :raw_candidate_count)))
        current["unique_valid_count"] = max(Int(current["unique_valid_count"]), Int(getfield(log, :unique_valid_count)))
        current["empty_after_filter"] = Bool(current["empty_after_filter"]) || Bool(getfield(log, :empty_after_filter))
        current["chosen_reward_delta"] = max(Float64(current["chosen_reward_delta"]), Float64(getfield(log, :chosen_reward_delta)))
        current["reward_q75"] = max(Float64(current["reward_q75"]), Float64(getfield(log, :reward_q75)))
        current["reward_max"] = max(Float64(current["reward_max"]), Float64(getfield(log, :reward_max)))
        current["proposal_positive"] = Bool(current["proposal_positive"]) || (Float64(getfield(log, :chosen_reward_delta)) > 0)
    end

    decision_summary = Dict{Tuple{UInt64,String,String,Int,Int,String}, Dict{String,Any}}()
    for log in decision_logs
        key = _attempt_key(
            getfield(log, :snapshot_id),
            getfield(log, :episode_id),
            getfield(log, :task_name),
            getfield(log, :step_index),
            getfield(log, :attempt_index),
            getfield(log, :basin_scaffold),
        )
        current = get!(decision_summary, key, Dict{String,Any}(
            "has_decision_log" => false,
            "decision_count" => 0,
            "max_reward_delta" => -Inf,
            "max_frontier_utility_delta" => -Inf,
            "enters_topk" => false,
        ))
        current["has_decision_log"] = true
        current["decision_count"] = Int(current["decision_count"]) + 1
        current["max_reward_delta"] = max(Float64(current["max_reward_delta"]), Float64(getfield(log, :reward_delta)))
        current["max_frontier_utility_delta"] = max(Float64(current["max_frontier_utility_delta"]), Float64(getfield(log, :frontier_utility_delta)))
        current["enters_topk"] = Bool(current["enters_topk"]) || Bool(getfield(log, :enters_topk))
    end

    all_keys = union(Set(keys(proposal_summary)), Set(keys(decision_summary)))
    summary = Dict{Tuple{UInt64,String,String,Int,Int,String}, BasinAttemptOutcomeSummary}()
    for key in all_keys
        snapshot_id, episode_id, task_name, step_index, attempt_index, basin_scaffold = key
        prop = get(proposal_summary, key, Dict{String,Any}())
        dec = get(decision_summary, key, Dict{String,Any}())
        has_proposal_log = get(prop, "has_proposal_log", false)
        has_decision_log = get(dec, "has_decision_log", false)
        raw_candidate_count = get(prop, "raw_candidate_count", 0)
        unique_valid_count = get(prop, "unique_valid_count", 0)
        empty_after_filter = get(prop, "empty_after_filter", false)
        chosen_reward_delta = get(prop, "chosen_reward_delta", 0.0)
        reward_q75 = get(prop, "reward_q75", 0.0)
        reward_max = get(prop, "reward_max", 0.0)
        proposal_positive = get(prop, "proposal_positive", false)
        decision_count = get(dec, "decision_count", 0)
        max_reward_delta = get(dec, "max_reward_delta", has_decision_log ? -Inf : 0.0)
        max_frontier_utility_delta = get(dec, "max_frontier_utility_delta", has_decision_log ? -Inf : 0.0)
        enters_topk = get(dec, "enters_topk", false)
        outcome_class = _classify_basin_outcome(
            has_proposal_log,
            has_decision_log,
            empty_after_filter,
            unique_valid_count,
            reward_max,
            chosen_reward_delta,
            isfinite(max_reward_delta) ? max_reward_delta : 0.0,
            isfinite(max_frontier_utility_delta) ? max_frontier_utility_delta : 0.0,
            enters_topk,
        )
        summary[key] = BasinAttemptOutcomeSummary(
            snapshot_id,
            episode_id,
            task_name,
            step_index,
            attempt_index,
            basin_scaffold,
            has_proposal_log,
            has_decision_log,
            raw_candidate_count,
            unique_valid_count,
            empty_after_filter,
            chosen_reward_delta,
            reward_q75,
            reward_max,
            proposal_positive,
            decision_count,
            isfinite(max_reward_delta) ? max_reward_delta : 0.0,
            isfinite(max_frontier_utility_delta) ? max_frontier_utility_delta : 0.0,
            enters_topk,
            outcome_class,
        )
    end
    return summary
end

function compute_basin_target(summary::BasinAttemptOutcomeSummary;
                              mode::Symbol=:blended)
    if mode == :binary_productive
        return summary.outcome_class == "productive" ? 1.0 : 0.0
    elseif mode == :frontier_utility
        return summary.max_frontier_utility_delta
    elseif mode == :reward_delta
        return summary.max_reward_delta
    elseif mode == :ordinal_productivity
        return if summary.outcome_class == "productive"
            1.0
        elseif summary.outcome_class == "weak_productive"
            0.35
        elseif summary.outcome_class == "null"
            0.0
        elseif summary.outcome_class == "degenerate"
            -1.0
        else
            -0.5
        end
    elseif mode == :risk_adjusted_utility
        target = 0.0
        target += summary.max_frontier_utility_delta
        target += 0.25 * summary.max_reward_delta
        target += summary.enters_topk ? 0.15 : 0.0
        target += summary.outcome_class == "weak_productive" ? 0.05 : 0.0
        target -= summary.outcome_class == "null" ? 0.15 : 0.0
        target -= summary.outcome_class == "degenerate" ? 1.0 : 0.0
        target -= summary.outcome_class == "missing_proposal" ? 0.6 : 0.0
        target -= summary.empty_after_filter ? 0.35 : 0.0
        return target
    else
        target = 0.0
        target += summary.max_frontier_utility_delta
        target += 0.35 * summary.max_reward_delta
        target += summary.enters_topk ? 0.25 : 0.0
        target += summary.proposal_positive ? 0.1 : 0.0
        target -= summary.empty_after_filter ? 0.5 : 0.0
        target -= (!summary.has_decision_log && summary.has_proposal_log && !summary.proposal_positive) ? 0.2 : 0.0
        target -= (!summary.has_proposal_log) ? 0.3 : 0.0
        return target
    end
end

function audit_basin_dataset_coverage(basin_logs::Vector{BasinDecisionLog},
                                      proposal_logs,
                                      decision_logs)
    outcome_summary = summarize_basin_attempt_outcomes(proposal_logs, decision_logs)
    class_counts = Dict{String,Int}()
    dropped_missing_attempt = 0
    with_proposal = 0
    with_decision = 0
    with_empty = 0

    for log in basin_logs
        key = _attempt_key(log)
        if !haskey(outcome_summary, key)
            dropped_missing_attempt += 1
            continue
        end
        outcome = outcome_summary[key]
        with_proposal += outcome.has_proposal_log ? 1 : 0
        with_decision += outcome.has_decision_log ? 1 : 0
        with_empty += outcome.empty_after_filter ? 1 : 0
        class_counts[outcome.outcome_class] = get(class_counts, outcome.outcome_class, 0) + 1
    end

    total = length(basin_logs)
    matched = total - dropped_missing_attempt
    return Dict(
        "basin_logs" => total,
        "matched_attempt_outcomes" => matched,
        "missing_attempt_outcomes" => dropped_missing_attempt,
        "proposal_coverage_fraction" => total == 0 ? 0.0 : with_proposal / total,
        "decision_coverage_fraction" => total == 0 ? 0.0 : with_decision / total,
        "empty_after_filter_fraction" => total == 0 ? 0.0 : with_empty / total,
        "class_counts" => class_counts,
    )
end

function extract_basin_controller_dataset(basin_logs::Vector{BasinDecisionLog},
                                          proposal_logs,
                                          decision_logs;
                                          target_mode::Symbol=:blended,
                                          feature_mode::Symbol=:basic,
                                          min_candidates::Int=2)
    outcome_summary = summarize_basin_attempt_outcomes(proposal_logs, decision_logs)
    dataset = BasinControllerDataset()

    for log in basin_logs
        candidate_count = length(log.candidate_basins)
        candidate_count < min_candidates && continue
        !(1 <= log.chosen_index <= candidate_count) && continue

        key = _attempt_key(log)
        outcome = get(outcome_summary, key, BasinAttemptOutcomeSummary(
            log.snapshot_id,
            log.episode_id,
            log.task_name,
            log.step_index,
            log.attempt_index,
            log.chosen_basin_scaffold,
            false,
            false,
            0,
            0,
            false,
            0.0,
            0.0,
            0.0,
            false,
            0,
            0.0,
            0.0,
            false,
            "missing_proposal",
        ))

        candidate_features = [basin_candidate_feature_vector(log, candidate;
            all_candidates=log.candidate_basins,
            candidate_index=idx,
            feature_mode=feature_mode) for (idx, candidate) in enumerate(log.candidate_basins)]
        candidate_scaffolds = [candidate.scaffold for candidate in log.candidate_basins]
        heuristic_scores = Float32[candidate.heuristic_score for candidate in log.candidate_basins]
        frontier_features = frontier_feature_vector(
            log.frontier_size,
            log.frontier_top1,
            log.frontier_top10_mean,
            log.frontier_scaffold_count,
            log.budget_remaining,
            log.step_index,
            candidate_count,
        )

        target_value = Float32(compute_basin_target(outcome; mode=target_mode))
        success = target_mode == :binary_productive ? target_value > 0.5f0 : target_value > 0.0f0

        push!(dataset.records, BasinDecisionRecord(
            log.snapshot_id,
            log.episode_id,
            log.task_name,
            log.step_index,
            log.attempt_index,
            frontier_features,
            candidate_features,
            candidate_scaffolds,
            heuristic_scores,
            log.chosen_index,
            Float32(outcome.max_reward_delta),
            Float32(outcome.max_frontier_utility_delta),
            outcome.enters_topk,
            target_value,
            success,
            outcome.has_proposal_log,
            outcome.has_decision_log,
            outcome.empty_after_filter,
            outcome.unique_valid_count,
            outcome.outcome_class,
        ))
    end

    return dataset
end

function split_basin_controller_dataset(dataset::BasinControllerDataset;
                                        train_fraction::Float64=0.8,
                                        rng::AbstractRNG=Random.MersenneTwister(0))
    isempty(dataset) && return BasinControllerDataset(), BasinControllerDataset()
    idxs = collect(eachindex(dataset.records))
    Random.shuffle!(rng, idxs)
    n_train = clamp(round(Int, train_fraction * length(idxs)), 1, length(idxs))
    train_idxs = idxs[1:n_train]
    val_idxs = idxs[(n_train + 1):end]
    return BasinControllerDataset(dataset.records[train_idxs]), BasinControllerDataset(dataset.records[val_idxs])
end

function basin_controller_dataset_stats(dataset::BasinControllerDataset)
    if isempty(dataset)
        return Dict(
            "size" => 0,
            "positive_fraction" => 0.0,
            "mean_candidates" => 0.0,
            "feature_dim" => 0,
            "mean_reward_delta" => 0.0,
            "mean_frontier_utility_delta" => 0.0,
            "proposal_coverage_fraction" => 0.0,
            "decision_coverage_fraction" => 0.0,
            "empty_after_filter_fraction" => 0.0,
            "class_counts" => Dict{String,Int}(),
        )
    end

    class_counts = Dict{String,Int}()
    for record in dataset.records
        class_counts[record.outcome_class] = get(class_counts, record.outcome_class, 0) + 1
    end

    return Dict(
        "size" => length(dataset.records),
        "positive_fraction" => mean(Float64[r.success_label for r in dataset.records]),
        "mean_candidates" => mean(Float64[length(r.candidate_features) for r in dataset.records]),
        "feature_dim" => length(dataset.records[1].candidate_features[1]),
        "mean_reward_delta" => mean(Float64[r.chosen_reward_delta for r in dataset.records]),
        "mean_frontier_utility_delta" => mean(Float64[r.chosen_frontier_utility_delta for r in dataset.records]),
        "proposal_coverage_fraction" => mean(Float64[r.has_proposal_log for r in dataset.records]),
        "decision_coverage_fraction" => mean(Float64[r.has_decision_log for r in dataset.records]),
        "empty_after_filter_fraction" => mean(Float64[r.empty_after_filter for r in dataset.records]),
        "class_counts" => class_counts,
    )
end
