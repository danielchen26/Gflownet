# Parent Controller Dataset Utilities
#
# Batch 1B′ scope:
# - capture truthful parent candidate sets from the frozen frontier snapshot
# - align parent choices with attempt-level proposal + decision outcomes
# - expose a more causal controller-learning object than basin-only scoring

using Random
using Statistics

struct ParentAttemptOutcomeSummary
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    parent_smiles::String
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

struct ParentDecisionRecord
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    context_features::Vector{Float32}
    candidate_features::Vector{Vector{Float32}}
    candidate_smiles::Vector{String}
    heuristic_scores::Vector{Float32}
    chosen_index::Int
    chosen_heuristic_score::Float32
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
    heuristic_top_index::Int
    learned_top_index::Int
    heuristic_margin::Float32
    learned_margin::Float32
    learned_advantage_vs_heuristic::Float32
    heuristic_entropy::Float32
    learned_entropy::Float32
    override_applied::Bool
    abstained_to_heuristic::Bool
    selection_reason::String
end

mutable struct ParentControllerDataset
    records::Vector{ParentDecisionRecord}
end

ParentControllerDataset() = ParentControllerDataset(ParentDecisionRecord[])
Base.length(dataset::ParentControllerDataset) = length(dataset.records)
Base.isempty(dataset::ParentControllerDataset) = isempty(dataset.records)

_parent_attempt_key(snapshot_id::UInt64,
                    episode_id::String,
                    task_name::String,
                    step_index::Int,
                    attempt_index::Int,
                    parent_smiles::String) =
    (snapshot_id, episode_id, task_name, step_index, attempt_index, parent_smiles)

_parent_attempt_key(log::ParentDecisionLog) = _parent_attempt_key(
    log.snapshot_id,
    log.episode_id,
    log.task_name,
    log.step_index,
    log.attempt_index,
    log.chosen_parent_smiles,
)

function _classify_parent_outcome(has_proposal_log::Bool,
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

function summarize_parent_attempt_outcomes(proposal_logs,
                                           decision_logs)
    proposal_summary = Dict{Tuple{UInt64,String,String,Int,Int,String}, Dict{String,Any}}()
    for log in proposal_logs
        key = _parent_attempt_key(
            getfield(log, :snapshot_id),
            getfield(log, :episode_id),
            getfield(log, :task_name),
            getfield(log, :step_index),
            getfield(log, :attempt_index),
            getfield(log, :parent_smiles),
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
        key = _parent_attempt_key(
            getfield(log, :snapshot_id),
            getfield(log, :episode_id),
            getfield(log, :task_name),
            getfield(log, :step_index),
            getfield(log, :attempt_index),
            getfield(log, :parent_smiles),
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
    summary = Dict{Tuple{UInt64,String,String,Int,Int,String}, ParentAttemptOutcomeSummary}()
    for key in all_keys
        snapshot_id, episode_id, task_name, step_index, attempt_index, parent_smiles = key
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
        outcome_class = _classify_parent_outcome(
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
        summary[key] = ParentAttemptOutcomeSummary(
            snapshot_id,
            episode_id,
            task_name,
            step_index,
            attempt_index,
            parent_smiles,
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

function _base_parent_target(summary::ParentAttemptOutcomeSummary)
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

function _normalized_heuristic_stats(log::ParentDecisionLog)
    scores = Float64[c.heuristic_score for c in log.candidate_parents]
    isempty(scores) && return (chosen=0.0, mean=0.0, top=0.0)
    lo = minimum(scores)
    hi = maximum(scores)
    span = max(hi - lo, 1e-8)
    normalized = [(s - lo) / span for s in scores]
    chosen = (1 <= log.chosen_index <= length(normalized)) ? normalized[log.chosen_index] : normalized[1]
    return (chosen=chosen, mean=mean(normalized), top=maximum(normalized))
end

function compute_parent_target(summary::ParentAttemptOutcomeSummary,
                               log::ParentDecisionLog;
                               mode::Symbol=:blended)
    if mode == :ordinal_productivity
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
    end

    base_target = _base_parent_target(summary)
    heur = _normalized_heuristic_stats(log)

    if mode == :heuristic_adjusted_blended
        return base_target - 0.30 * heur.chosen
    elseif mode == :relative_blended
        return base_target - 0.35 * (heur.chosen - heur.mean)
    elseif mode == :risk_adjusted_advantage
        risk_penalty = summary.empty_after_filter ? 0.35 : 0.0
        risk_penalty += summary.unique_valid_count <= 1 ? 0.15 : 0.0
        return base_target - 0.25 * max(heur.chosen - heur.mean, 0.0) - risk_penalty
    else
        return base_target
    end
end

function _score_entropy(scores::Vector{Float64})
    isempty(scores) && return 0.0
    shifted = scores .- maximum(scores)
    weights = exp.(shifted)
    total = sum(weights)
    total <= 0 && return 0.0
    probs = weights ./ total
    entropy = 0.0
    for p in probs
        p <= 0 && continue
        entropy -= p * log(p)
    end
    return entropy
end

function _score_margin(scores::Vector{Float64})
    isempty(scores) && return 0.0
    if length(scores) == 1
        return scores[1]
    end
    order = sortperm(scores, rev=true)
    return scores[order[1]] - scores[order[2]]
end

function parent_context_feature_vector(log::ParentDecisionLog)
    candidate_count = length(log.candidate_parents)
    max_reward = isempty(log.candidate_parents) ? 0.0 : maximum(c.reward for c in log.candidate_parents)
    mean_reward = isempty(log.candidate_parents) ? 0.0 : mean(c.reward for c in log.candidate_parents)
    heuristic_scores = Float64[c.heuristic_score for c in log.candidate_parents]
    return Float32[
        log.basin_score,
        log.step_index,
        candidate_count,
        max_reward,
        mean_reward,
        _score_margin(heuristic_scores),
        _score_entropy(heuristic_scores),
    ]
end

function _parent_rank_features(heuristic_scores::Vector{Float64}, idx::Int)
    isempty(heuristic_scores) && return Float32[1.0, 0.0, 0.0, 0.0]
    order = sortperm(heuristic_scores, rev=true)
    rank = something(findfirst(==(idx), order), 1)
    top_score = heuristic_scores[order[1]]
    next_score = rank < length(order) ? heuristic_scores[order[rank + 1]] : heuristic_scores[order[end]]
    current = heuristic_scores[idx]
    rank_fraction = length(order) == 1 ? 0.0 : (rank - 1) / (length(order) - 1)
    return Float32[rank, rank_fraction, current - top_score, current - next_score]
end

function parent_candidate_feature_vector(log::ParentDecisionLog,
                                         candidate::ParentDecisionCandidate;
                                         all_candidates::Vector{ParentDecisionCandidate}=log.candidate_parents,
                                         candidate_index::Int=1,
                                         feature_mode::Symbol=:basic)
    context = parent_context_feature_vector(log)
    base = Float32[
        candidate.reward,
        candidate.novelty_score,
        candidate.tb_delta_abs,
        candidate.heuristic_score,
        candidate.visit_count,
        candidate.basin_match ? 1.0f0 : 0.0f0,
        candidate.target_match ? 1.0f0 : 0.0f0,
        candidate.source == "seed" ? 1.0f0 : 0.0f0,
        candidate.source == "warmup" ? 1.0f0 : 0.0f0,
        candidate.source == "edit" ? 1.0f0 : 0.0f0,
    ]
    if feature_mode == :basic
        return vcat(context, base)
    end
    heuristic_scores = Float64[c.heuristic_score for c in all_candidates]
    top_reward = isempty(all_candidates) ? 0.0 : maximum(c.reward for c in all_candidates)
    mean_reward = isempty(all_candidates) ? 0.0 : mean(c.reward for c in all_candidates)
    extra = Float32[
        candidate.reward - mean_reward,
        top_reward - candidate.reward,
        candidate.novelty_score,
        candidate.tb_delta_abs,
    ]
    return vcat(context, base, extra, _parent_rank_features(heuristic_scores, candidate_index))
end

function parent_candidate_feature_vector(snapshot::FrontierSnapshot,
                                         candidate::ScoredParentCandidate;
                                         step_index::Int=0,
                                         all_candidates::Vector{ScoredParentCandidate}=ScoredParentCandidate[candidate],
                                         candidate_index::Int=1,
                                         feature_mode::Symbol=:basic)
    heuristic_scores = Float64[c.score for c in all_candidates]
    context = Float32[
        step_index,
        length(all_candidates),
        isempty(snapshot.entries) ? 0.0 : maximum(e.reward for e in snapshot.entries),
        isempty(snapshot.entries) ? 0.0 : mean(e.reward for e in snapshot.entries),
        snapshot.budget_remaining,
        _score_margin(heuristic_scores),
        _score_entropy(heuristic_scores),
    ]
    base = Float32[
        candidate.entry.reward,
        candidate.entry.novelty_score,
        candidate.entry.tb_delta_abs,
        candidate.score,
        candidate.visit_count,
        candidate.basin_match ? 1.0f0 : 0.0f0,
        candidate.target_match ? 1.0f0 : 0.0f0,
        candidate.entry.source == :seed ? 1.0f0 : 0.0f0,
        candidate.entry.source == :warmup ? 1.0f0 : 0.0f0,
        candidate.entry.source == :edit ? 1.0f0 : 0.0f0,
    ]
    if feature_mode == :basic
        return vcat(context, base)
    end
    heuristic_scores = Float64[c.score for c in all_candidates]
    top_reward = isempty(all_candidates) ? 0.0 : maximum(c.entry.reward for c in all_candidates)
    mean_reward = isempty(all_candidates) ? 0.0 : mean(c.entry.reward for c in all_candidates)
    extra = Float32[
        candidate.entry.reward - mean_reward,
        top_reward - candidate.entry.reward,
        candidate.entry.novelty_score,
        candidate.entry.tb_delta_abs,
    ]
    return vcat(context, base, extra, _parent_rank_features(heuristic_scores, candidate_index))
end

function audit_parent_dataset_coverage(parent_logs::Vector{ParentDecisionLog},
                                       proposal_logs,
                                       decision_logs)
    outcome_summary = summarize_parent_attempt_outcomes(proposal_logs, decision_logs)
    class_counts = Dict{String,Int}()
    dropped_missing_attempt = 0
    with_proposal = 0
    with_decision = 0
    with_empty = 0

    for log in parent_logs
        key = _parent_attempt_key(log)
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

    total = length(parent_logs)
    matched = total - dropped_missing_attempt
    return Dict(
        "parent_logs" => total,
        "matched_attempt_outcomes" => matched,
        "missing_attempt_outcomes" => dropped_missing_attempt,
        "proposal_coverage_fraction" => total == 0 ? 0.0 : with_proposal / total,
        "decision_coverage_fraction" => total == 0 ? 0.0 : with_decision / total,
        "empty_after_filter_fraction" => total == 0 ? 0.0 : with_empty / total,
        "class_counts" => class_counts,
    )
end

function extract_parent_controller_dataset(parent_logs::Vector{ParentDecisionLog},
                                           proposal_logs,
                                           decision_logs;
                                           target_mode::Symbol=:blended,
                                           feature_mode::Symbol=:basic,
                                           min_candidates::Int=2)
    outcome_summary = summarize_parent_attempt_outcomes(proposal_logs, decision_logs)
    dataset = ParentControllerDataset()

    for log in parent_logs
        candidate_count = length(log.candidate_parents)
        candidate_count < min_candidates && continue
        !(1 <= log.chosen_index <= candidate_count) && continue

        key = _parent_attempt_key(log)
        outcome = get(outcome_summary, key, ParentAttemptOutcomeSummary(
            log.snapshot_id,
            log.episode_id,
            log.task_name,
            log.step_index,
            log.attempt_index,
            log.chosen_parent_smiles,
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

        candidate_features = [parent_candidate_feature_vector(log, candidate;
            all_candidates=log.candidate_parents,
            candidate_index=idx,
            feature_mode=feature_mode) for (idx, candidate) in enumerate(log.candidate_parents)]
        candidate_smiles = [candidate.smiles for candidate in log.candidate_parents]
        heuristic_scores = Float32[candidate.heuristic_score for candidate in log.candidate_parents]
        target_value = Float32(compute_parent_target(outcome, log; mode=target_mode))
        success = target_mode == :ordinal_productivity ? target_value > 0.0f0 : target_value > 0.0f0
        chosen_heuristic_score = (1 <= log.chosen_index <= length(heuristic_scores)) ? heuristic_scores[log.chosen_index] : heuristic_scores[1]

        push!(dataset.records, ParentDecisionRecord(
            log.snapshot_id,
            log.episode_id,
            log.task_name,
            log.step_index,
            log.attempt_index,
            parent_context_feature_vector(log),
            candidate_features,
            candidate_smiles,
            heuristic_scores,
            log.chosen_index,
            chosen_heuristic_score,
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
            log.heuristic_top_index,
            log.learned_top_index,
            Float32(log.heuristic_margin),
            Float32(log.learned_margin),
            Float32(log.learned_advantage_vs_heuristic),
            Float32(log.heuristic_entropy),
            Float32(log.learned_entropy),
            log.override_applied,
            log.abstained_to_heuristic,
            log.selection_reason,
        ))
    end

    return dataset
end

function split_parent_controller_dataset(dataset::ParentControllerDataset;
                                         train_fraction::Float64=0.8,
                                         rng::AbstractRNG=Random.MersenneTwister(0))
    isempty(dataset) && return ParentControllerDataset(), ParentControllerDataset()
    idxs = collect(eachindex(dataset.records))
    Random.shuffle!(rng, idxs)
    n_train = clamp(round(Int, train_fraction * length(idxs)), 1, length(idxs))
    train_idxs = idxs[1:n_train]
    val_idxs = idxs[(n_train + 1):end]
    return ParentControllerDataset(dataset.records[train_idxs]), ParentControllerDataset(dataset.records[val_idxs])
end

function parent_controller_dataset_stats(dataset::ParentControllerDataset)
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
            "mean_heuristic_margin" => 0.0,
            "mean_heuristic_entropy" => 0.0,
            "heuristic_ambiguous_fraction" => 0.0,
            "override_fraction" => 0.0,
            "abstained_fraction" => 0.0,
            "selection_reason_counts" => Dict{String,Int}(),
            "class_counts" => Dict{String,Int}(),
        )
    end

    class_counts = Dict{String,Int}()
    selection_reason_counts = Dict{String,Int}()
    for record in dataset.records
        class_counts[record.outcome_class] = get(class_counts, record.outcome_class, 0) + 1
        selection_reason_counts[record.selection_reason] = get(selection_reason_counts, record.selection_reason, 0) + 1
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
        "mean_heuristic_margin" => mean(Float64[r.heuristic_margin for r in dataset.records]),
        "mean_heuristic_entropy" => mean(Float64[r.heuristic_entropy for r in dataset.records]),
        "heuristic_ambiguous_fraction" => mean(Float64[r.heuristic_margin < 0.1f0 for r in dataset.records]),
        "override_fraction" => mean(Float64[r.override_applied for r in dataset.records]),
        "abstained_fraction" => mean(Float64[r.abstained_to_heuristic for r in dataset.records]),
        "selection_reason_counts" => selection_reason_counts,
        "class_counts" => class_counts,
    )
end
