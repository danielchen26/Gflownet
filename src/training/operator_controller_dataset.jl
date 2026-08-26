# Operator Controller Dataset Utilities
#
# Batch 1H scope:
# - preserve truthful operator candidate sets from live hierarchical-edit steps
# - split eligibility supervision from conditional operator ranking
# - keep probe-derived state labels as supervision only, not online features

using Random
using Statistics

struct OperatorAttemptOutcomeSummary
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    operator::Symbol
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

struct OperatorDecisionRecord
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    attempt_index::Int
    eligibility_features::Vector{Float32}
    context_features::Vector{Float32}
    candidate_features::Vector{Vector{Float32}}
    candidate_operators::Vector{Symbol}
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
    state_label::String
    controller_eligible::Bool
    predicted_eligible::Bool
    eligibility_score::Float32
    heuristic_top_index::Int
    learned_top_index::Int
    heuristic_margin::Float32
    learned_margin::Float32
    learned_advantage_vs_heuristic::Float32
    heuristic_entropy::Float32
    learned_entropy::Float32
    acted_on::Bool
    preserved_to_heuristic::Bool
    override_applied::Bool
    abstained_to_heuristic::Bool
    selection_reason::String
end

mutable struct OperatorControllerDataset
    records::Vector{OperatorDecisionRecord}
end

OperatorControllerDataset() = OperatorControllerDataset(OperatorDecisionRecord[])
Base.length(dataset::OperatorControllerDataset) = length(dataset.records)
Base.isempty(dataset::OperatorControllerDataset) = isempty(dataset.records)

_operator_attempt_key(snapshot_id::UInt64,
                      episode_id::String,
                      task_name::String,
                      step_index::Int,
                      attempt_index::Int,
                      operator::Symbol) =
    (snapshot_id, episode_id, task_name, step_index, attempt_index, operator)

_operator_attempt_key(log::OperatorDecisionLog) = _operator_attempt_key(
    log.snapshot_id,
    log.episode_id,
    log.task_name,
    log.step_index,
    log.attempt_index,
    log.chosen_operator,
)

function _classify_operator_outcome(has_proposal_log::Bool,
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

function summarize_operator_attempt_outcomes(proposal_logs,
                                             decision_logs)
    proposal_summary = Dict{Tuple{UInt64,String,String,Int,Int,Symbol}, Dict{String,Any}}()
    for log in proposal_logs
        key = _operator_attempt_key(
            getfield(log, :snapshot_id),
            getfield(log, :episode_id),
            getfield(log, :task_name),
            getfield(log, :step_index),
            getfield(log, :attempt_index),
            getfield(log, :operator),
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

    decision_summary = Dict{Tuple{UInt64,String,String,Int,Int,Symbol}, Dict{String,Any}}()
    for log in decision_logs
        key = _operator_attempt_key(
            getfield(log, :snapshot_id),
            getfield(log, :episode_id),
            getfield(log, :task_name),
            getfield(log, :step_index),
            getfield(log, :attempt_index),
            getfield(log, :operator),
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
    summary = Dict{Tuple{UInt64,String,String,Int,Int,Symbol}, OperatorAttemptOutcomeSummary}()
    for key in all_keys
        snapshot_id, episode_id, task_name, step_index, attempt_index, operator = key
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
        outcome_class = _classify_operator_outcome(
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
        summary[key] = OperatorAttemptOutcomeSummary(
            snapshot_id,
            episode_id,
            task_name,
            step_index,
            attempt_index,
            operator,
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

function compute_operator_target(summary::OperatorAttemptOutcomeSummary; mode::Symbol=:blended)
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

function _classify_operator_state(log::OperatorDecisionLog,
                                  summary::OperatorAttemptOutcomeSummary)
    heuristic_scores = Float64[c.heuristic_score for c in log.candidate_operators]
    spread = isempty(heuristic_scores) ? 0.0 : maximum(heuristic_scores) - minimum(heuristic_scores)
    if summary.outcome_class == "degenerate" || summary.outcome_class == "missing_proposal"
        return "degenerate", false
    elseif length(log.candidate_operators) <= 1 || spread < 1e-8
        return "invariant", false
    elseif log.heuristic_margin < 0.1
        return summary.outcome_class in ("productive", "weak_productive") ? "k1_only" : "ambiguous", summary.outcome_class in ("productive", "weak_productive")
    elseif summary.outcome_class in ("productive", "weak_productive")
        return "robust_operator", true
    else
        return "ambiguous", false
    end
end

function audit_operator_dataset_coverage(operator_logs::Vector{OperatorDecisionLog},
                                         proposal_logs::Vector{HierarchicalEditProposalLog},
                                         decision_logs::Vector{HierarchicalEditDecisionLog})
    summary = summarize_operator_attempt_outcomes(proposal_logs, decision_logs)
    matched = 0
    class_counts = Dict{String,Int}()
    state_counts = Dict{String,Int}()
    empty_after_filter = 0

    for log in operator_logs
        key = _operator_attempt_key(log)
        if haskey(summary, key)
            matched += 1
            outcome = summary[key]
            class_counts[outcome.outcome_class] = get(class_counts, outcome.outcome_class, 0) + 1
            state_label, _ = _classify_operator_state(log, outcome)
            state_counts[state_label] = get(state_counts, state_label, 0) + 1
            empty_after_filter += outcome.empty_after_filter ? 1 : 0
        end
    end

    n_logs = length(operator_logs)
    return Dict(
        "operator_logs" => n_logs,
        "matched_attempt_outcomes" => matched,
        "proposal_coverage_fraction" => n_logs == 0 ? 0.0 : matched / n_logs,
        "decision_coverage_fraction" => n_logs == 0 ? 0.0 : mean(Float64[
            haskey(summary, _operator_attempt_key(log)) && summary[_operator_attempt_key(log)].has_decision_log
            for log in operator_logs
        ]),
        "empty_after_filter_fraction" => matched == 0 ? 0.0 : empty_after_filter / matched,
        "class_counts" => class_counts,
        "state_counts" => state_counts,
    )
end

function extract_operator_controller_dataset(operator_logs::Vector{OperatorDecisionLog},
                                             proposal_logs::Vector{HierarchicalEditProposalLog},
                                             decision_logs::Vector{HierarchicalEditDecisionLog};
                                             target_mode::Symbol=:blended,
                                             feature_mode::Symbol=:basic)
    summary = summarize_operator_attempt_outcomes(proposal_logs, decision_logs)
    dataset = OperatorControllerDataset()
    for log in operator_logs
        key = _operator_attempt_key(log)
        haskey(summary, key) || continue
        outcome = summary[key]
        isempty(log.candidate_operators) && continue
        chosen_index = clamp(log.chosen_index, 1, length(log.candidate_operators))
        state_label, controller_eligible = _classify_operator_state(log, outcome)
        eligibility_features = operator_eligibility_feature_vector(log)
        context_features = operator_context_feature_vector(log)
        candidate_features = [
            operator_candidate_feature_vector(log, candidate;
                all_candidates=log.candidate_operators,
                candidate_index=idx,
                feature_mode=feature_mode)
            for (idx, candidate) in enumerate(log.candidate_operators)
        ]
        push!(dataset.records, OperatorDecisionRecord(
            log.snapshot_id,
            log.episode_id,
            log.task_name,
            log.step_index,
            log.attempt_index,
            eligibility_features,
            context_features,
            candidate_features,
            [candidate.operator for candidate in log.candidate_operators],
            Float32[candidate.heuristic_score for candidate in log.candidate_operators],
            chosen_index,
            Float32(log.chosen_heuristic_score),
            Float32(outcome.chosen_reward_delta),
            Float32(outcome.max_frontier_utility_delta),
            outcome.enters_topk,
            Float32(compute_operator_target(outcome; mode=target_mode)),
            outcome.outcome_class in ("productive", "weak_productive"),
            outcome.has_proposal_log,
            outcome.has_decision_log,
            outcome.empty_after_filter,
            outcome.unique_valid_count,
            outcome.outcome_class,
            state_label,
            controller_eligible,
            log.predicted_eligible,
            Float32(log.eligibility_score),
            log.heuristic_top_index,
            log.learned_top_index,
            Float32(log.heuristic_margin),
            Float32(log.learned_margin),
            Float32(log.learned_advantage_vs_heuristic),
            Float32(log.heuristic_entropy),
            Float32(log.learned_entropy),
            log.acted_on,
            log.preserved_to_heuristic,
            log.override_applied,
            log.abstained_to_heuristic,
            log.selection_reason,
        ))
    end
    return dataset
end

function split_operator_controller_dataset(dataset::OperatorControllerDataset;
                                           train_fraction::Float64=0.8,
                                           rng::AbstractRNG=Random.MersenneTwister(0))
    n = length(dataset.records)
    n == 0 && return OperatorControllerDataset(), OperatorControllerDataset()
    perm = randperm(rng, n)
    n_train = clamp(round(Int, train_fraction * n), 1, max(n - 1, 1))
    train_idx = perm[1:n_train]
    val_idx = perm[(n_train + 1):end]
    isempty(val_idx) && (val_idx = train_idx[end:end])
    return OperatorControllerDataset(dataset.records[train_idx]), OperatorControllerDataset(dataset.records[val_idx])
end

function operator_controller_dataset_stats(dataset::OperatorControllerDataset)
    if isempty(dataset.records)
        return Dict(
            "size" => 0,
            "positive_fraction" => 0.0,
            "feature_dim" => 0,
            "eligibility_dim" => 0,
            "class_counts" => Dict{String,Int}(),
            "state_counts" => Dict{String,Int}(),
            "controller_eligible_fraction" => 0.0,
            "predicted_eligible_fraction" => 0.0,
            "acted_on_fraction" => 0.0,
            "preserved_fraction" => 0.0,
            "mean_heuristic_margin" => 0.0,
            "override_fraction" => 0.0,
            "abstained_fraction" => 0.0,
            "selection_reason_counts" => Dict{String,Int}(),
        )
    end
    class_counts = Dict{String,Int}()
    state_counts = Dict{String,Int}()
    selection_reason_counts = Dict{String,Int}()
    for record in dataset.records
        class_counts[record.outcome_class] = get(class_counts, record.outcome_class, 0) + 1
        state_counts[record.state_label] = get(state_counts, record.state_label, 0) + 1
        selection_reason_counts[record.selection_reason] = get(selection_reason_counts, record.selection_reason, 0) + 1
    end
    return Dict(
        "size" => length(dataset.records),
        "positive_fraction" => mean(Float64[r.success_label for r in dataset.records]),
        "feature_dim" => isempty(dataset.records[1].candidate_features) ? 0 : length(dataset.records[1].candidate_features[1]),
        "eligibility_dim" => length(dataset.records[1].eligibility_features),
        "class_counts" => class_counts,
        "state_counts" => state_counts,
        "controller_eligible_fraction" => mean(Float64[r.controller_eligible for r in dataset.records]),
        "predicted_eligible_fraction" => mean(Float64[r.predicted_eligible for r in dataset.records]),
        "acted_on_fraction" => mean(Float64[r.acted_on for r in dataset.records]),
        "preserved_fraction" => mean(Float64[r.preserved_to_heuristic for r in dataset.records]),
        "mean_heuristic_margin" => mean(Float64[r.heuristic_margin for r in dataset.records]),
        "override_fraction" => mean(Float64[r.override_applied for r in dataset.records]),
        "abstained_fraction" => mean(Float64[r.abstained_to_heuristic for r in dataset.records]),
        "selection_reason_counts" => selection_reason_counts,
    )
end
