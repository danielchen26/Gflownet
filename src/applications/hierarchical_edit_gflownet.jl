# Frontier-Conditioned, Finite-Horizon, Hierarchical Edit GFlowNet
#
# Batch 2 / Stage A goal:
# - preserve a frozen frontier snapshot throughout each episode
# - use an honest trusted operator kernel by default
# - separate compact trajectory memory from richer diagnostic decision logs
# - instrument approximate frontier utility without claiming a final objective
# - diagnose and reduce the current static-frontier failure regime

using LinearAlgebra
using Random
using Statistics

struct HierarchicalEditConfig
    horizon::Int
    frontier_snapshot_size::Int
    allow_crossover::Bool
    allow_fragment_ops::Bool
    max_operator_candidates::Int
    topk_tracking::Int
    max_step_attempts::Int
    operators::Union{Nothing,Vector{Symbol}}
    min_exploration_per_operator::Int
    multi_child_min_reward_ratio::Float64
    operator_prior_strength::Float64
    use_operator_adaptation::Bool
    operator_sampling_weights::Union{Nothing,Dict{Symbol,Float64}}
    basin_candidate_limit::Int
    use_learned_basin::Bool
    learned_basin_controller::Any
    parent_candidate_limit::Int
    use_learned_parent::Bool
    learned_parent_controller::Any
    use_learned_operator::Bool
    learned_operator_controller::Any

    function HierarchicalEditConfig(;
        horizon::Int=3,
        frontier_snapshot_size::Int=128,
        allow_crossover::Bool=true,
        allow_fragment_ops::Bool=false,
        max_operator_candidates::Int=8,
        topk_tracking::Int=10,
        max_step_attempts::Int=3,
        operators::Union{Nothing,Vector{Symbol}}=nothing,
        min_exploration_per_operator::Int=5,
        multi_child_min_reward_ratio::Float64=0.2,
        operator_prior_strength::Float64=4.0,
        use_operator_adaptation::Bool=true,
        operator_sampling_weights::Union{Nothing,Dict{Symbol,Float64}}=nothing,
        basin_candidate_limit::Int=8,
        use_learned_basin::Bool=false,
        learned_basin_controller=nothing,
        parent_candidate_limit::Int=16,
        use_learned_parent::Bool=false,
        learned_parent_controller=nothing,
        use_learned_operator::Bool=false,
        learned_operator_controller=nothing,
    )
        new(horizon, frontier_snapshot_size, allow_crossover, allow_fragment_ops,
            max_operator_candidates, topk_tracking, max_step_attempts, operators,
            min_exploration_per_operator, multi_child_min_reward_ratio, operator_prior_strength,
            use_operator_adaptation, isnothing(operator_sampling_weights) ? nothing : copy(operator_sampling_weights),
            basin_candidate_limit, use_learned_basin, learned_basin_controller,
            parent_candidate_limit, use_learned_parent, learned_parent_controller,
            use_learned_operator, learned_operator_controller)
    end
end

struct FrontierCommitRecord
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
    child_smiles::String
    child_reward::Float64
    child_scaffold::String
    candidate_count::Int
    terminated::Bool
    metadata::Dict{String,Any}
end

struct HierarchicalEditStep
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    step_index::Int
    basin_scaffold::String
    basin_score::Float64
    parent_smiles::String
    operator::Symbol
    child_smiles::String
    reward::Float64
    frontier_utility_delta::Float64
    terminated::Bool
    family_transition_type::String
end

struct HierarchicalEditEpisode
    snapshot_id::UInt64
    episode_id::String
    task_name::String
    steps::Vector{HierarchicalEditStep}
    best_smiles::String
    best_reward::Float64
    frontier_size_before::Int
    frontier_size_after::Int
    improved_topk::Bool
    commits_applied::Int
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

mutable struct HierarchicalEditDiagnosticsBuffer
    logs::Vector{HierarchicalEditDecisionLog}
    proposal_logs::Vector{HierarchicalEditProposalLog}
    basin_logs::Vector{BasinDecisionLog}
    parent_logs::Vector{ParentDecisionLog}
    operator_logs::Vector{OperatorDecisionLog}
    max_size::Int
    function HierarchicalEditDiagnosticsBuffer(max_size::Int=10000)
        new(HierarchicalEditDecisionLog[], HierarchicalEditProposalLog[], BasinDecisionLog[], ParentDecisionLog[], OperatorDecisionLog[], max_size)
    end
end

Base.length(buf::HierarchicalEditDiagnosticsBuffer) = length(buf.logs)
Base.isempty(buf::HierarchicalEditDiagnosticsBuffer) = isempty(buf.logs)

function add_decision_log!(buffer::HierarchicalEditDiagnosticsBuffer,
                           log::HierarchicalEditDecisionLog)
    push!(buffer.logs, log)
    while length(buffer.logs) > buffer.max_size
        popfirst!(buffer.logs)
    end
    return nothing
end

function add_proposal_log!(buffer::HierarchicalEditDiagnosticsBuffer,
                           log::HierarchicalEditProposalLog)
    push!(buffer.proposal_logs, log)
    while length(buffer.proposal_logs) > buffer.max_size
        popfirst!(buffer.proposal_logs)
    end
    return nothing
end

function add_basin_log!(buffer::HierarchicalEditDiagnosticsBuffer,
                        log::BasinDecisionLog)
    push!(buffer.basin_logs, log)
    while length(buffer.basin_logs) > buffer.max_size
        popfirst!(buffer.basin_logs)
    end
    return nothing
end

function add_parent_log!(buffer::HierarchicalEditDiagnosticsBuffer,
                         log::ParentDecisionLog)
    push!(buffer.parent_logs, log)
    while length(buffer.parent_logs) > buffer.max_size
        popfirst!(buffer.parent_logs)
    end
    return nothing
end

function add_operator_log!(buffer::HierarchicalEditDiagnosticsBuffer,
                           log::OperatorDecisionLog)
    push!(buffer.operator_logs, log)
    while length(buffer.operator_logs) > buffer.max_size
        popfirst!(buffer.operator_logs)
    end
    return nothing
end

function decision_log_stats(buffer::HierarchicalEditDiagnosticsBuffer)
    if isempty(buffer)
        return Dict(
            "size" => 0,
            "mean_frontier_utility_delta" => 0.0,
            "positive_delta_fraction" => 0.0,
            "topk_entry_fraction" => 0.0,
        )
    end

    utility_deltas = [l.frontier_utility_delta for l in buffer.logs]
    reward_deltas = [l.reward_delta for l in buffer.logs]
    return Dict(
        "size" => length(buffer.logs),
        "mean_frontier_utility_delta" => mean(utility_deltas),
        "positive_delta_fraction" => mean(Float64[d > 0 for d in reward_deltas]),
        "topk_entry_fraction" => mean(Float64[l.enters_topk for l in buffer.logs]),
    )
end

function proposal_log_stats(buffer::HierarchicalEditDiagnosticsBuffer)
    if isempty(buffer.proposal_logs)
        return Dict(
            "size" => 0,
            "empty_after_filter_fraction" => 0.0,
            "mean_raw_candidate_count" => 0.0,
            "mean_unique_valid_count" => 0.0,
            "duplicate_fraction" => 0.0,
            "cached_fraction" => 0.0,
            "self_child_fraction" => 0.0,
            "same_family_fraction" => 0.0,
            "cross_family_fraction" => 0.0,
            "chosen_positive_delta_fraction" => 0.0,
        )
    end

    logs = buffer.proposal_logs
    raw_total = sum(l.raw_candidate_count for l in logs)
    valid_total = sum(max(l.unique_valid_count, 0) for l in logs)
    family_total = sum(l.same_family_count + l.cross_family_count + l.no_scaffold_count for l in logs)
    chosen_logs = [l for l in logs if !isempty(l.chosen_child_smiles)]

    return Dict(
        "size" => length(logs),
        "empty_after_filter_fraction" => mean(Float64[l.empty_after_filter for l in logs]),
        "mean_raw_candidate_count" => mean(Float64[l.raw_candidate_count for l in logs]),
        "mean_unique_valid_count" => mean(Float64[l.unique_valid_count for l in logs]),
        "duplicate_fraction" => raw_total == 0 ? 0.0 : sum(l.duplicate_candidate_count for l in logs) / raw_total,
        "cached_fraction" => valid_total == 0 ? 0.0 : sum(l.cached_child_count for l in logs) / valid_total,
        "self_child_fraction" => raw_total == 0 ? 0.0 : sum(l.self_child_count for l in logs) / raw_total,
        "same_family_fraction" => family_total == 0 ? 0.0 : sum(l.same_family_count for l in logs) / family_total,
        "cross_family_fraction" => family_total == 0 ? 0.0 : sum(l.cross_family_count for l in logs) / family_total,
        "chosen_positive_delta_fraction" => isempty(chosen_logs) ? 0.0 : mean(Float64[l.chosen_reward_delta > 0 for l in chosen_logs]),
    )
end

"""
    choose_operator(config; bias_structural, operator_override, operator_stats)

Choose an edit operator with two-phase adaptive selection:

**Phase 1 (Exploration)**: Until every non-terminate operator has been tried at
least `config.min_exploration_per_operator` times, prefer the least-tried
operator. This prevents Thompson sampling cold-start from starving crossover
when mutate gets early lucky successes.

**Phase 2 (Exploitation)**: Thompson sampling with informed Beta prior.
Prior strength `config.operator_prior_strength` adds pseudo-observations so that
each operator starts at 50% success rate and real data must accumulate before
the posterior shifts meaningfully.

Falls back to structural bias or uniform random when no stats are available.
"""
function choose_operator(config::HierarchicalEditConfig;
                         bias_structural::Bool=false,
                         operator_override::Union{Nothing,Symbol}=nothing,
                         operator_stats::Union{Nothing,Dict{Symbol,Dict{String,Int}}}=nothing)
    !isnothing(operator_override) && return operator_override

    ops = if !isnothing(config.operators)
        config.operators
    else
        available_edit_operators(; allow_crossover=config.allow_crossover,
                                   allow_fragment_ops=config.allow_fragment_ops)
    end

    non_terminate = filter(!=(:terminate), ops)
    isempty(non_terminate) && return :terminate

    function _sample_weighted(ops_local::Vector{Symbol}, weights::Vector{Float64})
        total_w = sum(weights)
        total_w <= 0 && return rand(ops_local)
        probs = weights ./ total_w
        cumulative = cumsum(probs)
        r = rand()
        idx = clamp(searchsortedfirst(cumulative, r), 1, length(ops_local))
        return ops_local[idx]
    end

    # Adaptive weighting from empirical performance
    if config.use_operator_adaptation && !isnothing(operator_stats)
        min_trials = config.min_exploration_per_operator
        unique_non_terminate = unique(non_terminate)

        # Phase 1: Exploration — ensure all operators get minimum trials
        under_explored = Symbol[]
        for op in unique_non_terminate
            total = get(get(operator_stats, op, Dict{String,Int}()), "total_count", 0)
            if total < min_trials
                push!(under_explored, op)
            end
        end

        if !isempty(under_explored)
            # Prefer least-tried operator for efficient exploration
            trial_counts = [get(get(operator_stats, op, Dict{String,Int}()), "total_count", 0) for op in under_explored]
            min_count = minimum(trial_counts)
            least_tried = [under_explored[i] for i in 1:length(under_explored) if trial_counts[i] == min_count]
            return rand(least_tried)
        end

        # Phase 2: Thompson sampling with informed prior
        prior = config.operator_prior_strength
        half_prior = prior / 2.0
        weights = Float64[]
        for op in unique_non_terminate
            stats = get(operator_stats, op, Dict("positive_delta_count" => 0, "total_count" => 0))
            successes = get(stats, "positive_delta_count", 0)
            total = get(stats, "total_count", 0)
            # Beta(half_prior + s, half_prior + f) → mean = (half_prior + s) / (prior + t)
            weight = (half_prior + successes) / (prior + total)
            push!(weights, weight)
        end
        return _sample_weighted(unique_non_terminate, weights)
    end

    # Static transparent weighting for bounded bridge-stage probes
    if !isnothing(config.operator_sampling_weights)
        unique_non_terminate = unique(non_terminate)
        weights = Float64[max(get(config.operator_sampling_weights, op, 0.0), 0.0) for op in unique_non_terminate]
        if any(>(0.0), weights)
            return _sample_weighted(unique_non_terminate, weights)
        end
    end

    # Fallback: structural bias or uniform
    if bias_structural && :crossover in non_terminate
        fallback_ops = filter(!=(:crossover), unique(non_terminate))
        if isempty(fallback_ops)
            return :crossover
        end
        return rand() < 0.45 ? :crossover : rand(fallback_ops)
    end
    return rand(unique(non_terminate))
end

function choose_basin(snapshot::FrontierSnapshot,
                      config::HierarchicalEditConfig;
                      step_index::Int=0)
    candidates = candidate_basins(snapshot; max_candidates=config.basin_candidate_limit)
    isempty(candidates) && return nothing, ScoredBasinCandidate[]

    chosen = if config.use_learned_basin && !isnothing(config.learned_basin_controller)
        select_basin(config.learned_basin_controller, snapshot, candidates; step_index=step_index)
    else
        sample_scored_basin(candidates)
    end
    chosen === nothing && return nothing, candidates
    return chosen, candidates
end

function _operator_score_margin(scores::AbstractVector{<:Real}, top_idx::Int)
    isempty(scores) && return 0.0f0
    if length(scores) == 1
        return Float32(scores[top_idx])
    end
    order = sortperm(Float32.(scores), rev=true)
    return Float32(scores[order[1]] - scores[order[2]])
end

function _operator_score_entropy(scores::AbstractVector{<:Real})
    isempty(scores) && return 0.0f0
    x = Float32.(scores)
    shifted = x .- maximum(x)
    weights = exp.(shifted)
    total = sum(weights)
    total <= 0 && return 0.0f0
    probs = weights ./ total
    entropy = 0.0f0
    for p in probs
        p <= 0 && continue
        entropy -= p * log(p)
    end
    return entropy
end

function build_operator_decision_candidates(config::HierarchicalEditConfig;
                                            bias_structural::Bool=false,
                                            operator_stats::Union{Nothing,Dict{Symbol,Dict{String,Int}}}=nothing)
    ops = if !isnothing(config.operators)
        config.operators
    else
        available_edit_operators(; allow_crossover=config.allow_crossover,
                                   allow_fragment_ops=config.allow_fragment_ops)
    end
    non_terminate = unique(filter(!=(:terminate), ops))
    isempty(non_terminate) && return OperatorDecisionCandidate[]

    candidates = OperatorDecisionCandidate[]
    if config.use_operator_adaptation && !isnothing(operator_stats)
        min_trials = config.min_exploration_per_operator
        under_counts = Dict{Symbol,Int}()
        for op in non_terminate
            total = get(get(operator_stats, op, Dict{String,Int}()), "total_count", 0)
            if total < min_trials
                under_counts[op] = total
            end
        end
        if !isempty(under_counts)
            min_count = minimum(values(under_counts))
            for op in non_terminate
                stats = get(operator_stats, op, Dict{String,Int}())
                total = get(stats, "total_count", 0)
                pos = get(stats, "positive_delta_count", 0)
                exploration_bonus = haskey(under_counts, op) && total == min_count ? 1.0 : 0.0
                push!(candidates, OperatorDecisionCandidate(op, exploration_bonus, total, pos, exploration_bonus, 0.0))
            end
            return candidates
        end

        prior = config.operator_prior_strength
        half_prior = prior / 2.0
        for op in non_terminate
            stats = get(operator_stats, op, Dict("positive_delta_count" => 0, "total_count" => 0))
            pos = get(stats, "positive_delta_count", 0)
            total = get(stats, "total_count", 0)
            score = (half_prior + pos) / (prior + total)
            push!(candidates, OperatorDecisionCandidate(op, score, total, pos, 0.0, 0.0))
        end
        return candidates
    end

    if !isnothing(config.operator_sampling_weights)
        for op in non_terminate
            score = max(get(config.operator_sampling_weights, op, 0.0), 0.0)
            push!(candidates, OperatorDecisionCandidate(op, score, 0, 0, 0.0, score))
        end
        return candidates
    end

    if bias_structural && :crossover in non_terminate
        fallback_ops = filter(!=(:crossover), non_terminate)
        fallback_weight = isempty(fallback_ops) ? 0.0 : 0.55 / length(fallback_ops)
        for op in non_terminate
            score = op == :crossover ? 0.45 : fallback_weight
            push!(candidates, OperatorDecisionCandidate(op, score, 0, 0, 0.0, score))
        end
        return candidates
    end

    for op in non_terminate
        push!(candidates, OperatorDecisionCandidate(op, 1.0, 0, 0, 0.0, 0.0))
    end
    return candidates
end

function choose_operator_action(snapshot::FrontierSnapshot,
                                basin::BasinSummary,
                                parent::FrontierSnapshotEntry,
                                config::HierarchicalEditConfig;
                                step_index::Int=0,
                                bias_structural::Bool=false,
                                operator_override::Union{Nothing,Symbol}=nothing,
                                operator_stats::Union{Nothing,Dict{Symbol,Dict{String,Int}}}=nothing)
    if operator_override == :terminate
        metadata = Dict{String,Any}(
            "chosen_index" => 0,
            "heuristic_top_index" => 0,
            "learned_top_index" => 0,
            "heuristic_margin" => 0.0,
            "learned_margin" => 0.0,
            "learned_advantage_vs_heuristic" => 0.0,
            "heuristic_entropy" => 0.0,
            "learned_entropy" => 0.0,
            "predicted_eligible" => false,
            "eligibility_score" => 0.0,
            "acted_on" => false,
            "preserved_to_heuristic" => false,
            "override_applied" => false,
            "abstained_to_heuristic" => false,
            "selection_reason" => "terminate_override",
        )
        return :terminate, OperatorDecisionCandidate[], metadata
    end

    candidates = build_operator_decision_candidates(config;
        bias_structural=bias_structural,
        operator_stats=operator_stats)
    isempty(candidates) && return nothing, OperatorDecisionCandidate[], Dict{String,Any}()

    heuristic_scores = Float32[c.heuristic_score for c in candidates]
    heuristic_idx = argmax(heuristic_scores)
    heuristic_margin = Float64(_operator_score_margin(heuristic_scores, heuristic_idx))
    heuristic_entropy = Float64(_operator_score_entropy(heuristic_scores))

    if !isnothing(operator_override)
        chosen_idx = something(findfirst(c -> c.operator == operator_override, candidates), heuristic_idx)
        metadata = Dict{String,Any}(
            "chosen_index" => Int(chosen_idx),
            "heuristic_top_index" => Int(heuristic_idx),
            "learned_top_index" => Int(chosen_idx),
            "heuristic_margin" => heuristic_margin,
            "learned_margin" => heuristic_margin,
            "learned_advantage_vs_heuristic" => 0.0,
            "heuristic_entropy" => heuristic_entropy,
            "learned_entropy" => heuristic_entropy,
            "predicted_eligible" => true,
            "eligibility_score" => 1.0,
            "acted_on" => true,
            "preserved_to_heuristic" => false,
            "override_applied" => chosen_idx != heuristic_idx,
            "abstained_to_heuristic" => false,
            "selection_reason" => chosen_idx == heuristic_idx ? "operator_override_agree" : "operator_override",
        )
        return candidates[chosen_idx].operator, candidates, metadata
    end

    if config.use_learned_operator && !isnothing(config.learned_operator_controller)
        metadata = operator_selection_metadata(config.learned_operator_controller, snapshot, basin, parent, candidates; step_index=step_index)
        chosen_idx = get(metadata, "chosen_index", 0)
        chosen = (1 <= chosen_idx <= length(candidates)) ? candidates[chosen_idx].operator : nothing
        return chosen, candidates, metadata
    end

    chosen = choose_operator(config; bias_structural=bias_structural, operator_override=operator_override, operator_stats=operator_stats)
    chosen_idx = something(findfirst(c -> c.operator == chosen, candidates), heuristic_idx)
    metadata = Dict{String,Any}(
        "chosen_index" => Int(chosen_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(heuristic_idx),
        "heuristic_margin" => heuristic_margin,
        "learned_margin" => heuristic_margin,
        "learned_advantage_vs_heuristic" => 0.0,
        "heuristic_entropy" => heuristic_entropy,
        "learned_entropy" => heuristic_entropy,
        "predicted_eligible" => false,
        "eligibility_score" => 0.0,
        "acted_on" => false,
        "preserved_to_heuristic" => false,
        "override_applied" => chosen_idx != heuristic_idx,
        "abstained_to_heuristic" => false,
        "selection_reason" => chosen_idx == heuristic_idx ? "agree" : "heuristic_sample",
    )
    return chosen, candidates, metadata
end


function build_parent_decision_candidates(candidates::Vector{ScoredParentCandidate})
    return ParentDecisionCandidate[
        ParentDecisionCandidate(
            item.entry.smiles,
            item.entry.scaffold,
            item.entry.reward,
            item.entry.novelty_score,
            item.entry.tb_delta_abs,
            String(item.entry.source),
            item.score,
            item.visit_count,
            item.basin_match,
            item.target_match,
        ) for item in candidates
    ]
end

function choose_parent(snapshot::FrontierSnapshot,
                       basin::Union{Nothing,BasinSummary},
                       config::HierarchicalEditConfig;
                       step_index::Int=0,
                       visit_counts::Union{Nothing,Dict{String,Int}}=nothing)
    candidates = candidate_parents(snapshot;
        basin=basin,
        max_candidates=config.parent_candidate_limit,
        visit_counts=visit_counts,
        restrict_to_basin=false)
    isempty(candidates) && return nothing, ScoredParentCandidate[], Dict{String,Any}()

    if config.use_learned_parent && !isnothing(config.learned_parent_controller)
        metadata = parent_selection_metadata(config.learned_parent_controller, snapshot, candidates; step_index=step_index)
        chosen_index = get(metadata, "chosen_index", 0)
        chosen = (1 <= chosen_index <= length(candidates)) ? candidates[chosen_index] : nothing
        chosen === nothing && return nothing, candidates, metadata
        return chosen, candidates, metadata
    end

    chosen = sample_scored_parent(candidates)
    if chosen === nothing
        return nothing, candidates, Dict{String,Any}()
    end
    heuristic_scores = Float32[c.score for c in candidates]
    heuristic_idx = argmax(heuristic_scores)
    heuristic_margin = length(heuristic_scores) >= 2 ? begin
        order = sortperm(heuristic_scores, rev=true)
        heuristic_scores[order[1]] - heuristic_scores[order[2]]
    end : heuristic_scores[heuristic_idx]
    metadata = Dict{String,Any}(
        "chosen_index" => something(findfirst(c -> c.entry.smiles == chosen.entry.smiles, candidates), heuristic_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(heuristic_idx),
        "heuristic_margin" => Float64(heuristic_margin),
        "learned_margin" => Float64(heuristic_margin),
        "learned_advantage_vs_heuristic" => 0.0,
        "heuristic_entropy" => 0.0,
        "learned_entropy" => 0.0,
        "override_applied" => false,
        "abstained_to_heuristic" => false,
        "selection_reason" => "heuristic_sample",
    )
    return chosen, candidates, metadata
end

function frontier_quality_summary(frontier_buffer::MolecularFrontierBuffer; topk::Int=10)
    top_entries = frontier_topk(frontier_buffer, topk; by=:reward)
    top_rewards = [e.reward for e in top_entries]
    top_smiles = Set{String}(e.smiles for e in top_entries)
    top_scaffolds = Set{String}(e.scaffold for e in top_entries if !isempty(e.scaffold))
    top_source_counts = Dict{String,Int}()
    for entry in top_entries
        source_key = String(entry.source)
        top_source_counts[source_key] = get(top_source_counts, source_key, 0) + 1
    end
    return Dict{String,Any}(
        "size" => length(frontier_buffer),
        "top1" => isempty(top_rewards) ? 0.0 : top_rewards[1],
        "top10_mean" => isempty(top_rewards) ? 0.0 : mean(top_rewards),
        "top_smiles" => top_smiles,
        "top_scaffolds" => top_scaffolds,
        "top_source_counts" => top_source_counts,
        "n_scaffolds" => length(frontier_buffer.scaffold_counts),
    )
end

function compute_frontier_utility_delta(before::Dict{String,Any},
                                        after::Dict{String,Any},
                                        child_smiles::String,
                                        child_scaffold::String;
                                        w_top1::Float64=1.0,
                                        w_top10::Float64=1.0,
                                        w_enters_topk::Float64=0.25,
                                        w_family::Float64=0.1)
    delta_top1 = Float64(after["top1"]) - Float64(before["top1"])
    delta_top10_mean = Float64(after["top10_mean"]) - Float64(before["top10_mean"])
    enters_topk = (child_smiles in after["top_smiles"]) && !(child_smiles in before["top_smiles"])
    family_novelty_bonus = (!isempty(child_scaffold) && !(child_scaffold in before["top_scaffolds"])) ? 1.0 : 0.0
    utility = w_top1 * delta_top1 +
              w_top10 * delta_top10_mean +
              w_enters_topk * (enters_topk ? 1.0 : 0.0) +
              w_family * family_novelty_bonus

    return Dict{String,Any}(
        "delta_top1" => delta_top1,
        "delta_top10_mean" => delta_top10_mean,
        "enters_topk" => enters_topk,
        "family_novelty_bonus" => family_novelty_bonus,
        "frontier_utility_delta" => utility,
    )
end

function _family_transition_type(parent_scaffold::String, child_scaffold::String)
    if isempty(parent_scaffold) || isempty(child_scaffold)
        return "no_scaffold"
    elseif parent_scaffold == child_scaffold
        return "same_family"
    else
        return "new_family"
    end
end

function _clone_frontier_buffer(buffer::MolecularFrontierBuffer)
    clone = MolecularFrontierBuffer(buffer.max_size)
    clone.entries = [MolecularFrontierEntry(
        entry.smiles,
        entry.scaffold,
        entry.reward,
        entry.source,
        entry.parent_smiles,
        entry.operator,
        entry.novelty_score,
        entry.tb_delta_abs,
        entry.visits,
    ) for entry in buffer.entries]
    clone.seen_smiles = copy(buffer.seen_smiles)
    clone.scaffold_counts = copy(buffer.scaffold_counts)
    clone.needs_refresh = buffer.needs_refresh
    return clone
end

function _estimate_single_child_frontier_utility(frontier_buffer::MolecularFrontierBuffer,
                                                 child_smiles::String,
                                                 child_reward::Float64,
                                                 child_scaffold::String;
                                                 topk_tracking::Int=10)
    before = frontier_quality_summary(frontier_buffer; topk=topk_tracking)
    cloned = _clone_frontier_buffer(frontier_buffer)
    add_to_frontier!(cloned, child_smiles;
        reward=child_reward,
        source=:edit,
        parent_smiles=nothing,
        operator=:sample)
    after = frontier_quality_summary(cloned; topk=topk_tracking)
    return compute_frontier_utility_delta(before, after, child_smiles, child_scaffold)
end

function _filter_probe_proposals(frontier_buffer::MolecularFrontierBuffer,
                                 proposals::Vector{EditProposal})
    filtered = EditProposal[]
    cached_child_count = 0
    for proposal in proposals
        child_identity = canonicalize_smiles_identity(proposal.child_smiles)
        if proposal.operator != :terminate && haskey(frontier_buffer.seen_smiles, child_identity)
            cached_child_count += 1
            continue
        end
        push!(filtered, EditProposal(
            proposal.operator,
            canonicalize_smiles_identity(proposal.parent_smiles),
            isnothing(proposal.partner_smiles) ? nothing : canonicalize_smiles_identity(proposal.partner_smiles),
            child_identity,
            copy(proposal.metadata),
        ))
    end
    return filtered, cached_child_count
end

_finite_probe_utility(x::Real) = isfinite(x) ? Float64(x) : -1.0e9

function _range_or_zero(values::Vector{Float64})
    isempty(values) && return 0.0
    return maximum(values) - minimum(values)
end

function _normalized_label_instability(labels::Vector{String})
    filtered = [label for label in labels if !isempty(label) && label != "none"]
    n = length(filtered)
    n <= 1 && return 0.0
    return (length(unique(filtered)) - 1) / (n - 1)
end

function _pair_effect_summary(observed::Vector{Tuple{Int,Int,Float64}})
    if isempty(observed)
        return Dict{String,Any}(
            "parent_main_effect" => 0.0,
            "operator_main_effect" => 0.0,
            "interaction_effect" => 0.0,
            "best_pair_interaction_residual" => 0.0,
            "best_parent_index" => 0,
            "best_operator_index" => 0,
            "best_pair_utility" => -Inf,
        )
    end

    parent_values = Dict{Int,Vector{Float64}}()
    operator_values = Dict{Int,Vector{Float64}}()
    utility_values = Float64[]
    for (parent_idx, operator_idx, utility) in observed
        push!(get!(parent_values, parent_idx, Float64[]), utility)
        push!(get!(operator_values, operator_idx, Float64[]), utility)
        push!(utility_values, utility)
    end

    parent_means = Dict(idx => mean(vals) for (idx, vals) in parent_values)
    operator_means = Dict(idx => mean(vals) for (idx, vals) in operator_values)
    grand_mean = mean(utility_values)

    best_idx = argmax(utility_values)
    best_parent_index, best_operator_index, best_pair_utility = observed[best_idx]
    interaction_residuals = Float64[]
    best_pair_interaction_residual = 0.0

    for (parent_idx, operator_idx, utility) in observed
        baseline = parent_means[parent_idx] + operator_means[operator_idx] - grand_mean
        residual = utility - baseline
        push!(interaction_residuals, abs(residual))
        if parent_idx == best_parent_index && operator_idx == best_operator_index && utility == best_pair_utility
            best_pair_interaction_residual = residual
        end
    end

    return Dict{String,Any}(
        "parent_main_effect" => _range_or_zero(collect(values(parent_means))),
        "operator_main_effect" => _range_or_zero(collect(values(operator_means))),
        "interaction_effect" => isempty(interaction_residuals) ? 0.0 : maximum(interaction_residuals),
        "best_pair_interaction_residual" => best_pair_interaction_residual,
        "best_parent_index" => best_parent_index,
        "best_operator_index" => best_operator_index,
        "best_pair_utility" => best_pair_utility,
    )
end

function _candidate_view_summary(candidate_records::Vector{Dict{String,Any}}, budget::Union{Nothing,Int})
    isempty(candidate_records) && return Dict{String,Any}(
        "count_used" => 0,
        "best_child_smiles" => "",
        "best_reward" => -Inf,
        "best_reward_delta" => -Inf,
        "best_frontier_utility_delta" => -Inf,
        "best_enters_topk" => false,
        "mean_frontier_utility_delta" => -Inf,
        "mean_reward_delta" => -Inf,
        "productive_fraction" => 0.0,
        "enters_topk_fraction" => 0.0,
    )

    count_used = isnothing(budget) ? length(candidate_records) : min(max(budget, 0), length(candidate_records))
    count_used == 0 && return Dict{String,Any}(
        "count_used" => 0,
        "best_child_smiles" => "",
        "best_reward" => -Inf,
        "best_reward_delta" => -Inf,
        "best_frontier_utility_delta" => -Inf,
        "best_enters_topk" => false,
        "mean_frontier_utility_delta" => -Inf,
        "mean_reward_delta" => -Inf,
        "productive_fraction" => 0.0,
        "enters_topk_fraction" => 0.0,
    )

    records = candidate_records[1:count_used]
    utilities = Float64[Float64(record["frontier_utility_delta"]) for record in records]
    reward_deltas = Float64[Float64(record["reward_delta"]) for record in records]
    enters_topk = Bool[Bool(record["enters_topk"]) for record in records]
    productive = Float64[(reward_deltas[i] > 0 || utilities[i] > 0 || enters_topk[i]) for i in eachindex(records)]
    best_idx = argmax(utilities)
    best = records[best_idx]
    return Dict{String,Any}(
        "count_used" => count_used,
        "best_child_smiles" => String(best["child_smiles"]),
        "best_reward" => Float64(best["reward"]),
        "best_reward_delta" => Float64(best["reward_delta"]),
        "best_frontier_utility_delta" => Float64(best["frontier_utility_delta"]),
        "best_enters_topk" => Bool(best["enters_topk"]),
        "mean_frontier_utility_delta" => mean(utilities),
        "mean_reward_delta" => mean(reward_deltas),
        "productive_fraction" => mean(productive),
        "enters_topk_fraction" => mean(Float64.(enters_topk)),
    )
end

function _view_effect_or_default(view_observed::Dict{String,Vector{Tuple{Int,Int,Float64}}}, key::String)
    return _pair_effect_summary(get(view_observed, key, Tuple{Int,Int,Float64}[]))
end

function probe_parent_interventions(frontier_buffer::MolecularFrontierBuffer,
                                    reward_fn,
                                    vocab;
                                    reward_fn_batch=nothing,
                                    config::HierarchicalEditConfig=HierarchicalEditConfig(),
                                    target_smiles::Union{Nothing,String}=nothing,
                                    budget_remaining::Int=0,
                                    created_at_step::Int=0,
                                    task_name::String="unknown",
                                    max_parents::Int=4,
                                    operators::Union{Nothing,Vector{Symbol}}=nothing,
                                    max_basins::Int=1,
                                    restrict_parents_to_basin::Bool=false)
    snapshot = create_frontier_snapshot(frontier_buffer;
        max_entries=config.frontier_snapshot_size,
        target_smiles=target_smiles,
        budget_remaining=budget_remaining,
        created_at_step=created_at_step)
    isempty(snapshot.entries) && return Dict{String,Any}(
        "snapshot_id" => string(snapshot.snapshot_id),
        "task_name" => task_name,
        "candidate_mode" => restrict_parents_to_basin ? "basin_slice" : "frontier",
        "basin_scaffold" => "",
        "parent_summaries" => Dict{String,Any}[],
        "basin_summaries" => Dict{String,Any}[],
    )

    basin_candidates = candidate_basins(snapshot; max_candidates=max(max(config.basin_candidate_limit, 1), max_basins))
    selected_basins = isempty(basin_candidates) ? ScoredBasinCandidate[] : basin_candidates[1:min(max_basins, length(basin_candidates))]

    ops = isnothing(operators) ? begin
        if !isnothing(config.operators)
            config.operators
        else
            available_edit_operators(; allow_crossover=config.allow_crossover,
                                     allow_fragment_ops=config.allow_fragment_ops)
        end
    end : operators

    basin_summaries = Dict{String,Any}[]
    for (basin_rank, basin_candidate) in enumerate(selected_basins)
        basin = basin_candidate.basin
        parent_candidates = candidate_parents(snapshot;
            basin=basin,
            max_candidates=max_parents,
            restrict_to_basin=restrict_parents_to_basin)

        if isempty(parent_candidates)
            push!(basin_summaries, Dict{String,Any}(
                "candidate_mode" => restrict_parents_to_basin ? "basin_slice" : "frontier",
                "basin_rank" => basin_rank,
                "basin_scaffold" => basin.scaffold,
                "basin_score" => basin_candidate.score,
                "matched_budget" => 0,
                "heuristic_margin" => 0.0,
                "heuristic_parent_index" => 0,
                "view_effects" => Dict{String,Any}(),
                "proposal_set_effects" => Dict{String,Any}(),
                "best_parent_index_raw" => 0,
                "best_parent_index_normalized" => 0,
                "best_joint_parent_index_raw" => 0,
                "best_joint_operator_raw" => "none",
                "best_joint_parent_index_normalized" => 0,
                "best_joint_operator_normalized" => "none",
                "heuristic_parent_optimistic_utility_raw" => -Inf,
                "best_parent_optimistic_utility_raw" => -Inf,
                "parent_gap_raw" => 0.0,
                "heuristic_parent_optimistic_utility_normalized" => -Inf,
                "best_parent_optimistic_utility_normalized" => -Inf,
                "parent_gap_normalized" => 0.0,
                "best_joint_utility_raw" => -Inf,
                "best_joint_utility_normalized" => -Inf,
                "joint_gap_vs_heuristic_parent_raw" => 0.0,
                "joint_gap_vs_heuristic_parent_normalized" => 0.0,
                "parent_main_effect_raw" => 0.0,
                "operator_main_effect_raw" => 0.0,
                "interaction_effect_raw" => 0.0,
                "parent_main_effect_normalized" => 0.0,
                "operator_main_effect_normalized" => 0.0,
                "interaction_effect_normalized" => 0.0,
                "best_pair_interaction_residual_normalized" => 0.0,
                "operator_instability_across_parents" => 0.0,
                "parent_instability_across_operators" => 0.0,
                "all_degenerate" => true,
                "parent_summaries" => Dict{String,Any}[],
            ))
            continue
        end

        parent_records = Dict{String,Any}[]
        for (parent_idx, candidate) in enumerate(parent_candidates)
            pair_records = Dict{String,Any}[]
            for (operator_idx, op) in enumerate(ops)
                partner = op == :crossover ? choose_partner(snapshot, candidate.entry.smiles; deterministic=true) : nothing
                proposals, proposal_diagnostics = propose_edit_with_diagnostics(candidate.entry.smiles, op, vocab;
                    partner_smiles=partner,
                    max_candidates=config.max_operator_candidates)
                filtered_proposals, cached_child_count = _filter_probe_proposals(frontier_buffer, proposals)

                candidate_records = Dict{String,Any}[]
                if !isempty(filtered_proposals)
                    smiles_batch = [proposal.child_smiles for proposal in filtered_proposals]
                    rewards = isnothing(reward_fn_batch) ? Float64[reward_fn(smiles) for smiles in smiles_batch] : Float64[reward_fn_batch(smiles_batch)...]
                    for (proposal, reward_value) in zip(filtered_proposals, rewards)
                        child_scaffold = get_scaffold(proposal.child_smiles)
                        utility = _estimate_single_child_frontier_utility(frontier_buffer,
                            proposal.child_smiles,
                            reward_value,
                            child_scaffold;
                            topk_tracking=config.topk_tracking)
                        push!(candidate_records, Dict{String,Any}(
                            "child_smiles" => proposal.child_smiles,
                            "reward" => Float64(reward_value),
                            "reward_delta" => Float64(reward_value - candidate.entry.reward),
                            "frontier_utility_delta" => Float64(utility["frontier_utility_delta"]),
                            "enters_topk" => Bool(utility["enters_topk"]),
                        ))
                    end
                end

                push!(pair_records, Dict{String,Any}(
                    "operator" => String(op),
                    "operator_index" => operator_idx,
                    "partner_smiles" => partner,
                    "raw_candidate_count" => Int(proposal_diagnostics["raw_candidate_count"]),
                    "filtered_candidate_count" => length(filtered_proposals),
                    "cached_child_count" => cached_child_count,
                    "degenerate" => isempty(filtered_proposals),
                    "candidate_records" => candidate_records,
                ))
            end

            push!(parent_records, Dict{String,Any}(
                "parent_index" => parent_idx,
                "candidate" => candidate,
                "pair_records" => pair_records,
            ))
        end

        nondegenerate_counts = Int[
            Int(pair_record["filtered_candidate_count"])
            for parent_record in parent_records for pair_record in parent_record["pair_records"]
            if !Bool(pair_record["degenerate"])
        ]
        matched_budget = isempty(nondegenerate_counts) ? 0 : minimum(nondegenerate_counts)

        best_view_limits = Dict(
            "raw" => nothing,
            "strict_min" => matched_budget,
            "k1" => 1,
            "k2" => 2,
            "k4" => 4,
        )
        mean_view_limits = Dict(
            "mean_top2" => 2,
            "mean_top4" => 4,
        )
        observed_best_by_view = Dict{String,Vector{Tuple{Int,Int,Float64}}}(key => Tuple{Int,Int,Float64}[] for key in keys(best_view_limits))
        observed_mean_by_view = Dict{String,Vector{Tuple{Int,Int,Float64}}}(key => Tuple{Int,Int,Float64}[] for key in keys(mean_view_limits))

        parent_summaries = Dict{String,Any}[]
        for parent_record in parent_records
            parent_idx = Int(parent_record["parent_index"])
            candidate = parent_record["candidate"]
            pair_records = parent_record["pair_records"]

            operator_summaries = Dict{String,Any}[]
            best_raw_operator = "none"
            best_raw_child_smiles = ""
            best_raw_reward_delta = -Inf
            best_raw_utility = -Inf
            best_raw_enters_topk = false
            best_normalized_operator = "none"
            best_normalized_child_smiles = ""
            best_normalized_reward_delta = -Inf
            best_normalized_utility = -Inf
            best_normalized_enters_topk = false
            productive_operator_count = 0
            normalized_productive_operator_count = 0
            degenerate_operator_count = 0

            best_operator_by_view = Dict(key => "none" for key in keys(best_view_limits))
            best_utility_by_view = Dict(key => -Inf for key in keys(best_view_limits))

            for pair_record in pair_records
                op = String(pair_record["operator"])
                op_idx = Int(pair_record["operator_index"])
                candidate_records = pair_record["candidate_records"]
                if Bool(pair_record["degenerate"])
                    empty_best = Dict{String,Any}()
                    empty_mean = Dict{String,Any}()
                    for key in keys(best_view_limits)
                        empty_best[key] = _candidate_view_summary(Dict{String,Any}[], nothing)
                    end
                    for key in keys(mean_view_limits)
                        empty_mean[key] = _candidate_view_summary(Dict{String,Any}[], nothing)
                    end
                    push!(operator_summaries, Dict{String,Any}(
                        "operator" => op,
                        "operator_index" => op_idx,
                        "partner_smiles" => pair_record["partner_smiles"],
                        "raw_candidate_count" => Int(pair_record["raw_candidate_count"]),
                        "filtered_candidate_count" => Int(pair_record["filtered_candidate_count"]),
                        "cached_child_count" => Int(pair_record["cached_child_count"]),
                        "matched_budget" => matched_budget,
                        "degenerate" => true,
                        "best_child_smiles" => "",
                        "best_reward" => -Inf,
                        "best_reward_delta" => -Inf,
                        "best_frontier_utility_delta" => -Inf,
                        "enters_topk" => false,
                        "positive_fraction" => 0.0,
                        "normalized_best_child_smiles" => "",
                        "normalized_best_reward" => -Inf,
                        "normalized_best_reward_delta" => -Inf,
                        "normalized_best_frontier_utility_delta" => -Inf,
                        "normalized_enters_topk" => false,
                        "candidate_smiles" => String[],
                        "candidate_reward_deltas" => Float64[],
                        "candidate_frontier_utilities" => Float64[],
                        "view_summaries" => empty_best,
                        "proposal_set_summaries" => empty_mean,
                    ))
                    degenerate_operator_count += 1
                    continue
                end

                reward_deltas = Float64[Float64(record["reward_delta"]) for record in candidate_records]
                raw_utilities = Float64[Float64(record["frontier_utility_delta"]) for record in candidate_records]
                positive_fraction = mean(Float64[delta > 0 for delta in reward_deltas])

                view_summaries = Dict{String,Any}()
                for (key, limit) in best_view_limits
                    summary = _candidate_view_summary(candidate_records, limit)
                    view_summaries[key] = summary
                    utility = _finite_probe_utility(summary["best_frontier_utility_delta"])
                    isfinite(utility) && push!(get!(observed_best_by_view, key, Tuple{Int,Int,Float64}[]), (parent_idx, op_idx, utility))
                    if utility > best_utility_by_view[key]
                        best_utility_by_view[key] = utility
                        best_operator_by_view[key] = op
                    end
                end

                proposal_set_summaries = Dict{String,Any}()
                for (key, limit) in mean_view_limits
                    summary = _candidate_view_summary(candidate_records, limit)
                    proposal_set_summaries[key] = summary
                    utility = _finite_probe_utility(summary["mean_frontier_utility_delta"])
                    isfinite(utility) && push!(get!(observed_mean_by_view, key, Tuple{Int,Int,Float64}[]), (parent_idx, op_idx, utility))
                end

                raw_view = view_summaries["raw"]
                strict_min_view = view_summaries["strict_min"]
                raw_productive = Float64(raw_view["best_reward_delta"]) > 0 || Float64(raw_view["best_frontier_utility_delta"]) > 0 || Bool(raw_view["best_enters_topk"])
                strict_min_productive = Float64(strict_min_view["best_reward_delta"]) > 0 || Float64(strict_min_view["best_frontier_utility_delta"]) > 0 || Bool(strict_min_view["best_enters_topk"])
                raw_productive && (productive_operator_count += 1)
                strict_min_productive && (normalized_productive_operator_count += 1)

                raw_best_utility = Float64(raw_view["best_frontier_utility_delta"])
                normalized_best_utility = Float64(strict_min_view["best_frontier_utility_delta"])
                if raw_best_utility > best_raw_utility
                    best_raw_operator = op
                    best_raw_child_smiles = String(raw_view["best_child_smiles"])
                    best_raw_reward_delta = Float64(raw_view["best_reward_delta"])
                    best_raw_utility = raw_best_utility
                    best_raw_enters_topk = Bool(raw_view["best_enters_topk"])
                end
                if normalized_best_utility > best_normalized_utility
                    best_normalized_operator = op
                    best_normalized_child_smiles = String(strict_min_view["best_child_smiles"])
                    best_normalized_reward_delta = Float64(strict_min_view["best_reward_delta"])
                    best_normalized_utility = normalized_best_utility
                    best_normalized_enters_topk = Bool(strict_min_view["best_enters_topk"])
                end

                push!(operator_summaries, Dict{String,Any}(
                    "operator" => op,
                    "operator_index" => op_idx,
                    "partner_smiles" => pair_record["partner_smiles"],
                    "raw_candidate_count" => Int(pair_record["raw_candidate_count"]),
                    "filtered_candidate_count" => Int(pair_record["filtered_candidate_count"]),
                    "cached_child_count" => Int(pair_record["cached_child_count"]),
                    "matched_budget" => matched_budget,
                    "degenerate" => false,
                    "best_child_smiles" => String(raw_view["best_child_smiles"]),
                    "best_reward" => Float64(raw_view["best_reward"]),
                    "best_reward_delta" => Float64(raw_view["best_reward_delta"]),
                    "best_frontier_utility_delta" => raw_best_utility,
                    "enters_topk" => Bool(raw_view["best_enters_topk"]),
                    "positive_fraction" => positive_fraction,
                    "normalized_best_child_smiles" => String(strict_min_view["best_child_smiles"]),
                    "normalized_best_reward" => Float64(strict_min_view["best_reward"]),
                    "normalized_best_reward_delta" => Float64(strict_min_view["best_reward_delta"]),
                    "normalized_best_frontier_utility_delta" => normalized_best_utility,
                    "normalized_enters_topk" => Bool(strict_min_view["best_enters_topk"]),
                    "candidate_smiles" => String[String(record["child_smiles"]) for record in candidate_records],
                    "candidate_reward_deltas" => reward_deltas,
                    "candidate_frontier_utilities" => raw_utilities,
                    "view_summaries" => view_summaries,
                    "proposal_set_summaries" => proposal_set_summaries,
                ))
            end

            all_degenerate = degenerate_operator_count == length(ops)
            push!(parent_summaries, Dict{String,Any}(
                "parent_smiles" => candidate.entry.smiles,
                "parent_reward" => candidate.entry.reward,
                "heuristic_rank" => parent_idx,
                "heuristic_score" => candidate.score,
                "visit_count" => candidate.visit_count,
                "basin_match" => candidate.basin_match,
                "target_match" => candidate.target_match,
                "best_operator" => best_raw_operator,
                "best_child_smiles" => best_raw_child_smiles,
                "best_reward_delta" => all_degenerate ? -Inf : best_raw_reward_delta,
                "best_frontier_utility_delta" => all_degenerate ? -Inf : best_raw_utility,
                "best_enters_topk" => all_degenerate ? false : best_raw_enters_topk,
                "best_normalized_operator" => best_normalized_operator,
                "best_normalized_child_smiles" => best_normalized_child_smiles,
                "best_normalized_reward_delta" => all_degenerate ? -Inf : best_normalized_reward_delta,
                "best_normalized_frontier_utility_delta" => all_degenerate ? -Inf : best_normalized_utility,
                "best_normalized_enters_topk" => all_degenerate ? false : best_normalized_enters_topk,
                "best_operator_by_view" => best_operator_by_view,
                "best_utility_by_view" => best_utility_by_view,
                "productive_operator_count" => productive_operator_count,
                "normalized_productive_operator_count" => normalized_productive_operator_count,
                "degenerate_operator_count" => degenerate_operator_count,
                "all_degenerate" => all_degenerate,
                "operator_summaries" => operator_summaries,
            ))
        end

        raw_parent_utilities = [_finite_probe_utility(parent_summary["best_frontier_utility_delta"]) for parent_summary in parent_summaries]
        normalized_parent_utilities = [_finite_probe_utility(parent_summary["best_normalized_frontier_utility_delta"]) for parent_summary in parent_summaries]
        heuristic_scores = Float64[Float64(parent_summary["heuristic_score"]) for parent_summary in parent_summaries]
        heuristic_margin = length(heuristic_scores) >= 2 ? heuristic_scores[1] - heuristic_scores[2] : (isempty(heuristic_scores) ? 0.0 : heuristic_scores[1])
        heuristic_parent_index = isempty(parent_summaries) ? 0 : 1

        raw_effects = _view_effect_or_default(observed_best_by_view, "raw")
        strict_min_effects = _view_effect_or_default(observed_best_by_view, "strict_min")
        k1_effects = _view_effect_or_default(observed_best_by_view, "k1")
        k2_effects = _view_effect_or_default(observed_best_by_view, "k2")
        k4_effects = _view_effect_or_default(observed_best_by_view, "k4")
        mean_top2_effects = _view_effect_or_default(observed_mean_by_view, "mean_top2")
        mean_top4_effects = _view_effect_or_default(observed_mean_by_view, "mean_top4")

        best_parent_per_operator = String[]
        for op in ops
            best_parent_label = "none"
            best_parent_utility = -Inf
            for parent_summary in parent_summaries
                pair_matches = [summary for summary in parent_summary["operator_summaries"] if String(summary["operator"]) == String(op) && !Bool(summary["degenerate"])]
                isempty(pair_matches) && continue
                utility = _finite_probe_utility(pair_matches[1]["view_summaries"]["strict_min"]["best_frontier_utility_delta"])
                if utility > best_parent_utility
                    best_parent_utility = utility
                    best_parent_label = String(parent_summary["parent_smiles"])
                end
            end
            push!(best_parent_per_operator, best_parent_label)
        end

        view_effects = Dict{String,Any}(
            "raw" => raw_effects,
            "strict_min" => strict_min_effects,
            "k1" => k1_effects,
            "k2" => k2_effects,
            "k4" => k4_effects,
        )
        proposal_set_effects = Dict{String,Any}(
            "mean_top2" => mean_top2_effects,
            "mean_top4" => mean_top4_effects,
        )

        push!(basin_summaries, Dict{String,Any}(
            "candidate_mode" => restrict_parents_to_basin ? "basin_slice" : "frontier",
            "basin_rank" => basin_rank,
            "basin_scaffold" => basin.scaffold,
            "basin_score" => basin_candidate.score,
            "matched_budget" => matched_budget,
            "heuristic_margin" => heuristic_margin,
            "heuristic_parent_index" => heuristic_parent_index,
            "view_effects" => view_effects,
            "proposal_set_effects" => proposal_set_effects,
            "best_parent_index_raw" => get(raw_effects, "best_parent_index", 0),
            "best_parent_index_normalized" => get(strict_min_effects, "best_parent_index", 0),
            "best_joint_parent_index_raw" => get(raw_effects, "best_parent_index", 0),
            "best_joint_operator_raw" => (1 <= get(raw_effects, "best_operator_index", 0) <= length(ops)) ? String(ops[get(raw_effects, "best_operator_index", 0)]) : "none",
            "best_joint_parent_index_normalized" => get(strict_min_effects, "best_parent_index", 0),
            "best_joint_operator_normalized" => (1 <= get(strict_min_effects, "best_operator_index", 0) <= length(ops)) ? String(ops[get(strict_min_effects, "best_operator_index", 0)]) : "none",
            "heuristic_parent_optimistic_utility_raw" => isempty(raw_parent_utilities) ? -Inf : raw_parent_utilities[1],
            "best_parent_optimistic_utility_raw" => isempty(raw_parent_utilities) ? -Inf : maximum(raw_parent_utilities),
            "parent_gap_raw" => isempty(raw_parent_utilities) ? 0.0 : maximum(raw_parent_utilities) - raw_parent_utilities[1],
            "heuristic_parent_optimistic_utility_normalized" => isempty(normalized_parent_utilities) ? -Inf : normalized_parent_utilities[1],
            "best_parent_optimistic_utility_normalized" => isempty(normalized_parent_utilities) ? -Inf : maximum(normalized_parent_utilities),
            "parent_gap_normalized" => isempty(normalized_parent_utilities) ? 0.0 : maximum(normalized_parent_utilities) - normalized_parent_utilities[1],
            "best_joint_utility_raw" => get(raw_effects, "best_pair_utility", -Inf),
            "best_joint_utility_normalized" => get(strict_min_effects, "best_pair_utility", -Inf),
            "joint_gap_vs_heuristic_parent_raw" => isempty(raw_parent_utilities) ? 0.0 : get(raw_effects, "best_pair_utility", -Inf) - raw_parent_utilities[1],
            "joint_gap_vs_heuristic_parent_normalized" => isempty(normalized_parent_utilities) ? 0.0 : get(strict_min_effects, "best_pair_utility", -Inf) - normalized_parent_utilities[1],
            "parent_main_effect_raw" => get(raw_effects, "parent_main_effect", 0.0),
            "operator_main_effect_raw" => get(raw_effects, "operator_main_effect", 0.0),
            "interaction_effect_raw" => get(raw_effects, "interaction_effect", 0.0),
            "parent_main_effect_normalized" => get(strict_min_effects, "parent_main_effect", 0.0),
            "operator_main_effect_normalized" => get(strict_min_effects, "operator_main_effect", 0.0),
            "interaction_effect_normalized" => get(strict_min_effects, "interaction_effect", 0.0),
            "parent_main_effect_k1" => get(k1_effects, "parent_main_effect", 0.0),
            "operator_main_effect_k1" => get(k1_effects, "operator_main_effect", 0.0),
            "interaction_effect_k1" => get(k1_effects, "interaction_effect", 0.0),
            "parent_main_effect_k2" => get(k2_effects, "parent_main_effect", 0.0),
            "operator_main_effect_k2" => get(k2_effects, "operator_main_effect", 0.0),
            "interaction_effect_k2" => get(k2_effects, "interaction_effect", 0.0),
            "parent_main_effect_k4" => get(k4_effects, "parent_main_effect", 0.0),
            "operator_main_effect_k4" => get(k4_effects, "operator_main_effect", 0.0),
            "interaction_effect_k4" => get(k4_effects, "interaction_effect", 0.0),
            "operator_mean_top2_effect" => get(mean_top2_effects, "operator_main_effect", 0.0),
            "operator_mean_top4_effect" => get(mean_top4_effects, "operator_main_effect", 0.0),
            "interaction_mean_top2_effect" => get(mean_top2_effects, "interaction_effect", 0.0),
            "interaction_mean_top4_effect" => get(mean_top4_effects, "interaction_effect", 0.0),
            "best_pair_interaction_residual_normalized" => get(strict_min_effects, "best_pair_interaction_residual", 0.0),
            "operator_instability_across_parents" => _normalized_label_instability([String(get(parent_summary["best_operator_by_view"], "strict_min", "none")) for parent_summary in parent_summaries]),
            "parent_instability_across_operators" => _normalized_label_instability(best_parent_per_operator),
            "all_degenerate" => all(Bool(parent_summary["all_degenerate"]) for parent_summary in parent_summaries),
            "parent_summaries" => parent_summaries,
        ))
    end

    primary_basin_summary = isempty(basin_summaries) ? Dict{String,Any}() : basin_summaries[1]
    return Dict{String,Any}(
        "snapshot_id" => string(snapshot.snapshot_id),
        "task_name" => task_name,
        "candidate_mode" => restrict_parents_to_basin ? "basin_slice" : "frontier",
        "basin_scaffold" => get(primary_basin_summary, "basin_scaffold", ""),
        "matched_budget" => get(primary_basin_summary, "matched_budget", 0),
        "parent_summaries" => get(primary_basin_summary, "parent_summaries", Dict{String,Any}[]),
        "basin_summaries" => basin_summaries,
    )
end

function _heuristic_top_basin_candidate(snapshot::FrontierSnapshot,
                                      config::HierarchicalEditConfig;
                                      max_candidates::Int=max(config.basin_candidate_limit, 1))
    basin_candidates = candidate_basins(snapshot; max_candidates=max_candidates)
    return isempty(basin_candidates) ? nothing : basin_candidates[1]
end

function _heuristic_top_parent_candidate(snapshot::FrontierSnapshot,
                                         basin::Union{Nothing,BasinSummary};
                                         max_parents::Int=4,
                                         restrict_to_basin::Bool=false)
    parent_candidates = candidate_parents(snapshot;
        basin=basin,
        max_candidates=max_parents,
        restrict_to_basin=restrict_to_basin)
    return isempty(parent_candidates) ? (nothing, ScoredParentCandidate[]) : (parent_candidates[1], parent_candidates)
end

function _heuristic_top_operator_choice(config::HierarchicalEditConfig;
                                        bias_structural::Bool=false,
                                        operator_stats::Union{Nothing,Dict{Symbol,Dict{String,Int}}}=nothing)
    candidates = build_operator_decision_candidates(config;
        bias_structural=bias_structural,
        operator_stats=operator_stats)
    isempty(candidates) && return nothing, OperatorDecisionCandidate[], 0
    heuristic_scores = Float32[c.heuristic_score for c in candidates]
    heuristic_idx = argmax(heuristic_scores)
    return candidates[heuristic_idx].operator, candidates, heuristic_idx
end

function _probe_option_step!(cloned_frontier::MolecularFrontierBuffer,
                             snapshot::FrontierSnapshot,
                             reward_fn,
                             vocab,
                             parent::FrontierSnapshotEntry,
                             operator::Symbol;
                             reward_fn_batch=nothing,
                             config::HierarchicalEditConfig=HierarchicalEditConfig(),
                             topk_tracking::Int=config.topk_tracking)
    partner = operator == :crossover ? choose_partner(snapshot, parent.smiles; deterministic=true) : nothing
    proposals, proposal_diagnostics = propose_edit_with_diagnostics(parent.smiles, operator, vocab;
        partner_smiles=partner,
        max_candidates=config.max_operator_candidates)
    filtered_proposals, cached_child_count = _filter_probe_proposals(cloned_frontier, proposals)
    if isempty(filtered_proposals)
        return Dict{String,Any}(
            "degenerate" => true,
            "operator" => String(operator),
            "partner_smiles" => partner,
            "raw_candidate_count" => Int(proposal_diagnostics["raw_candidate_count"]),
            "filtered_candidate_count" => 0,
            "cached_child_count" => cached_child_count,
            "child_smiles" => "",
            "child_reward" => -Inf,
            "reward_delta" => -Inf,
            "incremental_frontier_utility_delta" => -Inf,
            "enters_topk" => false,
            "terminated" => false,
            "next_parent" => nothing,
        )
    end

    smiles_batch = [proposal.child_smiles for proposal in filtered_proposals]
    rewards = isnothing(reward_fn_batch) ? Float64[reward_fn(smiles) for smiles in smiles_batch] : Float64[reward_fn_batch(smiles_batch)...]
    best_idx = argmax(rewards)
    chosen = filtered_proposals[best_idx]
    chosen_reward = Float64(rewards[best_idx])
    child_scaffold = get_scaffold(chosen.child_smiles)
    utility = _estimate_single_child_frontier_utility(cloned_frontier,
        chosen.child_smiles,
        chosen_reward,
        child_scaffold;
        topk_tracking=topk_tracking)

    add_to_frontier!(cloned_frontier, chosen.child_smiles;
        reward=chosen_reward,
        source=:edit,
        parent_smiles=parent.smiles,
        operator=operator)

    terminated = (chosen.operator == :terminate)
    next_parent = terminated ? nothing : FrontierSnapshotEntry(
        chosen.child_smiles,
        child_scaffold,
        chosen_reward,
        0.0,
        0.0,
        :edit,
    )

    return Dict{String,Any}(
        "degenerate" => false,
        "operator" => String(operator),
        "partner_smiles" => partner,
        "raw_candidate_count" => Int(proposal_diagnostics["raw_candidate_count"]),
        "filtered_candidate_count" => length(filtered_proposals),
        "cached_child_count" => cached_child_count,
        "child_smiles" => chosen.child_smiles,
        "child_reward" => chosen_reward,
        "reward_delta" => chosen_reward - parent.reward,
        "incremental_frontier_utility_delta" => Float64(utility["frontier_utility_delta"]),
        "enters_topk" => Bool(utility["enters_topk"]),
        "parent_novelty_score" => parent.novelty_score,
        "parent_tb_delta_abs" => parent.tb_delta_abs,
        "terminated" => terminated,
        "next_parent" => next_parent,
    )
end

function _rollout_probe_option(frontier_buffer::MolecularFrontierBuffer,
                               snapshot::FrontierSnapshot,
                               reward_fn,
                               vocab;
                               reward_fn_batch=nothing,
                               config::HierarchicalEditConfig=HierarchicalEditConfig(),
                               target_smiles::Union{Nothing,String}=nothing,
                               initial_basin_candidate=nothing,
                               initial_parent=nothing,
                               initial_operator::Union{Nothing,Symbol}=nothing,
                               horizon::Int=3,
                               max_parents::Int=4,
                               restrict_parents_to_basin::Bool=false)
    cloned_frontier = _clone_frontier_buffer(frontier_buffer)
    current_parent = initial_parent isa ScoredParentCandidate ? initial_parent.entry : initial_parent
    initial_basin = initial_basin_candidate === nothing ? nothing : initial_basin_candidate.basin
    current_basin_candidate = initial_basin_candidate

    step_summaries = Dict{String,Any}[]
    cumulative_utility = 0.0
    best_cumulative_utility = -Inf
    step1_incremental_utility = -Inf
    best_reward_reached = -Inf
    nondegenerate_steps = 0
    terminated = false

    for step_idx in 1:horizon
        basin_candidate = if step_idx == 1 && current_basin_candidate !== nothing
            current_basin_candidate
        else
            _heuristic_top_basin_candidate(snapshot, config; max_candidates=max(config.basin_candidate_limit, 1))
        end
        basin_candidate === nothing && break
        basin = basin_candidate.basin

        parent_entry = if step_idx == 1 && current_parent !== nothing
            current_parent
        elseif current_parent !== nothing
            current_parent
        else
            scored_parent, _ = _heuristic_top_parent_candidate(snapshot, basin;
                max_parents=max_parents,
                restrict_to_basin=restrict_parents_to_basin)
            scored_parent === nothing ? nothing : scored_parent.entry
        end
        parent_entry === nothing && break

        operator = if step_idx == 1 && !isnothing(initial_operator)
            initial_operator
        else
            op, _, _ = _heuristic_top_operator_choice(config; bias_structural=!isnothing(target_smiles))
            op
        end
        isnothing(operator) && break

        step = _probe_option_step!(cloned_frontier, snapshot, reward_fn, vocab, parent_entry, operator;
            reward_fn_batch=reward_fn_batch,
            config=config,
            topk_tracking=config.topk_tracking)
        step["step_index"] = step_idx
        step["basin_scaffold"] = basin.scaffold
        step["basin_score"] = basin_candidate.score
        step["parent_smiles"] = parent_entry.smiles
        step["parent_reward"] = parent_entry.reward
        push!(step_summaries, step)

        if Bool(step["degenerate"])
            break
        end

        nondegenerate_steps += 1
        incremental_utility = Float64(step["incremental_frontier_utility_delta"])
        cumulative_utility += incremental_utility
        step["cumulative_frontier_utility_delta"] = cumulative_utility
        step1_incremental_utility = step_idx == 1 ? incremental_utility : step1_incremental_utility
        best_cumulative_utility = max(best_cumulative_utility, cumulative_utility)
        best_reward_reached = max(best_reward_reached, Float64(step["child_reward"]))
        terminated = Bool(step["terminated"])
        current_parent = get(step, "next_parent", nothing)
        terminated && break
        current_basin_candidate = nothing
    end

    all_degenerate = nondegenerate_steps == 0
    return Dict{String,Any}(
        "all_degenerate" => all_degenerate,
        "initial_basin_scaffold" => isnothing(initial_basin) ? "" : initial_basin.scaffold,
        "initial_parent_smiles" => current_parent === nothing && initial_parent === nothing ? "" : String((initial_parent isa ScoredParentCandidate ? initial_parent.entry.smiles : initial_parent.smiles)),
        "initial_operator" => isnothing(initial_operator) ? "none" : String(initial_operator),
        "step_summaries" => step_summaries,
        "step_count" => length(step_summaries),
        "nondegenerate_step_count" => nondegenerate_steps,
        "terminated" => terminated,
        "step1_incremental_utility" => all_degenerate ? -Inf : step1_incremental_utility,
        "best_cumulative_frontier_utility_delta" => all_degenerate ? -Inf : best_cumulative_utility,
        "continuation_gain" => all_degenerate ? 0.0 : best_cumulative_utility - step1_incremental_utility,
        "best_reward_reached" => all_degenerate ? -Inf : best_reward_reached,
        "topk_hit_fraction" => all_degenerate ? 0.0 : mean(Float64[Bool(get(step, "enters_topk", false)) for step in step_summaries if !Bool(step["degenerate"])]),
    )
end

function _best_rollout_summary(rollouts::Vector{Dict{String,Any}})
    isempty(rollouts) && return Dict{String,Any}(
        "all_degenerate" => true,
        "best_cumulative_frontier_utility_delta" => -Inf,
        "step1_incremental_utility" => -Inf,
        "continuation_gain" => 0.0,
        "best_reward_reached" => -Inf,
        "topk_hit_fraction" => 0.0,
        "rollout" => Dict{String,Any}(),
    )
    best_idx = argmax([_finite_probe_utility(get(rollout, "best_cumulative_frontier_utility_delta", -Inf)) for rollout in rollouts])
    best = rollouts[best_idx]
    return Dict{String,Any}(
        "all_degenerate" => Bool(best["all_degenerate"]),
        "best_cumulative_frontier_utility_delta" => get(best, "best_cumulative_frontier_utility_delta", -Inf),
        "step1_incremental_utility" => get(best, "step1_incremental_utility", -Inf),
        "continuation_gain" => get(best, "continuation_gain", 0.0),
        "best_reward_reached" => get(best, "best_reward_reached", -Inf),
        "topk_hit_fraction" => get(best, "topk_hit_fraction", 0.0),
        "rollout" => best,
    )
end

function probe_coupled_hierarchy_options(frontier_buffer::MolecularFrontierBuffer,
                                         reward_fn,
                                         vocab;
                                         reward_fn_batch=nothing,
                                         config::HierarchicalEditConfig=HierarchicalEditConfig(),
                                         target_smiles::Union{Nothing,String}=nothing,
                                         budget_remaining::Int=0,
                                         created_at_step::Int=0,
                                         task_name::String="unknown",
                                         max_parents::Int=4,
                                         operators::Union{Nothing,Vector{Symbol}}=nothing,
                                         max_basins::Int=2,
                                         horizon::Int=3)
    snapshot = create_frontier_snapshot(frontier_buffer;
        max_entries=config.frontier_snapshot_size,
        target_smiles=target_smiles,
        budget_remaining=budget_remaining,
        created_at_step=created_at_step)
    isempty(snapshot.entries) && return Dict{String,Any}(
        "snapshot_id" => string(snapshot.snapshot_id),
        "task_name" => task_name,
        "horizon" => horizon,
        "heuristic_baseline" => Dict{String,Any}(),
        "family_a_rollouts" => Dict{String,Any}[],
        "family_b_rollouts" => Dict{String,Any}[],
        "family_c_rollouts" => Dict{String,Any}[],
        "summary" => Dict{String,Any}("all_degenerate" => true),
    )

    selected_basins = candidate_basins(snapshot; max_candidates=max(max(config.basin_candidate_limit, 1), max_basins))
    selected_basins = isempty(selected_basins) ? ScoredBasinCandidate[] : selected_basins[1:min(max_basins, length(selected_basins))]

    ops = isnothing(operators) ? begin
        if !isnothing(config.operators)
            config.operators
        else
            available_edit_operators(; allow_crossover=config.allow_crossover,
                                     allow_fragment_ops=config.allow_fragment_ops)
        end
    end : operators

    heuristic_basin_candidate = isempty(selected_basins) ? nothing : selected_basins[1]
    heuristic_parent_candidate, heuristic_parent_candidates = if heuristic_basin_candidate === nothing
        (nothing, ScoredParentCandidate[])
    else
        _heuristic_top_parent_candidate(snapshot, heuristic_basin_candidate.basin;
            max_parents=max_parents,
            restrict_to_basin=false)
    end
    heuristic_operator, _, _ = if heuristic_parent_candidate === nothing || heuristic_basin_candidate === nothing
        (nothing, OperatorDecisionCandidate[], 0)
    else
        _heuristic_top_operator_choice(config; bias_structural=!isnothing(target_smiles))
    end

    heuristic_baseline = if heuristic_basin_candidate === nothing || heuristic_parent_candidate === nothing || isnothing(heuristic_operator)
        Dict{String,Any}("all_degenerate" => true)
    else
        _rollout_probe_option(frontier_buffer, snapshot, reward_fn, vocab;
            reward_fn_batch=reward_fn_batch,
            config=config,
            target_smiles=target_smiles,
            initial_basin_candidate=heuristic_basin_candidate,
            initial_parent=heuristic_parent_candidate,
            initial_operator=heuristic_operator,
            horizon=horizon,
            max_parents=max_parents,
            restrict_parents_to_basin=false)
    end

    family_a_rollouts = Dict{String,Any}[]
    if heuristic_basin_candidate !== nothing && heuristic_parent_candidate !== nothing
        for op in ops
            rollout = _rollout_probe_option(frontier_buffer, snapshot, reward_fn, vocab;
                reward_fn_batch=reward_fn_batch,
                config=config,
                target_smiles=target_smiles,
                initial_basin_candidate=heuristic_basin_candidate,
                initial_parent=heuristic_parent_candidate,
                initial_operator=op,
                horizon=horizon,
                max_parents=max_parents,
                restrict_parents_to_basin=false)
            rollout["family"] = "A"
            rollout["object_id"] = "A::$(String(op))"
            push!(family_a_rollouts, rollout)
        end
    end

    family_b_rollouts = Dict{String,Any}[]
    if heuristic_basin_candidate !== nothing && !isempty(heuristic_parent_candidates)
        for (parent_idx, parent_candidate) in enumerate(heuristic_parent_candidates)
            for op in ops
                rollout = _rollout_probe_option(frontier_buffer, snapshot, reward_fn, vocab;
                    reward_fn_batch=reward_fn_batch,
                    config=config,
                    target_smiles=target_smiles,
                    initial_basin_candidate=heuristic_basin_candidate,
                    initial_parent=parent_candidate,
                    initial_operator=op,
                    horizon=horizon,
                    max_parents=max_parents,
                    restrict_parents_to_basin=false)
                rollout["family"] = "B"
                rollout["parent_index"] = parent_idx
                rollout["object_id"] = "B::p$(parent_idx)::$(String(op))"
                push!(family_b_rollouts, rollout)
            end
        end
    end

    family_c_rollouts = Dict{String,Any}[]
    for (basin_idx, basin_candidate) in enumerate(selected_basins)
        parent_candidate_top, parent_candidates = _heuristic_top_parent_candidate(snapshot, basin_candidate.basin;
            max_parents=max_parents,
            restrict_to_basin=false)
        for (parent_idx, parent_candidate) in enumerate(parent_candidates)
            for op in ops
                rollout = _rollout_probe_option(frontier_buffer, snapshot, reward_fn, vocab;
                    reward_fn_batch=reward_fn_batch,
                    config=config,
                    target_smiles=target_smiles,
                    initial_basin_candidate=basin_candidate,
                    initial_parent=parent_candidate,
                    initial_operator=op,
                    horizon=horizon,
                    max_parents=max_parents,
                    restrict_parents_to_basin=false)
                rollout["family"] = "C"
                rollout["basin_index"] = basin_idx
                rollout["parent_index"] = parent_idx
                rollout["object_id"] = "C::b$(basin_idx)::p$(parent_idx)::$(String(op))"
                push!(family_c_rollouts, rollout)
            end
        end
    end

    baseline_utility = get(heuristic_baseline, "best_cumulative_frontier_utility_delta", -Inf)
    a_best = _best_rollout_summary(family_a_rollouts)
    b_best = _best_rollout_summary(family_b_rollouts)
    c_best = _best_rollout_summary(family_c_rollouts)

    best_by_basin = Dict{String,Float64}()
    for rollout in family_c_rollouts
        basin_scaffold = String(get(rollout, "initial_basin_scaffold", ""))
        basin_scaffold == "" && continue
        utility = _finite_probe_utility(get(rollout, "best_cumulative_frontier_utility_delta", -Inf))
        best_by_basin[basin_scaffold] = max(get(best_by_basin, basin_scaffold, -Inf), utility)
    end

    summary = Dict{String,Any}(
        "all_degenerate" => Bool(get(heuristic_baseline, "all_degenerate", true)) && Bool(a_best["all_degenerate"]) && Bool(b_best["all_degenerate"]) && Bool(c_best["all_degenerate"]),
        "heuristic_baseline_utility" => baseline_utility,
        "heuristic_baseline_step1_utility" => get(heuristic_baseline, "step1_incremental_utility", -Inf),
        "heuristic_baseline_continuation_gain" => get(heuristic_baseline, "continuation_gain", 0.0),
        "family_a_best_utility" => a_best["best_cumulative_frontier_utility_delta"],
        "family_b_best_utility" => b_best["best_cumulative_frontier_utility_delta"],
        "family_c_best_utility" => c_best["best_cumulative_frontier_utility_delta"],
        "family_a_best_step1_utility" => a_best["step1_incremental_utility"],
        "family_b_best_step1_utility" => b_best["step1_incremental_utility"],
        "family_c_best_step1_utility" => c_best["step1_incremental_utility"],
        "family_a_best_continuation_gain" => a_best["continuation_gain"],
        "family_b_best_continuation_gain" => b_best["continuation_gain"],
        "family_c_best_continuation_gain" => c_best["continuation_gain"],
        "family_a_gain_vs_baseline" => _finite_probe_utility(a_best["best_cumulative_frontier_utility_delta"]) - _finite_probe_utility(baseline_utility),
        "family_b_gain_vs_baseline" => _finite_probe_utility(b_best["best_cumulative_frontier_utility_delta"]) - _finite_probe_utility(baseline_utility),
        "family_c_gain_vs_baseline" => _finite_probe_utility(c_best["best_cumulative_frontier_utility_delta"]) - _finite_probe_utility(baseline_utility),
        "parent_coupling_gain" => _finite_probe_utility(b_best["best_cumulative_frontier_utility_delta"]) - _finite_probe_utility(a_best["best_cumulative_frontier_utility_delta"]),
        "basin_coupling_gain" => _finite_probe_utility(c_best["best_cumulative_frontier_utility_delta"]) - _finite_probe_utility(b_best["best_cumulative_frontier_utility_delta"]),
        "family_c_best_reward_reached" => c_best["best_reward_reached"],
        "family_c_best_topk_hit_fraction" => c_best["topk_hit_fraction"],
        "best_basins_utility_range" => isempty(best_by_basin) ? 0.0 : _range_or_zero(collect(values(best_by_basin))),
        "best_basins" => best_by_basin,
        "best_family_a_rollout" => a_best["rollout"],
        "best_family_b_rollout" => b_best["rollout"],
        "best_family_c_rollout" => c_best["rollout"],
    )

    return Dict{String,Any}(
        "snapshot_id" => string(snapshot.snapshot_id),
        "task_name" => task_name,
        "horizon" => horizon,
        "heuristic_basin_scaffold" => heuristic_basin_candidate === nothing ? "" : heuristic_basin_candidate.basin.scaffold,
        "heuristic_parent_smiles" => heuristic_parent_candidate === nothing ? "" : heuristic_parent_candidate.entry.smiles,
        "heuristic_operator" => isnothing(heuristic_operator) ? "none" : String(heuristic_operator),
        "heuristic_baseline" => heuristic_baseline,
        "family_a_rollouts" => family_a_rollouts,
        "family_b_rollouts" => family_b_rollouts,
        "family_c_rollouts" => family_c_rollouts,
        "summary" => summary,
    )
end

function _safe_probe_correlation(xs::Vector{Float64}, ys::Vector{Float64})
    length(xs) == length(ys) || return 0.0
    length(xs) < 2 && return 0.0
    std(xs) <= 1f-12 && return 0.0
    std(ys) <= 1f-12 && return 0.0
    value = cor(xs, ys)
    return isfinite(value) ? Float64(value) : 0.0
end

function _rollout_to_subtrajectory_record(probe::Dict{String,Any},
                                          rollout::Dict{String,Any};
                                          baseline_utility::Float64)
    family = String(get(rollout, "family", "?"))
    basin_scaffold = String(get(rollout, "initial_basin_scaffold", ""))
    parent_smiles = String(get(rollout, "initial_parent_smiles", ""))
    operator = String(get(rollout, "initial_operator", "none"))
    object_id = String(get(rollout, "object_id", family))
    option_value = _finite_probe_utility(get(rollout, "best_cumulative_frontier_utility_delta", -Inf))
    step1_value = _finite_probe_utility(get(rollout, "step1_incremental_utility", -Inf))
    continuation_gain = Float64(get(rollout, "continuation_gain", 0.0))
    step_summaries = get(rollout, "step_summaries", Dict{String,Any}[])
    trajectory_signature = join([
        string(get(step, "step_index", 0), ":", get(step, "operator", "none"), ":", get(step, "child_smiles", ""))
        for step in step_summaries if !Bool(get(step, "degenerate", false))
    ], " => ")
    first_step = isempty(step_summaries) ? Dict{String,Any}() : step_summaries[1]

    return Dict{String,Any}(
        "snapshot_id" => String(get(probe, "snapshot_id", "")),
        "task_name" => String(get(probe, "task_name", "unknown")),
        "horizon" => Int(get(probe, "horizon", 0)),
        "family" => family,
        "object_id" => object_id,
        "entry_key" => string(basin_scaffold, "|", parent_smiles, "|", operator),
        "basin_scaffold" => basin_scaffold,
        "basin_score" => Float64(get(first_step, "basin_score", 0.0)),
        "parent_smiles" => parent_smiles,
        "parent_reward" => Float64(get(first_step, "parent_reward", 0.0)),
        "parent_novelty_score" => Float64(get(first_step, "parent_novelty_score", 0.0)),
        "parent_tb_delta_abs" => Float64(get(first_step, "parent_tb_delta_abs", 0.0)),
        "operator" => operator,
        "step_count" => Int(get(rollout, "step_count", 0)),
        "nondegenerate_step_count" => Int(get(rollout, "nondegenerate_step_count", 0)),
        "all_degenerate" => Bool(get(rollout, "all_degenerate", true)),
        "step1_local_utility" => step1_value,
        "option_value" => option_value,
        "continuation_gain" => continuation_gain,
        "best_reward_reached" => Float64(get(rollout, "best_reward_reached", -Inf)),
        "topk_hit_fraction" => Float64(get(rollout, "topk_hit_fraction", 0.0)),
        "gain_vs_baseline" => option_value - baseline_utility,
        "trajectory_signature" => trajectory_signature,
    )
end

function extract_option_subtrajectory_records(probe::Dict{String,Any})
    summary = get(probe, "summary", Dict{String,Any}())
    baseline_utility = _finite_probe_utility(get(summary, "heuristic_baseline_utility", -Inf))
    records = Dict{String,Any}[]
    for key in ["family_a_rollouts", "family_b_rollouts", "family_c_rollouts"]
        for rollout in get(probe, key, Dict{String,Any}[])
            push!(records, _rollout_to_subtrajectory_record(probe, rollout; baseline_utility=baseline_utility))
        end
    end
    return records
end

function _best_record_by_metric(records::Vector{Dict{String,Any}}, metric::String)
    valid = [record for record in records if !Bool(get(record, "all_degenerate", true))]
    isempty(valid) && return nothing
    values = [_finite_probe_utility(get(record, metric, -Inf)) for record in valid]
    return valid[argmax(values)]
end

function compare_option_value_surfaces(probe::Dict{String,Any})
    summary = get(probe, "summary", Dict{String,Any}())
    baseline_utility = _finite_probe_utility(get(summary, "heuristic_baseline_utility", -Inf))
    records = extract_option_subtrajectory_records(probe)
    family_a_records = [record for record in records if record["family"] == "A"]
    family_c_records = [record for record in records if record["family"] == "C"]

    local_best = _best_record_by_metric(family_a_records, "step1_local_utility")
    entry_best = _best_record_by_metric(family_c_records, "step1_local_utility")
    subtrajectory_best = _best_record_by_metric(family_c_records, "option_value")

    local_object_utility = isnothing(local_best) ? baseline_utility : _finite_probe_utility(local_best["option_value"])
    entry_context_utility = isnothing(entry_best) ? baseline_utility : _finite_probe_utility(entry_best["option_value"])
    subtrajectory_object_utility = isnothing(subtrajectory_best) ? baseline_utility : _finite_probe_utility(subtrajectory_best["option_value"])

    local_step1_values = Float64[_finite_probe_utility(record["step1_local_utility"]) for record in family_a_records if !Bool(record["all_degenerate"])]
    local_option_values = Float64[_finite_probe_utility(record["option_value"]) for record in family_a_records if !Bool(record["all_degenerate"])]
    entry_step1_values = Float64[_finite_probe_utility(record["step1_local_utility"]) for record in family_c_records if !Bool(record["all_degenerate"])]
    entry_option_values = Float64[_finite_probe_utility(record["option_value"]) for record in family_c_records if !Bool(record["all_degenerate"])]
    entry_continuations = Float64[Float64(record["continuation_gain"]) for record in family_c_records if !Bool(record["all_degenerate"])]

    reorder_fraction = if isnothing(entry_best) || isnothing(subtrajectory_best)
        0.0
    else
        entry_best["object_id"] == subtrajectory_best["object_id"] ? 0.0 : 1.0
    end

    return Dict{String,Any}(
        "all_degenerate" => isempty(records) || Bool(get(summary, "all_degenerate", true)),
        "record_count" => length(records),
        "family_a_record_count" => length(family_a_records),
        "family_c_record_count" => length(family_c_records),
        "heuristic_baseline_utility" => baseline_utility,
        "local_object_utility" => local_object_utility,
        "entry_context_object_utility" => entry_context_utility,
        "subtrajectory_object_utility" => subtrajectory_object_utility,
        "local_gain_vs_baseline" => local_object_utility - baseline_utility,
        "entry_context_gain_vs_baseline" => entry_context_utility - baseline_utility,
        "subtrajectory_gain_vs_baseline" => subtrajectory_object_utility - baseline_utility,
        "entry_context_gain_vs_local" => entry_context_utility - local_object_utility,
        "subtrajectory_gain_vs_entry" => subtrajectory_object_utility - entry_context_utility,
        "local_regret_vs_subtrajectory" => subtrajectory_object_utility - local_object_utility,
        "entry_context_regret_vs_subtrajectory" => subtrajectory_object_utility - entry_context_utility,
        "local_surface_correlation" => _safe_probe_correlation(local_step1_values, local_option_values),
        "entry_surface_correlation" => _safe_probe_correlation(entry_step1_values, entry_option_values),
        "mean_entry_continuation_gain" => isempty(entry_continuations) ? 0.0 : mean(entry_continuations),
        "max_entry_continuation_gain" => isempty(entry_continuations) ? 0.0 : maximum(entry_continuations),
        "entry_reorder_fraction" => reorder_fraction,
        "best_basin_utility_range" => Float64(get(summary, "best_basins_utility_range", 0.0)),
        "local_best_record" => isnothing(local_best) ? Dict{String,Any}() : local_best,
        "entry_best_record" => isnothing(entry_best) ? Dict{String,Any}() : entry_best,
        "subtrajectory_best_record" => isnothing(subtrajectory_best) ? Dict{String,Any}() : subtrajectory_best,
        "records" => records,
    )
end

function _frontier_probe_median(xs::Vector{Float64})
    isempty(xs) && return 0.0
    ys = sort(xs)
    n = length(ys)
    if isodd(n)
        return ys[(n + 1) ÷ 2]
    end
    return 0.5 * (ys[n ÷ 2] + ys[n ÷ 2 + 1])
end

function _frontier_region_groups(records::Vector{Dict{String,Any}}, family_name::String)
    valid = [record for record in records if !Bool(get(record, "all_degenerate", true))]
    isempty(valid) && return Dict{String,Vector{Dict{String,Any}}}()

    groups = Dict{String,Vector{Dict{String,Any}}}()
    if family_name == "basin"
        for record in valid
            key = String(get(record, "basin_scaffold", ""))
            key = isempty(key) ? "__NO_SCAFFOLD__" : key
            push!(get!(groups, key, Dict{String,Any}[]), record)
        end
    elseif family_name == "parent_novelty"
        novelty_values = Float64[Float64(get(record, "parent_novelty_score", 0.0)) for record in valid]
        threshold = _frontier_probe_median(novelty_values)
        for record in valid
            novelty_value = Float64(get(record, "parent_novelty_score", 0.0))
            key = novelty_value >= threshold ? "novelty_high" : "novelty_low"
            push!(get!(groups, key, Dict{String,Any}[]), record)
        end
    elseif family_name == "continuation"
        for record in valid
            continuation_gain = Float64(get(record, "continuation_gain", 0.0))
            step1_value = abs(Float64(get(record, "step1_local_utility", 0.0)))
            threshold = max(0.05, 0.15 * step1_value)
            key = continuation_gain > threshold ? "continuation_sensitive" : "local_dominant"
            push!(get!(groups, key, Dict{String,Any}[]), record)
        end
    else
        error("Unknown frontier region family: $(family_name)")
    end

    return Dict(key => value for (key, value) in groups if !isempty(value))
end

function _frontier_region_summary(region_key::String,
                                  records::Vector{Dict{String,Any}})
    option_utilities = sort(Float64[_finite_probe_utility(get(record, "option_value", -Inf)) for record in records]; rev=true)
    heuristic_utilities = Float64[_finite_probe_utility(get(record, "step1_local_utility", -Inf)) for record in records]
    parent_rewards = Float64[Float64(get(record, "parent_reward", 0.0)) for record in records]
    novelty_scores = Float64[Float64(get(record, "parent_novelty_score", 0.0)) for record in records]
    continuation_gains = Float64[Float64(get(record, "continuation_gain", 0.0)) for record in records]
    return Dict{String,Any}(
        "region_key" => region_key,
        "count" => length(records),
        "heuristic_region_score" => isempty(heuristic_utilities) ? -Inf : maximum(heuristic_utilities),
        "mean_heuristic_region_score" => isempty(heuristic_utilities) ? 0.0 : mean(heuristic_utilities),
        "best_option_utility" => isempty(option_utilities) ? -Inf : maximum(option_utilities),
        "mean_option_utility" => isempty(option_utilities) ? 0.0 : mean(option_utilities),
        "mean_parent_reward" => isempty(parent_rewards) ? 0.0 : mean(parent_rewards),
        "mean_parent_novelty_score" => isempty(novelty_scores) ? 0.0 : mean(novelty_scores),
        "mean_continuation_gain" => isempty(continuation_gains) ? 0.0 : mean(continuation_gains),
        "option_utilities_sorted" => option_utilities,
        "records" => records,
    )
end

function _allocate_region_budget(region_summaries::Vector{Dict{String,Any}},
                                 policy_name::String,
                                 matched_budget::Int)
    quotas = Dict{String,Int}(String(summary["region_key"]) => 0 for summary in region_summaries)
    isempty(region_summaries) && return quotas
    matched_budget <= 0 && return quotas

    if policy_name == "heuristic_top_region"
        chosen = argmax(Float64[Float64(summary["heuristic_region_score"]) for summary in region_summaries])
        quotas[String(region_summaries[chosen]["region_key"])] = matched_budget
    elseif policy_name == "best_region"
        chosen = argmax(Float64[Float64(summary["best_option_utility"]) for summary in region_summaries])
        quotas[String(region_summaries[chosen]["region_key"])] = matched_budget
    elseif policy_name == "anti_heuristic_region"
        chosen = argmin(Float64[Float64(summary["heuristic_region_score"]) for summary in region_summaries])
        quotas[String(region_summaries[chosen]["region_key"])] = matched_budget
    elseif policy_name == "uniform_regions"
        ordered_keys = sort(String[summary["region_key"] for summary in region_summaries])
        for i in 1:matched_budget
            key = ordered_keys[1 + mod(i - 1, length(ordered_keys))]
            quotas[key] = get(quotas, key, 0) + 1
        end
    else
        error("Unknown frontier allocation policy: $(policy_name)")
    end

    return quotas
end

function _evaluate_region_allocation(region_summaries::Vector{Dict{String,Any}},
                                     quotas::Dict{String,Int})
    aggregate_utility = 0.0
    best_discovered_utility = -Inf
    used_budget = 0
    region_results = Dict{String,Any}[]

    for summary in region_summaries
        key = String(summary["region_key"])
        requested = get(quotas, key, 0)
        option_utilities = Float64[Float64(value) for value in summary["option_utilities_sorted"]]
        used = min(requested, length(option_utilities))
        selected = used == 0 ? Float64[] : option_utilities[1:used]
        aggregate_utility += sum(selected)
        if !isempty(selected)
            best_discovered_utility = max(best_discovered_utility, maximum(selected))
        end
        used_budget += used
        push!(region_results, Dict{String,Any}(
            "region_key" => key,
            "requested_budget" => requested,
            "used_budget" => used,
            "selected_option_utilities" => selected,
            "aggregate_utility" => sum(selected),
            "best_discovered_utility" => isempty(selected) ? -Inf : maximum(selected),
        ))
    end

    return Dict{String,Any}(
        "aggregate_utility" => aggregate_utility,
        "best_discovered_utility" => best_discovered_utility,
        "used_budget" => used_budget,
        "region_results" => region_results,
    )
end

function _classify_frontier_allocation_state(family_summary::Dict{String,Any};
                                             sensitivity_threshold::Float64=0.05)
    Bool(get(family_summary, "all_degenerate", false)) && return "degenerate_frontier_state"
    matched_budget = Int(get(family_summary, "matched_budget", 0))
    n_regions = Int(get(family_summary, "n_regions", 0))
    (matched_budget < 2 || n_regions < 2) && return "degenerate_frontier_state"

    best_vs_heuristic = Float64(get(family_summary, "best_vs_heuristic_gap", 0.0))
    best_vs_uniform = Float64(get(family_summary, "best_vs_uniform_gap", 0.0))
    heuristic_vs_anti = Float64(get(family_summary, "heuristic_vs_anti_gap", 0.0))
    heuristic_region = String(get(family_summary, "heuristic_top_region", ""))
    best_region = String(get(family_summary, "best_region", ""))

    if abs(best_vs_heuristic) <= sensitivity_threshold && abs(best_vs_uniform) <= sensitivity_threshold
        return heuristic_region == best_region ? "heuristic_frontier_dominant_state" : "allocation_invariant_state"
    elseif best_vs_heuristic > sensitivity_threshold && heuristic_region != best_region
        return "opportunity_routing_state"
    elseif best_vs_heuristic > sensitivity_threshold || best_vs_uniform > sensitivity_threshold
        return "allocation_sensitive_state"
    elseif heuristic_vs_anti > sensitivity_threshold && heuristic_region == best_region
        return "heuristic_frontier_dominant_state"
    else
        return "taxonomy_ambiguous_state"
    end
end

function probe_frontier_allocation_opportunities(frontier_buffer::MolecularFrontierBuffer,
                                                 reward_fn,
                                                 vocab;
                                                 reward_fn_batch=nothing,
                                                 config::HierarchicalEditConfig=HierarchicalEditConfig(),
                                                 target_smiles::Union{Nothing,String}=nothing,
                                                 budget_remaining::Int=0,
                                                 created_at_step::Int=0,
                                                 task_name::String="unknown",
                                                 max_parents::Int=4,
                                                 max_basins::Int=2,
                                                 operators::Union{Nothing,Vector{Symbol}}=nothing,
                                                 horizon::Int=3,
                                                 region_families::Vector{String}=["basin", "parent_novelty", "continuation"],
                                                 max_allocation_budget::Int=2,
                                                 sensitivity_threshold::Float64=0.05)
    option_probe = probe_coupled_hierarchy_options(frontier_buffer, reward_fn, vocab;
        reward_fn_batch=reward_fn_batch,
        config=config,
        target_smiles=target_smiles,
        budget_remaining=budget_remaining,
        created_at_step=created_at_step,
        task_name=task_name,
        max_parents=max_parents,
        max_basins=max_basins,
        operators=operators,
        horizon=horizon)

    bridge = compare_option_value_surfaces(option_probe)
    all_records = get(bridge, "records", Dict{String,Any}[])
    family_c_records = [record for record in all_records if String(get(record, "family", "")) == "C" && !Bool(get(record, "all_degenerate", true))]

    family_summaries = Dict{String,Any}[]
    state_counts = Dict{String,Int}()
    for family_name in region_families
        region_groups = _frontier_region_groups(family_c_records, family_name)
        region_summaries = [_frontier_region_summary(key, region_records) for (key, region_records) in region_groups]
        sort!(region_summaries, by=summary -> (-Float64(summary["heuristic_region_score"]), String(summary["region_key"])))

        matched_budget = isempty(region_summaries) ? 0 : min(max_allocation_budget, minimum(Int(summary["count"]) for summary in region_summaries))
        policies = Dict{String,Any}()
        if matched_budget >= 2 && length(region_summaries) >= 2
            for policy_name in ["heuristic_top_region", "uniform_regions", "best_region", "anti_heuristic_region"]
                quotas = _allocate_region_budget(region_summaries, policy_name, matched_budget)
                policies[policy_name] = merge(Dict{String,Any}("quotas" => quotas), _evaluate_region_allocation(region_summaries, quotas))
            end
        end

        heuristic_top_region = isempty(region_summaries) ? "" : String(region_summaries[argmax(Float64[Float64(summary["heuristic_region_score"]) for summary in region_summaries])]["region_key"])
        best_region = isempty(region_summaries) ? "" : String(region_summaries[argmax(Float64[Float64(summary["best_option_utility"]) for summary in region_summaries])]["region_key"])
        heuristic_eval = get(policies, "heuristic_top_region", Dict{String,Any}("aggregate_utility" => -Inf))
        uniform_eval = get(policies, "uniform_regions", Dict{String,Any}("aggregate_utility" => -Inf))
        best_eval = get(policies, "best_region", Dict{String,Any}("aggregate_utility" => -Inf))
        anti_eval = get(policies, "anti_heuristic_region", Dict{String,Any}("aggregate_utility" => -Inf))

        family_summary = Dict{String,Any}(
            "family_name" => family_name,
            "n_regions" => length(region_summaries),
            "matched_budget" => matched_budget,
            "all_degenerate" => isempty(region_summaries) || matched_budget < 2 || length(region_summaries) < 2,
            "heuristic_top_region" => heuristic_top_region,
            "best_region" => best_region,
            "heuristic_region_score" => isempty(region_summaries) ? -Inf : maximum(Float64[Float64(summary["heuristic_region_score"]) for summary in region_summaries]),
            "best_region_utility" => isempty(region_summaries) ? -Inf : maximum(Float64[Float64(summary["best_option_utility"]) for summary in region_summaries]),
            "heuristic_allocation_utility" => Float64(get(heuristic_eval, "aggregate_utility", -Inf)),
            "uniform_allocation_utility" => Float64(get(uniform_eval, "aggregate_utility", -Inf)),
            "best_allocation_utility" => Float64(get(best_eval, "aggregate_utility", -Inf)),
            "anti_heuristic_allocation_utility" => Float64(get(anti_eval, "aggregate_utility", -Inf)),
            "best_vs_heuristic_gap" => Float64(get(best_eval, "aggregate_utility", -Inf)) - Float64(get(heuristic_eval, "aggregate_utility", -Inf)),
            "best_vs_uniform_gap" => Float64(get(best_eval, "aggregate_utility", -Inf)) - Float64(get(uniform_eval, "aggregate_utility", -Inf)),
            "heuristic_vs_anti_gap" => Float64(get(heuristic_eval, "aggregate_utility", -Inf)) - Float64(get(anti_eval, "aggregate_utility", -Inf)),
            "region_opportunity_range" => isempty(region_summaries) ? 0.0 : _range_or_zero(Float64[Float64(summary["best_option_utility"]) for summary in region_summaries]),
            "region_summaries" => region_summaries,
            "policies" => policies,
        )
        family_summary["state_label"] = _classify_frontier_allocation_state(family_summary; sensitivity_threshold=sensitivity_threshold)
        state_label = String(family_summary["state_label"])
        state_counts[state_label] = get(state_counts, state_label, 0) + 1
        push!(family_summaries, family_summary)
    end

    best_family = isempty(family_summaries) ? Dict{String,Any}() : family_summaries[argmax(Float64[
        Bool(get(summary, "all_degenerate", true)) ? -Inf : Float64(get(summary, "best_vs_heuristic_gap", -Inf))
        for summary in family_summaries
    ])]

    return Dict{String,Any}(
        "task_name" => task_name,
        "snapshot_id" => String(get(option_probe, "snapshot_id", "")),
        "horizon" => horizon,
        "region_families" => region_families,
        "max_allocation_budget" => max_allocation_budget,
        "option_probe" => option_probe,
        "surface_comparison" => bridge,
        "region_family_summaries" => family_summaries,
        "summary" => Dict{String,Any}(
            "all_degenerate" => isempty(family_summaries) || all(Bool(get(summary, "all_degenerate", true)) for summary in family_summaries),
            "state_counts" => state_counts,
            "best_family_name" => isempty(best_family) ? "" : String(get(best_family, "family_name", "")),
            "best_family_state" => isempty(best_family) ? "" : String(get(best_family, "state_label", "")),
            "best_family_gap" => isempty(best_family) ? 0.0 : Float64(get(best_family, "best_vs_heuristic_gap", 0.0)),
            "local_surface_utility" => Float64(get(bridge, "local_object_utility", -Inf)),
            "entry_context_surface_utility" => Float64(get(bridge, "entry_context_object_utility", -Inf)),
            "subtrajectory_surface_utility" => Float64(get(bridge, "subtrajectory_object_utility", -Inf)),
        ),
    )
end


struct FrontierAllocationRegionRecord
    region_key::String
    features::Vector{Float32}
    heuristic_region::Bool
    winning_region::Bool
    heuristic_region_score::Float32
    best_option_utility::Float32
    allocation_utility::Float32
    label::Float32
end

struct FrontierAllocationSnapshotRecord
    snapshot_id::String
    task_name::String
    family_name::String
    features::Vector{Float32}
    override_worth_it::Bool
    heuristic_region::String
    winning_region::String
    heuristic_utility::Float32
    best_utility::Float32
    matched_budget::Int
    regions::Vector{FrontierAllocationRegionRecord}
end

struct FrontierAllocationDataset
    snapshots::Vector{FrontierAllocationSnapshotRecord}
end

Base.length(dataset::FrontierAllocationDataset) = length(dataset.snapshots)
Base.isempty(dataset::FrontierAllocationDataset) = isempty(dataset.snapshots)

struct FrontierAllocationLinearModel
    weights::Vector{Float32}
    bias::Float32
    feature_mean::Vector{Float32}
    feature_scale::Vector{Float32}
end

struct SelectiveFrontierAllocator
    family_name::String
    override_model::FrontierAllocationLinearModel
    region_model::FrontierAllocationLinearModel
    override_threshold::Float32
    margin_threshold::Float32
end

function _frontier_sigmoid32(x::Real)
    xf = Float32(x)
    xf >= 0 ? 1.0f0 / (1.0f0 + exp(-xf)) : begin
        ex = exp(xf)
        ex / (1.0f0 + ex)
    end
end

function _frontier_region_records(summary::Dict{String,Any})
    return Vector{Dict{String,Any}}(get(summary, "records", Dict{String,Any}[]))
end

function _frontier_summary_mean(summary::Dict{String,Any}, key::String)
    return Float64(get(summary, key, 0.0))
end

function _frontier_summary_record_mean(summary::Dict{String,Any}, key::String)
    records = _frontier_region_records(summary)
    isempty(records) && return 0.0
    values = Float64[Float64(get(record, key, 0.0)) for record in records]
    return isempty(values) ? 0.0 : mean(values)
end

function _frontier_region_allocation_utility(summary::Dict{String,Any}, matched_budget::Int)
    option_utilities = Float64[Float64(value) for value in get(summary, "option_utilities_sorted", Float64[])]
    isempty(option_utilities) && return -Inf
    used = min(max(matched_budget, 0), length(option_utilities))
    used <= 0 && return -Inf
    return sum(option_utilities[1:used])
end

function _frontier_sorted_region_indices(region_summaries::Vector{Dict{String,Any}}, key::String)
    values = Float64[Float64(get(summary, key, 0.0)) for summary in region_summaries]
    return sortperm(values, rev=true)
end

function _frontier_snapshot_feature_vector(region_summaries::Vector{Dict{String,Any}},
                                           heuristic_region_key::String,
                                           matched_budget::Int)
    n_regions = length(region_summaries)
    heuristic_scores = Float64[_frontier_summary_mean(summary, "heuristic_region_score") for summary in region_summaries]
    mean_rewards = Float64[_frontier_summary_mean(summary, "mean_parent_reward") for summary in region_summaries]
    novelty_scores = Float64[_frontier_summary_mean(summary, "mean_parent_novelty_score") for summary in region_summaries]
    tb_delta_scores = Float64[_frontier_summary_record_mean(summary, "parent_tb_delta_abs") for summary in region_summaries]
    counts = Float64[Float64(get(summary, "count", 0)) for summary in region_summaries]
    total_count = max(sum(counts), 1.0)

    sorted_scores = sort(copy(heuristic_scores), rev=true)
    top_score = isempty(sorted_scores) ? 0.0 : sorted_scores[1]
    second_score = length(sorted_scores) >= 2 ? sorted_scores[2] : top_score

    heuristic_summary = nothing
    for summary in region_summaries
        if String(get(summary, "region_key", "")) == heuristic_region_key
            heuristic_summary = summary
            break
        end
    end
    heuristic_summary = isnothing(heuristic_summary) && !isempty(region_summaries) ? region_summaries[1] : heuristic_summary

    heuristic_count_frac = isnothing(heuristic_summary) ? 0.0 : Float64(get(heuristic_summary, "count", 0)) / total_count
    heuristic_mean_reward = isnothing(heuristic_summary) ? 0.0 : _frontier_summary_mean(heuristic_summary, "mean_parent_reward")
    heuristic_mean_novelty = isnothing(heuristic_summary) ? 0.0 : _frontier_summary_mean(heuristic_summary, "mean_parent_novelty_score")
    heuristic_mean_tb = isnothing(heuristic_summary) ? 0.0 : _frontier_summary_record_mean(heuristic_summary, "parent_tb_delta_abs")

    return Float32[
        Float32(n_regions),
        Float32(matched_budget),
        Float32(top_score),
        Float32(second_score),
        Float32(top_score - second_score),
        Float32(isempty(heuristic_scores) ? 0.0 : mean(heuristic_scores)),
        Float32(isempty(heuristic_scores) ? 0.0 : _range_or_zero(heuristic_scores)),
        Float32(heuristic_count_frac),
        Float32(heuristic_mean_reward),
        Float32(heuristic_mean_novelty),
        Float32(heuristic_mean_tb),
        Float32(isempty(mean_rewards) ? 0.0 : mean(mean_rewards)),
        Float32(isempty(mean_rewards) ? 0.0 : _range_or_zero(mean_rewards)),
        Float32(isempty(novelty_scores) ? 0.0 : mean(novelty_scores)),
        Float32(isempty(novelty_scores) ? 0.0 : _range_or_zero(novelty_scores)),
        Float32(isempty(tb_delta_scores) ? 0.0 : mean(tb_delta_scores)),
        Float32(isempty(tb_delta_scores) ? 0.0 : _range_or_zero(tb_delta_scores)),
    ]
end

function _frontier_region_feature_vector(region_summary::Dict{String,Any},
                                         region_summaries::Vector{Dict{String,Any}},
                                         heuristic_region_key::String)
    heuristic_scores = Float64[_frontier_summary_mean(summary, "heuristic_region_score") for summary in region_summaries]
    mean_rewards = Float64[_frontier_summary_mean(summary, "mean_parent_reward") for summary in region_summaries]
    counts = Float64[Float64(get(summary, "count", 0)) for summary in region_summaries]
    total_count = max(sum(counts), 1.0)
    region_key = String(get(region_summary, "region_key", ""))

    ordered = _frontier_sorted_region_indices(region_summaries, "heuristic_region_score")
    rank_idx = findfirst(==(findfirst(summary -> String(get(summary, "region_key", "")) == region_key, region_summaries)), ordered)
    rank_fraction = isnothing(rank_idx) || isempty(region_summaries) ? 1.0 : (rank_idx - 1) / max(length(region_summaries) - 1, 1)

    top_heuristic_score = isempty(heuristic_scores) ? 0.0 : maximum(heuristic_scores)
    top_mean_reward = isempty(mean_rewards) ? 0.0 : maximum(mean_rewards)
    current_count = Float64(get(region_summary, "count", 0))
    current_heuristic = _frontier_summary_mean(region_summary, "heuristic_region_score")
    current_mean_heuristic = _frontier_summary_mean(region_summary, "mean_heuristic_region_score")
    current_mean_reward = _frontier_summary_mean(region_summary, "mean_parent_reward")
    current_mean_novelty = _frontier_summary_mean(region_summary, "mean_parent_novelty_score")
    current_mean_tb = _frontier_summary_record_mean(region_summary, "parent_tb_delta_abs")
    current_mean_basin = _frontier_summary_record_mean(region_summary, "basin_score")
    heuristic_match = region_key == heuristic_region_key ? 1.0 : 0.0

    return Float32[
        Float32(current_count),
        Float32(current_count / total_count),
        Float32(current_heuristic),
        Float32(current_mean_heuristic),
        Float32(current_mean_reward),
        Float32(current_mean_novelty),
        Float32(current_mean_tb),
        Float32(current_mean_basin),
        Float32(rank_fraction),
        Float32(top_heuristic_score - current_heuristic),
        Float32(top_mean_reward - current_mean_reward),
        Float32(heuristic_match),
    ]
end

function extract_frontier_allocation_dataset(probe_runs::Vector{<:AbstractDict};
                                             family_name::String="basin",
                                             override_threshold::Float64=0.01)
    snapshots = FrontierAllocationSnapshotRecord[]
    for run in probe_runs
        probe = get(run, "probe", run)
        family_summary = nothing
        for summary in get(probe, "region_family_summaries", Dict{String,Any}[])
            if String(get(summary, "family_name", "")) == family_name
                family_summary = summary
                break
            end
        end
        isnothing(family_summary) && continue
        Bool(get(family_summary, "all_degenerate", true)) && continue
        region_summaries = Vector{Dict{String,Any}}(get(family_summary, "region_summaries", Dict{String,Any}[]))
        matched_budget = Int(get(family_summary, "matched_budget", 0))
        if matched_budget < 2 || length(region_summaries) < 2
            continue
        end

        heuristic_region = String(get(family_summary, "heuristic_top_region", ""))
        winning_region = String(get(family_summary, "best_region", ""))
        heuristic_utility = Float32(_finite_probe_utility(get(family_summary, "heuristic_allocation_utility", -Inf)))
        best_utility = Float32(_finite_probe_utility(get(family_summary, "best_allocation_utility", -Inf)))
        override_gap = Float64(get(family_summary, "best_vs_heuristic_gap", 0.0))
        override_worth_it = winning_region != "" && heuristic_region != winning_region && override_gap > override_threshold

        snapshot_features = _frontier_snapshot_feature_vector(region_summaries, heuristic_region, matched_budget)
        regions = FrontierAllocationRegionRecord[]
        for region_summary in region_summaries
            region_key = String(get(region_summary, "region_key", ""))
            region_features = _frontier_region_feature_vector(region_summary, region_summaries, heuristic_region)
            best_option_utility = Float32(_finite_probe_utility(get(region_summary, "best_option_utility", -Inf)))
            allocation_utility = Float32(_finite_probe_utility(_frontier_region_allocation_utility(region_summary, matched_budget)))
            push!(regions, FrontierAllocationRegionRecord(
                region_key,
                region_features,
                region_key == heuristic_region,
                region_key == winning_region,
                Float32(_frontier_summary_mean(region_summary, "heuristic_region_score")),
                best_option_utility,
                allocation_utility,
                region_key == winning_region ? 1.0f0 : 0.0f0,
            ))
        end

        push!(snapshots, FrontierAllocationSnapshotRecord(
            String(get(probe, "snapshot_id", "")),
            String(get(probe, "task_name", "unknown")),
            family_name,
            snapshot_features,
            override_worth_it,
            heuristic_region,
            winning_region,
            heuristic_utility,
            best_utility,
            matched_budget,
            regions,
        ))
    end
    return FrontierAllocationDataset(snapshots)
end

function frontier_allocation_dataset_stats(dataset::FrontierAllocationDataset)
    isempty(dataset) && return Dict{String,Any}(
        "size" => 0,
        "override_positive_fraction" => 0.0,
        "mean_region_count" => 0.0,
        "snapshot_feature_dim" => 0,
        "region_feature_dim" => 0,
    )

    region_counts = Float64[length(snapshot.regions) for snapshot in dataset.snapshots]
    positive = count(snapshot -> snapshot.override_worth_it, dataset.snapshots)
    snapshot_dim = length(dataset.snapshots[1].features)
    region_dim = isempty(dataset.snapshots[1].regions) ? 0 : length(dataset.snapshots[1].regions[1].features)
    return Dict{String,Any}(
        "size" => length(dataset),
        "override_positive_fraction" => positive / length(dataset),
        "mean_region_count" => mean(region_counts),
        "snapshot_feature_dim" => snapshot_dim,
        "region_feature_dim" => region_dim,
    )
end

function _fit_frontier_linear_model(features::Vector{Vector{Float32}},
                                    targets::Vector{Float32};
                                    lambda::Float64=0.1)
    isempty(features) && return FrontierAllocationLinearModel(Float32[], 0.0f0, Float32[], Float32[])
    n = length(features)
    d = length(features[1])
    X = zeros(Float32, n, d)
    for (i, x) in enumerate(features)
        X[i, :] = Float32.(x)
    end
    y = Float32.(targets)
    mean_vec = vec(mean(X; dims=1))
    scale_vec = vec(std(X; dims=1, corrected=false))
    scale_vec = Float32[(value > 1f-6 ? value : 1.0f0) for value in scale_vec]
    Xn = similar(X)
    for j in 1:d
        Xn[:, j] .= (X[:, j] .- mean_vec[j]) ./ scale_vec[j]
    end
    Xaug = hcat(Xn, ones(Float32, n))
    reg = Matrix{Float32}(I, d + 1, d + 1)
    reg[end, end] = 0.0f0
    β = (Xaug' * Xaug + Float32(lambda) * reg) \ (Xaug' * y)
    return FrontierAllocationLinearModel(Float32.(β[1:d]), Float32(β[end]), mean_vec, scale_vec)
end

function _frontier_linear_score(model::FrontierAllocationLinearModel,
                                features::AbstractVector{<:Real})
    isempty(model.weights) && return 0.0f0
    x = Float32.(features)
    xn = similar(x)
    for i in eachindex(x)
        scale = i <= length(model.feature_scale) ? model.feature_scale[i] : 1.0f0
        mean_value = i <= length(model.feature_mean) ? model.feature_mean[i] : 0.0f0
        xn[i] = (x[i] - mean_value) / max(scale, 1f-6)
    end
    return dot(model.weights, xn) + model.bias
end

function frontier_allocation_override_score(policy::SelectiveFrontierAllocator,
                                            features::AbstractVector{<:Real})
    return _frontier_sigmoid32(_frontier_linear_score(policy.override_model, features))
end

function frontier_allocation_region_score(policy::SelectiveFrontierAllocator,
                                          features::AbstractVector{<:Real})
    return _frontier_linear_score(policy.region_model, features)
end

function _split_frontier_allocation_dataset(dataset::FrontierAllocationDataset;
                                            train_fraction::Float64=0.8,
                                            rng::AbstractRNG=Random.MersenneTwister(0))
    n = length(dataset)
    n == 0 && return FrontierAllocationDataset(FrontierAllocationSnapshotRecord[]), FrontierAllocationDataset(FrontierAllocationSnapshotRecord[])
    if n == 1
        return dataset, dataset
    end
    idx = collect(1:n)
    shuffle!(rng, idx)
    n_train = clamp(round(Int, train_fraction * n), 1, n - 1)
    train_idx = idx[1:n_train]
    val_idx = idx[(n_train + 1):end]
    train = FrontierAllocationDataset(dataset.snapshots[train_idx])
    val = FrontierAllocationDataset(dataset.snapshots[val_idx])
    isempty(val) && (val = train)
    return train, val
end

function _choose_frontier_region(policy::SelectiveFrontierAllocator,
                                 snapshot::FrontierAllocationSnapshotRecord;
                                 mode::Symbol=:selective_override)
    heuristic_region = snapshot.heuristic_region
    best_region = heuristic_region
    best_score = -Inf32
    heuristic_score = -Inf32
    region_scores = Dict{String,Float32}()
    region_utilities = Dict{String,Float32}(region.region_key => region.allocation_utility for region in snapshot.regions)

    for region in snapshot.regions
        score = frontier_allocation_region_score(policy, region.features)
        region_scores[region.region_key] = score
        if region.region_key == heuristic_region
            heuristic_score = score
        end
        if score > best_score
            best_score = score
            best_region = region.region_key
        end
    end
    if !haskey(region_scores, heuristic_region)
        heuristic_score = best_score
    end
    predicted_margin = best_score - heuristic_score
    override_score = frontier_allocation_override_score(policy, snapshot.features)

    override = if mode == :always_override
        best_region != heuristic_region
    elseif mode == :selective_override
        override_score >= policy.override_threshold && best_region != heuristic_region && predicted_margin > policy.margin_threshold
    else
        false
    end
    selected_region = override ? best_region : heuristic_region
    selected_utility = get(region_utilities, selected_region, snapshot.heuristic_utility)

    return Dict{String,Any}(
        "selected_region" => selected_region,
        "selected_utility" => selected_utility,
        "predicted_region" => best_region,
        "predicted_region_score" => best_score,
        "heuristic_region_score_pred" => heuristic_score,
        "predicted_margin" => predicted_margin,
        "override_score" => override_score,
        "override_applied" => override,
        "region_scores" => region_scores,
    )
end

function evaluate_selective_frontier_allocator(policy::Union{SelectiveFrontierAllocator,Nothing},
                                               dataset::FrontierAllocationDataset;
                                               mode::Symbol=:selective_override)
    isempty(dataset) && return Dict{String,Any}(
        "n_snapshots" => 0,
        "mean_gain_vs_heuristic" => 0.0,
        "mean_regret_vs_best_region" => 0.0,
        "override_fraction" => 0.0,
        "abstention_fraction" => 0.0,
        "override_precision" => 0.0,
        "override_recall" => 0.0,
        "basin_choice_accuracy" => 0.0,
        "heuristic_match_fraction" => 0.0,
        "per_snapshot" => Dict{String,Any}[],
    )

    per_snapshot = Dict{String,Any}[]
    gains = Float64[]
    regrets = Float64[]
    override_scores = Float64[]
    margins = Float64[]
    override_count = 0
    abstain_count = 0
    basin_hits = 0
    heuristic_matches = 0
    tp = 0
    fp = 0
    fn = 0

    for snapshot in dataset.snapshots
        decision = if mode == :heuristic_only
            Dict{String,Any}(
                "selected_region" => snapshot.heuristic_region,
                "selected_utility" => snapshot.heuristic_utility,
                "predicted_region" => snapshot.heuristic_region,
                "predicted_region_score" => 0.0f0,
                "heuristic_region_score_pred" => 0.0f0,
                "predicted_margin" => 0.0f0,
                "override_score" => 0.0f0,
                "override_applied" => false,
            )
        elseif mode == :oracle_best_region
            Dict{String,Any}(
                "selected_region" => snapshot.winning_region,
                "selected_utility" => snapshot.best_utility,
                "predicted_region" => snapshot.winning_region,
                "predicted_region_score" => 1.0f0,
                "heuristic_region_score_pred" => 0.0f0,
                "predicted_margin" => 1.0f0,
                "override_score" => snapshot.override_worth_it ? 1.0f0 : 0.0f0,
                "override_applied" => snapshot.override_worth_it,
            )
        else
            isnothing(policy) && error("learned policy required for mode $(mode)")
            _choose_frontier_region(policy, snapshot; mode=mode)
        end

        selected_region = String(decision["selected_region"])
        selected_utility = Float64(decision["selected_utility"])
        gain = selected_utility - Float64(snapshot.heuristic_utility)
        regret = Float64(snapshot.best_utility) - selected_utility
        override = Bool(get(decision, "override_applied", false))
        predicted_region = String(get(decision, "predicted_region", snapshot.heuristic_region))
        basin_hit = predicted_region == snapshot.winning_region

        push!(gains, gain)
        push!(regrets, regret)
        push!(override_scores, Float64(get(decision, "override_score", 0.0)))
        push!(margins, Float64(get(decision, "predicted_margin", 0.0)))
        override_count += override ? 1 : 0
        abstain_count += override ? 0 : 1
        basin_hits += basin_hit ? 1 : 0
        heuristic_matches += selected_region == snapshot.heuristic_region ? 1 : 0
        if override
            if snapshot.override_worth_it
                tp += 1
            else
                fp += 1
            end
        elseif snapshot.override_worth_it
            fn += 1
        end

        push!(per_snapshot, Dict{String,Any}(
            "snapshot_id" => snapshot.snapshot_id,
            "task_name" => snapshot.task_name,
            "override_target" => snapshot.override_worth_it,
            "selected_region" => selected_region,
            "predicted_region" => predicted_region,
            "heuristic_region" => snapshot.heuristic_region,
            "winning_region" => snapshot.winning_region,
            "selected_utility" => selected_utility,
            "heuristic_utility" => Float64(snapshot.heuristic_utility),
            "best_utility" => Float64(snapshot.best_utility),
            "gain_vs_heuristic" => gain,
            "regret_vs_best_region" => regret,
            "override_applied" => override,
            "override_score" => Float64(get(decision, "override_score", 0.0)),
            "predicted_margin" => Float64(get(decision, "predicted_margin", 0.0)),
        ))
    end

    n = length(dataset)
    return Dict{String,Any}(
        "n_snapshots" => n,
        "mean_gain_vs_heuristic" => isempty(gains) ? 0.0 : mean(gains),
        "mean_regret_vs_best_region" => isempty(regrets) ? 0.0 : mean(regrets),
        "override_fraction" => n == 0 ? 0.0 : override_count / n,
        "abstention_fraction" => n == 0 ? 0.0 : abstain_count / n,
        "override_precision" => tp + fp == 0 ? 0.0 : tp / (tp + fp),
        "override_recall" => tp + fn == 0 ? 0.0 : tp / (tp + fn),
        "basin_choice_accuracy" => n == 0 ? 0.0 : basin_hits / n,
        "heuristic_match_fraction" => n == 0 ? 0.0 : heuristic_matches / n,
        "mean_override_score" => isempty(override_scores) ? 0.0 : mean(override_scores),
        "mean_predicted_margin" => isempty(margins) ? 0.0 : mean(margins),
        "per_snapshot" => per_snapshot,
    )
end

function train_selective_frontier_allocator(dataset::FrontierAllocationDataset;
                                            rng::AbstractRNG=Random.MersenneTwister(0),
                                            train_fraction::Float64=0.8,
                                            family_name::String="basin",
                                            ridge_lambda::Float64=0.1,
                                            override_thresholds::Vector{Float64}=collect(0.35:0.05:0.70),
                                            margin_thresholds::Vector{Float64}=[0.0, 0.01, 0.02, 0.05])
    stats = frontier_allocation_dataset_stats(dataset)
    if length(dataset) < 2
        return Dict{String,Any}(
            "model" => nothing,
            "dataset_stats" => stats,
            "reason" => "insufficient_snapshots",
        )
    end

    train_dataset, val_dataset = _split_frontier_allocation_dataset(dataset; train_fraction=train_fraction, rng=rng)
    snapshot_features = [snapshot.features for snapshot in train_dataset.snapshots]
    override_targets = Float32[snapshot.override_worth_it ? 1.0f0 : 0.0f0 for snapshot in train_dataset.snapshots]
    region_features = Vector{Float32}[]
    region_targets = Float32[]
    for snapshot in train_dataset.snapshots
        for region in snapshot.regions
            push!(region_features, region.features)
            push!(region_targets, region.label)
        end
    end

    override_model = _fit_frontier_linear_model(snapshot_features, override_targets; lambda=ridge_lambda)
    region_model = _fit_frontier_linear_model(region_features, region_targets; lambda=ridge_lambda)

    best_policy = nothing
    best_objective = -Inf
    best_threshold = 0.5f0
    best_margin = 0.0f0
    best_train_eval = Dict{String,Any}()

    for threshold in override_thresholds
        for margin in margin_thresholds
            policy = SelectiveFrontierAllocator(
                family_name,
                override_model,
                region_model,
                Float32(threshold),
                Float32(margin),
            )
            eval = evaluate_selective_frontier_allocator(policy, train_dataset; mode=:selective_override)
            override_fraction = Float64(get(eval, "override_fraction", 0.0))
            objective = Float64(get(eval, "mean_gain_vs_heuristic", 0.0)) - 0.25 * Float64(get(eval, "mean_regret_vs_best_region", 0.0))
            if 0.05 <= override_fraction <= 0.95
                objective += 1e-3
            end
            if objective > best_objective
                best_objective = objective
                best_policy = policy
                best_threshold = Float32(threshold)
                best_margin = Float32(margin)
                best_train_eval = eval
            end
        end
    end

    best_policy === nothing && (best_policy = SelectiveFrontierAllocator(family_name, override_model, region_model, best_threshold, best_margin))

    val_eval = evaluate_selective_frontier_allocator(best_policy, val_dataset; mode=:selective_override)
    heuristic_eval = evaluate_selective_frontier_allocator(nothing, val_dataset; mode=:heuristic_only)
    always_eval = evaluate_selective_frontier_allocator(best_policy, val_dataset; mode=:always_override)
    oracle_eval = evaluate_selective_frontier_allocator(nothing, val_dataset; mode=:oracle_best_region)
    final_eval = evaluate_selective_frontier_allocator(best_policy, dataset; mode=:selective_override)

    return Dict{String,Any}(
        "model" => best_policy,
        "dataset_stats" => stats,
        "train_dataset" => train_dataset,
        "val_dataset" => val_dataset,
        "train_eval" => best_train_eval,
        "val_eval" => val_eval,
        "heuristic_val_eval" => heuristic_eval,
        "always_override_val_eval" => always_eval,
        "oracle_val_eval" => oracle_eval,
        "final_eval" => final_eval,
        "override_threshold" => best_threshold,
        "margin_threshold" => best_margin,
    )
end


struct OpportunityStateRecord
    snapshot_id::String
    task_name::String
    family_name::String
    features::Vector{Float32}
    override_eligible::Bool
    regime_label::String
    heuristic_region::String
    winning_region::String
    best_vs_heuristic_gap::Float32
    best_vs_uniform_gap::Float32
    heuristic_vs_anti_gap::Float32
    region_opportunity_range::Float32
    heuristic_utility::Float32
    best_utility::Float32
    matched_budget::Int
end

struct OpportunityStateDataset
    records::Vector{OpportunityStateRecord}
end

Base.length(dataset::OpportunityStateDataset) = length(dataset.records)
Base.isempty(dataset::OpportunityStateDataset) = isempty(dataset.records)

struct OpportunityStateDetector
    model::FrontierAllocationLinearModel
    threshold::Float32
    family_name::String
end

function _opportunity_state_label(family_summary::Dict{String,Any};
                                  state_threshold::Float64=0.01)
    if Bool(get(family_summary, "all_degenerate", true))
        return false, "invariant_or_ambiguous_state"
    end
    heuristic_region = String(get(family_summary, "heuristic_top_region", ""))
    best_region = String(get(family_summary, "best_region", ""))
    best_vs_heuristic = Float64(get(family_summary, "best_vs_heuristic_gap", 0.0))
    heuristic_vs_anti = Float64(get(family_summary, "heuristic_vs_anti_gap", 0.0))

    if best_region != "" && best_region != heuristic_region && best_vs_heuristic > state_threshold
        return true, "routing_sensitive_state"
    elseif heuristic_region != "" && (best_region == heuristic_region || (abs(best_vs_heuristic) <= state_threshold && heuristic_vs_anti > state_threshold))
        return false, "heuristic_dominant_state"
    else
        return false, "invariant_or_ambiguous_state"
    end
end

function extract_opportunity_state_dataset(probe_runs::Vector{<:AbstractDict};
                                           family_name::String="basin",
                                           state_threshold::Float64=0.01)
    records = OpportunityStateRecord[]
    for run in probe_runs
        probe = get(run, "probe", run)
        family_summary = nothing
        for summary in get(probe, "region_family_summaries", Dict{String,Any}[])
            if String(get(summary, "family_name", "")) == family_name
                family_summary = summary
                break
            end
        end
        isnothing(family_summary) && continue
        region_summaries = Vector{Dict{String,Any}}(get(family_summary, "region_summaries", Dict{String,Any}[]))
        matched_budget = Int(get(family_summary, "matched_budget", 0))
        isempty(region_summaries) && continue

        heuristic_region = String(get(family_summary, "heuristic_top_region", ""))
        features = _frontier_snapshot_feature_vector(region_summaries, heuristic_region, matched_budget)
        override_eligible, regime_label = _opportunity_state_label(family_summary; state_threshold=state_threshold)
        push!(records, OpportunityStateRecord(
            String(get(probe, "snapshot_id", "")),
            String(get(probe, "task_name", "unknown")),
            family_name,
            features,
            override_eligible,
            regime_label,
            heuristic_region,
            String(get(family_summary, "best_region", "")),
            Float32(get(family_summary, "best_vs_heuristic_gap", 0.0)),
            Float32(get(family_summary, "best_vs_uniform_gap", 0.0)),
            Float32(get(family_summary, "heuristic_vs_anti_gap", 0.0)),
            Float32(get(family_summary, "region_opportunity_range", 0.0)),
            Float32(_finite_probe_utility(get(family_summary, "heuristic_allocation_utility", -Inf))),
            Float32(_finite_probe_utility(get(family_summary, "best_allocation_utility", -Inf))),
            matched_budget,
        ))
    end
    return OpportunityStateDataset(records)
end

function opportunity_state_dataset_stats(dataset::OpportunityStateDataset)
    isempty(dataset) && return Dict{String,Any}(
        "size" => 0,
        "override_eligible_fraction" => 0.0,
        "feature_dim" => 0,
        "regime_counts" => Dict{String,Int}(),
    )
    positives = count(record -> record.override_eligible, dataset.records)
    regime_counts = Dict{String,Int}()
    for record in dataset.records
        regime_counts[record.regime_label] = get(regime_counts, record.regime_label, 0) + 1
    end
    return Dict{String,Any}(
        "size" => length(dataset),
        "override_eligible_fraction" => positives / length(dataset),
        "feature_dim" => length(dataset.records[1].features),
        "regime_counts" => regime_counts,
    )
end

function opportunity_state_score(detector::OpportunityStateDetector,
                                 features::AbstractVector{<:Real})
    return _frontier_sigmoid32(_frontier_linear_score(detector.model, features))
end

function _opportunity_state_split_indices(n::Int;
                                          train_fraction::Float64=0.8,
                                          num_splits::Int=1,
                                          rng::AbstractRNG=Random.MersenneTwister(0))
    split_count = max(num_splits, 1)
    if n <= 0
        return Tuple{Vector{Int},Vector{Int}}[]
    elseif n == 1
        return [([1], [1]) for _ in 1:split_count]
    end

    splits = Tuple{Vector{Int},Vector{Int}}[]
    for _ in 1:split_count
        idx = collect(1:n)
        shuffle!(rng, idx)
        n_train = clamp(round(Int, train_fraction * n), 1, n - 1)
        train_idx = sort(idx[1:n_train])
        val_idx = sort(idx[(n_train + 1):end])
        isempty(val_idx) && (val_idx = copy(train_idx))
        push!(splits, (train_idx, val_idx))
    end
    return splits
end

function _split_opportunity_state_dataset(dataset::OpportunityStateDataset;
                                          train_fraction::Float64=0.8,
                                          rng::AbstractRNG=Random.MersenneTwister(0))
    splits = _opportunity_state_split_indices(length(dataset);
        train_fraction=train_fraction,
        num_splits=1,
        rng=rng)
    isempty(splits) && return OpportunityStateDataset(OpportunityStateRecord[]), OpportunityStateDataset(OpportunityStateRecord[])
    train_idx, val_idx = splits[1]
    train = OpportunityStateDataset(dataset.records[train_idx])
    val = OpportunityStateDataset(dataset.records[val_idx])
    isempty(val) && (val = train)
    return train, val
end

function evaluate_opportunity_state_detector(detector::Union{OpportunityStateDetector,Nothing},
                                             dataset::OpportunityStateDataset;
                                             mode::Symbol=:learned)
    isempty(dataset) && return Dict{String,Any}(
        "n_snapshots" => 0,
        "predicted_positive_fraction" => 0.0,
        "precision" => 0.0,
        "recall" => 0.0,
        "mean_gap_predicted_positive" => 0.0,
        "mean_gap_predicted_negative" => 0.0,
        "gap_separation" => 0.0,
        "routing_sensitive_fraction_in_predicted_positive" => 0.0,
        "heuristic_dominant_fraction_in_predicted_negative" => 0.0,
        "mean_score" => 0.0,
        "zero_positive_collapse" => false,
        "all_positive_collapse" => false,
        "per_snapshot" => Dict{String,Any}[],
    )

    per_snapshot = Dict{String,Any}[]
    tp = 0
    fp = 0
    fn = 0
    predicted_positive = 0
    scores = Float64[]
    positive_gaps = Float64[]
    negative_gaps = Float64[]
    positive_routing_sensitive = 0
    negative_heuristic_dominant = 0
    negative_count = 0

    for record in dataset.records
        score = if mode == :oracle
            record.override_eligible ? 1.0f0 : 0.0f0
        elseif mode == :always_positive
            1.0f0
        elseif mode == :always_negative
            0.0f0
        else
            isnothing(detector) && error("detector required for learned evaluation")
            opportunity_state_score(detector, record.features)
        end
        predicted = if mode == :always_positive
            true
        elseif mode == :always_negative
            false
        elseif mode == :oracle
            record.override_eligible
        else
            score >= detector.threshold
        end

        gap = Float64(record.best_vs_heuristic_gap)
        push!(scores, Float64(score))
        if predicted
            predicted_positive += 1
            push!(positive_gaps, gap)
            positive_routing_sensitive += record.regime_label == "routing_sensitive_state" ? 1 : 0
            if record.override_eligible
                tp += 1
            else
                fp += 1
            end
        else
            negative_count += 1
            push!(negative_gaps, gap)
            negative_heuristic_dominant += record.regime_label == "heuristic_dominant_state" ? 1 : 0
            if record.override_eligible
                fn += 1
            end
        end

        push!(per_snapshot, Dict{String,Any}(
            "snapshot_id" => record.snapshot_id,
            "task_name" => record.task_name,
            "override_eligible" => record.override_eligible,
            "regime_label" => record.regime_label,
            "predicted_positive" => predicted,
            "score" => Float64(score),
            "best_vs_heuristic_gap" => gap,
            "region_opportunity_range" => Float64(record.region_opportunity_range),
            "heuristic_region" => record.heuristic_region,
            "winning_region" => record.winning_region,
        ))
    end

    n = length(dataset)
    mean_gap_positive = isempty(positive_gaps) ? 0.0 : mean(positive_gaps)
    mean_gap_negative = isempty(negative_gaps) ? 0.0 : mean(negative_gaps)
    return Dict{String,Any}(
        "n_snapshots" => n,
        "predicted_positive_fraction" => n == 0 ? 0.0 : predicted_positive / n,
        "precision" => tp + fp == 0 ? 0.0 : tp / (tp + fp),
        "recall" => tp + fn == 0 ? 0.0 : tp / (tp + fn),
        "mean_gap_predicted_positive" => mean_gap_positive,
        "mean_gap_predicted_negative" => mean_gap_negative,
        "gap_separation" => mean_gap_positive - mean_gap_negative,
        "routing_sensitive_fraction_in_predicted_positive" => predicted_positive == 0 ? 0.0 : positive_routing_sensitive / predicted_positive,
        "heuristic_dominant_fraction_in_predicted_negative" => negative_count == 0 ? 0.0 : negative_heuristic_dominant / negative_count,
        "mean_score" => isempty(scores) ? 0.0 : mean(scores),
        "zero_positive_collapse" => predicted_positive == 0,
        "all_positive_collapse" => predicted_positive == n,
        "per_snapshot" => per_snapshot,
    )
end

function evaluate_opportunity_state_conditional_oracle(detector::Union{OpportunityStateDetector,Nothing},
                                                       dataset::OpportunityStateDataset;
                                                       mode::Symbol=:learned)
    isempty(dataset) && return Dict{String,Any}(
        "n_snapshots" => 0,
        "mean_gain_vs_heuristic" => 0.0,
        "mean_regret_vs_best_region" => 0.0,
        "oracle_on_positive_fraction" => 0.0,
        "per_snapshot" => Dict{String,Any}[],
    )

    gains = Float64[]
    regrets = Float64[]
    positives = 0
    per_snapshot = Dict{String,Any}[]
    for record in dataset.records
        score = if mode == :oracle
            record.override_eligible ? 1.0f0 : 0.0f0
        elseif mode == :always_positive
            1.0f0
        elseif mode == :always_negative
            0.0f0
        else
            isnothing(detector) && error("detector required for learned evaluation")
            opportunity_state_score(detector, record.features)
        end
        predicted_positive = if mode == :always_positive
            true
        elseif mode == :always_negative
            false
        elseif mode == :oracle
            record.override_eligible
        else
            score >= detector.threshold
        end
        selected_utility = predicted_positive ? Float64(record.best_utility) : Float64(record.heuristic_utility)
        gain = selected_utility - Float64(record.heuristic_utility)
        regret = Float64(record.best_utility) - selected_utility
        positives += predicted_positive ? 1 : 0
        push!(gains, gain)
        push!(regrets, regret)
        push!(per_snapshot, Dict{String,Any}(
            "snapshot_id" => record.snapshot_id,
            "predicted_positive" => predicted_positive,
            "score" => Float64(score),
            "gain_vs_heuristic" => gain,
            "regret_vs_best_region" => regret,
            "override_eligible" => record.override_eligible,
            "regime_label" => record.regime_label,
        ))
    end

    n = length(dataset)
    return Dict{String,Any}(
        "n_snapshots" => n,
        "mean_gain_vs_heuristic" => isempty(gains) ? 0.0 : mean(gains),
        "mean_regret_vs_best_region" => isempty(regrets) ? 0.0 : mean(regrets),
        "oracle_on_positive_fraction" => n == 0 ? 0.0 : positives / n,
        "per_snapshot" => per_snapshot,
    )
end

function _opportunity_state_objective(eval::Dict{String,Any})
    positive_fraction = Float64(get(eval, "predicted_positive_fraction", 0.0))
    objective = Float64(get(eval, "precision", 0.0)) + Float64(get(eval, "recall", 0.0))
    objective += Float64(get(eval, "gap_separation", 0.0))
    if 0.05 <= positive_fraction <= 0.95
        objective += 1e-3
    end
    return objective
end

function _fit_opportunity_state_detector(train_dataset::OpportunityStateDataset;
                                         family_name::String="basin",
                                         ridge_lambda::Float64=0.1,
                                         thresholds::Vector{Float64}=collect(0.35:0.05:0.70))
    features = [record.features for record in train_dataset.records]
    targets = Float32[record.override_eligible ? 1.0f0 : 0.0f0 for record in train_dataset.records]
    model = _fit_frontier_linear_model(features, targets; lambda=ridge_lambda)

    best_detector = nothing
    best_objective = -Inf
    best_train_eval = Dict{String,Any}()
    best_threshold = 0.5f0
    threshold_search = Dict{String,Any}[]
    for threshold in thresholds
        detector = OpportunityStateDetector(model, Float32(threshold), family_name)
        eval = evaluate_opportunity_state_detector(detector, train_dataset)
        objective = _opportunity_state_objective(eval)
        push!(threshold_search, Dict{String,Any}(
            "threshold" => Float64(threshold),
            "objective" => objective,
            "predicted_positive_fraction" => Float64(get(eval, "predicted_positive_fraction", 0.0)),
            "gap_separation" => Float64(get(eval, "gap_separation", 0.0)),
            "precision" => Float64(get(eval, "precision", 0.0)),
            "recall" => Float64(get(eval, "recall", 0.0)),
        ))
        if objective > best_objective
            best_objective = objective
            best_detector = detector
            best_train_eval = eval
            best_threshold = Float32(threshold)
        end
    end
    best_detector === nothing && (best_detector = OpportunityStateDetector(model, best_threshold, family_name))

    return Dict{String,Any}(
        "detector" => best_detector,
        "train_eval" => best_train_eval,
        "threshold" => best_threshold,
        "threshold_search" => threshold_search,
        "objective" => best_objective,
    )
end

function opportunity_state_threshold_stability(detector::OpportunityStateDetector,
                                               dataset::OpportunityStateDataset;
                                               threshold_perturbations::Vector{Float64}=[-0.05, 0.0, 0.05])
    isempty(dataset) && return Dict{String,Any}(
        "n_thresholds" => 0,
        "nondegenerate_fraction" => 0.0,
        "positive_gap_fraction" => 0.0,
        "nonnegative_conditional_gain_fraction" => 0.0,
        "robust_fraction" => 0.0,
        "per_threshold" => Dict{String,Any}[],
    )

    per_threshold = Dict{String,Any}[]
    nondegenerate_count = 0
    positive_gap_count = 0
    nonnegative_conditional_count = 0
    robust_count = 0
    for delta in threshold_perturbations
        threshold = clamp(Float64(detector.threshold) + Float64(delta), 0.0, 1.0)
        perturbed = OpportunityStateDetector(detector.model, Float32(threshold), detector.family_name)
        eval = evaluate_opportunity_state_detector(perturbed, dataset)
        cond_eval = evaluate_opportunity_state_conditional_oracle(perturbed, dataset)
        gap_separation = Float64(get(eval, "gap_separation", 0.0))
        cond_gain = Float64(get(cond_eval, "mean_gain_vs_heuristic", 0.0))
        nondegenerate = !(Bool(get(eval, "zero_positive_collapse", false)) || Bool(get(eval, "all_positive_collapse", false)))
        robust = nondegenerate && gap_separation > 0.0 && cond_gain >= 0.0
        nondegenerate_count += nondegenerate ? 1 : 0
        positive_gap_count += gap_separation > 0.0 ? 1 : 0
        nonnegative_conditional_count += cond_gain >= 0.0 ? 1 : 0
        robust_count += robust ? 1 : 0
        push!(per_threshold, Dict{String,Any}(
            "threshold" => threshold,
            "delta" => Float64(delta),
            "predicted_positive_fraction" => Float64(get(eval, "predicted_positive_fraction", 0.0)),
            "gap_separation" => gap_separation,
            "conditional_gain" => cond_gain,
            "nondegenerate" => nondegenerate,
            "robust" => robust,
        ))
    end

    n = length(per_threshold)
    return Dict{String,Any}(
        "n_thresholds" => n,
        "nondegenerate_fraction" => n == 0 ? 0.0 : nondegenerate_count / n,
        "positive_gap_fraction" => n == 0 ? 0.0 : positive_gap_count / n,
        "nonnegative_conditional_gain_fraction" => n == 0 ? 0.0 : nonnegative_conditional_count / n,
        "robust_fraction" => n == 0 ? 0.0 : robust_count / n,
        "per_threshold" => per_threshold,
    )
end

function evaluate_opportunity_state_repeatability(dataset::OpportunityStateDataset;
                                                  rng::AbstractRNG=Random.MersenneTwister(0),
                                                  num_splits::Int=5,
                                                  train_fraction::Float64=0.75,
                                                  family_name::String="basin",
                                                  ridge_lambda::Float64=0.1,
                                                  thresholds::Vector{Float64}=collect(0.35:0.05:0.70),
                                                  threshold_perturbations::Vector{Float64}=[-0.05, 0.0, 0.05])
    stats = opportunity_state_dataset_stats(dataset)
    if length(dataset) < 2
        return Dict{String,Any}(
            "dataset_stats" => stats,
            "splits" => Dict{String,Any}[],
            "summary" => Dict{String,Any}(
                "n_splits" => 0,
                "repeatability_safe" => false,
                "recommendation" => "STATE_SPLIT_NOT_HELD_OUT_SAFE",
            ),
            "reason" => "insufficient_snapshots",
        )
    end

    split_indices = _opportunity_state_split_indices(length(dataset);
        train_fraction=train_fraction,
        num_splits=num_splits,
        rng=rng)

    split_summaries = Dict{String,Any}[]
    thresholds_used = Float64[]
    predicted_positive_fracs = Float64[]
    gap_separations = Float64[]
    conditional_gains = Float64[]
    routing_fracs = Float64[]
    heuristic_fracs = Float64[]
    robust_fracs = Float64[]
    nondegenerate_count = 0
    zero_positive_count = 0
    all_positive_count = 0

    for (split_idx, (train_idx, val_idx)) in enumerate(split_indices)
        train_dataset = OpportunityStateDataset(dataset.records[train_idx])
        val_dataset = OpportunityStateDataset(dataset.records[val_idx])
        isempty(val_dataset) && (val_dataset = train_dataset)

        fit = _fit_opportunity_state_detector(train_dataset;
            family_name=family_name,
            ridge_lambda=ridge_lambda,
            thresholds=thresholds)
        detector = fit["detector"]
        val_eval = evaluate_opportunity_state_detector(detector, val_dataset)
        conditional_val = evaluate_opportunity_state_conditional_oracle(detector, val_dataset)
        oracle_val = evaluate_opportunity_state_detector(nothing, val_dataset; mode=:oracle)
        oracle_conditional_val = evaluate_opportunity_state_conditional_oracle(nothing, val_dataset; mode=:oracle)
        threshold_stability = opportunity_state_threshold_stability(detector, val_dataset;
            threshold_perturbations=threshold_perturbations)

        predicted_positive_fraction = Float64(get(val_eval, "predicted_positive_fraction", 0.0))
        gap_separation = Float64(get(val_eval, "gap_separation", 0.0))
        conditional_gain = Float64(get(conditional_val, "mean_gain_vs_heuristic", 0.0))
        routing_fraction = Float64(get(val_eval, "routing_sensitive_fraction_in_predicted_positive", 0.0))
        heuristic_fraction = Float64(get(val_eval, "heuristic_dominant_fraction_in_predicted_negative", 0.0))
        robust_fraction = Float64(get(threshold_stability, "robust_fraction", 0.0))
        nondegenerate = !(Bool(get(val_eval, "zero_positive_collapse", false)) || Bool(get(val_eval, "all_positive_collapse", false)))

        push!(thresholds_used, Float64(fit["threshold"]))
        push!(predicted_positive_fracs, predicted_positive_fraction)
        push!(gap_separations, gap_separation)
        push!(conditional_gains, conditional_gain)
        push!(routing_fracs, routing_fraction)
        push!(heuristic_fracs, heuristic_fraction)
        push!(robust_fracs, robust_fraction)
        nondegenerate_count += nondegenerate ? 1 : 0
        zero_positive_count += Bool(get(val_eval, "zero_positive_collapse", false)) ? 1 : 0
        all_positive_count += Bool(get(val_eval, "all_positive_collapse", false)) ? 1 : 0

        push!(split_summaries, Dict{String,Any}(
            "split_index" => split_idx,
            "train_size" => length(train_dataset),
            "val_size" => length(val_dataset),
            "train_snapshot_ids" => [record.snapshot_id for record in train_dataset.records],
            "val_snapshot_ids" => [record.snapshot_id for record in val_dataset.records],
            "threshold" => Float64(fit["threshold"]),
            "threshold_search" => fit["threshold_search"],
            "train_eval" => fit["train_eval"],
            "val_eval" => val_eval,
            "conditional_val_eval" => conditional_val,
            "oracle_val_eval" => oracle_val,
            "oracle_conditional_val_eval" => oracle_conditional_val,
            "threshold_stability" => threshold_stability,
            "gap_separation" => gap_separation,
            "nondegenerate" => nondegenerate,
            "zero_positive_collapse" => Bool(get(val_eval, "zero_positive_collapse", false)),
            "all_positive_collapse" => Bool(get(val_eval, "all_positive_collapse", false)),
        ))
    end

    split_count = length(split_summaries)
    median_gap_separation = isempty(gap_separations) ? 0.0 : median(gap_separations)
    median_conditional_gain = isempty(conditional_gains) ? 0.0 : median(conditional_gains)
    median_threshold = isempty(thresholds_used) ? 0.0 : median(thresholds_used)
    median_threshold_robust_fraction = isempty(robust_fracs) ? 0.0 : median(robust_fracs)
    nondegenerate_fraction = split_count == 0 ? 0.0 : nondegenerate_count / split_count
    mean_routing_fraction = isempty(routing_fracs) ? 0.0 : mean(routing_fracs)
    mean_heuristic_fraction = isempty(heuristic_fracs) ? 0.0 : mean(heuristic_fracs)
    repeatability_safe = split_count > 0 &&
        nondegenerate_fraction > 0.5 &&
        median_gap_separation > 0.02 &&
        median_conditional_gain >= 0.0 &&
        mean_routing_fraction > 0.5 &&
        mean_heuristic_fraction > 0.4 &&
        median_threshold_robust_fraction > 0.5

    recommendation = if repeatability_safe
        "STATE_SPLIT_REPEATABLE"
    elseif median_gap_separation > 0.0 && nondegenerate_fraction > 0.0
        "STATE_SPLIT_PRESENT_BUT_UNSTABLE"
    else
        "STATE_SPLIT_NOT_HELD_OUT_SAFE"
    end

    summary = Dict{String,Any}(
        "n_splits" => split_count,
        "nondegenerate_fraction" => nondegenerate_fraction,
        "zero_positive_collapse_fraction" => split_count == 0 ? 0.0 : zero_positive_count / split_count,
        "all_positive_collapse_fraction" => split_count == 0 ? 0.0 : all_positive_count / split_count,
        "mean_predicted_positive_fraction" => isempty(predicted_positive_fracs) ? 0.0 : mean(predicted_positive_fracs),
        "median_predicted_positive_fraction" => isempty(predicted_positive_fracs) ? 0.0 : median(predicted_positive_fracs),
        "mean_gap_separation" => isempty(gap_separations) ? 0.0 : mean(gap_separations),
        "median_gap_separation" => median_gap_separation,
        "mean_conditional_gain" => isempty(conditional_gains) ? 0.0 : mean(conditional_gains),
        "median_conditional_gain" => median_conditional_gain,
        "mean_routing_sensitive_fraction_in_positive" => mean_routing_fraction,
        "mean_heuristic_dominant_fraction_in_negative" => mean_heuristic_fraction,
        "mean_threshold" => isempty(thresholds_used) ? 0.0 : mean(thresholds_used),
        "median_threshold" => median_threshold,
        "mean_threshold_robust_fraction" => isempty(robust_fracs) ? 0.0 : mean(robust_fracs),
        "median_threshold_robust_fraction" => median_threshold_robust_fraction,
        "repeatability_safe" => repeatability_safe,
        "recommendation" => recommendation,
    )

    return Dict{String,Any}(
        "dataset_stats" => stats,
        "splits" => split_summaries,
        "summary" => summary,
    )
end


function _opportunity_state_eval_with_threshold(detector::OpportunityStateDetector,
                                                dataset::OpportunityStateDataset,
                                                threshold::Float64)
    adjusted = OpportunityStateDetector(detector.model, Float32(clamp(threshold, 0.0, 1.0)), detector.family_name)
    eval = evaluate_opportunity_state_detector(adjusted, dataset)
    cond_eval = evaluate_opportunity_state_conditional_oracle(adjusted, dataset)
    return Dict{String,Any}(
        "threshold" => Float64(adjusted.threshold),
        "eval" => eval,
        "conditional_eval" => cond_eval,
    )
end

function _sparse_positive_guard_status(eval::Dict{String,Any};
                                       min_positive_count::Int=1,
                                       max_positive_fraction::Float64=0.35,
                                       min_precision::Float64=0.75)
    n = Int(get(eval, "n_snapshots", 0))
    predicted_positive_fraction = Float64(get(eval, "predicted_positive_fraction", 0.0))
    predicted_positive_count = round(Int, predicted_positive_fraction * n)
    precision = Float64(get(eval, "precision", 0.0))
    zero_positive = Bool(get(eval, "zero_positive_collapse", false))
    all_positive = Bool(get(eval, "all_positive_collapse", false))
    valid = !zero_positive && !all_positive &&
        predicted_positive_count >= min_positive_count &&
        predicted_positive_fraction <= max_positive_fraction &&
        precision >= min_precision
    return Dict{String,Any}(
        "valid" => valid,
        "predicted_positive_count" => predicted_positive_count,
        "predicted_positive_fraction" => predicted_positive_fraction,
        "precision" => precision,
        "zero_positive_collapse" => zero_positive,
        "all_positive_collapse" => all_positive,
    )
end

function select_sparse_positive_operating_point(detector::OpportunityStateDetector,
                                                train_dataset::OpportunityStateDataset;
                                                rule::Symbol=:precision_guarded,
                                                fixed_thresholds::Vector{Float64}=[0.45, 0.50, 0.55, 0.60],
                                                max_positive_fraction::Float64=0.35,
                                                min_positive_count::Int=1,
                                                min_train_precision::Float64=0.75)
    isempty(train_dataset) && return Dict{String,Any}(
        "rule" => String(rule),
        "selected_threshold" => nothing,
        "status" => "insufficient_training_data",
        "candidates" => Dict{String,Any}[],
    )

    candidates = Dict{String,Any}[]
    for threshold in fixed_thresholds
        train_stats = _opportunity_state_eval_with_threshold(detector, train_dataset, threshold)
        eval = train_stats["eval"]
        cond_eval = train_stats["conditional_eval"]
        guard = _sparse_positive_guard_status(eval;
            min_positive_count=min_positive_count,
            max_positive_fraction=max_positive_fraction,
            min_precision=min_train_precision)
        push!(candidates, Dict{String,Any}(
            "threshold" => Float64(threshold),
            "eval" => eval,
            "conditional_eval" => cond_eval,
            "guard" => guard,
            "objective" => _opportunity_state_objective(eval),
            "gap_separation" => Float64(get(eval, "gap_separation", 0.0)),
            "conditional_gain" => Float64(get(cond_eval, "mean_gain_vs_heuristic", 0.0)),
        ))
    end

    valid_candidates = [candidate for candidate in candidates if Bool(get(candidate["guard"], "valid", false))]
    selected = nothing
    status = "no_valid_threshold"

    if rule == :fixed_threshold
        status = isempty(candidates) ? "no_thresholds" : "evaluated_only"
    elseif rule == :precision_guarded
        if !isempty(valid_candidates)
            selected = valid_candidates[argmax([Float64(candidate["threshold"]) for candidate in valid_candidates])]
            status = "selected"
        end
    elseif rule == :fraction_capped
        if !isempty(valid_candidates)
            scores = [Float64(candidate["gap_separation"]) + 1e-3 * Float64(candidate["guard"]["precision"]) for candidate in valid_candidates]
            selected = valid_candidates[argmax(scores)]
            status = "selected"
        end
    elseif rule == :guarded_fallback
        if !isempty(valid_candidates)
            selected = valid_candidates[argmax([Float64(candidate["threshold"]) for candidate in valid_candidates])]
            status = "selected"
        else
            status = "no_valid_threshold"
        end
    else
        error("Unknown sparse-positive operating-point rule: $(rule)")
    end

    return Dict{String,Any}(
        "rule" => String(rule),
        "selected_threshold" => isnothing(selected) ? nothing : Float64(selected["threshold"]),
        "status" => status,
        "selected_candidate" => selected,
        "candidates" => candidates,
        "valid_threshold_fraction" => isempty(candidates) ? 0.0 : length(valid_candidates) / length(candidates),
    )
end

function evaluate_sparse_positive_operating_point(detector::OpportunityStateDetector,
                                                  train_dataset::OpportunityStateDataset,
                                                  val_dataset::OpportunityStateDataset;
                                                  rule::Symbol=:precision_guarded,
                                                  fixed_thresholds::Vector{Float64}=[0.45, 0.50, 0.55, 0.60],
                                                  max_positive_fraction::Float64=0.35,
                                                  min_positive_count::Int=1,
                                                  min_train_precision::Float64=0.75)
    selection = select_sparse_positive_operating_point(detector, train_dataset;
        rule=rule,
        fixed_thresholds=fixed_thresholds,
        max_positive_fraction=max_positive_fraction,
        min_positive_count=min_positive_count,
        min_train_precision=min_train_precision)

    selected_threshold = get(selection, "selected_threshold", nothing)
    if isnothing(selected_threshold)
        val_eval = evaluate_opportunity_state_detector(nothing, val_dataset; mode=:always_negative)
        cond_eval = evaluate_opportunity_state_conditional_oracle(nothing, val_dataset; mode=:always_negative)
        guard = _sparse_positive_guard_status(val_eval;
            min_positive_count=min_positive_count,
            max_positive_fraction=max_positive_fraction,
            min_precision=0.0)
        return Dict{String,Any}(
            "rule" => String(rule),
            "selection" => selection,
            "selected_threshold" => nothing,
            "val_eval" => val_eval,
            "conditional_val_eval" => cond_eval,
            "val_guard" => guard,
            "promotion_safe" => false,
            "abstained" => true,
        )
    end

    val_stats = _opportunity_state_eval_with_threshold(detector, val_dataset, Float64(selected_threshold))
    val_eval = val_stats["eval"]
    cond_eval = val_stats["conditional_eval"]
    val_guard = _sparse_positive_guard_status(val_eval;
        min_positive_count=min_positive_count,
        max_positive_fraction=max_positive_fraction,
        min_precision=0.0)

    gap_separation = Float64(get(val_eval, "gap_separation", 0.0))
    cond_gain = Float64(get(cond_eval, "mean_gain_vs_heuristic", 0.0))
    routing_fraction = Float64(get(val_eval, "routing_sensitive_fraction_in_predicted_positive", 0.0))
    promotion_safe = Bool(get(val_guard, "valid", false)) && gap_separation > 0.0 && cond_gain >= 0.0 && routing_fraction > 0.5

    return Dict{String,Any}(
        "rule" => String(rule),
        "selection" => selection,
        "selected_threshold" => Float64(selected_threshold),
        "val_eval" => val_eval,
        "conditional_val_eval" => cond_eval,
        "val_guard" => val_guard,
        "promotion_safe" => promotion_safe,
        "abstained" => false,
    )
end

function evaluate_sparse_positive_operating_points(dataset::OpportunityStateDataset;
                                                   rng::AbstractRNG=Random.MersenneTwister(0),
                                                   num_splits::Int=5,
                                                   train_fraction::Float64=0.75,
                                                   family_name::String="basin",
                                                   ridge_lambda::Float64=0.1,
                                                   detector_thresholds::Vector{Float64}=collect(0.35:0.05:0.70),
                                                   rule_families::Vector{Symbol}=[:fixed_threshold, :precision_guarded, :fraction_capped, :guarded_fallback],
                                                   fixed_thresholds::Vector{Float64}=[0.45, 0.50, 0.55, 0.60],
                                                   max_positive_fraction::Float64=0.35,
                                                   min_positive_count::Int=1,
                                                   min_train_precision::Float64=0.75)
    stats = opportunity_state_dataset_stats(dataset)
    if length(dataset) < 2
        return Dict{String,Any}(
            "dataset_stats" => stats,
            "rule_summaries" => Dict{String,Any}(),
            "splits" => Dict{String,Any}[],
            "summary" => Dict{String,Any}(
                "n_splits" => 0,
                "recommendation" => "NO_SPARSE_POSITIVE_OPERATING_POINT",
            ),
            "reason" => "insufficient_snapshots",
        )
    end

    split_indices = _opportunity_state_split_indices(length(dataset);
        train_fraction=train_fraction,
        num_splits=num_splits,
        rng=rng)

    per_rule = Dict{String,Vector{Dict{String,Any}}}(String(rule) => Dict{String,Any}[] for rule in rule_families)
    split_summaries = Dict{String,Any}[]

    for (split_idx, (train_idx, val_idx)) in enumerate(split_indices)
        train_dataset = OpportunityStateDataset(dataset.records[train_idx])
        val_dataset = OpportunityStateDataset(dataset.records[val_idx])
        isempty(val_dataset) && (val_dataset = train_dataset)

        fit = _fit_opportunity_state_detector(train_dataset;
            family_name=family_name,
            ridge_lambda=ridge_lambda,
            thresholds=detector_thresholds)
        detector = fit["detector"]

        rule_results = Dict{String,Any}()
        for rule in rule_families
            result = evaluate_sparse_positive_operating_point(detector, train_dataset, val_dataset;
                rule=rule,
                fixed_thresholds=fixed_thresholds,
                max_positive_fraction=max_positive_fraction,
                min_positive_count=min_positive_count,
                min_train_precision=min_train_precision)
            rule_results[String(rule)] = result
            push!(per_rule[String(rule)], merge(Dict(
                "split_index" => split_idx,
                "train_snapshot_ids" => [record.snapshot_id for record in train_dataset.records],
                "val_snapshot_ids" => [record.snapshot_id for record in val_dataset.records],
            ), result))
        end

        push!(split_summaries, Dict{String,Any}(
            "split_index" => split_idx,
            "train_snapshot_ids" => [record.snapshot_id for record in train_dataset.records],
            "val_snapshot_ids" => [record.snapshot_id for record in val_dataset.records],
            "fit_threshold" => Float64(fit["threshold"]),
            "threshold_search" => fit["threshold_search"],
            "rules" => rule_results,
        ))
    end

    rule_summaries = Dict{String,Any}()
    best_rule = "none"
    best_score = -Inf
    for rule in rule_families
        key = String(rule)
        results = get(per_rule, key, Dict{String,Any}[])
        split_count = length(results)
        predicted_positive_fracs = Float64[get(result["val_eval"], "predicted_positive_fraction", 0.0) for result in results]
        precisions = Float64[get(result["val_eval"], "precision", 0.0) for result in results]
        recalls = Float64[get(result["val_eval"], "recall", 0.0) for result in results]
        gaps = Float64[get(result["val_eval"], "gap_separation", 0.0) for result in results]
        gains = Float64[get(result["conditional_val_eval"], "mean_gain_vs_heuristic", 0.0) for result in results]
        routing_fracs = Float64[get(result["val_eval"], "routing_sensitive_fraction_in_predicted_positive", 0.0) for result in results]
        heuristic_fracs = Float64[get(result["val_eval"], "heuristic_dominant_fraction_in_predicted_negative", 0.0) for result in results]
        thresholds = Float64[isnothing(result["selected_threshold"]) ? NaN : Float64(result["selected_threshold"]) for result in results]
        valid_threshold_fraction = split_count == 0 ? 0.0 : mean(Float64[!isnothing(result["selected_threshold"]) for result in results])
        nondegenerate_fraction = split_count == 0 ? 0.0 : mean(Float64[!(Bool(get(result["val_eval"], "zero_positive_collapse", false)) || Bool(get(result["val_eval"], "all_positive_collapse", false))) for result in results])
        zero_positive_fraction = split_count == 0 ? 0.0 : mean(Float64[Bool(get(result["val_eval"], "zero_positive_collapse", false)) for result in results])
        all_positive_fraction = split_count == 0 ? 0.0 : mean(Float64[Bool(get(result["val_eval"], "all_positive_collapse", false)) for result in results])
        promotion_safe_fraction = split_count == 0 ? 0.0 : mean(Float64[Bool(get(result, "promotion_safe", false)) for result in results])

        median_predicted_positive_fraction = isempty(predicted_positive_fracs) ? 0.0 : median(predicted_positive_fracs)
        median_precision = isempty(precisions) ? 0.0 : median(precisions)
        median_recall = isempty(recalls) ? 0.0 : median(recalls)
        median_gap = isempty(gaps) ? 0.0 : median(gaps)
        median_gain = isempty(gains) ? 0.0 : median(gains)
        mean_routing = isempty(routing_fracs) ? 0.0 : mean(routing_fracs)
        mean_heuristic = isempty(heuristic_fracs) ? 0.0 : mean(heuristic_fracs)
        finite_thresholds = Float64[t for t in thresholds if isfinite(t)]
        mean_threshold = isempty(finite_thresholds) ? NaN : mean(finite_thresholds)
        promotion_safe = promotion_safe_fraction > 0.5 &&
            nondegenerate_fraction > 0.5 &&
            median_predicted_positive_fraction > 0.0 &&
            median_predicted_positive_fraction <= max_positive_fraction &&
            median_gain >= 0.0 &&
            median_gap > 0.0 &&
            mean_routing > 0.5

        summary = Dict{String,Any}(
            "n_splits" => split_count,
            "valid_threshold_fraction" => valid_threshold_fraction,
            "nondegenerate_fraction" => nondegenerate_fraction,
            "zero_positive_collapse_fraction" => zero_positive_fraction,
            "all_positive_collapse_fraction" => all_positive_fraction,
            "median_predicted_positive_fraction" => median_predicted_positive_fraction,
            "median_precision" => median_precision,
            "median_recall" => median_recall,
            "median_gap_separation" => median_gap,
            "median_conditional_gain" => median_gain,
            "mean_routing_sensitive_fraction_in_positive" => mean_routing,
            "mean_heuristic_dominant_fraction_in_negative" => mean_heuristic,
            "mean_selected_threshold" => mean_threshold,
            "promotion_safe_fraction" => promotion_safe_fraction,
            "promotion_safe" => promotion_safe,
        )
        rule_summaries[key] = summary

        score = median_gain + median_gap + promotion_safe_fraction
        if promotion_safe && score > best_score
            best_score = score
            best_rule = key
        end
    end

    recommendation = best_rule == "none" ? "NO_SPARSE_POSITIVE_OPERATING_POINT" : "SPARSE_POSITIVE_OPERATING_POINT_FOUND"
    return Dict{String,Any}(
        "dataset_stats" => stats,
        "rule_summaries" => rule_summaries,
        "splits" => split_summaries,
        "summary" => Dict{String,Any}(
            "n_splits" => length(split_summaries),
            "best_rule" => best_rule,
            "recommendation" => recommendation,
        ),
    )
end


struct OpportunityRepairAuditRecord
    snapshot_id::String
    task_name::String
    family_name::String
    baseline_features::Vector{Float32}
    history_features::Vector{Float32}
    surface_features::Vector{Float32}
    robustness_features::Vector{Float32}
    phase_features::Vector{Float32}
    current_override_eligible::Bool
    robust_override_eligible::Bool
    abstain_eligible::Bool
    typed_label::String
    intervention_regime_label::String
    ordinal_regret::Float32
    best_vs_heuristic_gap::Float32
    heuristic_utility::Float32
    best_utility::Float32
    matched_budget::Int
end

struct OpportunityRepairAuditDataset
    records::Vector{OpportunityRepairAuditRecord}
end

Base.length(dataset::OpportunityRepairAuditDataset) = length(dataset.records)
Base.isempty(dataset::OpportunityRepairAuditDataset) = isempty(dataset.records)

function _safe_ratio(num::Real, denom::Real)
    denom_f = Float64(denom)
    return denom_f <= 0 ? 0.0 : Float64(num) / denom_f
end

function _safe_mean(values::Vector{Float64})
    return isempty(values) ? 0.0 : mean(values)
end

function _safe_median(values::Vector{Float64})
    return isempty(values) ? 0.0 : median(values)
end

function _top_two_gap(values::Vector{Float64})
    if isempty(values)
        return 0.0
    elseif length(values) == 1
        return values[1]
    end
    ordered = sort(copy(values), rev=true)
    return ordered[1] - ordered[2]
end

function _repair_history_feature_vector(run::AbstractDict)
    seed_stats = get(run, "seed_stats", Dict{String,Any}())
    warmup_stats = get(run, "warmup_stats", Dict{String,Any}())
    calls_used = Float64(get(run, "calls_used", 0.0))
    budget_total = max(Float64(get(run, "budget_total", max(calls_used, 1.0))), 1.0)
    seeded = Float64(get(seed_stats, "seeded", 0.0))
    augmented = Float64(get(seed_stats, "augmented", 0.0))
    evaluated = max(Float64(get(seed_stats, "evaluated", seeded + augmented)), 1.0)
    warmup_added = Float64(get(warmup_stats, "warmup_added", 0.0))
    warmup_evaluated = Float64(get(warmup_stats, "warmup_evaluated", 0.0))
    warmup_rounds = Float64(get(warmup_stats, "warmup_rounds", 0.0))
    created_at_step = Float64(get(run, "created_at_step", 0.0))
    return Float32[
        Float32(_safe_ratio(calls_used, budget_total)),
        Float32(_safe_ratio(seeded, evaluated)),
        Float32(_safe_ratio(augmented, seeded + augmented)),
        Float32(_safe_ratio(warmup_added, max(warmup_evaluated, 1.0))),
        Float32(_safe_ratio(warmup_evaluated, budget_total)),
        Float32(warmup_rounds),
        Float32(created_at_step),
    ]
end

function _repair_surface_feature_vector(family_summary::Dict{String,Any},
                                        region_summaries::Vector{Dict{String,Any}},
                                        matched_budget::Int)
    region_utilities = Float64[_finite_probe_utility(_frontier_region_allocation_utility(summary, matched_budget)) for summary in region_summaries]
    region_utilities = [u for u in region_utilities if isfinite(u)]
    counts = Float64[Float64(get(summary, "count", 0)) for summary in region_summaries]
    total_count = max(sum(counts), 1.0)
    return Float32[
        Float32(get(family_summary, "best_vs_heuristic_gap", 0.0)),
        Float32(get(family_summary, "best_vs_uniform_gap", 0.0)),
        Float32(get(family_summary, "heuristic_vs_anti_gap", 0.0)),
        Float32(get(family_summary, "region_opportunity_range", 0.0)),
        Float32(_top_two_gap(region_utilities)),
        Float32(_safe_mean(region_utilities)),
        Float32(_safe_median(region_utilities)),
        Float32(length(region_summaries)),
        Float32(_safe_ratio(maximum(vcat(counts, [0.0])), total_count)),
    ]
end

function _repair_robustness_feature_vector(family_summary::Dict{String,Any},
                                           region_summaries::Vector{Dict{String,Any}},
                                           matched_budget::Int)
    heuristic_utility = Float64(_finite_probe_utility(get(family_summary, "heuristic_allocation_utility", -Inf)))
    region_utilities = Float64[_finite_probe_utility(_frontier_region_allocation_utility(summary, matched_budget)) for summary in region_summaries]
    region_utilities = [u for u in region_utilities if isfinite(u)]
    if isempty(region_utilities)
        return Float32[0, 0, 0, 0, 0, 0]
    end
    gains = region_utilities .- heuristic_utility
    positive_gains = [max(g, 0.0) for g in gains]
    support_count = count(>(0.0), gains)
    total_positive_gain = sum(positive_gains)
    top_positive_gain = isempty(positive_gains) ? 0.0 : maximum(positive_gains)
    opportunity_range = max(Float64(get(family_summary, "region_opportunity_range", 0.0)), 1e-6)
    return Float32[
        Float32(_safe_ratio(count(>=(0.0), gains), length(gains))),
        Float32(_safe_ratio(support_count, length(gains))),
        Float32(_top_two_gap(region_utilities)),
        Float32(_safe_ratio(_top_two_gap(region_utilities), opportunity_range)),
        Float32(_safe_ratio(top_positive_gain, max(total_positive_gain, 1e-6))),
        Float32(std(region_utilities; corrected=false)),
    ]
end

function _repair_phase_feature_vector(run::AbstractDict,
                                      probe::AbstractDict,
                                      family_summary::Dict{String,Any},
                                      region_summaries::Vector{Dict{String,Any}},
                                      matched_budget::Int)
    calls_used = Float64(get(run, "calls_used", 0.0))
    budget_total = max(Float64(get(run, "budget_total", max(calls_used, 1.0))), 1.0)
    max_allocation_budget = Float64(get(probe, "max_allocation_budget", matched_budget))
    horizon = Float64(get(probe, "horizon", 0.0))
    n_regions = Float64(length(region_summaries))
    return Float32[
        Float32(matched_budget),
        Float32(max_allocation_budget),
        Float32(horizon),
        Float32(n_regions),
        Float32(_safe_ratio(matched_budget, max(max_allocation_budget, 1.0))),
        Float32(_safe_ratio(calls_used, budget_total)),
    ]
end

function extract_opportunity_repair_audit_dataset(probe_runs::Vector{<:AbstractDict};
                                                  family_name::String="basin",
                                                  state_threshold::Float64=0.01,
                                                  stability_threshold::Float64=0.05)
    records = OpportunityRepairAuditRecord[]
    for run in probe_runs
        probe = get(run, "probe", run)
        family_summary = _find_region_family_summary(probe, family_name)
        isnothing(family_summary) && continue
        region_summaries = Vector{Dict{String,Any}}(get(family_summary, "region_summaries", Dict{String,Any}[]))
        isempty(region_summaries) && continue

        matched_budget = Int(get(family_summary, "matched_budget", 0))
        heuristic_region = String(get(family_summary, "heuristic_top_region", ""))
        baseline_features = _frontier_snapshot_feature_vector(region_summaries, heuristic_region, matched_budget)
        history_features = _repair_history_feature_vector(run)
        surface_features = _repair_surface_feature_vector(family_summary, region_summaries, matched_budget)
        robustness_features = _repair_robustness_feature_vector(family_summary, region_summaries, matched_budget)
        phase_features = _repair_phase_feature_vector(run, probe, family_summary, region_summaries, matched_budget)

        current_override_eligible, _ = _opportunity_state_label(family_summary; state_threshold=state_threshold)
        regime_label, stable_positive, _ = _intervention_regime_label(family_summary;
            state_threshold=state_threshold,
            stability_threshold=stability_threshold)
        abstain_eligible = regime_label in ["ambiguous_routing_sensitive_state", "invariant_or_low_signal_state"]
        typed_label = stable_positive ? "robust_positive" :
            regime_label == "heuristic_dominant_state" ? "heuristic_negative" : "abstain_or_ambiguous"

        best_utility = Float32(_finite_probe_utility(get(family_summary, "best_allocation_utility", -Inf)))
        heuristic_utility = Float32(_finite_probe_utility(get(family_summary, "heuristic_allocation_utility", -Inf)))
        ordinal_regret = Float32(max(Float64(best_utility - heuristic_utility), 0.0))

        push!(records, OpportunityRepairAuditRecord(
            String(get(probe, "snapshot_id", "")),
            String(get(probe, "task_name", "unknown")),
            family_name,
            baseline_features,
            history_features,
            surface_features,
            robustness_features,
            phase_features,
            current_override_eligible,
            stable_positive,
            abstain_eligible,
            typed_label,
            regime_label,
            ordinal_regret,
            Float32(get(family_summary, "best_vs_heuristic_gap", 0.0)),
            heuristic_utility,
            best_utility,
            matched_budget,
        ))
    end
    return OpportunityRepairAuditDataset(records)
end

function opportunity_repair_audit_dataset_stats(dataset::OpportunityRepairAuditDataset)
    isempty(dataset) && return Dict{String,Any}(
        "size" => 0,
        "current_positive_fraction" => 0.0,
        "robust_positive_fraction" => 0.0,
        "abstain_fraction" => 0.0,
        "typed_counts" => Dict{String,Int}(),
        "baseline_feature_dim" => 0,
    )
    typed_counts = Dict{String,Int}()
    for record in dataset.records
        typed_counts[record.typed_label] = get(typed_counts, record.typed_label, 0) + 1
    end
    n = length(dataset)
    return Dict{String,Any}(
        "size" => n,
        "current_positive_fraction" => count(r -> r.current_override_eligible, dataset.records) / n,
        "robust_positive_fraction" => count(r -> r.robust_override_eligible, dataset.records) / n,
        "abstain_fraction" => count(r -> r.abstain_eligible, dataset.records) / n,
        "typed_counts" => typed_counts,
        "baseline_feature_dim" => length(dataset.records[1].baseline_features),
        "history_feature_dim" => length(dataset.records[1].history_features),
        "surface_feature_dim" => length(dataset.records[1].surface_features),
        "robustness_feature_dim" => length(dataset.records[1].robustness_features),
        "phase_feature_dim" => length(dataset.records[1].phase_features),
    )
end

function _repair_feature_vector(record::OpportunityRepairAuditRecord, branch::String)
    baseline = record.baseline_features
    history = record.history_features
    surface = record.surface_features
    robustness = record.robustness_features
    phase = record.phase_features
    if branch == "baseline"
        return baseline
    elseif branch == "baseline_plus_history"
        return vcat(baseline, history)
    elseif branch == "baseline_plus_surface"
        return vcat(baseline, surface)
    elseif branch == "baseline_plus_robustness"
        return vcat(baseline, robustness)
    elseif branch == "baseline_plus_phase"
        return vcat(baseline, phase)
    elseif branch == "full"
        return vcat(baseline, history, surface, robustness, phase)
    elseif branch == "full_minus_history"
        return vcat(baseline, surface, robustness, phase)
    elseif branch == "full_minus_surface"
        return vcat(baseline, history, robustness, phase)
    elseif branch == "full_minus_robustness"
        return vcat(baseline, history, surface, phase)
    elseif branch == "full_minus_phase"
        return vcat(baseline, history, surface, robustness)
    else
        error("Unknown repair feature branch: $(branch)")
    end
end

function _repair_binary_target(record::OpportunityRepairAuditRecord, semantics::String)
    if semantics == "current_binary"
        return record.current_override_eligible
    elseif semantics == "robust_positive"
        return record.robust_override_eligible
    elseif semantics == "abstain_ambiguous"
        return record.abstain_eligible
    else
        error("Unknown binary repair semantics: $(semantics)")
    end
end

function _binary_pairwise_accuracy(scores::Vector{Float64}, labels::Vector{Bool})
    pos_idx = findall(identity, labels)
    neg_idx = findall(!, labels)
    if isempty(pos_idx) || isempty(neg_idx)
        return 0.5
    end
    wins = 0.0
    total = 0
    for i in pos_idx, j in neg_idx
        total += 1
        if scores[i] > scores[j]
            wins += 1.0
        elseif scores[i] == scores[j]
            wins += 0.5
        end
    end
    return total == 0 ? 0.5 : wins / total
end

function _ordinal_pairwise_accuracy(scores::Vector{Float64}, targets::Vector{Float64}; eps::Float64=1e-8)
    n = length(scores)
    n <= 1 && return 0.5
    wins = 0.0
    total = 0
    for i in 1:n-1, j in i+1:n
        dt = targets[i] - targets[j]
        abs(dt) <= eps && continue
        ds = scores[i] - scores[j]
        total += 1
        if (dt > 0 && ds > 0) || (dt < 0 && ds < 0)
            wins += 1.0
        elseif ds == 0
            wins += 0.5
        end
    end
    return total == 0 ? 0.5 : wins / total
end

function _score_target_correlation(scores::Vector{Float64}, targets::Vector{Float64})
    length(scores) <= 1 && return 0.0
    score_mean = mean(scores)
    target_mean = mean(targets)
    score_centered = scores .- score_mean
    target_centered = targets .- target_mean
    denom = sqrt(sum(score_centered .^ 2) * sum(target_centered .^ 2))
    denom <= 1e-12 && return 0.0
    return sum(score_centered .* target_centered) / denom
end

function _evaluate_binary_repair_probe(model::FrontierAllocationLinearModel,
                                       records::Vector{OpportunityRepairAuditRecord},
                                       feature_branch::String,
                                       semantics::String)
    isempty(records) && return Dict{String,Any}(
        "n_records" => 0,
        "class_present" => false,
        "positive_fraction" => 0.0,
        "pairwise_accuracy" => 0.5,
        "mean_score_gap" => 0.0,
    )
    scores = Float64[_frontier_sigmoid32(_frontier_linear_score(model, _repair_feature_vector(record, feature_branch))) for record in records]
    labels = Bool[_repair_binary_target(record, semantics) for record in records]
    positives = [scores[i] for i in eachindex(scores) if labels[i]]
    negatives = [scores[i] for i in eachindex(scores) if !labels[i]]
    return Dict{String,Any}(
        "n_records" => length(records),
        "class_present" => !isempty(positives) && !isempty(negatives),
        "positive_fraction" => mean(Float64.(labels)),
        "pairwise_accuracy" => _binary_pairwise_accuracy(scores, labels),
        "mean_score_gap" => (isempty(positives) || isempty(negatives)) ? 0.0 : mean(positives) - mean(negatives),
        "mean_score" => mean(scores),
    )
end

function evaluate_opportunity_repair_binary_probe(dataset::OpportunityRepairAuditDataset;
                                                  feature_branch::String="baseline",
                                                  semantics::String="current_binary",
                                                  rng::AbstractRNG=Random.MersenneTwister(0),
                                                  num_splits::Int=5,
                                                  train_fraction::Float64=0.75,
                                                  ridge_lambda::Float64=0.1)
    stats = opportunity_repair_audit_dataset_stats(dataset)
    if length(dataset) < 2
        return Dict{String,Any}(
            "dataset_stats" => stats,
            "feature_branch" => feature_branch,
            "semantics" => semantics,
            "splits" => Dict{String,Any}[],
            "summary" => Dict{String,Any}("n_splits" => 0),
            "reason" => "insufficient_snapshots",
        )
    end

    split_indices = _opportunity_state_split_indices(length(dataset);
        train_fraction=train_fraction,
        num_splits=num_splits,
        rng=rng)

    split_summaries = Dict{String,Any}[]
    pairwise_accuracies = Float64[]
    score_gaps = Float64[]
    positive_fracs = Float64[]
    class_present_count = 0

    for (split_idx, (train_idx, val_idx)) in enumerate(split_indices)
        train_records = dataset.records[train_idx]
        val_records = dataset.records[val_idx]
        isempty(val_records) && (val_records = train_records)
        train_features = [_repair_feature_vector(record, feature_branch) for record in train_records]
        train_targets = Float32[_repair_binary_target(record, semantics) ? 1.0f0 : 0.0f0 for record in train_records]
        model = _fit_frontier_linear_model(train_features, train_targets; lambda=ridge_lambda)
        train_eval = _evaluate_binary_repair_probe(model, train_records, feature_branch, semantics)
        val_eval = _evaluate_binary_repair_probe(model, val_records, feature_branch, semantics)
        class_present = Bool(get(val_eval, "class_present", false))
        class_present_count += class_present ? 1 : 0
        push!(pairwise_accuracies, Float64(get(val_eval, "pairwise_accuracy", 0.5)))
        push!(score_gaps, Float64(get(val_eval, "mean_score_gap", 0.0)))
        push!(positive_fracs, Float64(get(val_eval, "positive_fraction", 0.0)))
        push!(split_summaries, Dict{String,Any}(
            "split_index" => split_idx,
            "train_snapshot_ids" => [record.snapshot_id for record in train_records],
            "val_snapshot_ids" => [record.snapshot_id for record in val_records],
            "train_eval" => train_eval,
            "val_eval" => val_eval,
        ))
    end

    split_count = length(split_summaries)
    summary = Dict{String,Any}(
        "n_splits" => split_count,
        "class_present_fraction" => split_count == 0 ? 0.0 : class_present_count / split_count,
        "mean_pairwise_accuracy" => isempty(pairwise_accuracies) ? 0.5 : mean(pairwise_accuracies),
        "median_pairwise_accuracy" => isempty(pairwise_accuracies) ? 0.5 : median(pairwise_accuracies),
        "mean_score_gap" => isempty(score_gaps) ? 0.0 : mean(score_gaps),
        "median_score_gap" => isempty(score_gaps) ? 0.0 : median(score_gaps),
        "mean_positive_fraction" => isempty(positive_fracs) ? 0.0 : mean(positive_fracs),
        "median_positive_fraction" => isempty(positive_fracs) ? 0.0 : median(positive_fracs),
    )
    return Dict{String,Any}(
        "dataset_stats" => stats,
        "feature_branch" => feature_branch,
        "semantics" => semantics,
        "splits" => split_summaries,
        "summary" => summary,
    )
end

function _evaluate_ordinal_repair_probe(model::FrontierAllocationLinearModel,
                                        records::Vector{OpportunityRepairAuditRecord},
                                        feature_branch::String)
    isempty(records) && return Dict{String,Any}(
        "n_records" => 0,
        "pairwise_rank_accuracy" => 0.5,
        "score_target_correlation" => 0.0,
        "top_half_target_gap" => 0.0,
    )
    scores = Float64[_frontier_linear_score(model, _repair_feature_vector(record, feature_branch)) for record in records]
    targets = Float64[Float64(record.ordinal_regret) for record in records]
    order = sortperm(scores, rev=true)
    half = max(cld(length(scores), 2), 1)
    top_targets = targets[order[1:half]]
    bottom_targets = targets[order[min(half + 1, length(scores)):end]]
    gap = isempty(bottom_targets) ? 0.0 : mean(top_targets) - mean(bottom_targets)
    return Dict{String,Any}(
        "n_records" => length(records),
        "pairwise_rank_accuracy" => _ordinal_pairwise_accuracy(scores, targets),
        "score_target_correlation" => _score_target_correlation(scores, targets),
        "top_half_target_gap" => gap,
        "mean_score" => mean(scores),
        "mean_target" => mean(targets),
    )
end

function evaluate_opportunity_repair_ordinal_probe(dataset::OpportunityRepairAuditDataset;
                                                   feature_branch::String="baseline",
                                                   rng::AbstractRNG=Random.MersenneTwister(0),
                                                   num_splits::Int=5,
                                                   train_fraction::Float64=0.75,
                                                   ridge_lambda::Float64=0.1)
    stats = opportunity_repair_audit_dataset_stats(dataset)
    if length(dataset) < 2
        return Dict{String,Any}(
            "dataset_stats" => stats,
            "feature_branch" => feature_branch,
            "splits" => Dict{String,Any}[],
            "summary" => Dict{String,Any}("n_splits" => 0),
            "reason" => "insufficient_snapshots",
        )
    end

    split_indices = _opportunity_state_split_indices(length(dataset);
        train_fraction=train_fraction,
        num_splits=num_splits,
        rng=rng)

    split_summaries = Dict{String,Any}[]
    pairwise_scores = Float64[]
    correlations = Float64[]
    target_gaps = Float64[]

    for (split_idx, (train_idx, val_idx)) in enumerate(split_indices)
        train_records = dataset.records[train_idx]
        val_records = dataset.records[val_idx]
        isempty(val_records) && (val_records = train_records)
        train_features = [_repair_feature_vector(record, feature_branch) for record in train_records]
        train_targets = Float32[record.ordinal_regret for record in train_records]
        model = _fit_frontier_linear_model(train_features, train_targets; lambda=ridge_lambda)
        train_eval = _evaluate_ordinal_repair_probe(model, train_records, feature_branch)
        val_eval = _evaluate_ordinal_repair_probe(model, val_records, feature_branch)
        push!(pairwise_scores, Float64(get(val_eval, "pairwise_rank_accuracy", 0.5)))
        push!(correlations, Float64(get(val_eval, "score_target_correlation", 0.0)))
        push!(target_gaps, Float64(get(val_eval, "top_half_target_gap", 0.0)))
        push!(split_summaries, Dict{String,Any}(
            "split_index" => split_idx,
            "train_snapshot_ids" => [record.snapshot_id for record in train_records],
            "val_snapshot_ids" => [record.snapshot_id for record in val_records],
            "train_eval" => train_eval,
            "val_eval" => val_eval,
        ))
    end

    summary = Dict{String,Any}(
        "n_splits" => length(split_summaries),
        "mean_pairwise_rank_accuracy" => isempty(pairwise_scores) ? 0.5 : mean(pairwise_scores),
        "median_pairwise_rank_accuracy" => isempty(pairwise_scores) ? 0.5 : median(pairwise_scores),
        "mean_score_target_correlation" => isempty(correlations) ? 0.0 : mean(correlations),
        "median_score_target_correlation" => isempty(correlations) ? 0.0 : median(correlations),
        "mean_top_half_target_gap" => isempty(target_gaps) ? 0.0 : mean(target_gaps),
        "median_top_half_target_gap" => isempty(target_gaps) ? 0.0 : median(target_gaps),
    )
    return Dict{String,Any}(
        "dataset_stats" => stats,
        "feature_branch" => feature_branch,
        "splits" => split_summaries,
        "summary" => summary,
    )
end

function _repair_binary_metric(result::Dict{String,Any})
    summary = get(result, "summary", Dict{String,Any}())
    return Float64(get(summary, "median_pairwise_accuracy", 0.5))
end

function _repair_ordinal_metric(result::Dict{String,Any})
    summary = get(result, "summary", Dict{String,Any}())
    return Float64(get(summary, "median_pairwise_rank_accuracy", 0.5))
end

function _repair_score_gap(result::Dict{String,Any})
    summary = get(result, "summary", Dict{String,Any}())
    return Float64(get(summary, "median_score_gap", 0.0))
end

function _repair_branch_best_metric(results::Dict{String,Dict{String,Any}}, branch::String)
    current = haskey(results, "current_binary") && haskey(results["current_binary"], branch) ? _repair_binary_metric(results["current_binary"][branch]) : 0.5
    robust = haskey(results, "robust_positive") && haskey(results["robust_positive"], branch) ? _repair_binary_metric(results["robust_positive"][branch]) : 0.5
    return max(current, robust)
end

function evaluate_opportunity_representation_semantics_repair(dataset::OpportunityRepairAuditDataset;
                                                              rng::AbstractRNG=Random.MersenneTwister(0),
                                                              num_splits::Int=5,
                                                              train_fraction::Float64=0.75,
                                                              ridge_lambda::Float64=0.1,
                                                              feature_branches::Vector{String}=[
                                                                  "baseline",
                                                                  "baseline_plus_history",
                                                                  "baseline_plus_surface",
                                                                  "baseline_plus_robustness",
                                                                  "baseline_plus_phase",
                                                                  "full",
                                                                  "full_minus_history",
                                                                  "full_minus_surface",
                                                                  "full_minus_robustness",
                                                                  "full_minus_phase",
                                                              ])
    stats = opportunity_repair_audit_dataset_stats(dataset)
    binary_semantics = ["current_binary", "robust_positive", "abstain_ambiguous"]
    binary_results = Dict{String,Dict{String,Any}}()
    for semantics in binary_semantics
        branch_results = Dict{String,Any}()
        for branch in feature_branches
            branch_results[branch] = evaluate_opportunity_repair_binary_probe(dataset;
                feature_branch=branch,
                semantics=semantics,
                rng=rng,
                num_splits=num_splits,
                train_fraction=train_fraction,
                ridge_lambda=ridge_lambda)
        end
        binary_results[semantics] = branch_results
    end

    ordinal_results = Dict{String,Any}()
    for branch in feature_branches
        ordinal_results[branch] = evaluate_opportunity_repair_ordinal_probe(dataset;
            feature_branch=branch,
            rng=rng,
            num_splits=num_splits,
            train_fraction=train_fraction,
            ridge_lambda=ridge_lambda)
    end

    baseline_current = _repair_binary_metric(binary_results["current_binary"]["baseline"])
    baseline_robust = _repair_binary_metric(binary_results["robust_positive"]["baseline"])
    baseline_abstain = _repair_binary_metric(binary_results["abstain_ambiguous"]["baseline"])
    baseline_ordinal = _repair_ordinal_metric(ordinal_results["baseline"])

    rep_candidate_branches = ["baseline_plus_history", "baseline_plus_surface", "baseline_plus_robustness", "baseline_plus_phase"]
    best_rep_branch = "baseline"
    best_rep_metric = max(baseline_current, baseline_robust)
    for branch in rep_candidate_branches
        metric = _repair_branch_best_metric(binary_results, branch)
        if metric > best_rep_metric
            best_rep_metric = metric
            best_rep_branch = branch
        end
    end
    representation_improvement = best_rep_metric - max(baseline_current, baseline_robust)

    full_metric = _repair_branch_best_metric(binary_results, "full")
    rep_drop = if best_rep_branch == "baseline_plus_history"
        full_metric - _repair_branch_best_metric(binary_results, "full_minus_history")
    elseif best_rep_branch == "baseline_plus_surface"
        full_metric - _repair_branch_best_metric(binary_results, "full_minus_surface")
    elseif best_rep_branch == "baseline_plus_robustness"
        full_metric - _repair_branch_best_metric(binary_results, "full_minus_robustness")
    elseif best_rep_branch == "baseline_plus_phase"
        full_metric - _repair_branch_best_metric(binary_results, "full_minus_phase")
    else
        0.0
    end

    current_best_branch = feature_branches[argmax([_repair_binary_metric(binary_results["current_binary"][branch]) for branch in feature_branches])]
    robust_best_branch = feature_branches[argmax([_repair_binary_metric(binary_results["robust_positive"][branch]) for branch in feature_branches])]
    abstain_best_branch = feature_branches[argmax([_repair_binary_metric(binary_results["abstain_ambiguous"][branch]) for branch in feature_branches])]
    ordinal_best_branch = feature_branches[argmax([_repair_ordinal_metric(ordinal_results[branch]) for branch in feature_branches])]

    current_best_metric = _repair_binary_metric(binary_results["current_binary"][current_best_branch])
    robust_best_metric = _repair_binary_metric(binary_results["robust_positive"][robust_best_branch])
    abstain_best_metric = _repair_binary_metric(binary_results["abstain_ambiguous"][abstain_best_branch])
    ordinal_best_metric = _repair_ordinal_metric(ordinal_results[ordinal_best_branch])

    semantics_improvement = robust_best_metric - current_best_metric
    geometry_improvement = max(abstain_best_metric, ordinal_best_metric) - max(current_best_metric, robust_best_metric)
    strongest_metric = max(best_rep_metric, robust_best_metric, abstain_best_metric, ordinal_best_metric)

    verdict = "V5_NO_DECISIVE_REPAIR_SIGNAL"
    decisive = false
    if strongest_metric < 0.62
        verdict = "V4_SNAPSHOT_INSUFFICIENCY_DOMINATES"
        decisive = true
    elseif representation_improvement >= 0.08 && rep_drop >= 0.04 && best_rep_metric >= 0.65
        verdict = "V1_REPRESENTATION_REPAIR_DOMINATES"
        decisive = true
    elseif semantics_improvement >= 0.08 && robust_best_metric >= 0.67 && representation_improvement < semantics_improvement
        verdict = "V2_SEMANTICS_REPAIR_DOMINATES"
        decisive = true
    elseif geometry_improvement >= 0.08 && max(abstain_best_metric, ordinal_best_metric) >= 0.67 && max(current_best_metric, robust_best_metric) < 0.67
        verdict = "V3_NONSCALAR_GEOMETRY_DOMINATES"
        decisive = true
    end

    summary = Dict{String,Any}(
        "n_splits" => num_splits,
        "baseline_current_metric" => baseline_current,
        "baseline_robust_metric" => baseline_robust,
        "baseline_abstain_metric" => baseline_abstain,
        "baseline_ordinal_metric" => baseline_ordinal,
        "best_representation_branch" => best_rep_branch,
        "best_representation_metric" => best_rep_metric,
        "representation_improvement" => representation_improvement,
        "representation_ablation_drop" => rep_drop,
        "current_best_branch" => current_best_branch,
        "current_best_metric" => current_best_metric,
        "robust_best_branch" => robust_best_branch,
        "robust_best_metric" => robust_best_metric,
        "semantics_improvement" => semantics_improvement,
        "abstain_best_branch" => abstain_best_branch,
        "abstain_best_metric" => abstain_best_metric,
        "ordinal_best_branch" => ordinal_best_branch,
        "ordinal_best_metric" => ordinal_best_metric,
        "geometry_improvement" => geometry_improvement,
        "strongest_metric" => strongest_metric,
        "verdict" => verdict,
        "decisive" => decisive,
    )

    return Dict{String,Any}(
        "dataset_stats" => stats,
        "binary_results" => binary_results,
        "ordinal_results" => ordinal_results,
        "summary" => summary,
    )
end

function train_opportunity_state_detector(dataset::OpportunityStateDataset;
                                          rng::AbstractRNG=Random.MersenneTwister(0),
                                          train_fraction::Float64=0.8,
                                          family_name::String="basin",
                                          ridge_lambda::Float64=0.1,
                                          thresholds::Vector{Float64}=collect(0.35:0.05:0.70))
    stats = opportunity_state_dataset_stats(dataset)
    if length(dataset) < 2
        return Dict{String,Any}(
            "model" => nothing,
            "dataset_stats" => stats,
            "reason" => "insufficient_snapshots",
        )
    end

    train_dataset, val_dataset = _split_opportunity_state_dataset(dataset; train_fraction=train_fraction, rng=rng)
    fit = _fit_opportunity_state_detector(train_dataset;
        family_name=family_name,
        ridge_lambda=ridge_lambda,
        thresholds=thresholds)
    best_detector = fit["detector"]

    val_eval = evaluate_opportunity_state_detector(best_detector, val_dataset)
    final_eval = evaluate_opportunity_state_detector(best_detector, dataset)
    conditional_val = evaluate_opportunity_state_conditional_oracle(best_detector, val_dataset)
    conditional_final = evaluate_opportunity_state_conditional_oracle(best_detector, dataset)
    always_negative_val = evaluate_opportunity_state_detector(nothing, val_dataset; mode=:always_negative)
    always_positive_val = evaluate_opportunity_state_detector(nothing, val_dataset; mode=:always_positive)
    oracle_val = evaluate_opportunity_state_detector(nothing, val_dataset; mode=:oracle)
    heuristic_conditional_val = evaluate_opportunity_state_conditional_oracle(nothing, val_dataset; mode=:always_negative)
    oracle_conditional_val = evaluate_opportunity_state_conditional_oracle(nothing, val_dataset; mode=:oracle)

    return Dict{String,Any}(
        "model" => best_detector,
        "dataset_stats" => stats,
        "train_dataset" => train_dataset,
        "val_dataset" => val_dataset,
        "train_eval" => fit["train_eval"],
        "val_eval" => val_eval,
        "final_eval" => final_eval,
        "conditional_val_eval" => conditional_val,
        "conditional_final_eval" => conditional_final,
        "always_negative_val_eval" => always_negative_val,
        "always_positive_val_eval" => always_positive_val,
        "oracle_val_eval" => oracle_val,
        "heuristic_conditional_val_eval" => heuristic_conditional_val,
        "oracle_conditional_val_eval" => oracle_conditional_val,
        "threshold" => fit["threshold"],
        "threshold_search" => fit["threshold_search"],
    )
end


function _find_region_family_summary(probe::AbstractDict, family_name::String)
    for summary in get(probe, "region_family_summaries", Dict{String,Any}[])
        if String(get(summary, "family_name", "")) == family_name
            return summary
        end
    end
    return nothing
end

function _intervention_regime_label(family_summary::Dict{String,Any};
                                    state_threshold::Float64=0.01,
                                    stability_threshold::Float64=0.05,
                                    allow_regime_mismatch::Bool=false)
    if Bool(get(family_summary, "all_degenerate", true))
        return "invariant_or_low_signal_state", false, false
    end

    heuristic_region = String(get(family_summary, "heuristic_top_region", ""))
    best_region = String(get(family_summary, "best_region", ""))
    best_vs_heuristic = Float64(get(family_summary, "best_vs_heuristic_gap", 0.0))
    heuristic_vs_anti = Float64(get(family_summary, "heuristic_vs_anti_gap", 0.0))
    opportunity_range = Float64(get(family_summary, "region_opportunity_range", 0.0))
    _, pooled_label = _opportunity_state_label(family_summary; state_threshold=state_threshold)

    strong_region_change = best_region != "" && best_region != heuristic_region
    stable_candidate = strong_region_change &&
        best_vs_heuristic > state_threshold &&
        opportunity_range > stability_threshold &&
        pooled_label != "heuristic_dominant_state"
    ambiguous_positive = strong_region_change && best_vs_heuristic > 0.0 && !stable_candidate
    heuristic_dominant = heuristic_region != "" &&
        (best_region == heuristic_region ||
         (best_vs_heuristic <= state_threshold && heuristic_vs_anti > state_threshold))
    low_signal = opportunity_range <= stability_threshold && best_vs_heuristic <= state_threshold
    regime_mismatch = allow_regime_mismatch &&
        strong_region_change &&
        best_vs_heuristic > state_threshold &&
        opportunity_range <= 0.5 * stability_threshold &&
        heuristic_vs_anti <= 0.0

    regime_label = if stable_candidate
        "stable_routing_sensitive_state"
    elseif heuristic_dominant
        "heuristic_dominant_state"
    elseif regime_mismatch
        "regime_mismatch_state"
    elseif low_signal
        "invariant_or_low_signal_state"
    elseif ambiguous_positive
        "ambiguous_routing_sensitive_state"
    else
        "invariant_or_low_signal_state"
    end

    return regime_label, stable_candidate, ambiguous_positive
end

function _intervention_geometry_subset_summary(records::Vector{Dict{String,Any}}, total_count::Int)
    isempty(records) && return Dict{String,Any}(
        "count" => 0,
        "fraction" => 0.0,
        "mean_best_vs_heuristic_gap" => 0.0,
        "median_best_vs_heuristic_gap" => 0.0,
        "gap_std" => 0.0,
        "mean_heuristic_vs_anti_gap" => 0.0,
        "mean_region_opportunity_range" => 0.0,
        "override_eligible_fraction" => 0.0,
        "heuristic_dominant_fraction" => 0.0,
        "stable_intervention_fraction" => 0.0,
        "ambiguous_positive_fraction" => 0.0,
        "snapshot_ids" => String[],
    )

    gaps = Float64[Float64(record["best_vs_heuristic_gap"]) for record in records]
    anti_gaps = Float64[Float64(record["heuristic_vs_anti_gap"]) for record in records]
    opportunity_ranges = Float64[Float64(record["region_opportunity_range"]) for record in records]
    override_flags = Float64[Bool(record["override_eligible"]) for record in records]
    heuristic_flags = Float64[String(record["regime_label"]) == "heuristic_dominant_state" for record in records]
    stable_flags = Float64[Bool(record["stable_intervention_candidate"]) for record in records]
    ambiguous_flags = Float64[Bool(record["ambiguous_positive"]) for record in records]

    return Dict{String,Any}(
        "count" => length(records),
        "fraction" => total_count == 0 ? 0.0 : length(records) / total_count,
        "mean_best_vs_heuristic_gap" => mean(gaps),
        "median_best_vs_heuristic_gap" => median(gaps),
        "gap_std" => length(gaps) > 1 ? std(gaps) : 0.0,
        "mean_heuristic_vs_anti_gap" => mean(anti_gaps),
        "mean_region_opportunity_range" => mean(opportunity_ranges),
        "override_eligible_fraction" => mean(override_flags),
        "heuristic_dominant_fraction" => mean(heuristic_flags),
        "stable_intervention_fraction" => mean(stable_flags),
        "ambiguous_positive_fraction" => mean(ambiguous_flags),
        "snapshot_ids" => String[String(record["snapshot_id"]) for record in records],
    )
end

function compare_intervention_geometry_atlas(atlas::Dict{String,Any})
    records = get(atlas, "records", Dict{String,Any}[])
    total_count = length(records)

    pooled_positive = Dict{String,Any}[record for record in records if Bool(record["override_eligible"])]
    stable_positive = Dict{String,Any}[record for record in records if String(record["regime_label"]) == "stable_routing_sensitive_state"]
    ambiguous_positive = Dict{String,Any}[record for record in records if String(record["regime_label"]) == "ambiguous_routing_sensitive_state"]
    heuristic_dominant = Dict{String,Any}[record for record in records if String(record["regime_label"]) == "heuristic_dominant_state"]
    invariant_low = Dict{String,Any}[record for record in records if String(record["regime_label"]) == "invariant_or_low_signal_state"]
    regime_mismatch = Dict{String,Any}[record for record in records if String(record["regime_label"]) == "regime_mismatch_state"]

    pooled_summary = _intervention_geometry_subset_summary(pooled_positive, total_count)
    stable_summary = _intervention_geometry_subset_summary(stable_positive, total_count)
    ambiguous_summary = _intervention_geometry_subset_summary(ambiguous_positive, total_count)
    heuristic_summary = _intervention_geometry_subset_summary(heuristic_dominant, total_count)
    invariant_summary = _intervention_geometry_subset_summary(invariant_low, total_count)
    mismatch_summary = _intervention_geometry_subset_summary(regime_mismatch, total_count)

    stable_gap_delta = Float64(stable_summary["mean_best_vs_heuristic_gap"]) - Float64(pooled_summary["mean_best_vs_heuristic_gap"])
    stable_gap_std_improvement = Float64(pooled_summary["gap_std"]) - Float64(stable_summary["gap_std"])
    stable_range_delta = Float64(stable_summary["mean_region_opportunity_range"]) - Float64(pooled_summary["mean_region_opportunity_range"])
    ambiguous_gap_mean = Float64(ambiguous_summary["mean_best_vs_heuristic_gap"])
    pooled_ambiguous_fraction = isempty(pooled_positive) ? 0.0 : mean(Float64[Bool(record["ambiguous_positive"]) for record in pooled_positive])
    compact_taxonomy = count(summary -> Int(summary["count"]) > 0,
        [stable_summary, ambiguous_summary, heuristic_summary, invariant_summary, mismatch_summary]) <= 4
    mixture_explains_instability = !isempty(stable_positive) && !isempty(ambiguous_positive) &&
        stable_gap_delta > 0.0 && stable_gap_std_improvement >= 0.0
    regime_split_useful = compact_taxonomy && !isempty(stable_positive) &&
        (mixture_explains_instability || (stable_gap_delta > 0.0 && stable_range_delta > 0.0))

    return Dict{String,Any}(
        "pooled_positive" => pooled_summary,
        "stable_routing_sensitive_state" => stable_summary,
        "ambiguous_routing_sensitive_state" => ambiguous_summary,
        "heuristic_dominant_state" => heuristic_summary,
        "invariant_or_low_signal_state" => invariant_summary,
        "regime_mismatch_state" => mismatch_summary,
        "stable_vs_pooled_gap_mean_delta" => stable_gap_delta,
        "stable_vs_pooled_gap_std_improvement" => stable_gap_std_improvement,
        "stable_vs_pooled_opportunity_range_delta" => stable_range_delta,
        "pooled_positive_ambiguous_fraction" => pooled_ambiguous_fraction,
        "ambiguous_positive_mean_gap" => ambiguous_gap_mean,
        "mixture_explains_instability" => mixture_explains_instability,
        "compact_taxonomy" => compact_taxonomy,
        "regime_split_useful" => regime_split_useful,
    )
end

function intervention_geometry_atlas_stats(atlas::Dict{String,Any})
    records = get(atlas, "records", Dict{String,Any}[])
    isempty(records) && return Dict{String,Any}(
        "size" => 0,
        "override_eligible_fraction" => 0.0,
        "stable_intervention_fraction" => 0.0,
        "ambiguous_positive_fraction" => 0.0,
        "feature_dim" => 0,
        "regime_counts" => Dict{String,Int}(),
        "n_active_regimes" => 0,
    )

    regime_counts = Dict{String,Int}()
    stable_count = 0
    ambiguous_count = 0
    override_count = 0
    for record in records
        label = String(record["regime_label"])
        regime_counts[label] = get(regime_counts, label, 0) + 1
        stable_count += Bool(record["stable_intervention_candidate"]) ? 1 : 0
        ambiguous_count += Bool(record["ambiguous_positive"]) ? 1 : 0
        override_count += Bool(record["override_eligible"]) ? 1 : 0
    end

    return Dict{String,Any}(
        "size" => length(records),
        "override_eligible_fraction" => override_count / length(records),
        "stable_intervention_fraction" => stable_count / length(records),
        "ambiguous_positive_fraction" => ambiguous_count / length(records),
        "feature_dim" => isempty(records) ? 0 : length(get(records[1], "features", Float32[])),
        "regime_counts" => regime_counts,
        "n_active_regimes" => count(>(0), values(regime_counts)),
    )
end

function extract_intervention_geometry_atlas(probe_runs::Vector{<:AbstractDict};
                                             family_name::String="basin",
                                             state_threshold::Float64=0.01,
                                             stability_threshold::Float64=0.05,
                                             allow_regime_mismatch::Bool=false)
    records = Dict{String,Any}[]
    for run in probe_runs
        probe = get(run, "probe", run)
        family_summary = _find_region_family_summary(probe, family_name)
        isnothing(family_summary) && continue
        region_summaries = Vector{Dict{String,Any}}(get(family_summary, "region_summaries", Dict{String,Any}[]))
        isempty(region_summaries) && continue
        matched_budget = Int(get(family_summary, "matched_budget", 0))
        heuristic_region = String(get(family_summary, "heuristic_top_region", ""))
        override_eligible, pooled_label = _opportunity_state_label(family_summary; state_threshold=state_threshold)
        regime_label, stable_candidate, ambiguous_positive = _intervention_regime_label(family_summary;
            state_threshold=state_threshold,
            stability_threshold=stability_threshold,
            allow_regime_mismatch=allow_regime_mismatch)
        features = _frontier_snapshot_feature_vector(region_summaries, heuristic_region, matched_budget)
        push!(records, Dict{String,Any}(
            "snapshot_id" => String(get(probe, "snapshot_id", "")),
            "task_name" => String(get(probe, "task_name", "unknown")),
            "family_name" => family_name,
            "features" => features,
            "override_eligible" => override_eligible,
            "state_label" => pooled_label,
            "regime_label" => regime_label,
            "heuristic_region" => heuristic_region,
            "best_region" => String(get(family_summary, "best_region", "")),
            "best_vs_heuristic_gap" => Float64(get(family_summary, "best_vs_heuristic_gap", 0.0)),
            "best_vs_uniform_gap" => Float64(get(family_summary, "best_vs_uniform_gap", 0.0)),
            "heuristic_vs_anti_gap" => Float64(get(family_summary, "heuristic_vs_anti_gap", 0.0)),
            "region_opportunity_range" => Float64(get(family_summary, "region_opportunity_range", 0.0)),
            "matched_budget" => matched_budget,
            "stable_intervention_candidate" => stable_candidate,
            "ambiguous_positive" => ambiguous_positive,
            "all_degenerate" => Bool(get(family_summary, "all_degenerate", false)),
        ))
    end

    atlas = Dict{String,Any}(
        "family_name" => family_name,
        "records" => records,
    )
    stats = intervention_geometry_atlas_stats(atlas)
    comparison = compare_intervention_geometry_atlas(atlas)
    interpretable = Bool(get(comparison, "regime_split_useful", false)) && Int(get(stats, "n_active_regimes", 0)) <= 4
    recommendation = if interpretable
        "REGIME_STRUCTURE_INTERPRETABLE"
    elseif Float64(get(stats, "override_eligible_fraction", 0.0)) > 0.0
        "POOLED_STATE_STILL_MIXED"
    else
        "NO_INTERVENTION_GEOMETRY"
    end

    atlas["stats"] = stats
    atlas["comparison"] = comparison
    atlas["interpretable"] = interpretable
    atlas["recommendation"] = recommendation
    return atlas
end

function _make_episode_id(snapshot_id::UInt64, created_at_step::Int)
    return string("snapshot-", snapshot_id, "-step-", created_at_step, "-", rand(UInt32))
end

function _trajectory_metadata(record::FrontierCommitRecord,
                              utility::Dict{String,Any},
                              family_transition_type::String)
    return merge(copy(record.metadata), Dict{String,Any}(
        "snapshot_id" => string(record.snapshot_id),
        "episode_id" => record.episode_id,
        "task_name" => record.task_name,
        "attempt_index" => record.attempt_index,
        "candidate_count" => record.candidate_count,
        "parent_scaffold" => record.parent_scaffold,
        "child_scaffold" => record.child_scaffold,
        "family_transition_type" => family_transition_type,
        "delta_top1" => utility["delta_top1"],
        "delta_top10_mean" => utility["delta_top10_mean"],
        "enters_topk" => utility["enters_topk"],
        "family_novelty_bonus" => utility["family_novelty_bonus"],
        "frontier_utility_delta" => utility["frontier_utility_delta"],
    ))
end

function _quantile_or_zero(values::Vector{Float64}, q::Float64)
    isempty(values) && return 0.0
    return Float64(quantile(values, q))
end

function _family_counts(parent_scaffold::String, proposals::Vector{EditProposal})
    same_family = 0
    cross_family = 0
    no_scaffold = 0
    for proposal in proposals
        child_scaffold = get_scaffold(proposal.child_smiles)
        family_type = _family_transition_type(parent_scaffold, child_scaffold)
        if family_type == "same_family"
            same_family += 1
        elseif family_type == "new_family"
            cross_family += 1
        else
            no_scaffold += 1
        end
    end
    return same_family, cross_family, no_scaffold
end

function _commit_episode_to_frontier!(frontier_buffer::MolecularFrontierBuffer,
                                      trajectory_buffer::EditTrajectoryBuffer,
                                      diagnostics_buffer::Union{Nothing,HierarchicalEditDiagnosticsBuffer},
                                      pending_commits::Vector{FrontierCommitRecord};
                                      topk_tracking::Int=10)
    steps = HierarchicalEditStep[]
    improved_topk = false

    for record in pending_commits
        before_summary = frontier_quality_summary(frontier_buffer; topk=topk_tracking)
        add_to_frontier!(frontier_buffer, record.child_smiles;
            reward=record.child_reward,
            source=:edit,
            parent_smiles=record.parent_smiles,
            operator=record.operator)
        after_summary = frontier_quality_summary(frontier_buffer; topk=topk_tracking)

        utility = compute_frontier_utility_delta(before_summary, after_summary,
            record.child_smiles, record.child_scaffold)
        family_transition_type = _family_transition_type(record.parent_scaffold, record.child_scaffold)

        add_edit_trajectory!(trajectory_buffer,
            record.basin_scaffold,
            record.parent_smiles,
            record.parent_reward,
            record.operator,
            record.child_smiles,
            record.child_reward;
            step_index=record.step_index,
            terminated=record.terminated,
            metadata=_trajectory_metadata(record, utility, family_transition_type))

        push!(steps, HierarchicalEditStep(
            record.snapshot_id,
            record.episode_id,
            record.task_name,
            record.step_index,
            record.basin_scaffold,
            record.basin_score,
            record.parent_smiles,
            record.operator,
            record.child_smiles,
            record.child_reward,
            Float64(utility["frontier_utility_delta"]),
            record.terminated,
            family_transition_type,
        ))

        if !isnothing(diagnostics_buffer)
            add_decision_log!(diagnostics_buffer, HierarchicalEditDecisionLog(
                record.snapshot_id,
                record.episode_id,
                record.task_name,
                record.step_index,
                record.attempt_index,
                record.basin_scaffold,
                record.basin_score,
                record.parent_smiles,
                record.parent_reward,
                record.operator,
                record.candidate_count,
                record.child_smiles,
                record.child_reward,
                record.child_reward - record.parent_reward,
                record.terminated,
                record.parent_scaffold,
                record.child_scaffold,
                family_transition_type,
                Bool(utility["enters_topk"]),
                Float64(utility["delta_top1"]),
                Float64(utility["delta_top10_mean"]),
                Float64(utility["family_novelty_bonus"]),
                Float64(utility["frontier_utility_delta"]),
                true,
            ))
        end

        improved_topk |= Bool(utility["enters_topk"]) || Float64(utility["delta_top1"]) > 0 || Float64(utility["delta_top10_mean"]) > 0
    end

    return steps, improved_topk
end

"""
    run_hierarchical_edit_episode!(frontier_buffer, trajectory_buffer, reward_fn, vocab; ...)

Frozen frontier-guided search episode:
1. freeze frontier snapshot
2. sample basin
3. sample parent
4. choose operator
5. propose children
6. score candidates
7. buffer accepted children locally
8. commit buffered children to the live frontier after the episode ends
"""
function run_hierarchical_edit_episode!(frontier_buffer::MolecularFrontierBuffer,
                                        trajectory_buffer::EditTrajectoryBuffer,
                                        reward_fn,
                                        vocab;
                                        reward_fn_batch=nothing,
                                        diagnostics_buffer::Union{Nothing,HierarchicalEditDiagnosticsBuffer}=nothing,
                                        config::HierarchicalEditConfig=HierarchicalEditConfig(),
                                        target_smiles::Union{Nothing,String}=nothing,
                                        budget_remaining::Int=0,
                                        created_at_step::Int=0,
                                        task_name::String="unknown",
                                        episode_id::Union{Nothing,String}=nothing,
                                        operator_override::Union{Nothing,Symbol}=nothing)
    snapshot = create_frontier_snapshot(frontier_buffer;
        max_entries=config.frontier_snapshot_size,
        target_smiles=target_smiles,
        budget_remaining=budget_remaining,
        created_at_step=created_at_step)

    episode_id_local = isnothing(episode_id) ? _make_episode_id(snapshot.snapshot_id, created_at_step) : episode_id
    frontier_before = frontier_quality_summary(frontier_buffer; topk=config.topk_tracking)

    if isempty(snapshot.entries)
        return HierarchicalEditEpisode(
            snapshot.snapshot_id,
            episode_id_local,
            task_name,
            HierarchicalEditStep[],
            "",
            0.0,
            Int(frontier_before["size"]),
            Int(frontier_before["size"]),
            false,
            0,
        )
    end

    pending_commits = FrontierCommitRecord[]
    pending_child_smiles = Set{String}()
    best_smiles = ""
    best_reward = -Inf
    current_parent = nothing
    terminated_episode = false
    parent_visit_counts = Dict{String,Int}()
    operator_stats_local = Dict{Symbol,Dict{String,Int}}()

    for step_idx in 1:config.horizon
        committed_step = false

        for attempt_idx in 1:config.max_step_attempts
            scored_basin, basin_candidates = choose_basin(snapshot, config; step_index=step_idx)
            scored_basin === nothing && break
            basin = scored_basin.basin
            basin_score_value = scored_basin.score

            if !isnothing(diagnostics_buffer)
                basin_logs = build_basin_decision_candidates(snapshot, basin_candidates)
                chosen_index = findfirst(c -> c.scaffold == basin.scaffold && isapprox(c.heuristic_score, basin_score_value; atol=1e-8), basin_logs)
                if isnothing(chosen_index)
                    chosen_index = findfirst(c -> c.scaffold == basin.scaffold, basin_logs)
                end
                add_basin_log!(diagnostics_buffer, BasinDecisionLog(
                    snapshot.snapshot_id,
                    episode_id_local,
                    task_name,
                    step_idx,
                    attempt_idx,
                    budget_remaining,
                    created_at_step,
                    length(snapshot.entries),
                    isempty(snapshot.entries) ? 0.0 : maximum(e.reward for e in snapshot.entries),
                    isempty(snapshot.entries) ? 0.0 : mean(sort([e.reward for e in snapshot.entries], rev=true)[1:min(10, length(snapshot.entries))]),
                    length(Set(e.scaffold for e in snapshot.entries if !isempty(e.scaffold))),
                    basin_logs,
                    something(chosen_index, 1),
                    basin.scaffold,
                    basin_score_value,
                ))
            end

            parent_candidate_logs = ParentDecisionCandidate[]
            parent = current_parent
            if isnothing(current_parent)
                scored_parent, parent_candidates, parent_selection = choose_parent(snapshot, basin, config;
                    step_index=step_idx,
                    visit_counts=parent_visit_counts)
                scored_parent === nothing && break
                parent = scored_parent.entry
                parent_candidate_logs = build_parent_decision_candidates(parent_candidates)
                if !isnothing(diagnostics_buffer)
                    chosen_parent_index = findfirst(c -> c.smiles == parent.smiles, parent_candidate_logs)
                    add_parent_log!(diagnostics_buffer, ParentDecisionLog(
                        snapshot.snapshot_id,
                        episode_id_local,
                        task_name,
                        step_idx,
                        attempt_idx,
                        basin.scaffold,
                        basin_score_value,
                        parent_candidate_logs,
                        something(chosen_parent_index, get(parent_selection, "chosen_index", 1)),
                        parent.smiles,
                        scored_parent.score,
                        get(parent_selection, "heuristic_top_index", 1),
                        get(parent_selection, "learned_top_index", 1),
                        get(parent_selection, "heuristic_margin", 0.0),
                        get(parent_selection, "learned_margin", 0.0),
                        get(parent_selection, "learned_advantage_vs_heuristic", 0.0),
                        get(parent_selection, "heuristic_entropy", 0.0),
                        get(parent_selection, "learned_entropy", 0.0),
                        get(parent_selection, "override_applied", false),
                        get(parent_selection, "abstained_to_heuristic", false),
                        get(parent_selection, "selection_reason", "heuristic_sample"),
                    ))
                end
            end
            parent === nothing && break
            parent_visit_counts[parent.smiles] = get(parent_visit_counts, parent.smiles, 0) + 1

            bias_structural = !isnothing(target_smiles)
            op, operator_candidates, operator_selection = choose_operator_action(snapshot, basin, parent, config;
                step_index=step_idx,
                bias_structural=bias_structural,
                operator_override=operator_override,
                operator_stats=operator_stats_local)
            op === nothing && break
            if !isnothing(diagnostics_buffer)
                chosen_operator_index = something(findfirst(c -> c.operator == op, operator_candidates), get(operator_selection, "chosen_index", 1))
                chosen_operator_score = (1 <= chosen_operator_index <= length(operator_candidates)) ? operator_candidates[chosen_operator_index].heuristic_score : 0.0
                add_operator_log!(diagnostics_buffer, OperatorDecisionLog(
                    snapshot.snapshot_id,
                    episode_id_local,
                    task_name,
                    step_idx,
                    attempt_idx,
                    basin.scaffold,
                    basin_score_value,
                    parent.smiles,
                    parent.reward,
                    parent.scaffold,
                    parent.novelty_score,
                    parent.tb_delta_abs,
                    String(parent.source),
                    operator_candidates,
                    chosen_operator_index,
                    op,
                    chosen_operator_score,
                    get(operator_selection, "predicted_eligible", false),
                    get(operator_selection, "eligibility_score", 0.0),
                    get(operator_selection, "acted_on", false),
                    get(operator_selection, "preserved_to_heuristic", false),
                    get(operator_selection, "heuristic_top_index", 1),
                    get(operator_selection, "learned_top_index", 1),
                    get(operator_selection, "heuristic_margin", 0.0),
                    get(operator_selection, "learned_margin", 0.0),
                    get(operator_selection, "learned_advantage_vs_heuristic", 0.0),
                    get(operator_selection, "heuristic_entropy", 0.0),
                    get(operator_selection, "learned_entropy", 0.0),
                    get(operator_selection, "override_applied", false),
                    get(operator_selection, "abstained_to_heuristic", false),
                    get(operator_selection, "selection_reason", "heuristic_sample"),
                ))
            end
            partner = op == :crossover ? choose_partner(snapshot, parent.smiles) : nothing
            proposals, proposal_diagnostics = propose_edit_with_diagnostics(parent.smiles, op, vocab;
                partner_smiles=partner,
                max_candidates=config.max_operator_candidates)

            cached_child_count = 0
            filtered_proposals = EditProposal[]
            for proposal in proposals
                child_identity = canonicalize_smiles_identity(proposal.child_smiles)
                if proposal.operator != :terminate &&
                   (haskey(frontier_buffer.seen_smiles, child_identity) || (child_identity in pending_child_smiles))
                    cached_child_count += 1
                    continue
                end
                push!(filtered_proposals, EditProposal(
                    proposal.operator,
                    canonicalize_smiles_identity(proposal.parent_smiles),
                    isnothing(proposal.partner_smiles) ? nothing : canonicalize_smiles_identity(proposal.partner_smiles),
                    child_identity,
                    copy(proposal.metadata),
                ))
            end

            same_family_count, cross_family_count, no_scaffold_count = _family_counts(parent.scaffold, filtered_proposals)

            chosen_child_smiles = ""
            chosen_reward = 0.0
            chosen_reward_delta = 0.0
            reward_q25 = 0.0
            reward_q50 = 0.0
            reward_q75 = 0.0
            reward_max = 0.0

            if !isempty(filtered_proposals)
                smiles_batch = [p.child_smiles for p in filtered_proposals]
                rewards = isnothing(reward_fn_batch) ? Float64[reward_fn(s) for s in smiles_batch] : Float64[reward_fn_batch(smiles_batch)...]

                reward_q25 = _quantile_or_zero(rewards, 0.25)
                reward_q50 = _quantile_or_zero(rewards, 0.50)
                reward_q75 = _quantile_or_zero(rewards, 0.75)
                reward_max = isempty(rewards) ? 0.0 : maximum(rewards)

                best_idx = argmax(rewards)
                chosen = filtered_proposals[best_idx]
                chosen_child_smiles = chosen.child_smiles
                chosen_reward = rewards[best_idx]
                chosen_reward_delta = chosen_reward - parent.reward
                child_scaffold = get_scaffold(chosen.child_smiles)
                terminated = (chosen.operator == :terminate)

                if !isnothing(diagnostics_buffer)
                    add_proposal_log!(diagnostics_buffer, HierarchicalEditProposalLog(
                        snapshot.snapshot_id,
                        episode_id_local,
                        task_name,
                        step_idx,
                        attempt_idx,
                        basin.scaffold,
                        basin_score_value,
                        parent.smiles,
                        parent.reward,
                        parent.scaffold,
                        op,
                        partner,
                        Int(proposal_diagnostics["raw_candidate_count"]),
                        Int(proposal_diagnostics["duplicate_candidate_count"]),
                        Int(proposal_diagnostics["empty_child_count"]),
                        Int(proposal_diagnostics["self_child_count"]),
                        cached_child_count,
                        length(filtered_proposals),
                        same_family_count,
                        cross_family_count,
                        no_scaffold_count,
                        chosen_child_smiles,
                        chosen_reward,
                        chosen_reward_delta,
                        reward_q25,
                        reward_q50,
                        reward_q75,
                        reward_max,
                        false,
                    ))
                end

                # Track operator success for adaptive weighting
                if !isnothing(operator_stats_local)
                    op_entry = get!(operator_stats_local, op, Dict("positive_delta_count" => 0, "total_count" => 0))
                    op_entry["total_count"] += 1
                    if chosen_reward_delta > 0
                        op_entry["positive_delta_count"] += 1
                    end
                end

                # Multi-child acceptance: commit proposals with reward above a
                # quality floor. The floor prevents frontier flooding with
                # very low-quality children while keeping the multi-child benefit
                # (A1 showed commits/episode 1-3→5-15 was the biggest win).
                min_ratio = config.multi_child_min_reward_ratio
                quality_floor = parent.reward > 0.0 ? parent.reward * min_ratio : 0.0
                accepted_indices = sortperm(rewards, rev=true)
                for accept_idx in accepted_indices
                    accept_prop = filtered_proposals[accept_idx]
                    accept_reward = rewards[accept_idx]
                    accept_reward <= 0.0 && continue
                    # Quality gate: skip proposals far below parent quality
                    accept_reward < quality_floor && continue
                    accept_child_scaffold = get_scaffold(accept_prop.child_smiles)
                    accept_terminated = (accept_prop.operator == :terminate)

                    accept_metadata = merge(copy(accept_prop.metadata), Dict{String,Any}(
                        "attempt_index" => attempt_idx,
                        "partner_smiles" => partner,
                        "raw_candidate_count" => Int(proposal_diagnostics["raw_candidate_count"]),
                        "duplicate_candidate_count" => Int(proposal_diagnostics["duplicate_candidate_count"]),
                        "empty_child_count" => Int(proposal_diagnostics["empty_child_count"]),
                        "self_child_count" => Int(proposal_diagnostics["self_child_count"]),
                        "cached_child_count" => cached_child_count,
                        "unique_valid_count" => length(filtered_proposals),
                        "same_family_count" => same_family_count,
                        "cross_family_count" => cross_family_count,
                        "no_scaffold_count" => no_scaffold_count,
                        "reward_q25" => reward_q25,
                        "reward_q50" => reward_q50,
                        "reward_q75" => reward_q75,
                        "reward_max" => reward_max,
                        "multi_accept" => (accept_idx != best_idx),
                    ))

                    push!(pending_commits, FrontierCommitRecord(
                        snapshot.snapshot_id,
                        episode_id_local,
                        task_name,
                        step_idx,
                        attempt_idx,
                        basin.scaffold,
                        basin_score_value,
                        parent.smiles,
                        parent.reward,
                        parent.scaffold,
                        accept_prop.operator,
                        accept_prop.child_smiles,
                        accept_reward,
                        accept_child_scaffold,
                        length(filtered_proposals),
                        accept_terminated,
                        accept_metadata,
                    ))
                    push!(pending_child_smiles, canonicalize_smiles_identity(accept_prop.child_smiles))
                end

                if chosen_reward > best_reward
                    best_reward = chosen_reward
                    best_smiles = chosen.child_smiles
                end

                terminated_episode = terminated
                committed_step = true

                # Chain to the best child as next parent
                if !terminated
                    current_parent = FrontierSnapshotEntry(
                        chosen.child_smiles,
                        child_scaffold,
                        chosen_reward,
                        0.0,
                        0.0,
                        :edit,
                    )
                end
                break
            else
                if !isnothing(diagnostics_buffer)
                    add_proposal_log!(diagnostics_buffer, HierarchicalEditProposalLog(
                        snapshot.snapshot_id,
                        episode_id_local,
                        task_name,
                        step_idx,
                        attempt_idx,
                        basin.scaffold,
                        basin_score_value,
                        parent.smiles,
                        parent.reward,
                        parent.scaffold,
                        op,
                        partner,
                        Int(proposal_diagnostics["raw_candidate_count"]),
                        Int(proposal_diagnostics["duplicate_candidate_count"]),
                        Int(proposal_diagnostics["empty_child_count"]),
                        Int(proposal_diagnostics["self_child_count"]),
                        cached_child_count,
                        0,
                        same_family_count,
                        cross_family_count,
                        no_scaffold_count,
                        chosen_child_smiles,
                        chosen_reward,
                        chosen_reward_delta,
                        reward_q25,
                        reward_q50,
                        reward_q75,
                        reward_max,
                        true,
                    ))
                end
                current_parent = nothing
                continue
            end
        end

        terminated_episode && break
        committed_step || (current_parent = nothing)
    end

    steps, improved_topk = _commit_episode_to_frontier!(
        frontier_buffer,
        trajectory_buffer,
        diagnostics_buffer,
        pending_commits;
        topk_tracking=config.topk_tracking,
    )

    frontier_after = frontier_quality_summary(frontier_buffer; topk=config.topk_tracking)
    return HierarchicalEditEpisode(
        snapshot.snapshot_id,
        episode_id_local,
        task_name,
        steps,
        best_smiles,
        isfinite(best_reward) ? best_reward : 0.0,
        Int(frontier_before["size"]),
        Int(frontier_after["size"]),
        improved_topk,
        length(pending_commits),
    )
end
