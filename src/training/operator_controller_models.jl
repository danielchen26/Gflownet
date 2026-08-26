# Operator Controller Models
#
# Batch 1H scope:
# - separate operator eligibility detection from conditional operator ranking
# - preserve a deterministic fair heuristic baseline
# - support eligibility-gated and anchored operator control

using Serialization
using Random

struct LearnedOperatorEligibilityModel
    weights::Vector{Float32}
    bias::Float32
    input_dim::Int
end

struct LearnedOperatorController
    weights::Vector{Float32}
    bias::Float32
    input_dim::Int
    feature_mode::Symbol
end

struct HeuristicTopOperatorController end

struct AnchoredOperatorController{T}
    base_controller::T
    override_margin::Float32
    preserve_margin::Float32
    learned_confidence_margin::Float32
end

struct EligibilityGatedOperatorController{E,T}
    eligibility_model::E
    base_controller::T
    eligibility_threshold::Float32
end

function create_learned_operator_eligibility_model(input_dim::Int;
                                                    rng::AbstractRNG=Random.MersenneTwister(0))
    weights = 0.01f0 .* randn(rng, Float32, input_dim)
    return LearnedOperatorEligibilityModel(weights, 0.0f0, input_dim)
end

function create_learned_operator_controller(input_dim::Int;
                                            feature_mode::Symbol=:basic,
                                            rng::AbstractRNG=Random.MersenneTwister(0))
    weights = 0.01f0 .* randn(rng, Float32, input_dim)
    return LearnedOperatorController(weights, 0.0f0, input_dim, feature_mode)
end

create_anchored_operator_controller(base_controller;
                                    override_margin::Float32=0.05f0,
                                    preserve_margin::Float32=0.15f0,
                                    learned_confidence_margin::Float32=0.05f0) =
    AnchoredOperatorController(base_controller, override_margin, preserve_margin, learned_confidence_margin)

create_gated_operator_controller(eligibility_model, base_controller;
                                 eligibility_threshold::Float32=0.50f0) =
    EligibilityGatedOperatorController(eligibility_model, base_controller, eligibility_threshold)

_sigmoid(x::Real) = 1.0f0 / (1.0f0 + exp(-clamp(Float32(x), -20.0f0, 20.0f0)))

_opctrl_score_margin(scores::AbstractVector{<:Real}, top_idx::Int) =
    isempty(scores) ? 0.0f0 :
    length(scores) == 1 ? Float32(scores[top_idx]) : begin
        order = sortperm(Float32.(scores), rev=true)
        Float32(scores[order[1]] - scores[order[2]])
    end

function _opctrl_score_entropy(scores::AbstractVector{<:Real})
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

function _operator_rank_features(heuristic_scores::Vector{Float64}, idx::Int)
    isempty(heuristic_scores) && return Float32[1.0, 0.0, 0.0, 0.0]
    order = sortperm(heuristic_scores, rev=true)
    rank = something(findfirst(==(idx), order), 1)
    top_score = heuristic_scores[order[1]]
    next_score = rank < length(order) ? heuristic_scores[order[rank + 1]] : heuristic_scores[order[end]]
    current = heuristic_scores[idx]
    rank_fraction = length(order) == 1 ? 0.0 : (rank - 1) / (length(order) - 1)
    return Float32[rank, rank_fraction, current - top_score, current - next_score]
end

function operator_eligibility_feature_vector(log::OperatorDecisionLog)
    heuristic_scores = Float64[c.heuristic_score for c in log.candidate_operators]
    return Float32[
        log.basin_score,
        log.step_index,
        log.parent_reward,
        log.parent_novelty_score,
        log.parent_tb_delta_abs,
        length(log.candidate_operators),
        _opctrl_score_margin(heuristic_scores, clamp(log.heuristic_top_index, 1, max(length(heuristic_scores), 1))),
        _opctrl_score_entropy(heuristic_scores),
    ]
end

operator_context_feature_vector(log::OperatorDecisionLog) = operator_eligibility_feature_vector(log)

function operator_eligibility_feature_vector(snapshot::FrontierSnapshot,
                                             basin::BasinSummary,
                                             parent::FrontierSnapshotEntry,
                                             candidates::Vector{OperatorDecisionCandidate};
                                             step_index::Int=0)
    heuristic_scores = Float64[c.heuristic_score for c in candidates]
    heuristic_idx = isempty(heuristic_scores) ? 1 : argmax(Float32.(heuristic_scores))
    return Float32[
        basin.best_reward,
        step_index,
        parent.reward,
        parent.novelty_score,
        parent.tb_delta_abs,
        length(candidates),
        _opctrl_score_margin(heuristic_scores, heuristic_idx),
        _opctrl_score_entropy(heuristic_scores),
    ]
end

function operator_candidate_feature_vector(log::OperatorDecisionLog,
                                           candidate::OperatorDecisionCandidate;
                                           all_candidates::Vector{OperatorDecisionCandidate}=log.candidate_operators,
                                           candidate_index::Int=1,
                                           feature_mode::Symbol=:basic)
    context = operator_context_feature_vector(log)
    positive_fraction = candidate.total_count == 0 ? 0.0f0 : Float32(candidate.positive_delta_count / max(candidate.total_count, 1))
    heuristic_scores = Float64[c.heuristic_score for c in all_candidates]
    rank_features = _operator_rank_features(heuristic_scores, candidate_index)
    op_flags = Float32[
        candidate.operator == :mutate ? 1.0 : 0.0,
        candidate.operator == :crossover ? 1.0 : 0.0,
    ]
    base = Float32[
        context...,
        candidate.heuristic_score,
        candidate.total_count,
        positive_fraction,
        candidate.exploration_bonus,
        candidate.structural_bias,
    ]
    return feature_mode == :augmented ? vcat(base, rank_features, op_flags) : base
end

function operator_candidate_feature_vector(snapshot::FrontierSnapshot,
                                           basin::BasinSummary,
                                           parent::FrontierSnapshotEntry,
                                           candidate::OperatorDecisionCandidate;
                                           all_candidates::Vector{OperatorDecisionCandidate},
                                           candidate_index::Int=1,
                                           step_index::Int=0,
                                           feature_mode::Symbol=:basic)
    context = operator_eligibility_feature_vector(snapshot, basin, parent, all_candidates; step_index=step_index)
    positive_fraction = candidate.total_count == 0 ? 0.0f0 : Float32(candidate.positive_delta_count / max(candidate.total_count, 1))
    heuristic_scores = Float64[c.heuristic_score for c in all_candidates]
    rank_features = _operator_rank_features(heuristic_scores, candidate_index)
    op_flags = Float32[
        candidate.operator == :mutate ? 1.0 : 0.0,
        candidate.operator == :crossover ? 1.0 : 0.0,
    ]
    base = Float32[
        context...,
        candidate.heuristic_score,
        candidate.total_count,
        positive_fraction,
        candidate.exploration_bonus,
        candidate.structural_bias,
    ]
    return feature_mode == :augmented ? vcat(base, rank_features, op_flags) : base
end

function operator_eligibility_score(model::LearnedOperatorEligibilityModel,
                                    features::AbstractVector{<:Real})
    x = Float32.(features)
    return _sigmoid(dot(model.weights, x) + model.bias)
end

function operator_eligibility_score(model::LearnedOperatorEligibilityModel,
                                    snapshot::FrontierSnapshot,
                                    basin::BasinSummary,
                                    parent::FrontierSnapshotEntry,
                                    candidates::Vector{OperatorDecisionCandidate};
                                    step_index::Int=0)
    return operator_eligibility_score(model,
        operator_eligibility_feature_vector(snapshot, basin, parent, candidates; step_index=step_index))
end

function operator_candidate_score(controller::LearnedOperatorController,
                                  features::AbstractVector{<:Real})
    x = Float32.(features)
    return dot(controller.weights, x) + controller.bias
end

function score_operator_candidates(controller,
                                   snapshot::FrontierSnapshot,
                                   basin::BasinSummary,
                                   parent::FrontierSnapshotEntry,
                                   candidates::Vector{OperatorDecisionCandidate};
                                   step_index::Int=0)
    isempty(candidates) && return Float32[]
    feature_mode = hasproperty(controller, :feature_mode) ? getfield(controller, :feature_mode) : :basic
    return Float32[
        operator_candidate_score(controller,
            operator_candidate_feature_vector(snapshot, basin, parent, candidate;
                all_candidates=candidates,
                candidate_index=idx,
                step_index=step_index,
                feature_mode=feature_mode))
        for (idx, candidate) in enumerate(candidates)
    ]
end

function operator_selection_metadata(controller,
                                     snapshot::FrontierSnapshot,
                                     basin::BasinSummary,
                                     parent::FrontierSnapshotEntry,
                                     candidates::Vector{OperatorDecisionCandidate};
                                     step_index::Int=0)
    isempty(candidates) && return Dict{String,Any}(
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
        "selection_reason" => "empty",
    )
    heuristic_scores = Float32[c.heuristic_score for c in candidates]
    heuristic_idx = argmax(heuristic_scores)
    chosen = select_operator(controller, snapshot, basin, parent, candidates; step_index=step_index)
    chosen_idx = something(findfirst(c -> c.operator == chosen.operator, candidates), heuristic_idx)
    return Dict{String,Any}(
        "chosen_index" => Int(chosen_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(chosen_idx),
        "heuristic_margin" => Float64(_opctrl_score_margin(heuristic_scores, heuristic_idx)),
        "learned_margin" => 0.0,
        "learned_advantage_vs_heuristic" => 0.0,
        "heuristic_entropy" => Float64(_opctrl_score_entropy(heuristic_scores)),
        "learned_entropy" => 0.0,
        "predicted_eligible" => true,
        "eligibility_score" => 1.0,
        "acted_on" => true,
        "preserved_to_heuristic" => false,
        "override_applied" => chosen_idx != heuristic_idx,
        "abstained_to_heuristic" => false,
        "selection_reason" => chosen_idx == heuristic_idx ? "agree" : "custom_override",
    )
end

function operator_selection_metadata(::HeuristicTopOperatorController,
                                     snapshot::FrontierSnapshot,
                                     basin::BasinSummary,
                                     parent::FrontierSnapshotEntry,
                                     candidates::Vector{OperatorDecisionCandidate};
                                     step_index::Int=0)
    isempty(candidates) && return operator_selection_metadata(nothing, snapshot, basin, parent, candidates; step_index=step_index)
    heuristic_scores = Float32[c.heuristic_score for c in candidates]
    heuristic_idx = argmax(heuristic_scores)
    return Dict{String,Any}(
        "chosen_index" => Int(heuristic_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(heuristic_idx),
        "heuristic_margin" => Float64(_opctrl_score_margin(heuristic_scores, heuristic_idx)),
        "learned_margin" => Float64(_opctrl_score_margin(heuristic_scores, heuristic_idx)),
        "learned_advantage_vs_heuristic" => 0.0,
        "heuristic_entropy" => Float64(_opctrl_score_entropy(heuristic_scores)),
        "learned_entropy" => Float64(_opctrl_score_entropy(heuristic_scores)),
        "predicted_eligible" => false,
        "eligibility_score" => 0.0,
        "acted_on" => false,
        "preserved_to_heuristic" => true,
        "override_applied" => false,
        "abstained_to_heuristic" => false,
        "selection_reason" => "heuristic_top",
    )
end

function operator_selection_metadata(controller::LearnedOperatorController,
                                     snapshot::FrontierSnapshot,
                                     basin::BasinSummary,
                                     parent::FrontierSnapshotEntry,
                                     candidates::Vector{OperatorDecisionCandidate};
                                     step_index::Int=0)
    isempty(candidates) && return operator_selection_metadata(HeuristicTopOperatorController(), snapshot, basin, parent, candidates; step_index=step_index)
    heuristic_scores = Float32[c.heuristic_score for c in candidates]
    learned_scores = score_operator_candidates(controller, snapshot, basin, parent, candidates; step_index=step_index)
    heuristic_idx = argmax(heuristic_scores)
    learned_idx = argmax(learned_scores)
    return Dict{String,Any}(
        "chosen_index" => Int(learned_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(learned_idx),
        "heuristic_margin" => Float64(_opctrl_score_margin(heuristic_scores, heuristic_idx)),
        "learned_margin" => Float64(_opctrl_score_margin(learned_scores, learned_idx)),
        "learned_advantage_vs_heuristic" => Float64(learned_scores[learned_idx] - learned_scores[heuristic_idx]),
        "heuristic_entropy" => Float64(_opctrl_score_entropy(heuristic_scores)),
        "learned_entropy" => Float64(_opctrl_score_entropy(learned_scores)),
        "predicted_eligible" => true,
        "eligibility_score" => 1.0,
        "acted_on" => true,
        "preserved_to_heuristic" => false,
        "override_applied" => learned_idx != heuristic_idx,
        "abstained_to_heuristic" => false,
        "selection_reason" => learned_idx == heuristic_idx ? "eligible_agree" : "learned_override",
    )
end

function operator_selection_metadata(controller::AnchoredOperatorController,
                                     snapshot::FrontierSnapshot,
                                     basin::BasinSummary,
                                     parent::FrontierSnapshotEntry,
                                     candidates::Vector{OperatorDecisionCandidate};
                                     step_index::Int=0)
    isempty(candidates) && return operator_selection_metadata(HeuristicTopOperatorController(), snapshot, basin, parent, candidates; step_index=step_index)
    heuristic_scores = Float32[c.heuristic_score for c in candidates]
    learned_scores = score_operator_candidates(controller.base_controller, snapshot, basin, parent, candidates; step_index=step_index)
    heuristic_idx = argmax(heuristic_scores)
    learned_idx = argmax(learned_scores)
    heuristic_margin = _opctrl_score_margin(heuristic_scores, heuristic_idx)
    learned_margin = _opctrl_score_margin(learned_scores, learned_idx)
    learned_advantage = learned_scores[learned_idx] - learned_scores[heuristic_idx]

    chosen_idx = learned_idx
    abstained = false
    preserved = false
    acted_on = true
    reason = learned_idx == heuristic_idx ? "eligible_agree" : "anchored_override"
    if learned_idx != heuristic_idx && heuristic_margin >= controller.preserve_margin
        chosen_idx = heuristic_idx
        abstained = true
        preserved = true
        acted_on = false
        reason = "preserve_strong_heuristic"
    elseif learned_idx != heuristic_idx && learned_margin < controller.learned_confidence_margin
        chosen_idx = heuristic_idx
        abstained = true
        preserved = true
        acted_on = false
        reason = "low_learned_confidence"
    elseif learned_idx != heuristic_idx && learned_advantage < controller.override_margin
        chosen_idx = heuristic_idx
        abstained = true
        preserved = true
        acted_on = false
        reason = "low_learned_advantage"
    end

    return Dict{String,Any}(
        "chosen_index" => Int(chosen_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(learned_idx),
        "heuristic_margin" => Float64(heuristic_margin),
        "learned_margin" => Float64(learned_margin),
        "learned_advantage_vs_heuristic" => Float64(learned_advantage),
        "heuristic_entropy" => Float64(_opctrl_score_entropy(heuristic_scores)),
        "learned_entropy" => Float64(_opctrl_score_entropy(learned_scores)),
        "predicted_eligible" => true,
        "eligibility_score" => 1.0,
        "acted_on" => acted_on,
        "preserved_to_heuristic" => preserved,
        "override_applied" => chosen_idx != heuristic_idx,
        "abstained_to_heuristic" => abstained,
        "selection_reason" => reason,
    )
end

function operator_selection_metadata(controller::EligibilityGatedOperatorController,
                                     snapshot::FrontierSnapshot,
                                     basin::BasinSummary,
                                     parent::FrontierSnapshotEntry,
                                     candidates::Vector{OperatorDecisionCandidate};
                                     step_index::Int=0)
    isempty(candidates) && return operator_selection_metadata(HeuristicTopOperatorController(), snapshot, basin, parent, candidates; step_index=step_index)
    eligibility_score = operator_eligibility_score(controller.eligibility_model, snapshot, basin, parent, candidates; step_index=step_index)
    predicted_eligible = eligibility_score >= controller.eligibility_threshold
    heuristic_scores = Float32[c.heuristic_score for c in candidates]
    heuristic_idx = argmax(heuristic_scores)

    if !predicted_eligible
        return Dict{String,Any}(
            "chosen_index" => Int(heuristic_idx),
            "heuristic_top_index" => Int(heuristic_idx),
            "learned_top_index" => Int(heuristic_idx),
            "heuristic_margin" => Float64(_opctrl_score_margin(heuristic_scores, heuristic_idx)),
            "learned_margin" => 0.0,
            "learned_advantage_vs_heuristic" => 0.0,
            "heuristic_entropy" => Float64(_opctrl_score_entropy(heuristic_scores)),
            "learned_entropy" => 0.0,
            "predicted_eligible" => false,
            "eligibility_score" => Float64(eligibility_score),
            "acted_on" => false,
            "preserved_to_heuristic" => true,
            "override_applied" => false,
            "abstained_to_heuristic" => true,
            "selection_reason" => "ineligible_preserve",
        )
    end

    base_meta = operator_selection_metadata(controller.base_controller, snapshot, basin, parent, candidates; step_index=step_index)
    return Dict{String,Any}(
        "chosen_index" => get(base_meta, "chosen_index", heuristic_idx),
        "heuristic_top_index" => get(base_meta, "heuristic_top_index", heuristic_idx),
        "learned_top_index" => get(base_meta, "learned_top_index", heuristic_idx),
        "heuristic_margin" => get(base_meta, "heuristic_margin", 0.0),
        "learned_margin" => get(base_meta, "learned_margin", 0.0),
        "learned_advantage_vs_heuristic" => get(base_meta, "learned_advantage_vs_heuristic", 0.0),
        "heuristic_entropy" => get(base_meta, "heuristic_entropy", 0.0),
        "learned_entropy" => get(base_meta, "learned_entropy", 0.0),
        "predicted_eligible" => true,
        "eligibility_score" => Float64(eligibility_score),
        "acted_on" => !Bool(get(base_meta, "preserved_to_heuristic", false)),
        "preserved_to_heuristic" => Bool(get(base_meta, "preserved_to_heuristic", false)),
        "override_applied" => Bool(get(base_meta, "override_applied", false)),
        "abstained_to_heuristic" => Bool(get(base_meta, "abstained_to_heuristic", false)),
        "selection_reason" => String(get(base_meta, "selection_reason", "eligible_agree")),
    )
end

function select_operator(::HeuristicTopOperatorController,
                         snapshot::FrontierSnapshot,
                         basin::BasinSummary,
                         parent::FrontierSnapshotEntry,
                         candidates::Vector{OperatorDecisionCandidate};
                         step_index::Int=0)
    isempty(candidates) && return nothing
    heuristic_scores = Float32[c.heuristic_score for c in candidates]
    return candidates[argmax(heuristic_scores)]
end

function select_operator(controller::LearnedOperatorController,
                         snapshot::FrontierSnapshot,
                         basin::BasinSummary,
                         parent::FrontierSnapshotEntry,
                         candidates::Vector{OperatorDecisionCandidate};
                         step_index::Int=0)
    isempty(candidates) && return nothing
    meta = operator_selection_metadata(controller, snapshot, basin, parent, candidates; step_index=step_index)
    return candidates[meta["chosen_index"]]
end

function select_operator(controller::AnchoredOperatorController,
                         snapshot::FrontierSnapshot,
                         basin::BasinSummary,
                         parent::FrontierSnapshotEntry,
                         candidates::Vector{OperatorDecisionCandidate};
                         step_index::Int=0)
    isempty(candidates) && return nothing
    meta = operator_selection_metadata(controller, snapshot, basin, parent, candidates; step_index=step_index)
    return candidates[meta["chosen_index"]]
end

function select_operator(controller::EligibilityGatedOperatorController,
                         snapshot::FrontierSnapshot,
                         basin::BasinSummary,
                         parent::FrontierSnapshotEntry,
                         candidates::Vector{OperatorDecisionCandidate};
                         step_index::Int=0)
    isempty(candidates) && return nothing
    meta = operator_selection_metadata(controller, snapshot, basin, parent, candidates; step_index=step_index)
    return candidates[meta["chosen_index"]]
end

function save_learned_operator_controller(path::String, controller)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, controller)
    end
    return path
end

function load_learned_operator_controller(path::String)
    open(path, "r") do io
        return deserialize(io)
    end
end
