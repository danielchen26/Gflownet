# Parent Controller Models
#
# Batch 1C scope:
# - keep the first parent controller simple and stable
# - compare linear / small MLP scorers
# - support principled heuristic-anchored override variants
# - expose decision metadata for abstention / transfer-safety analysis

using Serialization
using Random

struct LearnedParentController
    weights::Vector{Float32}
    bias::Float32
    input_dim::Int
    feature_mode::Symbol
end

struct MLPParentController
    W1::Matrix{Float32}
    b1::Vector{Float32}
    W2::Vector{Float32}
    b2::Float32
    input_dim::Int
    hidden_dim::Int
    feature_mode::Symbol
end

struct HeuristicTopParentController end

struct AnchoredParentController{T}
    base_controller::T
    override_margin::Float32
    preserve_margin::Float32
    learned_confidence_margin::Float32
end

function create_learned_parent_controller(input_dim::Int;
                                          hidden_dim::Int=32,
                                          feature_mode::Symbol=:basic,
                                          rng::AbstractRNG=Random.MersenneTwister(0))
    weights = 0.01f0 .* randn(rng, Float32, input_dim)
    return LearnedParentController(weights, 0.0f0, input_dim, feature_mode)
end

function create_mlp_parent_controller(input_dim::Int;
                                      hidden_dim::Int=32,
                                      feature_mode::Symbol=:basic,
                                      rng::AbstractRNG=Random.MersenneTwister(0))
    W1 = 0.05f0 .* randn(rng, Float32, hidden_dim, input_dim)
    b1 = zeros(Float32, hidden_dim)
    W2 = 0.05f0 .* randn(rng, Float32, hidden_dim)
    b2 = 0.0f0
    return MLPParentController(W1, b1, W2, b2, input_dim, hidden_dim, feature_mode)
end

create_anchored_parent_controller(base_controller;
                                  override_margin::Float32=0.05f0,
                                  preserve_margin::Float32=0.15f0,
                                  learned_confidence_margin::Float32=0.05f0) =
    AnchoredParentController(base_controller, override_margin, preserve_margin, learned_confidence_margin)

function parent_candidate_score(controller::LearnedParentController,
                                features::AbstractVector{<:Real})
    x = Float32.(features)
    return dot(controller.weights, x) + controller.bias
end

function parent_candidate_score(controller::MLPParentController,
                                features::AbstractVector{<:Real})
    x = Float32.(features)
    h = tanh.(controller.W1 * x .+ controller.b1)
    return dot(controller.W2, h) + controller.b2
end

function score_parent_candidates(controller,
                                 snapshot::FrontierSnapshot,
                                 candidates::Vector{ScoredParentCandidate};
                                 step_index::Int=0)
    isempty(candidates) && return Float32[]
    return Float32[
        parent_candidate_score(controller,
            parent_candidate_feature_vector(snapshot, candidate;
                step_index=step_index,
                all_candidates=candidates,
                candidate_index=idx,
                feature_mode=getfield(controller, :feature_mode)))
        for (idx, candidate) in enumerate(candidates)
    ]
end

_parent_score_margin(scores::AbstractVector{<:Real}, top_idx::Int) =
    isempty(scores) ? 0.0f0 :
    length(scores) == 1 ? Float32(scores[top_idx]) : begin
        order = sortperm(Float32.(scores), rev=true)
        top = Float32(scores[order[1]])
        second = Float32(scores[order[2]])
        top - second
    end

function _parent_score_entropy(scores::AbstractVector{<:Real})
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

function parent_selection_metadata(controller,
                                   snapshot::FrontierSnapshot,
                                   candidates::Vector{ScoredParentCandidate};
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
        "override_applied" => false,
        "abstained_to_heuristic" => false,
        "selection_reason" => "empty",
    )
    heuristic_scores = Float32[c.score for c in candidates]
    heuristic_idx = argmax(heuristic_scores)
    chosen = select_parent(controller, snapshot, candidates; step_index=step_index)
    chosen_idx = something(findfirst(c -> c.entry.smiles == chosen.entry.smiles, candidates), heuristic_idx)
    return Dict{String,Any}(
        "chosen_index" => Int(chosen_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(chosen_idx),
        "heuristic_margin" => Float64(_parent_score_margin(heuristic_scores, heuristic_idx)),
        "learned_margin" => 0.0,
        "learned_advantage_vs_heuristic" => 0.0,
        "heuristic_entropy" => Float64(_parent_score_entropy(heuristic_scores)),
        "learned_entropy" => 0.0,
        "override_applied" => chosen_idx != heuristic_idx,
        "abstained_to_heuristic" => false,
        "selection_reason" => chosen_idx == heuristic_idx ? "agree" : "custom_override",
    )
end

function parent_selection_metadata(controller::LearnedParentController,
                                   snapshot::FrontierSnapshot,
                                   candidates::Vector{ScoredParentCandidate};
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
        "override_applied" => false,
        "abstained_to_heuristic" => false,
        "selection_reason" => "empty",
    )
    heuristic_scores = Float32[c.score for c in candidates]
    learned_scores = score_parent_candidates(controller, snapshot, candidates; step_index=step_index)
    heuristic_idx = argmax(heuristic_scores)
    learned_idx = argmax(learned_scores)
    return Dict{String,Any}(
        "chosen_index" => Int(learned_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(learned_idx),
        "heuristic_margin" => Float64(_parent_score_margin(heuristic_scores, heuristic_idx)),
        "learned_margin" => Float64(_parent_score_margin(learned_scores, learned_idx)),
        "learned_advantage_vs_heuristic" => Float64(learned_scores[learned_idx] - learned_scores[heuristic_idx]),
        "heuristic_entropy" => Float64(_parent_score_entropy(heuristic_scores)),
        "learned_entropy" => Float64(_parent_score_entropy(learned_scores)),
        "override_applied" => learned_idx != heuristic_idx,
        "abstained_to_heuristic" => false,
        "selection_reason" => learned_idx == heuristic_idx ? "agree" : "learned_override",
    )
end

function parent_selection_metadata(controller::MLPParentController,
                                   snapshot::FrontierSnapshot,
                                   candidates::Vector{ScoredParentCandidate};
                                   step_index::Int=0)
    isempty(candidates) && return parent_selection_metadata(create_learned_parent_controller(1), snapshot, candidates; step_index=step_index)
    heuristic_scores = Float32[c.score for c in candidates]
    learned_scores = score_parent_candidates(controller, snapshot, candidates; step_index=step_index)
    heuristic_idx = argmax(heuristic_scores)
    learned_idx = argmax(learned_scores)
    return Dict{String,Any}(
        "chosen_index" => Int(learned_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(learned_idx),
        "heuristic_margin" => Float64(_parent_score_margin(heuristic_scores, heuristic_idx)),
        "learned_margin" => Float64(_parent_score_margin(learned_scores, learned_idx)),
        "learned_advantage_vs_heuristic" => Float64(learned_scores[learned_idx] - learned_scores[heuristic_idx]),
        "heuristic_entropy" => Float64(_parent_score_entropy(heuristic_scores)),
        "learned_entropy" => Float64(_parent_score_entropy(learned_scores)),
        "override_applied" => learned_idx != heuristic_idx,
        "abstained_to_heuristic" => false,
        "selection_reason" => learned_idx == heuristic_idx ? "agree" : "learned_override",
    )
end

function parent_selection_metadata(controller::AnchoredParentController,
                                   snapshot::FrontierSnapshot,
                                   candidates::Vector{ScoredParentCandidate};
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
        "override_applied" => false,
        "abstained_to_heuristic" => false,
        "selection_reason" => "empty",
    )

    heuristic_scores = Float32[c.score for c in candidates]
    learned_scores = score_parent_candidates(controller.base_controller, snapshot, candidates; step_index=step_index)
    heuristic_idx = argmax(heuristic_scores)
    learned_idx = argmax(learned_scores)
    heuristic_margin = _parent_score_margin(heuristic_scores, heuristic_idx)
    learned_margin = _parent_score_margin(learned_scores, learned_idx)
    learned_advantage_vs_heuristic = learned_scores[learned_idx] - learned_scores[heuristic_idx]

    chosen_idx = learned_idx
    abstained = false
    reason = learned_idx == heuristic_idx ? "agree" : "anchored_override"
    if learned_idx != heuristic_idx && heuristic_margin >= controller.preserve_margin
        chosen_idx = heuristic_idx
        abstained = true
        reason = "preserve_strong_heuristic"
    elseif learned_idx != heuristic_idx && learned_margin < controller.learned_confidence_margin
        chosen_idx = heuristic_idx
        abstained = true
        reason = "low_learned_confidence"
    elseif learned_idx != heuristic_idx && learned_advantage_vs_heuristic < controller.override_margin
        chosen_idx = heuristic_idx
        abstained = true
        reason = "low_learned_advantage"
    end

    return Dict{String,Any}(
        "chosen_index" => Int(chosen_idx),
        "heuristic_top_index" => Int(heuristic_idx),
        "learned_top_index" => Int(learned_idx),
        "heuristic_margin" => Float64(heuristic_margin),
        "learned_margin" => Float64(learned_margin),
        "learned_advantage_vs_heuristic" => Float64(learned_advantage_vs_heuristic),
        "heuristic_entropy" => Float64(_parent_score_entropy(heuristic_scores)),
        "learned_entropy" => Float64(_parent_score_entropy(learned_scores)),
        "override_applied" => chosen_idx != heuristic_idx,
        "abstained_to_heuristic" => abstained,
        "selection_reason" => reason,
    )
end

function select_parent(::HeuristicTopParentController,
                       snapshot::FrontierSnapshot,
                       candidates::Vector{ScoredParentCandidate};
                       step_index::Int=0)
    isempty(candidates) && return nothing
    heuristic_scores = Float32[c.score for c in candidates]
    return candidates[argmax(heuristic_scores)]
end

function select_parent(controller::LearnedParentController,
                       snapshot::FrontierSnapshot,
                       candidates::Vector{ScoredParentCandidate};
                       step_index::Int=0)
    isempty(candidates) && return nothing
    meta = parent_selection_metadata(controller, snapshot, candidates; step_index=step_index)
    return candidates[meta["chosen_index"]]
end

function select_parent(controller::MLPParentController,
                       snapshot::FrontierSnapshot,
                       candidates::Vector{ScoredParentCandidate};
                       step_index::Int=0)
    isempty(candidates) && return nothing
    meta = parent_selection_metadata(controller, snapshot, candidates; step_index=step_index)
    return candidates[meta["chosen_index"]]
end

function select_parent(controller::AnchoredParentController,
                       snapshot::FrontierSnapshot,
                       candidates::Vector{ScoredParentCandidate};
                       step_index::Int=0)
    isempty(candidates) && return nothing
    meta = parent_selection_metadata(controller, snapshot, candidates; step_index=step_index)
    return candidates[meta["chosen_index"]]
end

function save_learned_parent_controller(path::String,
                                        controller)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, controller)
    end
    return path
end

function load_learned_parent_controller(path::String)
    open(path, "r") do io
        return deserialize(io)
    end
end
