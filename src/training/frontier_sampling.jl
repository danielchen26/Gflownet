# Frontier Sampling — snapshot-based parent/basin allocation for hierarchical edit search
#
# Design principle:
# - the live frontier is mutable across episodes
# - each episode should condition on a FROZEN snapshot for cleaner training semantics

using Statistics
using Random

"""
    FrontierSnapshotEntry

A frozen frontier entry used during one search episode.
"""
struct FrontierSnapshotEntry
    smiles::String
    scaffold::String
    reward::Float64
    novelty_score::Float64
    tb_delta_abs::Float64
    source::Symbol
end

"""
    FrontierSnapshot

Frozen frontier context for one finite-horizon search episode.
"""
struct FrontierSnapshot
    entries::Vector{FrontierSnapshotEntry}
    target_smiles::Union{Nothing,String}
    target_scaffold::Union{Nothing,String}
    budget_remaining::Int
    created_at_step::Int
    snapshot_id::UInt64
end

"""
    BasinSummary

Aggregate family/basin statistics derived from a frontier snapshot.
"""
struct BasinSummary
    scaffold::String
    count::Int
    best_reward::Float64
    mean_reward::Float64
    mean_novelty::Float64
    mean_delta::Float64
end

"""
    ScoredBasinCandidate

Deterministic scored basin candidate used for offline dataset capture and
optional learned basin selection.
"""
struct ScoredBasinCandidate
    basin::BasinSummary
    score::Float64
end

"""
    ScoredParentCandidate

Deterministic scored parent candidate used for online parent choice and offline
parent-controller dataset capture.
"""
struct ScoredParentCandidate
    entry::FrontierSnapshotEntry
    score::Float64
    visit_count::Int
    basin_match::Bool
    target_match::Bool
end

"""
    compute_frontier_snapshot_id(entries, target_smiles, budget_remaining, created_at_step)

Compute a stable identifier for a frozen snapshot.
"""
function compute_frontier_snapshot_id(entries::Vector{FrontierSnapshotEntry},
                                      target_smiles::Union{Nothing,String},
                                      budget_remaining::Int,
                                      created_at_step::Int)::UInt64
    basis = Tuple[(
        e.smiles,
        e.scaffold,
        round(e.reward, digits=8),
        round(e.novelty_score, digits=8),
        round(e.tb_delta_abs, digits=8),
        e.source,
    ) for e in entries]
    return UInt64(hash((basis, something(target_smiles, ""), budget_remaining, created_at_step)))
end

"""
    create_frontier_snapshot(buffer; max_entries=128, target_smiles=nothing,
                             budget_remaining=0, created_at_step=0)

Freeze the top region of frontier memory into a read-only snapshot.
"""
function create_frontier_snapshot(buffer::MolecularFrontierBuffer;
                                  max_entries::Int=128,
                                  target_smiles::Union{Nothing,String}=nothing,
                                  budget_remaining::Int=0,
                                  created_at_step::Int=0)
    if isempty(buffer)
        return FrontierSnapshot(
            FrontierSnapshotEntry[],
            target_smiles,
            nothing,
            budget_remaining,
            created_at_step,
            compute_frontier_snapshot_id(FrontierSnapshotEntry[], target_smiles, budget_remaining, created_at_step),
        )
    end

    top = frontier_topk(buffer, min(max_entries, length(buffer)); by=:reward)
    target_scaffold = isnothing(target_smiles) ? nothing : get_scaffold(target_smiles)
    entries = FrontierSnapshotEntry[
        FrontierSnapshotEntry(e.smiles, e.scaffold, e.reward, e.novelty_score, e.tb_delta_abs, e.source)
        for e in top
    ]
    snapshot_id = compute_frontier_snapshot_id(entries, target_smiles, budget_remaining, created_at_step)
    return FrontierSnapshot(entries, target_smiles, target_scaffold, budget_remaining, created_at_step, snapshot_id)
end

"""
    summarize_basins(snapshot)

Group snapshot entries by scaffold/family.
"""
function summarize_basins(snapshot::FrontierSnapshot)
    isempty(snapshot.entries) && return BasinSummary[]

    grouped = Dict{String, Vector{FrontierSnapshotEntry}}()
    for entry in snapshot.entries
        key = isempty(entry.scaffold) ? "__NO_SCAFFOLD__" : entry.scaffold
        if !haskey(grouped, key)
            grouped[key] = FrontierSnapshotEntry[]
        end
        push!(grouped[key], entry)
    end

    basins = BasinSummary[]
    for (scaffold, entries) in grouped
        rewards = [e.reward for e in entries]
        novelties = [e.novelty_score for e in entries]
        deltas = [e.tb_delta_abs for e in entries]
        push!(basins, BasinSummary(
            scaffold,
            length(entries),
            maximum(rewards),
            mean(rewards),
            mean(novelties),
            mean(deltas),
        ))
    end
    return sort(basins, by=b -> (-b.best_reward, -b.mean_novelty, -b.mean_delta, b.scaffold))
end

"""
    basin_score(snapshot, basin; reward_weight=1.0, novelty_weight=0.5,
                delta_weight=0.25, target_bonus=0.4)

Compute the basin sampling score used for diagnostic logging.
"""
function basin_score(snapshot::FrontierSnapshot,
                     basin::BasinSummary;
                     reward_weight::Float64=1.0,
                     novelty_weight::Float64=0.5,
                     delta_weight::Float64=0.25,
                     target_bonus::Float64=0.4)
    score = reward_weight * basin.best_reward +
            novelty_weight * basin.mean_novelty +
            delta_weight * basin.mean_delta
    if !isnothing(snapshot.target_scaffold) && basin.scaffold == snapshot.target_scaffold
        score *= (1.0 + target_bonus)
    end
    return max(score, 1e-8)
end

"""
    candidate_basins(snapshot; max_candidates=0, reward_weight=1.0,
                     novelty_weight=0.5, delta_weight=0.25, target_bonus=0.4)

Create a deterministic ranked candidate list of basin summaries for a frozen
snapshot. This is used both by the online heuristic/learned controller and by
offline dataset extraction to ensure action-space consistency.
"""
function candidate_basins(snapshot::FrontierSnapshot;
                          max_candidates::Int=0,
                          reward_weight::Float64=1.0,
                          novelty_weight::Float64=0.5,
                          delta_weight::Float64=0.25,
                          target_bonus::Float64=0.4)
    basins = summarize_basins(snapshot)
    isempty(basins) && return ScoredBasinCandidate[]

    scored = ScoredBasinCandidate[]
    for basin in basins
        push!(scored, ScoredBasinCandidate(
            basin,
            basin_score(snapshot, basin;
                reward_weight=reward_weight,
                novelty_weight=novelty_weight,
                delta_weight=delta_weight,
                target_bonus=target_bonus),
        ))
    end

    sort!(scored, by=c -> (-c.score, -c.basin.best_reward, -c.basin.mean_reward, c.basin.scaffold))
    if max_candidates > 0
        return scored[1:min(max_candidates, length(scored))]
    end
    return scored
end

"""
    sample_scored_basin(candidates)

Sample a basin from an already-computed scored candidate list.
"""
function sample_scored_basin(candidates::Vector{ScoredBasinCandidate})
    isempty(candidates) && return nothing
    scores = Float64[max(c.score, 1e-8) for c in candidates]
    idx = _sample_index_from_scores(scores)
    return candidates[idx]
end

"""
    sample_basin(snapshot; reward_weight=1.0, novelty_weight=0.5,
                 delta_weight=0.25, target_bonus=0.4, max_candidates=0)

Choose a scaffold/family basin from the frontier snapshot.
"""
function sample_basin(snapshot::FrontierSnapshot;
                      reward_weight::Float64=1.0,
                      novelty_weight::Float64=0.5,
                      delta_weight::Float64=0.25,
                      target_bonus::Float64=0.4,
                      max_candidates::Int=0)
    candidates = candidate_basins(snapshot;
        max_candidates=max_candidates,
        reward_weight=reward_weight,
        novelty_weight=novelty_weight,
        delta_weight=delta_weight,
        target_bonus=target_bonus)
    chosen = sample_scored_basin(candidates)
    return isnothing(chosen) ? nothing : chosen.basin
end

"""
    parent_score(snapshot, entry; basin=nothing, reward_weight=1.0,
                 novelty_weight=0.35, delta_weight=0.15, source_bonus=0.1,
                 target_bonus=0.2, visit_counts=nothing, visit_decay=0.5)

Compute the heuristic parent-selection score used for deterministic candidate
capture and heuristic/learned parent selection.
"""
function parent_score(snapshot::FrontierSnapshot,
                      entry::FrontierSnapshotEntry;
                      basin::Union{Nothing,BasinSummary}=nothing,
                      reward_weight::Float64=1.0,
                      novelty_weight::Float64=0.35,
                      delta_weight::Float64=0.15,
                      source_bonus::Float64=0.1,
                      target_bonus::Float64=0.2,
                      visit_counts::Union{Nothing,Dict{String,Int}}=nothing,
                      visit_decay::Float64=0.5)
    score = reward_weight * entry.reward +
            novelty_weight * entry.novelty_score +
            delta_weight * entry.tb_delta_abs
    if entry.source == :seed
        score *= (1.0 + source_bonus)
    end
    if !isnothing(snapshot.target_smiles) && entry.smiles == snapshot.target_smiles
        score *= (1.0 + target_bonus)
    end
    if !isnothing(snapshot.target_scaffold) && !isempty(entry.scaffold) && entry.scaffold == snapshot.target_scaffold
        score *= (1.0 + 0.5 * target_bonus)
    end
    if !isnothing(visit_counts)
        visits = get(visit_counts, entry.smiles, 0)
        score *= 1.0 / (1.0 + visit_decay * visits)
    end
    return max(score, 1e-8)
end

"""
    candidate_parents(snapshot; basin=nothing, max_candidates=0, reward_weight=1.0,
                      novelty_weight=0.35, delta_weight=0.15, source_bonus=0.1,
                      target_bonus=0.2, visit_counts=nothing, visit_decay=0.5)

Create a deterministic ranked candidate list of concrete parents from a frozen
frontier snapshot. This is used both by the online parent controller and by the
offline parent-controller dataset extraction path.
"""
function candidate_parents(snapshot::FrontierSnapshot;
                           basin::Union{Nothing,BasinSummary}=nothing,
                           max_candidates::Int=0,
                           reward_weight::Float64=1.0,
                           novelty_weight::Float64=0.35,
                           delta_weight::Float64=0.15,
                           source_bonus::Float64=0.1,
                           target_bonus::Float64=0.2,
                           visit_counts::Union{Nothing,Dict{String,Int}}=nothing,
                           visit_decay::Float64=0.5,
                           restrict_to_basin::Bool=false)
    entries = if !isnothing(basin) && restrict_to_basin
        [e for e in snapshot.entries if (isempty(e.scaffold) ? "__NO_SCAFFOLD__" : e.scaffold) == basin.scaffold]
    else
        snapshot.entries
    end
    isempty(entries) && return ScoredParentCandidate[]

    scored = ScoredParentCandidate[]
    for entry in entries
        visits = isnothing(visit_counts) ? 0 : get(visit_counts, entry.smiles, 0)
        push!(scored, ScoredParentCandidate(
            entry,
            parent_score(snapshot, entry;
                basin=basin,
                reward_weight=reward_weight,
                novelty_weight=novelty_weight,
                delta_weight=delta_weight,
                source_bonus=source_bonus,
                target_bonus=target_bonus,
                visit_counts=visit_counts,
                visit_decay=visit_decay),
            visits,
            !isnothing(basin) && ((isempty(entry.scaffold) ? "__NO_SCAFFOLD__" : entry.scaffold) == basin.scaffold),
            !isnothing(snapshot.target_smiles) && entry.smiles == snapshot.target_smiles,
        ))
    end

    sort!(scored, by=c -> (-c.score, -c.entry.reward, -c.entry.novelty_score, c.entry.smiles))
    if max_candidates > 0
        return scored[1:min(max_candidates, length(scored))]
    end
    return scored
end

"""
    sample_scored_parent(candidates)

Sample a parent from an already-computed scored parent candidate list.
"""
function sample_scored_parent(candidates::Vector{ScoredParentCandidate})
    isempty(candidates) && return nothing
    scores = Float64[max(c.score, 1e-8) for c in candidates]
    idx = _sample_index_from_scores(scores)
    return candidates[idx]
end

"""
    sample_parent(snapshot; basin=nothing, reward_weight=1.0,
                  novelty_weight=0.35, delta_weight=0.15, source_bonus=0.1,
                  visit_counts=nothing, visit_decay=0.5, max_candidates=0)

Select a concrete parent molecule from a frontier snapshot, optionally restricted
to a chosen basin. Parents that have been selected many times get downweighted
by `1/(1 + visit_decay * visits)` to encourage exploration of underused parents.
"""
function sample_parent(snapshot::FrontierSnapshot;
                       basin::Union{Nothing,BasinSummary}=nothing,
                       reward_weight::Float64=1.0,
                       novelty_weight::Float64=0.35,
                       delta_weight::Float64=0.15,
                       source_bonus::Float64=0.1,
                       visit_counts::Union{Nothing,Dict{String,Int}}=nothing,
                       visit_decay::Float64=0.5,
                       max_candidates::Int=0)
    candidates = candidate_parents(snapshot;
        basin=basin,
        max_candidates=max_candidates,
        reward_weight=reward_weight,
        novelty_weight=novelty_weight,
        delta_weight=delta_weight,
        source_bonus=source_bonus,
        visit_counts=visit_counts,
        visit_decay=visit_decay,
        restrict_to_basin=true)
    chosen = sample_scored_parent(candidates)
    return isnothing(chosen) ? nothing : chosen.entry
end

function _sample_index_from_scores(scores::Vector{Float64})
    total = sum(scores)
    probs = total > 0 ? scores ./ total : fill(1.0 / length(scores), length(scores))
    cumulative = cumsum(probs)
    r = rand()
    return clamp(searchsortedfirst(cumulative, r), 1, length(scores))
end
