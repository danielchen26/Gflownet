# Molecular Frontier Buffer — first-class search memory for Hierarchical Edit-Flow GFlowNets
#
# This extends the simple reward replay buffer into a frontier-oriented memory that tracks:
# - reward / top-molecule exploitation
# - scaffold novelty / structural diversity
# - provenance (model, seed, GA, edit operator)
# - local flow imbalance signal (|δ|)
#
# The intent is to support a pivot from token-only de novo generation toward
# frontier-conditioned molecular search.

using Random
using Statistics
using PythonCall

# =============================================================================
# Graph Identity Helper
# =============================================================================

"""
    canonicalize_smiles_identity(smiles) -> String

Return a canonical graph-identity key for a SMILES string.
Falls back to the stripped raw string if canonicalization fails.
"""
function canonicalize_smiles_identity(smiles::AbstractString)::String
    stripped = strip(smiles)
    isempty(stripped) && return ""
    try
        Chem = pyimport("rdkit.Chem")
        mol = Chem.MolFromSmiles(stripped)
        pyis(mol, pybuiltins.None) && return stripped
        return pyconvert(String, Chem.MolToSmiles(mol))
    catch
        return stripped
    end
end

# =============================================================================
# Frontier Entry
# =============================================================================

"""
    MolecularFrontierEntry

A single molecule stored in frontier memory.

Fields capture not just reward, but also where the molecule came from and
why it should be revisited.
"""
mutable struct MolecularFrontierEntry
    smiles::String
    scaffold::String
    reward::Float64
    source::Symbol                  # :model, :seed, :ga, :mutation, :crossover, :edit
    parent_smiles::Union{Nothing,String}
    operator::Symbol                # :sample, :seed, :augment, :mutation, :crossover, ...
    novelty_score::Float64          # Higher = underrepresented scaffold / structurally novel
    tb_delta_abs::Float64           # |TB error| or local flow mismatch proxy
    visits::Int                     # Number of times sampled from frontier
end

# =============================================================================
# Frontier Buffer
# =============================================================================

"""
    MolecularFrontierBuffer

Frontier-oriented molecular memory for edit/search-based GFlowNet training.

Unlike `SMILESReplayBuffer`, which focuses on reward-ranked teacher-forced replay,
this buffer is designed to support *search decisions*:
- which molecule/scaffold to expand next,
- which frontier basins are underexplored,
- which entries are high reward but also information-rich.
"""
mutable struct MolecularFrontierBuffer
    entries::Vector{MolecularFrontierEntry}
    max_size::Int
    seen_smiles::Dict{String, Int}
    scaffold_counts::Dict{String, Int}
    needs_refresh::Bool

    function MolecularFrontierBuffer(max_size::Int=5000)
        new(MolecularFrontierEntry[], max_size, Dict{String, Int}(), Dict{String, Int}(), false)
    end
end

Base.length(buf::MolecularFrontierBuffer) = length(buf.entries)
Base.isempty(buf::MolecularFrontierBuffer) = isempty(buf.entries)

# =============================================================================
# Core Operations
# =============================================================================

"""
    add_to_frontier!(buffer, smiles; reward, source=:model, parent_smiles=nothing,
                     operator=:sample, tb_delta_abs=0.0)

Add or update a molecule in the frontier. Novelty is scaffold-based and recomputed
from scaffold counts.
Graph identity is canonicalized before insertion so randomized SMILES variants of
one molecular graph collapse to a single frontier entry.
"""
function add_to_frontier!(buffer::MolecularFrontierBuffer, smiles::String;
                          reward::Float64,
                          source::Symbol=:model,
                          parent_smiles::Union{Nothing,String}=nothing,
                          operator::Symbol=:sample,
                          tb_delta_abs::Float64=0.0)
    identity_smiles = canonicalize_smiles_identity(smiles)
    isempty(identity_smiles) && return nothing
    reward <= 0.0 && return nothing

    canonical_parent = isnothing(parent_smiles) ? nothing : canonicalize_smiles_identity(parent_smiles)
    scaffold = get_scaffold(identity_smiles)
    novelty = _compute_scaffold_novelty(buffer, scaffold)

    if haskey(buffer.seen_smiles, identity_smiles)
        idx = buffer.seen_smiles[identity_smiles]
        if idx <= length(buffer.entries)
            entry = buffer.entries[idx]
            entry.reward = max(entry.reward, reward)
            entry.tb_delta_abs = max(entry.tb_delta_abs, abs(tb_delta_abs))
            entry.novelty_score = max(entry.novelty_score, novelty)
            if reward >= entry.reward
                entry.source = source
                entry.parent_smiles = canonical_parent
                entry.operator = operator
            end
        end
    else
        entry = MolecularFrontierEntry(
            identity_smiles,
            scaffold,
            reward,
            source,
            canonical_parent,
            operator,
            novelty,
            abs(tb_delta_abs),
            0,
        )
        push!(buffer.entries, entry)
        buffer.seen_smiles[identity_smiles] = length(buffer.entries)
        if !isempty(scaffold)
            buffer.scaffold_counts[scaffold] = get(buffer.scaffold_counts, scaffold, 0) + 1
        end
        if length(buffer.entries) > buffer.max_size
            _evict_frontier!(buffer)
        end
    end

    buffer.needs_refresh = true
    return nothing
end

"""
    update_frontier_delta!(buffer, smiles, tb_delta_abs)

Update the local flow-imbalance magnitude for a molecule already in frontier memory.
"""
function update_frontier_delta!(buffer::MolecularFrontierBuffer,
                                smiles::String,
                                tb_delta_abs::Float64)
    identity_smiles = canonicalize_smiles_identity(smiles)
    if haskey(buffer.seen_smiles, identity_smiles)
        idx = buffer.seen_smiles[identity_smiles]
        if idx <= length(buffer.entries)
            buffer.entries[idx].tb_delta_abs = abs(tb_delta_abs)
        end
    end
    return nothing
end

"""
    sample_frontier(buffer, n; reward_weight=1.0, novelty_weight=0.5,
                    delta_weight=0.25, target_scaffold=nothing)

Sample frontier entries using a mixed priority score.

Priority components:
- reward rank: exploit high-value molecules
- novelty rank: revisit underrepresented scaffold basins
- delta rank: replay entries with high local flow mismatch
- target bonus: upweight entries matching a target scaffold for structural tasks
"""
function sample_frontier(buffer::MolecularFrontierBuffer, n::Int;
                         reward_weight::Float64=1.0,
                         novelty_weight::Float64=0.5,
                         delta_weight::Float64=0.25,
                         target_scaffold::Union{Nothing,String}=nothing)
    isempty(buffer) && return MolecularFrontierEntry[]

    _refresh_frontier!(buffer)

    n_available = length(buffer.entries)
    if n_available == 1
        buffer.entries[1].visits += 1
        return [buffer.entries[1] for _ in 1:n]
    end

    rewards = [entry.reward for entry in buffer.entries]
    novelties = [entry.novelty_score for entry in buffer.entries]
    deltas = [entry.tb_delta_abs for entry in buffer.entries]

    reward_ranks = _rank_descending(rewards)
    novelty_ranks = _rank_descending(novelties)
    delta_ranks = _rank_descending(deltas)

    scores = Float64[]
    for i in 1:n_available
        entry = buffer.entries[i]
        score = reward_weight * (1.0 / reward_ranks[i]) +
                novelty_weight * (1.0 / novelty_ranks[i]) +
                delta_weight * (1.0 / delta_ranks[i])

        if !isnothing(target_scaffold) && !isempty(entry.scaffold) && entry.scaffold == target_scaffold
            score *= 1.25
        end
        push!(scores, score)
    end

    total = sum(scores)
    probs = total > 0 ? scores ./ total : fill(1.0 / n_available, n_available)
    cumulative = cumsum(probs)

    sampled = MolecularFrontierEntry[]
    for _ in 1:n
        r = rand()
        idx = clamp(searchsortedfirst(cumulative, r), 1, n_available)
        buffer.entries[idx].visits += 1
        push!(sampled, buffer.entries[idx])
    end

    return sampled
end

"""
    frontier_topk(buffer, k; by=:reward)

Return top-k frontier entries ranked by reward, novelty, or delta magnitude.
"""
function frontier_topk(buffer::MolecularFrontierBuffer, k::Int; by::Symbol=:reward)
    isempty(buffer) && return MolecularFrontierEntry[]
    _refresh_frontier!(buffer)
    metric = if by == :reward
        e -> -e.reward
    elseif by == :novelty
        e -> -e.novelty_score
    elseif by == :delta
        e -> -e.tb_delta_abs
    else
        e -> -e.reward
    end
    n = min(k, length(buffer.entries))
    return sort(buffer.entries, by=metric)[1:n]
end

"""
    frontier_stats(buffer)

Summary statistics for the frontier memory.
"""
function frontier_stats(buffer::MolecularFrontierBuffer)
    if isempty(buffer)
        return Dict(
            "size" => 0,
            "n_scaffolds" => 0,
            "mean_reward" => 0.0,
            "max_reward" => 0.0,
            "mean_novelty" => 0.0,
            "mean_delta" => 0.0,
            "seed_fraction" => 0.0,
            "ga_fraction" => 0.0,
            "source_counts" => Dict{String,Int}(),
            "operator_counts" => Dict{String,Int}(),
        )
    end

    rewards = [e.reward for e in buffer.entries]
    novelties = [e.novelty_score for e in buffer.entries]
    deltas = [e.tb_delta_abs for e in buffer.entries]
    sources = [e.source for e in buffer.entries]

    source_counts = Dict{String,Int}()
    operator_counts = Dict{String,Int}()
    for entry in buffer.entries
        source_key = String(entry.source)
        source_counts[source_key] = get(source_counts, source_key, 0) + 1
        operator_key = String(entry.operator)
        operator_counts[operator_key] = get(operator_counts, operator_key, 0) + 1
    end

    graph_unique = length(buffer.seen_smiles)
    entry_count = length(buffer.entries)

    return Dict(
        "size" => entry_count,
        "graph_unique_count" => graph_unique,
        "string_vs_graph_ratio" => entry_count > 0 ? entry_count / max(graph_unique, 1) : 1.0,
        "n_scaffolds" => length(buffer.scaffold_counts),
        "mean_reward" => mean(rewards),
        "max_reward" => maximum(rewards),
        "mean_novelty" => mean(novelties),
        "mean_delta" => mean(deltas),
        "seed_fraction" => mean(Float64[s == :seed for s in sources]),
        "ga_fraction" => mean(Float64[s in (:ga, :mutation, :crossover) for s in sources]),
        "source_counts" => source_counts,
        "operator_counts" => operator_counts,
    )
end

"""
    frontier_source_summary(buffer; topk=10)

Report source attribution for the full frontier and the current top-k frontier.
"""
function frontier_source_summary(buffer::MolecularFrontierBuffer; topk::Int=10)
    top_entries = frontier_topk(buffer, topk; by=:reward)
    overall_sources = Dict{String,Int}()
    overall_operators = Dict{String,Int}()
    topk_sources = Dict{String,Int}()
    topk_operators = Dict{String,Int}()
    topk_edit_operators = Dict{String,Int}()

    for entry in buffer.entries
        source_key = String(entry.source)
        op_key = String(entry.operator)
        overall_sources[source_key] = get(overall_sources, source_key, 0) + 1
        overall_operators[op_key] = get(overall_operators, op_key, 0) + 1
    end
    for entry in top_entries
        source_key = String(entry.source)
        op_key = String(entry.operator)
        topk_sources[source_key] = get(topk_sources, source_key, 0) + 1
        topk_operators[op_key] = get(topk_operators, op_key, 0) + 1
        if entry.source == :edit
            topk_edit_operators[op_key] = get(topk_edit_operators, op_key, 0) + 1
        end
    end

    top1_entry = isempty(top_entries) ? nothing : top_entries[1]
    return Dict(
        "overall" => overall_sources,
        "overall_operators" => overall_operators,
        "topk" => topk_sources,
        "topk_operators" => topk_operators,
        "topk_edit_operators" => topk_edit_operators,
        "topk_size" => length(top_entries),
        "top1_source" => isnothing(top1_entry) ? "none" : String(top1_entry.source),
        "top1_operator" => isnothing(top1_entry) ? "none" : String(top1_entry.operator),
    )
end

# =============================================================================
# Helpers
# =============================================================================

function _compute_scaffold_novelty(buffer::MolecularFrontierBuffer, scaffold::String)
    isempty(scaffold) && return 0.5
    count = get(buffer.scaffold_counts, scaffold, 0)
    return count == 0 ? 1.0 : 1.0 / (1.0 + count)
end

function _rank_descending(values::Vector{Float64})
    perm = sortperm(values, rev=true)
    ranks = similar(perm)
    for (rank, idx) in enumerate(perm)
        ranks[idx] = rank
    end
    return ranks
end

function _refresh_frontier!(buffer::MolecularFrontierBuffer)
    if !buffer.needs_refresh
        return nothing
    end

    for entry in buffer.entries
        entry.novelty_score = _compute_scaffold_novelty(buffer, entry.scaffold)
    end
    buffer.needs_refresh = false
    return nothing
end

function _evict_frontier!(buffer::MolecularFrontierBuffer)
    _refresh_frontier!(buffer)

    sort!(buffer.entries, by=e -> (-(0.65 * e.reward + 0.20 * e.novelty_score + 0.15 * e.tb_delta_abs), -e.reward))
    while length(buffer.entries) > buffer.max_size
        removed = pop!(buffer.entries)
        delete!(buffer.seen_smiles, removed.smiles)
        if !isempty(removed.scaffold) && haskey(buffer.scaffold_counts, removed.scaffold)
            new_count = buffer.scaffold_counts[removed.scaffold] - 1
            if new_count <= 0
                delete!(buffer.scaffold_counts, removed.scaffold)
            else
                buffer.scaffold_counts[removed.scaffold] = new_count
            end
        end
    end

    empty!(buffer.seen_smiles)
    for (i, entry) in enumerate(buffer.entries)
        buffer.seen_smiles[entry.smiles] = i
    end
    return nothing
end
