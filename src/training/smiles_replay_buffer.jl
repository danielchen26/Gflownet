# SMILES Replay Buffer — Rank-Based Experience Replay for CAFE-GFN
#
# Inspired by Genetic GFN (Kim et al., NeurIPS 2024):
# - Store (SMILES, tokens, reward) tuples from all training iterations
# - Sample replay batches with rank-based priority: P(i) ∝ 1/rank(i)
# - Replay 4-8x per new sample for drastically improved sample efficiency
#
# Key difference from core ReplayBuffer: stores SMILES token sequences
# (not GFlowNet Trajectory objects) for teacher-forced loss computation.

using Random
using Statistics

# =============================================================================
# SMILES Replay Entry
# =============================================================================

"""
    SMILESReplayEntry

A single entry in the SMILES replay buffer.
"""
struct SMILESReplayEntry
    smiles::String       # Canonical SMILES
    tokens::Vector{Int}  # Token sequence (for teacher-forced loss)
    reward::Float64      # Oracle reward value
end

# =============================================================================
# SMILES Replay Buffer
# =============================================================================

"""
    SMILESReplayBuffer

Rank-based experience replay buffer for SMILES GFlowNet finetuning.

Stores (SMILES, tokens, reward) tuples and supports rank-based priority
sampling where P(i) ∝ 1/rank(i), ensuring high-reward molecules are
replayed more frequently.

# Design choices (from Genetic GFN, NeurIPS 2024)
- Rank-based priority > reward-proportional priority (more robust to scale)
- Buffer size 5000-10000 (large enough for diversity, small enough for memory)
- Deduplicated by SMILES string (same molecule stored once with max reward)
- Sorted by reward after each insertion batch

# Usage
```julia
buffer = SMILESReplayBuffer(5000)

# After each training step, add new molecules
for (smi, tok, rew) in zip(smiles_list, tokens_list, rewards)
    add_to_replay!(buffer, smi, tok, rew)
end

# Sample replay batch for additional training
replay_entries = sample_replay(buffer, 128)  # rank-weighted
```
"""
mutable struct SMILESReplayBuffer
    entries::Vector{SMILESReplayEntry}
    max_size::Int
    seen_smiles::Dict{String, Int}  # SMILES → index in entries (deduplication)
    needs_sort::Bool                # Whether entries need re-sorting after additions
    # --- Novel: |δ|-priority tracking (Direction B) ---
    tb_deltas::Dict{String, Float64}  # SMILES → |TB error| for δ-priority sampling

    function SMILESReplayBuffer(max_size::Int=5000)
        new(SMILESReplayEntry[], max_size, Dict{String, Int}(), false, Dict{String, Float64}())
    end
end

Base.length(buf::SMILESReplayBuffer) = length(buf.entries)
Base.isempty(buf::SMILESReplayBuffer) = isempty(buf.entries)

"""
    add_to_replay!(buffer, smiles, tokens, reward)

Add a molecule to the replay buffer. If the molecule already exists,
update its reward if the new reward is higher (keep best observation).
"""
function add_to_replay!(buffer::SMILESReplayBuffer, smiles::String,
                         tokens::Vector{Int}, reward::Float64)
    # Skip invalid entries
    isempty(smiles) && return nothing
    reward <= 0.0 && return nothing
    length(tokens) < 2 && return nothing

    if haskey(buffer.seen_smiles, smiles)
        # Already exists — update reward if higher
        idx = buffer.seen_smiles[smiles]
        if idx <= length(buffer.entries) && reward > buffer.entries[idx].reward
            buffer.entries[idx] = SMILESReplayEntry(smiles, tokens, reward)
            buffer.needs_sort = true
        end
    else
        # New molecule
        entry = SMILESReplayEntry(smiles, tokens, reward)
        push!(buffer.entries, entry)
        buffer.seen_smiles[smiles] = length(buffer.entries)
        buffer.needs_sort = true

        # Evict lowest-reward entries if over capacity
        if length(buffer.entries) > buffer.max_size
            _evict_lowest!(buffer)
        end
    end
    return nothing
end

"""
    add_batch_to_replay!(buffer, smiles_list, tokens_list, rewards)

Add a batch of molecules to the replay buffer.
"""
function add_batch_to_replay!(buffer::SMILESReplayBuffer,
                               smiles_list::Vector{String},
                               tokens_list::Vector{Vector{Int}},
                               rewards::Vector{Float64})
    for (smi, tok, rew) in zip(smiles_list, tokens_list, rewards)
        add_to_replay!(buffer, smi, tok, rew)
    end
    # Sort once after batch insertion
    if buffer.needs_sort
        _sort_and_reindex!(buffer)
    end
end

"""
    sample_replay(buffer, n; rank_weighted=true) → Vector{SMILESReplayEntry}

Sample n entries from the replay buffer.

If `rank_weighted=true`, uses rank-based priority: P(i) ∝ 1/rank(i).
This means the top-ranked molecule has ~N× higher probability than the
lowest-ranked one, providing strong exploitation signal.

If `rank_weighted=false`, samples uniformly.
"""
function sample_replay(buffer::SMILESReplayBuffer, n::Int;
                        rank_weighted::Bool=true)
    if isempty(buffer)
        return SMILESReplayEntry[]
    end

    # Ensure sorted
    if buffer.needs_sort
        _sort_and_reindex!(buffer)
    end

    n_available = length(buffer.entries)

    if !rank_weighted || n_available <= 1
        # Uniform sampling (cap at buffer size)
        n = min(n, n_available)
        indices = rand(1:n_available, n)
    else
        # Rank-based priority: P(i) ∝ 1/rank(i)
        # rank 1 (best) gets weight 1.0, rank N gets weight 1/N
        weights = [1.0 / i for i in 1:n_available]
        total_w = sum(weights)
        probs = weights ./ total_w

        # Sample with replacement (allows replaying top molecules multiple times)
        indices = Int[]
        cumulative = cumsum(probs)
        for _ in 1:n
            r = rand()
            idx = searchsortedfirst(cumulative, r)
            idx = clamp(idx, 1, n_available)
            push!(indices, idx)
        end
    end

    return [buffer.entries[i] for i in indices]
end

"""
    get_top_molecules(buffer, k) → Vector{SMILESReplayEntry}

Return the top-k molecules by reward.
"""
function get_top_molecules(buffer::SMILESReplayBuffer, k::Int)
    if buffer.needs_sort
        _sort_and_reindex!(buffer)
    end
    n = min(k, length(buffer.entries))
    return buffer.entries[1:n]
end

"""
    replay_stats(buffer) → Dict

Return summary statistics of the replay buffer.
"""
function replay_stats(buffer::SMILESReplayBuffer)
    if isempty(buffer)
        return Dict("size" => 0, "mean_reward" => 0.0, "max_reward" => 0.0,
                     "min_reward" => 0.0, "unique_smiles" => 0)
    end
    rewards = [e.reward for e in buffer.entries]
    return Dict(
        "size" => length(buffer.entries),
        "mean_reward" => mean(rewards),
        "max_reward" => maximum(rewards),
        "min_reward" => minimum(rewards),
        "unique_smiles" => length(buffer.seen_smiles),
    )
end

# =============================================================================
# |δ|-Priority Replay (Direction B: Novel, Convergence-Safe)
# =============================================================================

"""
    update_deltas!(buffer, smiles_list, delta_values)

Update |TB error| values for molecules in the buffer. These are used for
δ-priority sampling, which is the GFlowNet analogue of Prioritized Experience
Replay (Schaul et al., 2015).

Unlike reward augmentation (Loss-Guided GFN, 2025), δ-priority replay does NOT
modify the training objective — it only affects which molecules are replayed.
This preserves TB convergence guarantees unconditionally.
"""
function update_deltas!(buffer::SMILESReplayBuffer,
                         smiles_list::Vector{String},
                         delta_values::Vector{Float64})
    for (smi, delta) in zip(smiles_list, delta_values)
        if haskey(buffer.seen_smiles, smi)
            buffer.tb_deltas[smi] = abs(delta)
        end
    end
end

"""
    sample_replay(buffer, n; rank_weighted=true, delta_priority=false)

Sample n entries from the replay buffer.

Priority modes:
- `rank_weighted=true, delta_priority=false`: Rank by reward (default, Genetic GFN)
- `rank_weighted=true, delta_priority=true`: Rank by |δ| (novel, Direction B)
- `rank_weighted=false`: Uniform sampling

|δ|-priority: P(i) ∝ |δ_i|^α where α is implicit in the rank transform.
Molecules with high flow imbalance are replayed more, improving balance accuracy.
At convergence δ→0 for all molecules, so priority becomes uniform — self-correcting.
"""
function sample_replay_with_delta(buffer::SMILESReplayBuffer, n::Int;
                                    rank_weighted::Bool=true,
                                    delta_priority::Bool=false)
    if isempty(buffer)
        return SMILESReplayEntry[]
    end

    if buffer.needs_sort
        _sort_and_reindex!(buffer)
    end

    n_available = length(buffer.entries)

    if !rank_weighted || n_available <= 1
        n_capped = min(n, n_available)
        indices = rand(1:n_available, n_capped)
    elseif delta_priority && !isempty(buffer.tb_deltas)
        # |δ|-priority: sort by |TB error| descending, then rank-weight
        delta_vals = Float64[]
        for entry in buffer.entries
            d = get(buffer.tb_deltas, entry.smiles, 0.0)
            push!(delta_vals, d)
        end
        # Sort indices by |δ| descending
        sorted_indices = sortperm(delta_vals, rev=true)
        # Rank-based weights on δ-sorted order
        weights = [1.0 / i for i in 1:n_available]
        total_w = sum(weights)
        probs = weights ./ total_w
        cumulative = cumsum(probs)
        indices = Int[]
        for _ in 1:n
            r = rand()
            rank_idx = searchsortedfirst(cumulative, r)
            rank_idx = clamp(rank_idx, 1, n_available)
            push!(indices, sorted_indices[rank_idx])
        end
    else
        # Standard rank-by-reward (already sorted by reward descending)
        weights = [1.0 / i for i in 1:n_available]
        total_w = sum(weights)
        probs = weights ./ total_w
        cumulative = cumsum(probs)
        indices = Int[]
        for _ in 1:n
            r = rand()
            idx = searchsortedfirst(cumulative, r)
            idx = clamp(idx, 1, n_available)
            push!(indices, idx)
        end
    end

    return [buffer.entries[i] for i in indices]
end

# =============================================================================
# Internal Helpers
# =============================================================================

"""Sort entries by reward (descending) and rebuild the SMILES→index mapping."""
function _sort_and_reindex!(buffer::SMILESReplayBuffer)
    sort!(buffer.entries, by=e -> -e.reward)
    empty!(buffer.seen_smiles)
    for (i, entry) in enumerate(buffer.entries)
        buffer.seen_smiles[entry.smiles] = i
    end
    buffer.needs_sort = false
end

"""Remove lowest-reward entries to bring buffer back to max_size."""
function _evict_lowest!(buffer::SMILESReplayBuffer)
    _sort_and_reindex!(buffer)
    # Keep only top max_size entries
    while length(buffer.entries) > buffer.max_size
        removed = pop!(buffer.entries)
        delete!(buffer.seen_smiles, removed.smiles)
    end
end
