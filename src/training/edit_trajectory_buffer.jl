# Edit Trajectory Buffer — explicit memory for frontier-guided molecular search

using Random
using Statistics

struct EditTrajectoryEntry
    basin_scaffold::String
    parent_smiles::String
    parent_reward::Float64
    operator::Symbol
    child_smiles::String
    child_reward::Float64
    reward_delta::Float64
    step_index::Int
    terminated::Bool
    metadata::Dict{String,Any}
end

mutable struct EditTrajectoryBuffer
    entries::Vector{EditTrajectoryEntry}
    max_size::Int
    function EditTrajectoryBuffer(max_size::Int=10000)
        new(EditTrajectoryEntry[], max_size)
    end
end

Base.length(buf::EditTrajectoryBuffer) = length(buf.entries)
Base.isempty(buf::EditTrajectoryBuffer) = isempty(buf.entries)

function add_edit_trajectory!(buffer::EditTrajectoryBuffer,
                              basin_scaffold::String,
                              parent_smiles::String,
                              parent_reward::Float64,
                              operator::Symbol,
                              child_smiles::String,
                              child_reward::Float64;
                              step_index::Int=0,
                              terminated::Bool=false,
                              metadata::Dict{String,Any}=Dict{String,Any}())
    entry = EditTrajectoryEntry(
        basin_scaffold,
        parent_smiles,
        parent_reward,
        operator,
        child_smiles,
        child_reward,
        child_reward - parent_reward,
        step_index,
        terminated,
        metadata,
    )
    push!(buffer.entries, entry)
    while length(buffer.entries) > buffer.max_size
        popfirst!(buffer.entries)
    end
    return nothing
end

function sample_edit_trajectories(buffer::EditTrajectoryBuffer, n::Int;
                                  prioritize_positive_delta::Bool=true)
    isempty(buffer) && return EditTrajectoryEntry[]

    if !prioritize_positive_delta
        idxs = rand(1:length(buffer.entries), min(n, length(buffer.entries)))
        return [buffer.entries[i] for i in idxs]
    end

    weights = Float64[]
    for e in buffer.entries
        w = max(e.child_reward, 0.0) + max(e.reward_delta, 0.0) + (e.terminated ? 0.05 : 0.0)
        push!(weights, max(w, 1e-6))
    end
    total = sum(weights)
    probs = weights ./ total
    cumulative = cumsum(probs)

    sampled = EditTrajectoryEntry[]
    for _ in 1:n
        r = rand()
        idx = clamp(searchsortedfirst(cumulative, r), 1, length(buffer.entries))
        push!(sampled, buffer.entries[idx])
    end
    return sampled
end

function edit_trajectory_stats(buffer::EditTrajectoryBuffer)
    if isempty(buffer)
        return Dict(
            "size" => 0,
            "mean_child_reward" => 0.0,
            "mean_delta" => 0.0,
            "positive_delta_fraction" => 0.0,
        )
    end

    child_rewards = [e.child_reward for e in buffer.entries]
    deltas = [e.reward_delta for e in buffer.entries]
    return Dict(
        "size" => length(buffer.entries),
        "mean_child_reward" => mean(child_rewards),
        "mean_delta" => mean(deltas),
        "positive_delta_fraction" => mean(Float64[d > 0 for d in deltas]),
    )
end
