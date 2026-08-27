# Universal GFlowNet metrics computation
# Domain-agnostic quality metrics for all GFlowNet models

using GFlowNet: GFlowNetModel, Trajectory, reward
using Statistics: mean, std

"""
    compute_gflownet_metrics(model, trajectories)

Compute universal GFlowNet quality metrics that apply to ALL domains.
Returns a Dict with reward metrics, diversity metrics, and model metrics.

# Arguments
- `model::GFlowNetModel`: The GFlowNet model
- `trajectories::Vector{Trajectory}`: Sampled trajectories

# Returns
- `Dict`: Metrics including mean_reward, diversity_ratio, unique_terminals, etc.
"""
function compute_gflownet_metrics(model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    if isempty(trajectories)
        return Dict("error" => "No trajectories")
    end

    terminals = [t.states[end] for t in trajectories]
    rewards   = Float64[reward(s) for s in terminals]
    unique_count = length(unique(terminals))
    lengths = [length(t.actions) for t in trajectories]

    Z = if model.log_partition_function !== nothing
        exp(model.log_partition_function)
    else
        1.0
    end

    return Dict(
        # Reward metrics
        "mean_reward"       => mean(rewards),
        "max_reward"        => maximum(rewards),
        "min_reward"        => minimum(rewards),
        "reward_std"        => length(rewards) > 1 ? std(rewards) : 0.0,
        # Diversity metrics
        "unique_terminals"  => unique_count,
        "diversity_ratio"   => unique_count / length(trajectories),
        # Trajectory metrics
        "mean_length"       => mean(lengths),
        "max_length"        => maximum(lengths),
        # Model metrics. `mean_log_Z` was CONSUMED but never PRODUCED: the server
        # did get(universal_metrics, "mean_log_Z", 0.0), so the dashboard's
        # "Mean log Z" readout displayed a hardcoded 0.000 forever. That hid the
        # one quantity whose convergence was actually broken -- log Z used to be
        # fed to Adam through a scaled gradient, which is a provable no-op, so it
        # crawled at exactly the base learning rate. Emitting the real value.
        "partition_function" => Z,
        "mean_log_Z"         => model.log_partition_function === nothing ? 0.0 :
                                Float64(model.log_partition_function),
        # Sample size
        "n_trajectories"    => length(trajectories)
    )
end
