# Experience Replay Buffer for Off-Policy GFlowNet Training
# Based on GFlowNet Foundations (JMLR 2023) - Off-policy learning with importance sampling
#
# Key References:
# - Bengio et al. "GFlowNet Foundations" JMLR 2023 - Off-policy training advantages
# - Schaul et al. "Prioritized Experience Replay" ICLR 2016 - Priority-based sampling

using Random
using StatsBase: sample, Weights

using ..GFlowNet: Trajectory

# =============================================================================
# Replay Buffer Implementation
# =============================================================================

"""
    ReplayBuffer

Prioritized experience replay buffer for GFlowNet trajectories.

Enables off-policy learning by storing and resampling past trajectories,
improving sample efficiency by 2-5x (per GFlowNet Foundations, JMLR 2023).

# Fields
- `trajectories::Vector{Trajectory}`: Storage for trajectories
- `priorities::Vector{Float64}`: Priority scores for sampling (higher = more likely to sample)
- `max_size::Int`: Maximum buffer capacity
- `alpha::Float64`: Priority exponent (0 = uniform sampling, 1 = full prioritization)
- `current_size::Int`: Current number of trajectories in buffer

# Priority Sampling
With priority exponent α, sampling probability is:
    P(i) = p_i^α / Σ_j p_j^α

where p_i is the priority of trajectory i.

# Example
```julia
buffer = ReplayBuffer(10000; alpha=0.6)

# Add trajectories with priority based on terminal reward
for traj in trajectories
    traj_reward = reward(traj.states[end])
    priority = compute_trajectory_priority(traj_reward)
    add!(buffer, traj, priority)
end

# Sample with importance weights for off-policy correction
sampled_trajs, weights, indices = sample_with_weights(buffer, 32)
```
"""
mutable struct ReplayBuffer
    trajectories::Vector{Trajectory}
    priorities::Vector{Float64}
    max_size::Int
    alpha::Float64
    current_size::Int
    position::Int  # Circular buffer position

    function ReplayBuffer(max_size::Int=10000; alpha::Float64=0.6)
        if max_size <= 0
            throw(ArgumentError("max_size must be positive"))
        end
        if alpha < 0.0 || alpha > 1.0
            throw(ArgumentError("alpha must be in [0.0, 1.0]"))
        end

        new(
            Vector{Trajectory}(undef, max_size),
            zeros(Float64, max_size),
            max_size,
            alpha,
            0,
            1  # Start at position 1 (Julia 1-indexing)
        )
    end
end

"""
    add!(buffer::ReplayBuffer, trajectory::Trajectory, priority::Float64=1.0)

Add trajectory to replay buffer with given priority.

If buffer is full, overwrites oldest trajectory (circular buffer).
"""
function add!(buffer::ReplayBuffer, trajectory::Trajectory, priority::Float64=1.0)
    # Store at current position
    buffer.trajectories[buffer.position] = trajectory
    buffer.priorities[buffer.position] = max(priority, 1e-8)  # Ensure non-zero priority

    # Update size
    if buffer.current_size < buffer.max_size
        buffer.current_size += 1
    end

    # Move position (circular)
    buffer.position = buffer.position % buffer.max_size + 1

    return nothing
end

"""
    sample_with_weights(buffer::ReplayBuffer, n::Int)

Sample n trajectories with importance weights for off-policy correction.

Returns tuple of (trajectories, importance_weights, indices).

Indices are drawn i.i.d. WITH replacement from
    P(i) = p_i^alpha / sum_j p_j^alpha
so a trajectory may appear more than once in one batch, and exactly `n`
trajectories are always returned as long as the buffer is non-empty.

The importance weights are then
    w_i = (N * P(i))^(-1)
normalized by max weight to ensure w_max = 1.0.

# Why with replacement
This draw used to be `replace=false`, which broke the function in two ways at
once.

First, the weight formula above is Schaul et al. (2016) with beta = 1, and its
precondition is that index i is selected with probability P(i). That holds for an
i.i.d. draw. Under sampling without replacement the marginal inclusion
probability of i is not P(i) -- it depends on n and on every other priority --
so the correction applied was not the correction derived.

Second, and this is the reachable one: without replacement the draw needs
n <= current_size, so `n` was clamped to the buffer size, and when n reached the
buffer size the draw returned EVERY stored trajectory regardless of priority.
`replay_priority_alpha` then had no effect at all. That is reachable from the
public config: train_gflownet only enters the replay branch once
length(buffer) >= batch_size, and replay_ratio = 1.0 asks for n = batch_size.
MEASURED on a 10-entry buffer with priorities [1,1,1,1,1,1,1,1,1,100], drawing
n = 10 two thousand times: the high-priority entry's share of the drawn indices
was 0.100 at alpha = 0.0 AND 0.100 at alpha = 1.0 -- the uniform 1/10 both
times, because every draw returned the whole buffer. With the i.i.d. draw below
the same measurement gives 0.100 at alpha = 0.0 and 0.917 at alpha = 1.0,
against the closed-form 0.1 and 100/109 = 0.9174.
"""
function sample_with_weights(buffer::ReplayBuffer, n::Int)
    if buffer.current_size == 0
        return Trajectory[], Float64[], Int[]
    end

    # Get active priorities
    active_priorities = buffer.priorities[1:buffer.current_size]

    # Compute sampling probabilities with priority exponent
    probs = active_priorities .^ buffer.alpha
    probs ./= sum(probs)

    # Sample indices i.i.d. from P(i) -- see "Why with replacement" above.
    indices = sample(1:buffer.current_size, Weights(probs), n; replace=true)

    # Compute importance sampling weights
    # w_i = (N × P(i))^(-β) where β=1 for full correction
    N = buffer.current_size
    weights = (N .* probs[indices]) .^ (-1)
    weights ./= maximum(weights)  # Normalize so max weight is 1

    # Get trajectories
    trajectories = [buffer.trajectories[i] for i in indices]

    return trajectories, weights, indices
end

"""
    sample_uniform(buffer::ReplayBuffer, n::Int)

Sample n trajectories uniformly (no prioritization).

Returns vector of trajectories.
"""
function sample_uniform(buffer::ReplayBuffer, n::Int)
    if buffer.current_size == 0
        return Trajectory[]
    end

    n = min(n, buffer.current_size)
    indices = sample(1:buffer.current_size, n; replace=false)

    return [buffer.trajectories[i] for i in indices]
end

"""
    update_priorities!(buffer::ReplayBuffer, indices::Vector{Int}, new_priorities::Vector{Float64})

Update priorities for sampled trajectories based on their TD errors.

Typically called after computing loss for sampled trajectories:
```julia
trajs, weights, indices = sample_with_weights(buffer, 32)
td_errors = compute_td_errors(model, trajs)
update_priorities!(buffer, indices, abs.(td_errors) .+ 1e-6)
```
"""
function update_priorities!(buffer::ReplayBuffer, indices::Vector{Int}, new_priorities::Vector{Float64})
    for (i, idx) in enumerate(indices)
        if idx <= buffer.current_size
            buffer.priorities[idx] = max(new_priorities[i], 1e-8)
        end
    end
    return nothing
end

"""
    length(buffer::ReplayBuffer)

Return current number of trajectories in buffer.
"""
Base.length(buffer::ReplayBuffer) = buffer.current_size

"""
    isempty(buffer::ReplayBuffer)

Check if buffer is empty.
"""
Base.isempty(buffer::ReplayBuffer) = buffer.current_size == 0

"""
    clear!(buffer::ReplayBuffer)

Clear all trajectories from buffer.
"""
function clear!(buffer::ReplayBuffer)
    buffer.current_size = 0
    buffer.position = 1
    fill!(buffer.priorities, 0.0)
    return nothing
end

# =============================================================================
# Priority Computation Helpers
# =============================================================================

"""
    compute_trajectory_priority(value::Float64)

Compute priority for a trajectory based on a scalar value (reward or loss).

Higher value = higher priority = more likely to be replayed.
Typically called with the trajectory's terminal reward so that
high-reward trajectories are replayed more often (mode retention).

Uses sqrt for stability: priority = sqrt(|value|) + ε
"""
function compute_trajectory_priority(value::Float64)
    return sqrt(abs(value)) + 1e-6
end

# =============================================================================
# Exports
# =============================================================================

export ReplayBuffer
export add!, sample_with_weights, sample_uniform, update_priorities!, clear!
export compute_trajectory_priority
