# QGFN: Q-Function Guided GFlowNet Inference (NeurIPS 2024)
#
# The Q-function estimates the expected log-reward reachable from a state-action pair:
#   Q(s, a) ≈ E[log R(x) | s, a]
#
# At inference time, actions with Q-values below the p-quantile are masked out.
# This focuses generation on high-reward regions without modifying training.
#
# Key properties:
# - Training is decoupled from inference (Q trains alongside policy with no extra oracle calls)
# - p=0 gives standard GFlowNet sampling (no masking)
# - p→1 gives greedy exploitation
# - p can be scheduled: start exploratory (p=0), increase over budget

using Lux
using Random
using Zygote
using ComponentArrays
using Statistics

# =============================================================================
# Q-Function Network
# =============================================================================

"""
    QFunctionNetwork

Wraps a Lux MLP that estimates Q(s, a) = E[log R(x) | s, a].

The network takes hidden state features from the GRU and outputs
Q-values for each action in the vocabulary.
"""
struct QFunctionNetwork
    model::Lux.Chain
end

"""
    create_q_function(hidden_dim::Int, vocab_size::Int; rng=Random.default_rng())

Create a Q-function network.

# Returns
Tuple of (QFunctionNetwork, parameters, states)
"""
function create_q_function(hidden_dim::Int, vocab_size::Int; rng=Random.default_rng())
    q_net = Lux.Chain(
        Lux.Dense(hidden_dim => hidden_dim, Lux.relu),
        Lux.Dense(hidden_dim => hidden_dim, Lux.relu),
        Lux.Dense(hidden_dim => vocab_size)  # Q-value per action
    )

    ps, st = Lux.setup(rng, q_net)
    return QFunctionNetwork(q_net), ps, st
end

# =============================================================================
# Q-Value Computation
# =============================================================================

"""
    compute_q_values(q_net::QFunctionNetwork, hidden_state, ps, st)

Compute Q-values for all actions given a GRU hidden state.

# Arguments
- `q_net`: Q-function network
- `hidden_state`: GRU hidden state vector (from the last GRU layer)
- `ps`: Q-function parameters
- `st`: Q-function states

# Returns
Vector of Q-values (one per vocabulary token)
"""
function compute_q_values(q_net::QFunctionNetwork, hidden_state::AbstractVector, ps, st)
    q_values, new_st = q_net.model(hidden_state, ps, st)
    return q_values, new_st
end

# =============================================================================
# p-Quantile Masking
# =============================================================================

"""
    apply_q_masking(logits, q_values, p_quantile; applicable_mask=nothing)

Apply p-quantile masking to action logits using Q-values.

Actions with Q-values below the p-quantile of applicable Q-values are masked
(set to -Inf in logits). If all actions would be masked, keeps the argmax.

# Arguments
- `logits`: Action logits from the forward policy
- `q_values`: Q-values from the Q-function
- `p_quantile`: Quantile threshold in [0, 1] (0 = no masking, 1 = only best)
- `applicable_mask`: Optional boolean mask of applicable actions

# Returns
Masked logits
"""
function apply_q_masking(logits::AbstractVector, q_values::AbstractVector,
                          p_quantile::Float64; applicable_mask=nothing)
    if p_quantile <= 0.0
        return logits  # No masking
    end

    # Get Q-values for applicable actions
    if !isnothing(applicable_mask)
        applicable_q = q_values[applicable_mask]
    else
        applicable_q = q_values
    end

    if isempty(applicable_q) || all(isnan, applicable_q)
        return logits
    end

    # Compute p-quantile threshold
    sorted_q = sort(filter(!isnan, applicable_q))
    if isempty(sorted_q)
        return logits
    end

    threshold_idx = max(1, ceil(Int, p_quantile * length(sorted_q)))
    threshold_idx = min(threshold_idx, length(sorted_q))
    q_threshold = sorted_q[threshold_idx]

    # Mask actions below threshold
    masked_logits = copy(logits)
    n_kept = 0

    for i in eachindex(masked_logits)
        if !isnothing(applicable_mask) && !applicable_mask[i]
            continue  # Already not applicable
        end
        if q_values[i] < q_threshold
            masked_logits[i] = -Inf
        else
            n_kept += 1
        end
    end

    # Fallback: if all masked, keep the argmax
    if n_kept == 0
        best_idx = argmax(q_values)
        masked_logits[best_idx] = logits[best_idx]
    end

    return masked_logits
end

# =============================================================================
# Q-Function Training (n-step TD)
# =============================================================================

"""
    QTrainingBuffer

Buffer for storing (hidden_state, action, reward, next_hidden_state) transitions
for Q-function training.
"""
mutable struct QTrainingBuffer
    hidden_states::Vector{Vector{Float32}}
    actions::Vector{Int}
    rewards::Vector{Float64}
    next_hidden_states::Vector{Vector{Float32}}
    is_terminal::Vector{Bool}
    capacity::Int
    position::Int
    full::Bool

    function QTrainingBuffer(capacity::Int=10000)
        new(
            Vector{Float32}[], Int[], Float64[], Vector{Float32}[], Bool[],
            capacity, 0, false
        )
    end
end

"""Add a transition to the Q-training buffer."""
function add_q_transition!(buffer::QTrainingBuffer, hidden::Vector{Float32},
                           action::Int, reward::Float64,
                           next_hidden::Vector{Float32}, terminal::Bool)
    if length(buffer.hidden_states) < buffer.capacity
        push!(buffer.hidden_states, hidden)
        push!(buffer.actions, action)
        push!(buffer.rewards, reward)
        push!(buffer.next_hidden_states, next_hidden)
        push!(buffer.is_terminal, terminal)
    else
        buffer.position = (buffer.position % buffer.capacity) + 1
        buffer.full = true
        buffer.hidden_states[buffer.position] = hidden
        buffer.actions[buffer.position] = action
        buffer.rewards[buffer.position] = reward
        buffer.next_hidden_states[buffer.position] = next_hidden
        buffer.is_terminal[buffer.position] = terminal
    end
end

"""Sample a batch from the Q-training buffer."""
function sample_q_batch(buffer::QTrainingBuffer, batch_size::Int)
    n = length(buffer.hidden_states)
    if n == 0
        return nothing
    end

    indices = rand(1:n, min(batch_size, n))
    return (
        hidden_states=[buffer.hidden_states[i] for i in indices],
        actions=[buffer.actions[i] for i in indices],
        rewards=[buffer.rewards[i] for i in indices],
        next_hidden_states=[buffer.next_hidden_states[i] for i in indices],
        is_terminal=[buffer.is_terminal[i] for i in indices],
    )
end

"""
    train_q_function!(q_net, q_params, q_states, buffer, optimizer;
                       batch_size=32, gamma=0.99)

Train the Q-function using TD learning.

Q-target: y = r + γ * max_a' Q(s', a')  (terminal: y = r)
Loss: L = (Q(s, a) - y)²

# Arguments
- `q_net`: QFunctionNetwork
- `q_params`: Q-function parameters (will be mutated)
- `q_states`: Q-function Lux states
- `buffer`: QTrainingBuffer with transitions
- `optimizer`: Optimisers.jl optimizer state
- `batch_size`: Training batch size
- `gamma`: Discount factor

# Returns
Tuple of (loss, updated_params, updated_optimizer)
"""
function train_q_function!(q_net::QFunctionNetwork, q_params, q_states,
                            buffer::QTrainingBuffer, optimizer;
                            batch_size::Int=32, gamma::Float64=0.99)
    batch = sample_q_batch(buffer, batch_size)
    if isnothing(batch)
        return 0.0, q_params, optimizer
    end

    # Compute TD targets (outside gradient)
    targets = Float64[]
    for i in eachindex(batch.rewards)
        if batch.is_terminal[i]
            push!(targets, batch.rewards[i])
        else
            next_q, _ = compute_q_values(q_net, batch.next_hidden_states[i], q_params, q_states)
            push!(targets, batch.rewards[i] + gamma * maximum(next_q))
        end
    end

    # Compute loss and gradients
    loss_fn = ps -> begin
        total_loss = 0.0
        for i in eachindex(batch.hidden_states)
            q_values, _ = q_net.model(batch.hidden_states[i], ps, q_states)
            action_idx = batch.actions[i] + 1  # 0-indexed → 1-indexed
            if 1 <= action_idx <= length(q_values)
                td_error = q_values[action_idx] - targets[i]
                total_loss += td_error^2
            end
        end
        total_loss / length(batch.hidden_states)
    end

    loss_val, grads = Zygote.withgradient(loss_fn, q_params)

    if grads[1] !== nothing
        optimizer, q_params = Optimisers.update(optimizer, q_params, grads[1])
    end

    return loss_val, q_params, optimizer
end

# =============================================================================
# p-Quantile Schedule
# =============================================================================

"""
    compute_p_quantile(budget_used::Int, total_budget::Int;
                        p_start=0.0, p_end=0.8, warmup_fraction=0.2)

Compute p-quantile based on oracle budget progress.

Schedule: p increases from p_start to p_end after warmup_fraction of budget.

# Arguments
- `budget_used`: Oracle calls used so far
- `total_budget`: Total oracle budget
- `p_start`: Starting p-quantile (usually 0.0 for pure exploration)
- `p_end`: Final p-quantile (0.5-0.9 typical)
- `warmup_fraction`: Fraction of budget for warmup (p stays at p_start)

# Returns
Current p-quantile value
"""
function compute_p_quantile(budget_used::Int, total_budget::Int;
                             p_start::Float64=0.0, p_end::Float64=0.8,
                             warmup_fraction::Float64=0.2)
    progress = budget_used / max(total_budget, 1)

    if progress <= warmup_fraction
        return p_start
    end

    # Linear ramp from p_start to p_end
    ramp_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
    ramp_progress = clamp(ramp_progress, 0.0, 1.0)

    return p_start + ramp_progress * (p_end - p_start)
end

# =============================================================================
# QGFN Integration Helpers
# =============================================================================

"""
    fill_transitions_with_reward!(buffer::QTrainingBuffer, transitions, log_reward::Float64)

Backfill transitions from `sample_smiles_autoregressive(collect_transitions=true)`
into the Q-training buffer with the terminal reward.

For GFlowNets, Q(s,a) ≈ E[log R(x) | s,a]. Intermediate rewards are 0;
only the terminal transition gets the actual log_reward.

# Arguments
- `buffer`: Q-training buffer to fill
- `transitions`: Vector of (hidden, action, next_hidden, is_terminal) from sampling
- `log_reward`: log R(x) for the completed molecule
"""
function fill_transitions_with_reward!(buffer::QTrainingBuffer,
                                        transitions::AbstractVector,
                                        log_reward::Float64)
    for (i, t) in enumerate(transitions)
        reward = t.is_terminal ? log_reward : 0.0
        hidden = Vector{Float32}(t.hidden)
        next_hidden = Vector{Float32}(t.next_hidden)
        add_q_transition!(buffer, hidden, t.action, reward, next_hidden, t.is_terminal)
    end
end

"""
    collect_and_fill_q_buffer!(buffer, policy_model, params, states, vocab,
                                reward_fn; n_samples=32, max_length=150,
                                reward_exponent=1.0, min_reward=0.01)

Sample molecules, compute rewards, and fill the Q-training buffer.
Convenience function for QGFN integration with the fine-tuning pipeline.

# Returns
Tuple of (n_valid, mean_reward, smiles_list)
"""
function collect_and_fill_q_buffer!(buffer::QTrainingBuffer,
                                     policy_model, params, states, vocab,
                                     reward_fn;
                                     n_samples::Int=32, max_length::Int=150,
                                     reward_exponent::Float64=1.0,
                                     min_reward::Float64=0.01)
    n_valid = 0
    rewards = Float64[]
    smiles_list = String[]

    for _ in 1:n_samples
        smiles, tokens, log_prob, transitions = sample_smiles_autoregressive(
            policy_model, params, states, vocab;
            max_length=max_length, temperature=1.0, collect_transitions=true,
            constrained=false
        )

        if isempty(smiles) || length(tokens) < 2
            continue
        end

        raw_r = try Float64(reward_fn(smiles)) catch; 0.0 end
        if raw_r > 0.0
            r_shaped = max(raw_r, min_reward) ^ reward_exponent
            log_r = log(r_shaped)
            fill_transitions_with_reward!(buffer, transitions, log_r)
            n_valid += 1
            push!(rewards, raw_r)
            push!(smiles_list, smiles)
        end
    end

    mean_r = isempty(rewards) ? 0.0 : mean(rewards)
    return n_valid, mean_r, smiles_list
end
