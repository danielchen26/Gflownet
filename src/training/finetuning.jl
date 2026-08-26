# TB Fine-Tuning for SMILES GFlowNet (CAFE-GFN)
#
# After MLE pretraining (learn SMILES grammar), fine-tune with actual oracle rewards.
# Uses Trajectory Balance: L_TB = (log Z + log P_F(τ) - log R(x))²
# with KL regularization against the pretrained reference policy π₀.
#
# Fine-tuning combines:
# 1. TB loss with oracle rewards (not R=1)
# 2. KL(π_θ || π₀) to maintain chemical validity
# 3. Optional Q-function guidance for budget-efficient sampling
# 4. Experience replay with rank-based priority (Genetic GFN, NeurIPS 2024)
# 5. SMILES augmentation for training data multiplication

using Lux
using Zygote
using Optimisers
using Random
using Statistics
using NNlib

# =============================================================================
# Fine-Tuning Configuration
# =============================================================================

"""
    FinetuningConfig

Configuration for CAFE-GFN fine-tuning with oracle rewards.

# Key parameters
- `kl_weight`: Controls deviation from pretrained policy (higher = more conservative)
  Genetic GFN uses 0.001; our default 0.01 balances exploration vs stability.
- `reward_exponent`: Reward shaping R(x)^β — higher β focuses on high-reward modes
- `kl_decay_schedule`: `:none`, `:cosine`, or `:linear` decay of KL weight

# Replay parameters (Genetic GFN, NeurIPS 2024)
- `use_replay`: Enable rank-based experience replay
- `replay_ratio`: Number of replay training steps per new-sample step (4-8 typical)

# QGFN parameters
- `use_qgfn_sampling`: Apply Q-value masking during training sampling
"""
struct FinetuningConfig
    n_iterations::Int           # Number of fine-tuning steps
    sample_batch_size::Int      # Trajectories sampled per step
    learning_rate::Float64      # Learning rate for policy
    gradient_clip_norm::Float64 # Gradient clipping threshold
    kl_weight::Float64          # λ for KL regularization
    kl_decay_schedule::Symbol   # :none, :cosine, :linear
    loss_type::Symbol           # :mse or :shifted_cosh
    cosh_threshold::Float64     # Shifted-cosh hybrid threshold
    max_length::Int             # Maximum SMILES length
    temperature::Float64        # Sampling temperature
    epsilon::Float64            # ε-uniform exploration rate
    log_frequency::Int          # Iterations between log output
    reward_exponent::Float64    # R(x)^β reward shaping
    min_reward::Float64         # Floor for reward values
    training_mode::Symbol       # :tb (trajectory balance) or :rwmle (reward-weighted MLE)
    constructive_only::Bool     # TB only: skip molecules where gradient would decrease probability
    freeze_gru::Bool            # Freeze GRU layers; only fine-tune output MLP (preserves grammar)
    reward_weighted::Bool       # TB only: weight each molecule's loss by normalized reward
    unfreeze_top_gru::Bool      # When freeze_gru=true, also unfreeze gru_3 for more capacity
    # --- Replay buffer (Genetic GFN) ---
    use_replay::Bool            # Enable rank-based experience replay
    replay_ratio::Int           # Replay steps per on-policy step (Genetic GFN uses 8)
    # --- QGFN ---
    use_qgfn_sampling::Bool     # Apply Q-value masking during training sampling
    # --- Novel: β-scheduling (Direction A) ---
    beta_schedule::Symbol       # :none, :linear_ramp — schedule reward_exponent over training
    beta_start::Float64         # Starting β value (0.0 = standard TB, explore broadly)
    beta_end::Float64           # Ending β value (8.0 = focused on high-reward, exploit)
    # --- Novel: |δ|-priority replay (Direction B) ---
    delta_priority_replay::Bool # Use |TB error| as replay priority instead of reward rank
    # --- Log_Z learning rate ---
    lr_z_multiplier::Float64   # log_Z LR = learning_rate × lr_z_multiplier (Genetic GFN uses 0.1/0.0005 = 200)
    lr_z::Float64              # Direct log_Z LR (overrides lr_z_multiplier when > 0)
    log_z_grad_clip::Float64   # Gradient clipping for log_Z updates (element-wise clamp)
    # --- LR warmup ---
    warmup_iters::Int          # Linear warmup from 0.1×LR to LR over this many iterations (0 = disabled)

    function FinetuningConfig(;
        n_iterations::Int=1000,
        sample_batch_size::Int=64,
        learning_rate::Float64=1e-5,
        gradient_clip_norm::Float64=1.0,
        kl_weight::Float64=0.01,           # CHANGED: 0.1 → 0.01 (Genetic GFN uses 0.001)
        kl_decay_schedule::Symbol=:none,    # CHANGED: :cosine → :none (cosine→0 causes collapse)
        loss_type::Symbol=:shifted_cosh,
        cosh_threshold::Float64=2.0,   # cosh(15)≈1.6M was catastrophic; 2.0 is sane
        max_length::Int=150,
        temperature::Float64=1.0,
        epsilon::Float64=0.05,
        log_frequency::Int=50,
        reward_exponent::Float64=1.0,
        min_reward::Float64=0.01,     # 1e-8 with R^β gives log=-73, causing gradient explosion
        training_mode::Symbol=:tb,   # :tb or :rwmle
        constructive_only::Bool=false, # TB only: skip destructive gradient (δ>0)
        freeze_gru::Bool=false,       # Freeze GRU/embedding; only update output layer
        reward_weighted::Bool=false,  # TB only: weight loss by normalized reward (RWMLE-style focus)
        unfreeze_top_gru::Bool=false, # When freeze_gru, also unfreeze gru_3 for more capacity
        # --- Replay buffer ---
        use_replay::Bool=false,       # Enable experience replay
        replay_ratio::Int=4,          # Replay batches per on-policy batch
        # --- QGFN ---
        use_qgfn_sampling::Bool=false, # Apply Q-masking during sampling
        # --- β-scheduling (Direction A: proved theoretically sound) ---
        beta_schedule::Symbol=:none,  # :none or :linear_ramp
        beta_start::Float64=0.0,      # Starting β (0 = standard TB, max diversity)
        beta_end::Float64=8.0,        # Ending β (8 = focused on top molecules)
        # --- |δ|-priority replay (Direction B: novel, convergence-safe) ---
        delta_priority_replay::Bool=false,  # Use |TB error| for replay priority
        # --- Log_Z learning rate ---
        lr_z_multiplier::Float64=10.0,  # log_Z LR = LR × this. Genetic GFN uses ~200
        lr_z::Float64=0.0,              # Direct log_Z LR (0.0 = use multiplier)
        log_z_grad_clip::Float64=1.0,   # Clamp log_Z gradients to ±this
        # --- LR warmup ---
        warmup_iters::Int=0             # Linear warmup iterations (0 = disabled)
    )
        new(n_iterations, sample_batch_size, learning_rate, gradient_clip_norm,
            kl_weight, kl_decay_schedule, loss_type, cosh_threshold, max_length,
            temperature, epsilon, log_frequency, reward_exponent, min_reward,
            training_mode, constructive_only, freeze_gru, reward_weighted, unfreeze_top_gru,
            use_replay, replay_ratio, use_qgfn_sampling,
            beta_schedule, beta_start, beta_end, delta_priority_replay, lr_z_multiplier,
            lr_z, log_z_grad_clip, warmup_iters)
    end
end

"""
    FinetuningHistory

Records metrics during fine-tuning.
"""
mutable struct FinetuningHistory
    tb_losses::Vector{Float64}
    kl_losses::Vector{Float64}
    combined_losses::Vector{Float64}
    mean_rewards::Vector{Float64}
    max_rewards::Vector{Float64}
    log_Z_values::Vector{Float64}
    gradient_norms::Vector{Float64}
    n_valid::Vector{Int}
    unique_smiles::Set{String}
end

FinetuningHistory() = FinetuningHistory(
    Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Int[], Set{String}()
)

# =============================================================================
# Reward Evaluation Helpers
# =============================================================================

"""
    _evaluate_reward_batch(smiles_list, reward_fn; reward_fn_batch=nothing)

Evaluate a batch of SMILES rewards outside the differentiation path.
When `reward_fn_batch` is provided, this performs a single batched reward call.
Falls back to per-molecule evaluation for compatibility.
"""
function _evaluate_reward_batch(smiles_list::Vector{String}, reward_fn;
                                reward_fn_batch=nothing)
    isempty(smiles_list) && return Float64[]

    if !isnothing(reward_fn_batch)
        try
            scores = reward_fn_batch(smiles_list)
            return Float64[s for s in scores]
        catch
            # Fall back to per-molecule scoring if batch path fails.
        end
    end

    scores = Float64[]
    for smiles in smiles_list
        raw_r = try
            Float64(reward_fn(smiles))
        catch
            0.0
        end
        push!(scores, raw_r)
    end
    return scores
end

# =============================================================================
# TB Fine-Tuning Loss with Oracle Rewards
# =============================================================================

"""
    compute_tb_finetune_loss(policy_model, vocab, ps, st, log_Z, reward_fn;
                              batch_size=64, max_length=150, ...,
                              q_net=nothing, q_params=nothing, q_states=nothing,
                              p_quantile=0.0)

Compute TB loss on sampled trajectories with oracle rewards.

Unlike pretraining (R=1), this uses actual reward values from the oracle function.
TB error: log Z + log P_F(τ) - log R(x)

Optionally uses QGFN Q-value masking during sampling to bias toward
high-reward regions (off-policy TB is mathematically valid).

# Returns
Tuple of (loss_value, sampled_smiles, rewards, all_tokens)
- `all_tokens`: Vector of token sequences for replay buffer storage
"""
function compute_tb_finetune_loss(
    policy_model, vocab, ps, st, log_Z, reward_fn;
    batch_size::Int=64, max_length::Int=150,
    loss_type::Symbol=:shifted_cosh, threshold::Float64=15.0,
    temperature::Float64=1.0, epsilon::Float64=0.05,
    reward_exponent::Float64=1.0, min_reward::Float64=1e-8,
    constructive_only::Bool=false,
    reward_weighted::Bool=false,
    reward_fn_batch=nothing,
    # QGFN masking during sampling
    q_net=nothing, q_params=nothing, q_states=nothing,
    p_quantile::Float64=0.0
)
    # Sample trajectories and evaluate rewards (NON-DIFFERENTIABLE)
    # Only valid molecules (reward > 0) participate in TB loss.
    sampled_data = Zygote.@ignore begin
        sampled = NamedTuple{(:tokens, :smiles), Tuple{Vector{Int}, String}}[]
        for _ in 1:batch_size
            smiles, tokens, _ = sample_smiles_autoregressive(
                policy_model, ps, st, vocab;
                max_length=max_length, temperature=temperature, epsilon=epsilon,
                constrained=false,  # Must be false for TB correctness
                q_net=q_net, q_params=q_params, q_states=q_states,
                p_quantile=p_quantile
            )
            if length(tokens) >= 2 && !isempty(smiles)
                push!(sampled, (tokens=tokens, smiles=smiles))
            end
        end

        smiles_batch = [d.smiles for d in sampled]
        raw_rewards = _evaluate_reward_batch(smiles_batch, reward_fn; reward_fn_batch=reward_fn_batch)

        data = NamedTuple{(:tokens, :reward, :raw_reward, :smiles, :is_valid), Tuple{Vector{Int}, Float64, Float64, String, Bool}}[]
        for (d, raw_r) in zip(sampled, raw_rewards)
            is_valid = raw_r > 0.0
            r = max(raw_r, min_reward) ^ reward_exponent
            push!(data, (tokens=d.tokens, reward=r, raw_reward=raw_r, smiles=d.smiles, is_valid=is_valid))
        end
        data
    end

    n_total = Zygote.@ignore length(sampled_data)
    n_valid_for_tb = Zygote.@ignore count(d -> d.is_valid, sampled_data)

    if n_total == 0
        return 0.0, String[], Float64[], Vector{Int}[]
    end

    # Compute TB loss only on VALID molecules (DIFFERENTIABLE w.r.t. ps and log_Z)
    total_loss = 0.0
    n_loss_samples = Zygote.@ignore Ref(0)
    total_weight_sum = Zygote.@ignore Ref(0.0)
    if n_valid_for_tb > 0
        for d in sampled_data
            skip = Zygote.@ignore !d.is_valid
            if skip
                continue
            end
            log_prob_sum, _ = compute_log_probs_teacher_forced(
                policy_model, d.tokens, ps, st;
                detach_end_nonterminal=true
            )
            log_reward = Zygote.@ignore log(d.reward)
            tb_error = log_Z + log_prob_sum - log_reward

            skip_destructive = Zygote.@ignore (constructive_only && Float64(log_Z) + Float64(log_prob_sum) - Float64(log_reward) > 0.0)
            if skip_destructive
                continue
            end
            Zygote.@ignore (n_loss_samples[] += 1)

            w = Zygote.@ignore (reward_weighted ? Float64(d.reward) : 1.0)
            Zygote.@ignore (total_weight_sum[] += w)
            total_loss += w * apply_tb_loss(tb_error, loss_type; threshold=threshold)
        end
        n_div = Zygote.@ignore (reward_weighted ? max(total_weight_sum[], 1e-8) : Float64(max(n_loss_samples[], 1)))
        total_loss = total_loss / n_div
    end

    sampled_smiles = Zygote.@ignore [d.smiles for d in sampled_data if d.is_valid]
    # Return RAW rewards (not R^β) — replay buffer stores raw oracle scores,
    # and compute_replay_loss applies R^β once at use time (principled data contract)
    rewards = Zygote.@ignore [d.raw_reward for d in sampled_data if d.is_valid]
    all_tokens = Zygote.@ignore [d.tokens for d in sampled_data if d.is_valid]
    return total_loss, sampled_smiles, rewards, all_tokens
end

# =============================================================================
# Reward-Weighted MLE Loss (Grammar-Preserving Alternative to TB)
# =============================================================================

"""
    compute_rwmle_loss(policy_model, vocab, ps, st, reward_fn;
                        batch_size=64, max_length=150, ...,
                        q_net=nothing, q_params=nothing, q_states=nothing,
                        p_quantile=0.0)

Compute reward-weighted maximum likelihood estimation loss.

Loss: L = -Σ_k w_k · log P_θ(x_k)
where w_k = R(x_k)^β / Σ_j R(x_j)^β (normalized reward weights)

# Returns
Tuple of (loss_value, sampled_smiles, rewards, all_tokens)
"""
function compute_rwmle_loss(
    policy_model, vocab, ps, st, reward_fn;
    batch_size::Int=64, max_length::Int=150,
    temperature::Float64=1.0, epsilon::Float64=0.05,
    reward_exponent::Float64=1.0, min_reward::Float64=0.01,
    reward_fn_batch=nothing,
    # QGFN masking during sampling
    q_net=nothing, q_params=nothing, q_states=nothing,
    p_quantile::Float64=0.0
)
    # Sample trajectories and evaluate rewards (NON-DIFFERENTIABLE)
    sampled_data = Zygote.@ignore begin
        sampled = NamedTuple{(:tokens, :smiles), Tuple{Vector{Int}, String}}[]
        for _ in 1:batch_size
            smiles, tokens, _ = sample_smiles_autoregressive(
                policy_model, ps, st, vocab;
                max_length=max_length, temperature=temperature, epsilon=epsilon,
                constrained=false,
                q_net=q_net, q_params=q_params, q_states=q_states,
                p_quantile=p_quantile
            )
            if length(tokens) >= 2 && !isempty(smiles)
                push!(sampled, (tokens=tokens, smiles=smiles))
            end
        end

        smiles_batch = [d.smiles for d in sampled]
        raw_rewards = _evaluate_reward_batch(smiles_batch, reward_fn; reward_fn_batch=reward_fn_batch)

        data = NamedTuple{(:tokens, :weight, :raw_reward, :smiles),
                          Tuple{Vector{Int}, Float64, Float64, String}}[]
        for (d, raw_r) in zip(sampled, raw_rewards)
            if raw_r > 0.0
                w = max(raw_r, min_reward) ^ reward_exponent
                push!(data, (tokens=d.tokens, weight=w, raw_reward=raw_r, smiles=d.smiles))
            end
        end
        data
    end

    n_valid = Zygote.@ignore length(sampled_data)
    if n_valid == 0
        return 0.0, String[], Float64[], Vector{Int}[]
    end

    total_weight = Zygote.@ignore sum(d.weight for d in sampled_data)

    # Compute weighted NLL (DIFFERENTIABLE w.r.t. ps)
    total_loss = 0.0
    for d in sampled_data
        log_prob_sum, _ = compute_log_probs_teacher_forced(policy_model, d.tokens, ps, st)
        w = Zygote.@ignore (d.weight / total_weight)
        total_loss += -w * log_prob_sum
    end

    sampled_smiles = Zygote.@ignore [d.smiles for d in sampled_data]
    rewards = Zygote.@ignore [d.raw_reward for d in sampled_data]
    all_tokens = Zygote.@ignore [d.tokens for d in sampled_data]
    return total_loss, sampled_smiles, rewards, all_tokens
end

# =============================================================================
# Replay Loss — Teacher-Forced Loss on Stored Trajectories
# =============================================================================

"""
    compute_replay_loss(policy_model, ps, st, log_Z, replay_entries, config)

Compute loss on replayed (stored) trajectories via teacher forcing.

For TB: L_replay = Σ shifted_cosh(log_Z + log_PF(τ_stored) - log_R_stored)
For RWMLE: L_replay = -Σ w_k · log P_θ(x_k)

The token sequences are fixed (from when the molecule was first sampled),
but log P_θ is recomputed under the current policy. This is mathematically
valid for both TB (off-policy compatible) and RWMLE.

# Arguments
- `replay_entries`: Vector{SMILESReplayEntry} from the replay buffer
- `config`: FinetuningConfig for loss parameters

# Returns
Scalar loss value (differentiable w.r.t. ps and log_Z)
"""
function compute_replay_loss(
    policy_model, ps, st, log_Z,
    replay_entries,
    config::FinetuningConfig
)
    if isempty(replay_entries)
        return 0.0
    end

    if config.training_mode == :tb
        # TB replay: teacher-forced log_PF on stored tokens, with stored reward
        total_loss = 0.0
        n_valid = Zygote.@ignore Ref(0)
        total_weight_sum = Zygote.@ignore Ref(0.0)

        for entry in replay_entries
            tokens = Zygote.@ignore entry.tokens
            reward = Zygote.@ignore max(entry.reward, config.min_reward) ^ config.reward_exponent

            if Zygote.@ignore (length(tokens) < 2)
                continue
            end

            log_prob_sum, _ = compute_log_probs_teacher_forced(
                policy_model, tokens, ps, st;
                detach_end_nonterminal=true
            )
            log_reward = Zygote.@ignore log(reward)
            tb_error = log_Z + log_prob_sum - log_reward

            # Constructive-only filtering for replay too
            skip_destructive = Zygote.@ignore (config.constructive_only &&
                Float64(log_Z) + Float64(log_prob_sum) - Float64(log_reward) > 0.0)
            if skip_destructive
                continue
            end

            Zygote.@ignore (n_valid[] += 1)
            w = Zygote.@ignore (config.reward_weighted ? Float64(reward) : 1.0)
            Zygote.@ignore (total_weight_sum[] += w)
            total_loss += w * apply_tb_loss(tb_error, config.loss_type; threshold=config.cosh_threshold)
        end

        n_div = Zygote.@ignore (config.reward_weighted ?
            max(total_weight_sum[], 1e-8) : Float64(max(n_valid[], 1)))
        return total_loss / n_div
    else
        # RWMLE replay: weighted NLL on stored tokens
        total_weight = Zygote.@ignore begin
            s = 0.0
            for entry in replay_entries
                s += max(entry.reward, config.min_reward) ^ config.reward_exponent
            end
            max(s, 1e-8)
        end

        total_loss = 0.0
        for entry in replay_entries
            tokens = Zygote.@ignore entry.tokens
            if Zygote.@ignore (length(tokens) < 2)
                continue
            end
            log_prob_sum, _ = compute_log_probs_teacher_forced(policy_model, tokens, ps, st)
            w = Zygote.@ignore begin
                ew = max(entry.reward, config.min_reward) ^ config.reward_exponent
                ew / total_weight
            end
            total_loss += -w * log_prob_sum
        end

        return total_loss
    end
end

# =============================================================================
# KL Regularization for SMILES Policies
# =============================================================================

"""
    compute_kl_smiles_loss(policy_model, sampled_tokens, ps, st, ref_ps, ref_st)

Compute KL divergence between current and reference SMILES policies on sampled trajectories.

KL(π_θ || π₀) ≈ (1/K) Σ_k [log P_θ(τ_k) - log P₀(τ_k)]
"""
function compute_kl_smiles_loss(
    policy_model, sampled_tokens::Vector{Vector{Int}},
    ps, st, ref_ps, ref_st
)
    if isempty(sampled_tokens)
        return 0.0
    end

    total_kl = 0.0
    n_valid = 0

    for tokens in sampled_tokens
        if length(tokens) < 2
            continue
        end

        # Current policy log probs (DIFFERENTIABLE)
        log_prob_current, _ = compute_log_probs_teacher_forced(policy_model, tokens, ps, st)

        # Reference policy log probs (FROZEN — no gradient)
        log_prob_ref = Zygote.@ignore begin
            lp, _ = compute_log_probs_teacher_forced(policy_model, tokens, ref_ps, ref_st)
            Float64(lp)
        end

        # Per-trajectory KL: log P_θ(τ) - log P₀(τ)
        kl = log_prob_current - log_prob_ref
        total_kl += kl
        n_valid += 1
    end

    return n_valid > 0 ? total_kl / n_valid : 0.0
end

# =============================================================================
# Gradient Masking for Selective Layer Fine-Tuning
# =============================================================================

"""Recursively zero all arrays in a nested parameter structure."""
function _zero_nested(x)
    if x isa AbstractArray
        return zero(x)
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(_zero_nested, values(x)))
    elseif x isa Tuple
        return map(_zero_nested, x)
    else
        return x
    end
end

"""
    _freeze_gru_grads(grads; keep_top_gru=false)

Zero out gradients for GRU and embedding layers, keeping only output layer gradients.
If `keep_top_gru=true`, also keep gradients for `gru_3`.
"""
function _freeze_gru_grads(grads; keep_top_gru::Bool=false)
    if grads isa NamedTuple
        kk = keys(grads)
        vals = map(kk) do k
            if k == :output
                grads[k]
            elseif keep_top_gru && k == :gru_3
                grads[k]
            else
                _zero_nested(grads[k])
            end
        end
        return NamedTuple{kk}(vals)
    else
        return grads
    end
end

# =============================================================================
# Fine-Tuning Loop (with Replay, QGFN, Augmentation)
# =============================================================================

"""
    finetune_smiles_gflownet(policy_model, vocab, params, states,
                              ref_params, ref_states, reward_fn, config;
                              verbose=true,
                              replay_buffer=nothing,
                              q_net=nothing, q_params=nothing,
                              q_states=nothing, q_optimizer=nothing,
                              budget_used=0, total_budget=10000,
                              augment_fn=nothing,
                              scaffold_filter=nothing)

Fine-tune a pretrained SMILES GFlowNet with oracle rewards.

This implements the full CAFE-GFN fine-tuning phase with all improvements:
- TB/RWMLE loss with actual oracle rewards
- KL regularization against pretrained reference policy
- Learnable log_Z parameter (jointly optimized, TB only)
- **Rank-based experience replay** (Genetic GFN): replay stored molecules 4-8× per new sample
- **QGFN Q-value masking**: guide sampling toward high-reward regions (budget efficiency)
- **SMILES augmentation**: multiply training data via randomized SMILES (no oracle cost)

# Required Arguments
- `policy_model`: SMILESPolicyModel (pretrained)
- `vocab`: SMILESVocabulary
- `params`: Initial model parameters (from pretraining)
- `states`: Model states (from pretraining)
- `ref_params`: Frozen reference policy parameters
- `ref_states`: Frozen reference states
- `reward_fn`: Oracle function `SMILES::String -> Float64`
- `config`: FinetuningConfig

# Optional Arguments (new features)
- `replay_buffer`: SMILESReplayBuffer for experience replay (persists across segments)
- `q_net, q_params, q_states, q_optimizer`: QGFN Q-function for guided sampling
- `budget_used, total_budget`: For QGFN p-quantile scheduling
- `augment_fn`: Function `(smiles) → Vector{(smiles, tokens)}` for SMILES augmentation
- `scaffold_filter`: ScaffoldFilter for diversity management

# Returns
Named tuple with: params, states, log_Z, history, q_params, q_optimizer
"""
function finetune_smiles_gflownet(
    policy_model, vocab, params, states,
    ref_params, ref_states,
    reward_fn,
    config::FinetuningConfig;
    reward_fn_batch=nothing,
    log_Z_init::Float64=0.0,
    verbose::Bool=true,
    # --- Experience replay ---
    replay_buffer=nothing,          # SMILESReplayBuffer (persists across calls)
    # --- QGFN ---
    q_net=nothing,                  # QFunctionNetwork
    q_params=nothing,               # Q-function parameters
    q_states=nothing,               # Q-function Lux states
    q_optimizer=nothing,            # Q-function optimizer state
    budget_used::Int=0,             # Current oracle budget usage (for p_quantile schedule)
    total_budget::Int=10000,        # Total oracle budget
    # --- Augmentation ---
    augment_fn=nothing,             # fn(smiles) → Vector{(smiles, tokens)}
    # --- Scaffold diversity ---
    scaffold_filter=nothing,        # ScaffoldFilter
    # --- Per-iteration GA (matches Genetic GFN's per-step GA) ---
    ga_fn=nothing                   # fn(replay_buffer, reward_fn) → adds GA offspring to buffer
)
    history = FinetuningHistory()

    # Auto-create replay buffer if use_replay is enabled but none provided
    if config.use_replay && isnothing(replay_buffer)
        replay_buffer = SMILESReplayBuffer(5000)
    end

    # Determine if QGFN masking should be active
    use_qgfn = config.use_qgfn_sampling && !isnothing(q_net) && !isnothing(q_params)

    # Initialize log_Z (only used in TB mode)
    log_Z_param = Float32[0.0f0]
    opt_state_lz = nothing

    if config.training_mode == :tb
        actual_log_Z_init = if log_Z_init == 0.0 && verbose
            println("Estimating initial log_Z from TB equilibrium samples...")
            sampled = NamedTuple{(:smiles, :tokens), Tuple{String, Vector{Int}}}[]
            for _ in 1:min(config.sample_batch_size, 32)
                smi, tokens, _ = sample_smiles_autoregressive(
                    policy_model, params, states, vocab;
                    max_length=config.max_length, temperature=config.temperature,
                    constrained=false
                )
                if !isempty(smi) && length(tokens) >= 2
                    push!(sampled, (smiles=smi, tokens=tokens))
                end
            end

            smiles_batch = [x.smiles for x in sampled]
            raw_rewards = _evaluate_reward_batch(smiles_batch, reward_fn; reward_fn_batch=reward_fn_batch)
            tb_estimates = Float64[]
            for (sample, r) in zip(sampled, raw_rewards)
                r_shaped = max(r, config.min_reward) ^ config.reward_exponent
                log_pf_tf, _ = compute_log_probs_teacher_forced(
                    policy_model, sample.tokens, params, states
                )
                push!(tb_estimates, log(r_shaped) - Float64(log_pf_tf))
            end
            if !isempty(tb_estimates)
                est = mean(tb_estimates)
                println("  Estimated log_Z = $(round(est, digits=2)) from $(length(tb_estimates)) samples")
                println("  (range: $(round(minimum(tb_estimates), digits=1)) to $(round(maximum(tb_estimates), digits=1)))")
                flush(stdout)
                est
            else
                0.0
            end
        else
            log_Z_init
        end
        log_Z_param = Float32[Float32(actual_log_Z_init)]
        lr_z_val = config.lr_z > 0 ? config.lr_z : config.learning_rate * config.lr_z_multiplier
        opt_lz = Optimisers.Adam(lr_z_val)
        opt_state_lz = Optimisers.setup(opt_lz, log_Z_param)
    end

    # Setup policy optimizer
    opt = Optimisers.Adam(config.learning_rate)
    opt_state = Optimisers.setup(opt, params)

    if verbose
        mode_str = config.training_mode == :rwmle ? "RWMLE (reward-weighted MLE)" : "TB (trajectory balance)"
        println("Starting CAFE-GFN fine-tuning ($mode_str)...")
        println("  Iterations: $(config.n_iterations)")
        println("  Sample batch: $(config.sample_batch_size)")
        println("  KL weight: $(config.kl_weight) ($(config.kl_decay_schedule) decay)")
        println("  Loss type: $(config.loss_type)")
        println("  Reward exponent: $(config.reward_exponent)")
        println("  Temperature: $(config.temperature), ε: $(config.epsilon)")
        println("  Replay: $(config.use_replay ? "ON (ratio=$(config.replay_ratio))" : "OFF")")
        println("  QGFN sampling: $(use_qgfn ? "ON" : "OFF")")
        println("  Augmentation: $(!isnothing(augment_fn) ? "ON" : "OFF")")
        lr_z_eff = config.lr_z > 0 ? config.lr_z : config.learning_rate * config.lr_z_multiplier
        println("  log_Z LR: $(lr_z_eff) (clip=±$(config.log_z_grad_clip))")
        println("  Warmup: $(config.warmup_iters > 0 ? "$(config.warmup_iters) iters" : "OFF")")
        println("  Initial log_Z: $(round(Float64(log_Z_param[1]), digits=4))")
        flush(stdout)
    end

    best_mean_reward = -Inf

    for iter in 1:config.n_iterations
        step_start = time()

        # LR warmup: linear ramp from 0.1×LR to LR over warmup_iters
        if config.warmup_iters > 0 && iter <= config.warmup_iters
            warmup_factor = 0.1 + 0.9 * (iter / config.warmup_iters)
            Optimisers.adjust!(opt_state, eta = config.learning_rate * warmup_factor)
        end

        # Compute KL weight with optional decay
        kl_w = _compute_kl_decay(config.kl_weight, config.kl_decay_schedule,
                                  iter, config.n_iterations)

        # Compute current β from schedule (Direction A: β-scheduling)
        # Uses global budget progress so β ramps across ALL segments, not per-segment
        current_beta = _compute_current_beta(config, iter, config.n_iterations;
                                             budget_used=budget_used, total_budget=total_budget)

        # Compute QGFN p-quantile based on budget progress
        current_p_quantile = if use_qgfn
            compute_p_quantile(budget_used + iter * config.sample_batch_size,
                             total_budget; p_start=0.0, p_end=0.6, warmup_fraction=0.2)
        else
            0.0
        end

        # Side-channel for metrics extraction
        metrics_ref = Ref{NamedTuple}((smiles=String[], rewards=Float64[], tokens=Vector{Int}[]))

        # Shared KL computation helper
        function _kl_block(ps)
            if kl_w > 0.0
                kl_tokens = Zygote.@ignore begin
                    toks = Vector{Vector{Int}}()
                    for _ in 1:min(config.sample_batch_size, 16)
                        _, t, _ = sample_smiles_autoregressive(
                            policy_model, ps, states, vocab;
                            max_length=config.max_length, temperature=1.0,
                            constrained=false
                        )
                        if length(t) >= 2
                            push!(toks, t)
                        end
                    end
                    toks
                end
                compute_kl_smiles_loss(policy_model, kl_tokens, ps, states, ref_params, ref_states)
            else
                0.0
            end
        end

        # =====================================================================
        # STEP 1: On-policy loss (sample new molecules + compute loss)
        # =====================================================================
        local loss_val, policy_grads

        if config.training_mode == :rwmle
            loss_fn = ps -> begin
                rwmle_loss, smiles_out, rewards_out, tokens_out = compute_rwmle_loss(
                    policy_model, vocab, ps, states, reward_fn;
                    batch_size=config.sample_batch_size,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    epsilon=config.epsilon,
                    reward_exponent=current_beta,
                    min_reward=config.min_reward,
                    reward_fn_batch=reward_fn_batch,
                    q_net=use_qgfn ? q_net : nothing,
                    q_params=use_qgfn ? q_params : nothing,
                    q_states=use_qgfn ? q_states : nothing,
                    p_quantile=current_p_quantile
                )
                Zygote.@ignore begin
                    metrics_ref[] = (smiles=smiles_out, rewards=rewards_out, tokens=tokens_out)
                end
                rwmle_loss + kl_w * _kl_block(ps)
            end
            loss_val, grads = Zygote.withgradient(loss_fn, params)
            policy_grads = grads[1]
        else
            loss_fn = (ps, lz) -> begin
                tb_loss, smiles_out, rewards_out, tokens_out = compute_tb_finetune_loss(
                    policy_model, vocab, ps, states, lz[1], reward_fn;
                    batch_size=config.sample_batch_size,
                    max_length=config.max_length,
                    loss_type=config.loss_type,
                    threshold=config.cosh_threshold,
                    temperature=config.temperature,
                    epsilon=config.epsilon,
                    reward_exponent=current_beta,
                    min_reward=config.min_reward,
                    constructive_only=config.constructive_only,
                    reward_weighted=config.reward_weighted,
                    reward_fn_batch=reward_fn_batch,
                    q_net=use_qgfn ? q_net : nothing,
                    q_params=use_qgfn ? q_params : nothing,
                    q_states=use_qgfn ? q_states : nothing,
                    p_quantile=current_p_quantile
                )
                Zygote.@ignore begin
                    metrics_ref[] = (smiles=smiles_out, rewards=rewards_out, tokens=tokens_out)
                end
                tb_loss + kl_w * _kl_block(ps)
            end
            loss_val, grads = Zygote.withgradient(loss_fn, params, log_Z_param)
            policy_grads = grads[1]
            if grads[2] !== nothing
                lz_grad = clamp.(grads[2], -config.log_z_grad_clip, config.log_z_grad_clip)
                opt_state_lz, log_Z_param = Optimisers.update(opt_state_lz, log_Z_param, lz_grad)
            end
        end

        if policy_grads === nothing
            @warn "Nil gradients at iteration $iter — skipping update"
            continue
        end

        # Freeze GRU layers if configured
        if config.freeze_gru
            policy_grads = _freeze_gru_grads(policy_grads; keep_top_gru=config.unfreeze_top_gru)
        end

        # Gradient norm and clipping
        grad_norm = sqrt(_grad_norm(policy_grads))
        scaled_grads = if grad_norm > config.gradient_clip_norm && grad_norm > 0
            _scale_grads(policy_grads, config.gradient_clip_norm / grad_norm)
        else
            policy_grads
        end

        # Update policy parameters
        opt_state, params = Optimisers.update(opt_state, params, scaled_grads)

        # Extract metrics from training call
        sampled_smiles = metrics_ref[].smiles
        rewards = metrics_ref[].rewards
        sampled_tokens = metrics_ref[].tokens

        # =====================================================================
        # STEP 2: Add new molecules to replay buffer + augment
        # =====================================================================
        if !isnothing(replay_buffer) && !isempty(sampled_smiles)
            for (smi, tok, rew) in zip(sampled_smiles, sampled_tokens, rewards)
                # Scaffold diversity check
                if !isnothing(scaffold_filter)
                    if should_add_molecule(scaffold_filter, smi)
                        add_to_replay!(replay_buffer, smi, tok, rew)
                        register_molecule!(scaffold_filter, smi)
                    end
                else
                    add_to_replay!(replay_buffer, smi, tok, rew)
                end

                # SMILES augmentation: add randomized SMILES with same reward
                if !isnothing(augment_fn) && rew > 0.0
                    try
                        aug_results = augment_fn(smi)
                        for (aug_smi, aug_tok) in aug_results
                            add_to_replay!(replay_buffer, aug_smi, aug_tok, rew)
                        end
                    catch
                        # Augmentation failure is non-critical
                    end
                end
            end
        end

        # =====================================================================
        # STEP 2.5: Per-iteration GA (Genetic GFN-style per-step GA)
        # =====================================================================
        # Genetic GFN generates 16 GA offspring PER STEP (not per segment).
        # This is the PRIMARY structural discovery mechanism for hard tasks.
        if !isnothing(ga_fn) && !isnothing(replay_buffer) && length(replay_buffer) >= 10
            Zygote.@ignore begin
                try
                    ga_fn(replay_buffer, reward_fn)
                catch e
                    # GA failure is non-critical — RDKit may not be available
                end
            end
        end

        # =====================================================================
        # STEP 3: Replay training (additional gradient steps on stored molecules)
        # =====================================================================
        if config.use_replay && !isnothing(replay_buffer) && length(replay_buffer) >= config.sample_batch_size
            for _replay_step in 1:config.replay_ratio
                # Direction B: |δ|-priority replay or standard rank-by-reward
                replay_entries = if config.delta_priority_replay
                    sample_replay_with_delta(replay_buffer, config.sample_batch_size;
                        rank_weighted=true, delta_priority=true)
                else
                    sample_replay(replay_buffer, config.sample_batch_size; rank_weighted=true)
                end

                if config.training_mode == :rwmle
                    replay_loss_fn = ps -> begin
                        compute_replay_loss(policy_model, ps, states, 0.0, replay_entries, config) +
                        kl_w * _kl_block(ps)
                    end
                    replay_loss_val, replay_grads = Zygote.withgradient(replay_loss_fn, params)
                    replay_policy_grads = replay_grads[1]
                else
                    replay_loss_fn = (ps, lz) -> begin
                        compute_replay_loss(policy_model, ps, states, lz[1], replay_entries, config) +
                        kl_w * _kl_block(ps)
                    end
                    replay_loss_val, replay_grads = Zygote.withgradient(replay_loss_fn, params, log_Z_param)
                    replay_policy_grads = replay_grads[1]
                    if replay_grads[2] !== nothing
                        lz_grad_r = clamp.(replay_grads[2], -config.log_z_grad_clip, config.log_z_grad_clip)
                        opt_state_lz, log_Z_param = Optimisers.update(opt_state_lz, log_Z_param, lz_grad_r)
                    end
                end

                if replay_policy_grads !== nothing
                    if config.freeze_gru
                        replay_policy_grads = _freeze_gru_grads(replay_policy_grads; keep_top_gru=config.unfreeze_top_gru)
                    end
                    rg_norm = sqrt(_grad_norm(replay_policy_grads))
                    replay_scaled = if rg_norm > config.gradient_clip_norm && rg_norm > 0
                        _scale_grads(replay_policy_grads, config.gradient_clip_norm / rg_norm)
                    else
                        replay_policy_grads
                    end
                    opt_state, params = Optimisers.update(opt_state, params, replay_scaled)
                end

                # Track |δ| values for δ-priority replay (Direction B)
                if config.delta_priority_replay && config.training_mode == :tb
                    Zygote.@ignore begin
                        delta_smiles = String[]
                        delta_vals = Float64[]
                        for entry in replay_entries
                            if length(entry.tokens) >= 2
                                log_pf, _ = compute_log_probs_teacher_forced(
                                    policy_model, entry.tokens, params, states)
                                r_shaped = max(entry.reward, config.min_reward) ^ current_beta
                                delta = Float64(log_Z_param[1]) + Float64(log_pf) - log(r_shaped)
                                push!(delta_smiles, entry.smiles)
                                push!(delta_vals, delta)
                            end
                        end
                        if !isempty(delta_smiles)
                            update_deltas!(replay_buffer, delta_smiles, delta_vals)
                        end
                    end
                end
            end
        end

        # =====================================================================
        # STEP 4: QGFN Q-function training (if available)
        # =====================================================================
        if use_qgfn && !isnothing(q_optimizer) && !isnothing(q_states)
            # Collect transitions from recent samples for Q-function training
            q_buffer_local = QTrainingBuffer(5000)
            n_q_samples = min(config.sample_batch_size, 16)
            collect_and_fill_q_buffer!(q_buffer_local, policy_model,
                params, states, vocab, reward_fn;
                n_samples=n_q_samples, max_length=config.max_length,
                reward_exponent=config.reward_exponent, min_reward=config.min_reward)

            if length(q_buffer_local.hidden_states) > 32
                for _ in 1:5  # 5 Q-function update steps
                    _, q_params, q_optimizer = train_q_function!(
                        q_net, q_params, q_states, q_buffer_local, q_optimizer;
                        batch_size=32, gamma=0.99)
                end
            end
        end

        # =====================================================================
        # STEP 5: Periodic log_Z re-estimation (TB only)
        # =====================================================================
        # Skip re-estimation for β > 2 — the dynamic range of log_R is too large,
        # making E[log_R - log_PF] unreliable with small samples. Let Adam optimize log_Z instead.
        if config.training_mode == :tb && iter % 25 == 0 && config.reward_exponent <= 2.0
            new_log_Z_est = Zygote.@ignore begin
                sampled = NamedTuple{(:smiles, :tokens), Tuple{String, Vector{Int}}}[]
                for _ in 1:min(config.sample_batch_size, 32)
                    smi, tokens, _ = sample_smiles_autoregressive(
                        policy_model, params, states, vocab;
                        max_length=config.max_length, temperature=config.temperature,
                        constrained=false
                    )
                    if !isempty(smi) && length(tokens) >= 2
                        push!(sampled, (smiles=smi, tokens=tokens))
                    end
                end
                smiles_batch = [x.smiles for x in sampled]
                raw_rewards = _evaluate_reward_batch(smiles_batch, reward_fn; reward_fn_batch=reward_fn_batch)
                tb_ests = Float64[]
                for (sample, r) in zip(sampled, raw_rewards)
                    r_shaped = max(r, config.min_reward) ^ config.reward_exponent
                    log_pf_tf, _ = compute_log_probs_teacher_forced(
                        policy_model, sample.tokens, params, states
                    )
                    push!(tb_ests, log(r_shaped) - Float64(log_pf_tf))
                end
                isempty(tb_ests) ? nothing : Float32(mean(tb_ests))
            end
            if new_log_Z_est !== nothing
                old_lz = log_Z_param[1]
                log_Z_param = Float32[new_log_Z_est]
                lr_z_fresh = config.lr_z > 0 ? config.lr_z : config.learning_rate * config.lr_z_multiplier
                opt_lz_fresh = Optimisers.Adam(lr_z_fresh)
                opt_state_lz = Optimisers.setup(opt_lz_fresh, log_Z_param)
                if verbose && iter % config.log_frequency == 0
                    println("    log_Z re-estimated: $(round(Float64(old_lz), digits=2)) → $(round(Float64(new_log_Z_est), digits=2))")
                    flush(stdout)
                end
            end
        end

        # =====================================================================
        # Metrics and Logging
        # =====================================================================
        mean_r = isempty(rewards) ? 0.0 : mean(rewards)
        max_r = isempty(rewards) ? 0.0 : maximum(rewards)
        n_valid = length(sampled_smiles)

        for s in sampled_smiles
            push!(history.unique_smiles, s)
        end

        push!(history.combined_losses, Float64(loss_val))
        push!(history.mean_rewards, mean_r)
        push!(history.max_rewards, max_r)
        push!(history.log_Z_values, Float64(log_Z_param[1]))
        push!(history.gradient_norms, grad_norm)
        push!(history.n_valid, n_valid)

        if mean_r > best_mean_reward
            best_mean_reward = mean_r
        end

        if verbose && iter % config.log_frequency == 0
            elapsed = (time() - step_start) * 1000.0
            replay_info = !isnothing(replay_buffer) ? ", buf=$(length(replay_buffer))" : ""
            qgfn_info = use_qgfn ? ", p_q=$(round(current_p_quantile, digits=2))" : ""
            beta_info = config.beta_schedule != :none ? ", β=$(round(current_beta, digits=2))" : ""
            println("  Iter $iter: loss=$(round(Float64(loss_val), digits=4)), " *
                    "R_mean=$(round(mean_r, digits=4)), R_max=$(round(max_r, digits=4)), " *
                    "log_Z=$(round(Float64(log_Z_param[1]), digits=3)), " *
                    "KL_w=$(round(kl_w, digits=4)), " *
                    "unique=$(length(history.unique_smiles))$(replay_info)$(qgfn_info)$(beta_info), " *
                    "$(round(elapsed, digits=0))ms")
            flush(stdout)
        end
    end

    if verbose
        println("Fine-tuning complete: $(config.n_iterations) iters")
        println("  Best mean reward: $(round(best_mean_reward, digits=4))")
        println("  Unique molecules: $(length(history.unique_smiles))")
        println("  Final log_Z: $(round(Float64(log_Z_param[1]), digits=4))")
        if !isnothing(replay_buffer)
            println("  Replay buffer: $(length(replay_buffer)) molecules")
        end
        flush(stdout)
    end

    return (
        params=params,
        states=states,
        log_Z=Float64(log_Z_param[1]),
        history=history,
        q_params=q_params,
        q_optimizer=q_optimizer,
    )
end

# =============================================================================
# KL Decay Schedule
# =============================================================================

function _compute_kl_decay(base_weight::Float64, schedule::Symbol,
                            iteration::Int, n_iterations::Int)
    if base_weight <= 0.0
        return 0.0
    end

    progress = iteration / max(n_iterations, 1)

    if schedule == :none
        return base_weight
    elseif schedule == :cosine
        return base_weight * 0.5 * (1.0 + cos(π * progress))
    elseif schedule == :linear
        return base_weight * (1.0 - progress)
    else
        return base_weight
    end
end

# =============================================================================
# β-Scheduling (Direction A: Reward-Weighted TB Divergence Family)
# =============================================================================

"""
    _compute_current_beta(config, iteration, total_iterations)

Compute the current β (reward exponent) based on schedule.

Theoretically proved: β-scheduling from low (exploration) to high (exploitation)
preserves convergence, analogous to simulated annealing. The gradient is
a consistent estimator with O(1/K) bias at each β value.

- β=0: Standard TB (maximum diversity, P(x)∝R(x))
- β=4-8: Reward-focused TB (emphasizes top molecules)
- β→∞: Best-of-batch only (maximum exploitation)
"""
function _compute_current_beta(config::FinetuningConfig, iteration::Int, total_iterations::Int;
                               budget_used::Int=0, total_budget::Int=0)
    if config.beta_schedule == :none
        return config.reward_exponent  # Use fixed reward_exponent
    end

    # Use GLOBAL budget progress if available (across all segments),
    # otherwise fall back to per-segment progress
    progress = if total_budget > 0
        # Global: ramp β across entire PMO budget
        estimated_budget = budget_used + iteration * config.sample_batch_size
        clamp(estimated_budget / total_budget, 0.0, 1.0)
    else
        clamp(iteration / max(total_iterations, 1), 0.0, 1.0)
    end

    if config.beta_schedule == :linear_ramp
        # Linear ramp: β_start → β_end over training
        return config.beta_start + (config.beta_end - config.beta_start) * progress
    else
        return config.reward_exponent
    end
end
