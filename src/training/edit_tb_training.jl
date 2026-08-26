# ============================================================================
# Edit-TB Training Loop
# ============================================================================
#
# Supports both RWMLE (primary) and TB-style (upgrade) objectives.
# Uses GROUPED validation splits by (task, run) — not random episode splits.
# Episodes from the same run are too correlated for random splits.

using Optimisers
using Zygote
using Random
using Statistics
using Serialization

# ============================================================================
# Gradient Helpers (local copies — originals in pretraining.jl are not exported)
# ============================================================================

"""Recursively compute sum of squared gradient values."""
function _edit_tb_grad_norm(x)
    if x isa AbstractArray
        return Float64(sum(abs2, x))
    elseif x isa NamedTuple
        s = 0.0
        for v in values(x)
            s += _edit_tb_grad_norm(v)
        end
        return s
    elseif x isa Tuple
        s = 0.0
        for v in x
            s += _edit_tb_grad_norm(v)
        end
        return s
    elseif x isa Number
        return Float64(abs2(x))
    else
        return 0.0
    end
end

"""Recursively scale all arrays in a nested NamedTuple."""
function _edit_tb_scale_grads(x, factor)
    if x isa AbstractArray
        return factor .* x
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(v -> _edit_tb_scale_grads(v, factor), values(x)))
    elseif x isa Tuple
        return map(v -> _edit_tb_scale_grads(v, factor), x)
    elseif x isa Number
        return factor * x
    else
        return x
    end
end

# ============================================================================
# Grouped Train/Val Split
# ============================================================================

"""
Split trajectories into train/val by (task, run) group.

REQUIRED: episodes from the same run are too correlated for random splits.
Uses run_id field which encodes task__runN.
"""
function _edit_tb_grouped_split(trajectories::Vector{EditTBTrajectory}, val_fraction::Float64; rng=Random.default_rng())
    # Group by run_id (which is task__runN)
    groups = Dict{String, Vector{Int}}()
    for (i, traj) in enumerate(trajectories)
        push!(get!(groups, traj.run_id, Int[]), i)
    end

    group_keys = collect(keys(groups))
    shuffle!(rng, group_keys)
    n_val_groups = max(1, round(Int, length(group_keys) * val_fraction))

    val_keys = Set(group_keys[1:n_val_groups])
    val_indices = Int[]
    train_indices = Int[]
    for (k, idxs) in groups
        if k in val_keys
            append!(val_indices, idxs)
        else
            append!(train_indices, idxs)
        end
    end

    return trajectories[train_indices], trajectories[val_indices]
end

# ============================================================================
# Training Loop
# ============================================================================

"""
Train the edit policy using RWMLE or TB-style objective.

Uses grouped validation splits by (task, run).
Returns (best_params, best_log_Z, history).
"""
function train_edit_policy!(
    model, params, states,
    dataset::EditTBDataset,
    config::EditTBConfig;
    verbose::Bool=true,
    rng::AbstractRNG=Random.default_rng()
)
    # ── Grouped split ────────────────────────────────
    train_trajs, val_trajs = _edit_tb_grouped_split(
        dataset.trajectories, config.val_fraction; rng=rng
    )

    if verbose
        println("[edit-tb] Split: $(length(train_trajs)) train, $(length(val_trajs)) val " *
                "(grouped by run_id)")
        flush(stdout)
    end

    # ── Initialize ──────────────────────────────────
    log_Z = if config.objective == :tb
        valid_rewards = [t.terminal_reward for t in train_trajs if t.terminal_reward > 0.001]
        init_lz = isempty(valid_rewards) ? 0.0f0 : Float32(mean(log.(valid_rewards)))
        Float32[init_lz]
    else
        Float32[0.0f0]
    end

    opt_policy = Optimisers.Adam(Float32(config.learning_rate))
    opt_state_policy = Optimisers.setup(opt_policy, params)

    opt_lz = Optimisers.Adam(Float32(config.lr_z))
    opt_state_lz = Optimisers.setup(opt_lz, log_Z)

    best_val_loss = Inf
    best_params = deepcopy(params)
    best_log_Z = copy(log_Z)

    history = Dict{String, Vector{Float64}}(
        "train_loss" => Float64[],
        "val_loss" => Float64[],
        "log_Z" => Float64[],
        "grad_norm" => Float64[],
    )

    for epoch in 1:config.n_epochs
        shuffle!(rng, train_trajs)
        epoch_loss = 0.0
        epoch_gnorm = 0.0
        n_batches = 0

        for batch_start in 1:config.batch_size:length(train_trajs)
            batch_end = min(batch_start + config.batch_size - 1, length(train_trajs))
            batch = train_trajs[batch_start:batch_end]

            if config.objective == :rwmle
                loss_val, grads = Zygote.withgradient(
                    ps -> compute_edit_rwmle_loss(model, ps, states, batch;
                        beta=config.beta,
                        task_names=dataset.task_names,
                        task_conditioning=config.task_conditioning),
                    params
                )
                policy_grads = grads[1]

                gnorm = sqrt(_edit_tb_grad_norm(policy_grads))
                if gnorm > config.gradient_clip
                    policy_grads = _edit_tb_scale_grads(policy_grads, config.gradient_clip / gnorm)
                end
                opt_state_policy, params = Optimisers.update(opt_state_policy, params, policy_grads)
                epoch_gnorm += gnorm

            elseif config.objective == :tb
                loss_fn = (ps, lz) -> compute_edit_tb_style_loss(
                    model, ps, states, lz, batch;
                    beta=config.beta,
                    threshold=config.threshold,
                    reward_weighted=true,
                    task_names=dataset.task_names,
                    task_conditioning=config.task_conditioning
                )

                loss_val, grads = Zygote.withgradient(loss_fn, params, log_Z)
                policy_grads = grads[1]
                lz_grads = grads[2]

                gnorm = sqrt(_edit_tb_grad_norm(policy_grads))
                if gnorm > config.gradient_clip
                    policy_grads = _edit_tb_scale_grads(policy_grads, config.gradient_clip / gnorm)
                end
                opt_state_policy, params = Optimisers.update(opt_state_policy, params, policy_grads)

                if lz_grads !== nothing
                    lz_grads = clamp.(lz_grads, -config.log_z_grad_clip, config.log_z_grad_clip)
                    opt_state_lz, log_Z = Optimisers.update(opt_state_lz, log_Z, lz_grads)
                end
                epoch_gnorm += gnorm
            end

            epoch_loss += Float64(loss_val)
            n_batches += 1
        end

        avg_train = epoch_loss / max(n_batches, 1)
        avg_gnorm = epoch_gnorm / max(n_batches, 1)

        # ── Validation ──
        val_loss = if config.objective == :rwmle
            Float64(compute_edit_rwmle_loss(model, params, states, val_trajs;
                beta=config.beta,
                task_names=dataset.task_names,
                task_conditioning=config.task_conditioning))
        else
            Float64(compute_edit_tb_style_loss(model, params, states, log_Z, val_trajs;
                beta=config.beta,
                threshold=config.threshold,
                task_names=dataset.task_names,
                task_conditioning=config.task_conditioning))
        end

        push!(history["train_loss"], avg_train)
        push!(history["val_loss"], val_loss)
        push!(history["log_Z"], Float64(log_Z[1]))
        push!(history["grad_norm"], avg_gnorm)

        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_params = deepcopy(params)
            best_log_Z = copy(log_Z)
        end

        if verbose && (epoch % 5 == 0 || epoch == 1)
            println("[$(config.objective)] epoch $epoch: train=$(round(avg_train, digits=4)) " *
                    "val=$(round(val_loss, digits=4)) log_Z=$(round(log_Z[1], digits=3)) " *
                    "gnorm=$(round(avg_gnorm, digits=3))")
            flush(stdout)
        end
    end

    if verbose
        println("[edit-tb] Training complete. Best val loss: $(round(best_val_loss, digits=4))")
        flush(stdout)
    end

    return best_params, best_log_Z, history
end

# ============================================================================
# Online Stabilizer — Interpolated Deployment
# ============================================================================

"""
Choose an action using the learned policy with online stabilizer.

Uses interpolation: (1-λ)·heuristic + λ·learned
Plus entropy floor enforcement to prevent collapse.

Returns (chosen_index, probs, entropy).
"""
function choose_with_edit_policy(
    model, params, states, config::EditTBConfig,
    context_features::Vector{Float32},
    candidate_features::Matrix{Float32},
    heuristic_scores::Vector{Float64},
    n_candidates::Int;
    head::Symbol  # :basin, :parent, or :operator
)
    h = getfield(model, head)
    ps = getfield(params, head)
    st = getfield(states, head)

    # Learned scores
    learned_logits = Vector{Float32}(undef, n_candidates)
    for i in 1:n_candidates
        input = vcat(context_features, candidate_features[:, i])
        score, _ = h(reshape(input, :, 1), ps, st)
        learned_logits[i] = score[1]
    end

    # Heuristic logits
    heuristic_logits = Float32.(log.(max.(heuristic_scores, 1e-10)))

    # Interpolation: (1-λ) * heuristic + λ * learned
    λ = Float32(config.heuristic_interpolation)
    mixed_logits = (1.0f0 - λ) .* heuristic_logits .+ λ .* learned_logits

    # Softmax
    max_logit = maximum(mixed_logits)
    exp_logits = exp.(mixed_logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)

    # Entropy
    eps32 = Float32(1e-10)
    entropy = -sum(p * log(max(p, eps32)) for p in probs)

    # Entropy floor enforcement
    if entropy < config.entropy_floor
        uniform = fill(1.0f0 / n_candidates, n_candidates)
        α = 0.3f0
        probs = (1.0f0 - α) .* probs .+ α .* uniform
        entropy = -sum(p * log(max(p, eps32)) for p in probs)
    end

    # Flow-proportional sampling (NOT greedy)
    chosen = _edit_tb_sample_categorical(probs)
    return chosen, probs, entropy
end

"""Sample from a categorical distribution defined by probability vector."""
function _edit_tb_sample_categorical(probs::Vector{Float32})
    u = rand(Float32)
    cumsum = 0.0f0
    for i in eachindex(probs)
        cumsum += probs[i]
        if u <= cumsum
            return i
        end
    end
    return length(probs)
end

function _edit_tb_margin_from_probs(probs::AbstractVector{<:Real}, top_idx::Int)::Float64
    isempty(probs) && return 0.0
    if length(probs) == 1
        return Float64(probs[top_idx])
    end
    order = sortperm(Float64.(probs), rev=true)
    return Float64(probs[order[1]] - probs[order[2]])
end

function _edit_tb_entropy_from_scores(scores::AbstractVector{<:Real})::Float64
    isempty(scores) && return 0.0
    x = Float64.(scores)
    shifted = x .- maximum(x)
    weights = exp.(shifted)
    total = sum(weights)
    total <= 0 && return 0.0
    probs = weights ./ total
    entropy = 0.0
    for p in probs
        p <= 0 && continue
        entropy -= p * log(p)
    end
    return entropy
end

# ============================================================================
# Checkpoint Save/Load
# ============================================================================

"""Save trained edit policy to disk."""
function save_edit_policy(filepath::String, params, log_Z, history, config::EditTBConfig)
    serialize(filepath, Dict(
        "params" => params,
        "log_Z" => log_Z,
        "history" => history,
        "config" => config,
        "version" => "edit_tb_v1"
    ))
end

"""Load trained edit policy from disk."""
function load_edit_policy(filepath::String)
    data = deserialize(filepath)
    return data["params"], data["log_Z"], data["history"], data["config"]
end

# ============================================================================
# Online Integration — Runtime Feature Builders + Controller
# ============================================================================
# These translate LIVE runtime types (FrontierSnapshot, ScoredBasinCandidate,
# ScoredParentCandidate, OperatorDecisionCandidate) into the same feature
# vectors used during offline training.

"""
Mutable controller wrapping a trained edit-TB policy for online use.

Set `use_learned_basin/parent/operator` in HierarchicalEditConfig and
pass this as the corresponding `learned_*_controller`.
"""
mutable struct EditTBPolicyController
    model::Any           # NamedTuple{(:basin, :parent, :operator)}
    params::Any          # trained parameters
    states::Any          # Lux states
    config::EditTBConfig # includes interpolation λ, entropy floor, etc.
    task_names::Vector{String}
    current_task_name::String
    _last_basin::Union{Nothing,BasinSummary}  # set by select_basin, read by parent_selection_metadata
end

EditTBPolicyController(model, params, states, config;
                       task_names::Vector{String}=String[],
                       current_task_name::String="") =
    EditTBPolicyController(model, params, states, config, copy(task_names), current_task_name, nothing)

"""Update the active task for runtime task-conditioned inference."""
function set_edit_tb_task!(controller::EditTBPolicyController, task_name::AbstractString)
    controller.current_task_name = String(task_name)
    return controller
end

"""Task one-hot features for the controller's current online task."""
function _controller_task_features(controller::EditTBPolicyController)::Vector{Float32}
    return _edit_tb_task_features(controller.current_task_name, controller.task_names;
        enabled=controller.config.task_conditioning)
end

"""Build 8-d frontier features from a live FrontierSnapshot (mirrors offline build_edit_tb_frontier_features)."""
function _runtime_frontier_features(snapshot::FrontierSnapshot; step_index::Int=0)::Vector{Float32}
    entries = snapshot.entries
    n = length(entries)
    rewards = [e.reward for e in entries]
    top1 = isempty(rewards) ? 0.0 : maximum(rewards)
    sorted_r = sort(rewards, rev=true)
    top10_mean = isempty(sorted_r) ? 0.0 : mean(sorted_r[1:min(10, length(sorted_r))])
    scaffold_count = length(unique(e.scaffold for e in entries))

    Float32[
        top1,                                             # 1
        top10_mean,                                       # 2
        n / 256.0f0,                                      # 3
        scaffold_count / max(n, 1),                       # 4
        snapshot.budget_remaining / 3000.0f0,             # 5
        top1 - top10_mean,                                # 6
        scaffold_count / 20.0f0,                          # 7
        step_index / 3.0f0                                # 8
    ]
end

"""Build basin candidate features from a live ScoredBasinCandidate (mirrors offline)."""
function _runtime_basin_candidate_features(c::ScoredBasinCandidate; include_heuristic::Bool=true)::Vector{Float32}
    b = c.basin
    feats = Float32[
        b.count / 20.0f0,      # 1
        b.best_reward,          # 2
        b.mean_reward,          # 3
        b.mean_novelty,         # 4
        b.mean_delta,           # 5
    ]
    if include_heuristic
        push!(feats, Float32(c.score))  # 6 — maps to heuristic_score
    end
    return feats
end

"""Build 4-d basin context features for the parent head (mirrors offline _edit_tb_extract_basin_features)."""
function _runtime_basin_context_features(basin::BasinSummary)::Vector{Float32}
    Float32[
        basin.count / 20.0f0,
        basin.best_reward,
        basin.mean_reward,
        basin.mean_novelty
    ]
end

"""Build parent candidate features from a live ScoredParentCandidate (mirrors offline)."""
function _runtime_parent_candidate_features(c::ScoredParentCandidate; include_heuristic::Bool=true)::Vector{Float32}
    feats = Float32[
        c.entry.reward,                          # 1
        c.visit_count / 10.0f0,                  # 2
        c.basin_match ? 1.0f0 : 0.0f0,           # 3
        c.target_match ? 1.0f0 : 0.0f0,          # 4
    ]
    if include_heuristic
        push!(feats, Float32(c.score))  # 5 — maps to heuristic_score
    end
    return feats
end

"""Build 6-d parent context features for the operator head (mirrors offline parent_ctx)."""
function _runtime_parent_context_features(
    parent::FrontierSnapshotEntry,
    snapshot::FrontierSnapshot;
    step_index::Int=0,
    attempt_index::Int=1
)::Vector{Float32}
    n = length(snapshot.entries)
    top1 = isempty(snapshot.entries) ? 0.01 : maximum(e.reward for e in snapshot.entries)
    Float32[
        parent.reward,                              # 1
        parent.reward / max(Float32(top1), 0.01f0), # 2
        step_index / 3.0f0,                         # 3
        attempt_index / 3.0f0,                      # 4
        n / 256.0f0,                                # 5
        snapshot.budget_remaining / 3000.0f0         # 6
    ]
end

"""Build operator candidate features from a live OperatorDecisionCandidate (mirrors offline)."""
function _runtime_operator_candidate_features(c::OperatorDecisionCandidate; include_heuristic::Bool=true)::Vector{Float32}
    feats = Float32[
        c.total_count / 50.0f0,                            # 1
        c.positive_delta_count / max(c.total_count, 1),    # 2
        c.exploration_bonus,                                # 3
    ]
    if include_heuristic
        push!(feats, Float32(c.heuristic_score))  # 4
    end
    return feats
end

# ============================================================================
# Dispatch Methods — plug into existing choose_basin / choose_parent / choose_operator_action
# ============================================================================

"""
select_basin(controller::EditTBPolicyController, snapshot, candidates; step_index)

Returns a ScoredBasinCandidate chosen by the learned policy, or nothing.
"""
function select_basin(controller::EditTBPolicyController,
                      snapshot::FrontierSnapshot,
                      candidates::Vector{ScoredBasinCandidate};
                      step_index::Int=0)
    isempty(candidates) && return nothing

    include_h = controller.config.include_heuristic_scores
    task_features = _controller_task_features(controller)
    context = _edit_tb_with_task_context(
        _runtime_frontier_features(snapshot; step_index=step_index),
        task_features,
    )
    n = length(candidates)
    feats = [_runtime_basin_candidate_features(c; include_heuristic=include_h) for c in candidates]
    d = length(feats[1])
    cand_mat = Matrix{Float32}(undef, d, n)
    for i in 1:n
        cand_mat[:, i] = feats[i]
    end
    heuristic_scores = [c.score for c in candidates]

    chosen_idx, probs, entropy = choose_with_edit_policy(
        controller.model, controller.params, controller.states, controller.config,
        context, cand_mat, heuristic_scores, n; head=:basin
    )

    chosen = (1 <= chosen_idx <= n) ? candidates[chosen_idx] : nothing
    # Store for parent head's basin context features
    controller._last_basin = chosen !== nothing ? chosen.basin : nothing
    return chosen
end

"""
parent_selection_metadata(controller::EditTBPolicyController, snapshot, candidates; step_index)

Returns Dict with "chosen_index" and diagnostic fields.
"""
function parent_selection_metadata(controller::EditTBPolicyController,
                                   snapshot::FrontierSnapshot,
                                   candidates::Vector{ScoredParentCandidate};
                                   step_index::Int=0)
    n = length(candidates)
    if n == 0
        return Dict{String,Any}("chosen_index" => 0, "selection_reason" => "empty_candidates")
    end

    # Basin context for parent head — stored by select_basin earlier in the same step.
    # Falls back to zeros if basin was not set (e.g., basin selection is heuristic).
    basin_ctx = if controller._last_basin !== nothing
        _runtime_basin_context_features(controller._last_basin)
    else
        zeros(Float32, 4)
    end

    include_h = controller.config.include_heuristic_scores
    task_features = _controller_task_features(controller)
    basin_ctx = _edit_tb_with_task_context(basin_ctx, task_features)
    feats = [_runtime_parent_candidate_features(c; include_heuristic=include_h) for c in candidates]
    d = length(feats[1])
    cand_mat = Matrix{Float32}(undef, d, n)
    for i in 1:n
        cand_mat[:, i] = feats[i]
    end
    heuristic_scores = [c.score for c in candidates]

    chosen_idx, probs, entropy = choose_with_edit_policy(
        controller.model, controller.params, controller.states, controller.config,
        basin_ctx, cand_mat, heuristic_scores, n; head=:parent
    )

    heuristic_idx = argmax(heuristic_scores)
    override = chosen_idx != heuristic_idx
    heuristic_entropy = _edit_tb_entropy_from_scores(heuristic_scores)
    learned_margin = _edit_tb_margin_from_probs(probs, chosen_idx)
    heuristic_margin = _edit_tb_margin_from_probs(heuristic_scores, heuristic_idx)
    return Dict{String,Any}(
        "chosen_index" => chosen_idx,
        "override_applied" => override,
        "abstained_to_heuristic" => false,
        "heuristic_top_index" => heuristic_idx,
        "learned_top_index" => chosen_idx,
        "heuristic_margin" => heuristic_margin,
        "learned_margin" => learned_margin,
        "learned_advantage_vs_heuristic" => Float64(probs[chosen_idx] - probs[heuristic_idx]),
        "heuristic_entropy" => heuristic_entropy,
        "learned_entropy" => Float64(entropy),
        "entropy" => Float64(entropy),
        "selection_reason" => override ? "edit_tb_policy_override" : "edit_tb_policy_agree",
    )
end

"""
operator_selection_metadata(controller::EditTBPolicyController, snapshot, basin, parent, candidates; step_index)

Returns Dict with "chosen_index" and diagnostic fields.
"""
function operator_selection_metadata(controller::EditTBPolicyController,
                                     snapshot::FrontierSnapshot,
                                     basin::BasinSummary,
                                     parent::FrontierSnapshotEntry,
                                     candidates::Vector{OperatorDecisionCandidate};
                                     step_index::Int=0)
    n = length(candidates)
    if n == 0
        return Dict{String,Any}("chosen_index" => 0, "selection_reason" => "empty_candidates")
    end

    include_h = controller.config.include_heuristic_scores
    task_features = _controller_task_features(controller)
    context = _edit_tb_with_task_context(
        _runtime_parent_context_features(parent, snapshot;
            step_index=step_index, attempt_index=1),
        task_features,
    )
    feats = [_runtime_operator_candidate_features(c; include_heuristic=include_h) for c in candidates]
    d = length(feats[1])
    cand_mat = Matrix{Float32}(undef, d, n)
    for i in 1:n
        cand_mat[:, i] = feats[i]
    end
    heuristic_scores = [c.heuristic_score for c in candidates]

    chosen_idx, probs, entropy = choose_with_edit_policy(
        controller.model, controller.params, controller.states, controller.config,
        context, cand_mat, heuristic_scores, n; head=:operator
    )

    heuristic_idx = argmax(heuristic_scores)
    override = chosen_idx != heuristic_idx
    heuristic_entropy = _edit_tb_entropy_from_scores(heuristic_scores)
    learned_margin = _edit_tb_margin_from_probs(probs, chosen_idx)
    heuristic_margin = _edit_tb_margin_from_probs(heuristic_scores, heuristic_idx)
    return Dict{String,Any}(
        "chosen_index" => chosen_idx,
        "override_applied" => override,
        "abstained_to_heuristic" => false,
        "heuristic_top_index" => heuristic_idx,
        "learned_top_index" => chosen_idx,
        "heuristic_margin" => heuristic_margin,
        "learned_margin" => learned_margin,
        "learned_advantage_vs_heuristic" => Float64(probs[chosen_idx] - probs[heuristic_idx]),
        "heuristic_entropy" => heuristic_entropy,
        "learned_entropy" => Float64(entropy),
        "entropy" => Float64(entropy),
        "selection_reason" => override ? "edit_tb_policy_override" : "edit_tb_policy_agree",
    )
end
