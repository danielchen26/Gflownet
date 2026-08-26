# ============================================================================
# Edit-TB Loss Functions
# ============================================================================
#
# RWMLE (primary): reward-weighted maximum likelihood estimation
# TB-style (upgrade): forward-only TB variant with log P_B = 0
#
# RWMLE first, TB second — avoids conflating policy failure with TB assumption
# violation from environment stochasticity in edit episodes.

using Zygote
using NNlib: logsoftmax

# ============================================================================
# Shared Helpers
# ============================================================================

function _edit_tb_with_task_context(context::Vector{Float32}, task_features::Vector{Float32})::Vector{Float32}
    isempty(task_features) && return context
    return vcat(task_features, context)
end

"""
Compute log P(chosen | context, candidates) via softmax over candidate set.

Each candidate is scored by concatenating context + candidate features, passing through
the head network, then applying logsoftmax over all candidates.

Uses batched matrix forward pass to avoid array mutation (Zygote compatibility).
"""
function _edit_tb_log_prob_from_candidates(
    head, params, states,
    context::Vector{Float32},
    candidates::Matrix{Float32},
    chosen_index::Int,
    n_candidates::Int
)
    ctx_repeated = repeat(context, 1, n_candidates)  # (d_ctx, n_candidates)
    input_mat = vcat(ctx_repeated, candidates)        # (d_ctx + d_cand, n_candidates)

    scores, _ = head(input_mat, params, states)
    logits_vec = vec(scores)
    log_probs = logits_vec .- NNlib.logsumexp(logits_vec)
    return log_probs[chosen_index]
end

"""
Compute full factored forward log-probability for a trajectory.

log P_θ(τ) = Σ_t [log P(basin|s_t) + log P(parent|basin,s_t) + log P(operator|parent,basin,s_t)]
"""
function _edit_tb_compute_log_pf(model, params, states, traj::EditTBTrajectory;
                                 task_names::Vector{String}=String[],
                                 task_conditioning::Bool=false)
    task_features = Zygote.@ignore _edit_tb_task_features(traj.task_name, task_names; enabled=task_conditioning)
    log_pf = 0.0f0
    for step in traj.steps
        basin_context = _edit_tb_with_task_context(step.basin.frontier_features, task_features)
        parent_context = _edit_tb_with_task_context(step.parent.basin_features, task_features)
        operator_context = _edit_tb_with_task_context(step.operator.parent_features, task_features)

        log_pf = log_pf + _edit_tb_log_prob_from_candidates(
            model.basin, params.basin, states.basin,
            basin_context, step.basin.candidate_features,
            step.basin.chosen_index, step.basin.n_candidates)
        log_pf = log_pf + _edit_tb_log_prob_from_candidates(
            model.parent, params.parent, states.parent,
            parent_context, step.parent.candidate_features,
            step.parent.chosen_index, step.parent.n_candidates)
        log_pf = log_pf + _edit_tb_log_prob_from_candidates(
            model.operator, params.operator, states.operator,
            operator_context, step.operator.candidate_features,
            step.operator.chosen_index, step.operator.n_candidates)
    end
    return log_pf
end

# ============================================================================
# RWMLE Loss (PRIMARY — no TB assumptions needed)
# ============================================================================

"""
Reward-Weighted Maximum Likelihood Estimation loss.

L = -Σ_k w_k · log P_θ(τ_k) / Σ_j w_j

where w_k = R(τ_k)^β.

No backward model, no log_Z, no flow conservation assumptions.
Environment stochasticity does not affect this objective.
If this fails, the policy object is definitively wrong.
"""
function compute_edit_rwmle_loss(
    model, params, states,
    trajectories::Vector{EditTBTrajectory};
    beta::Float64=4.0,
    task_names::Vector{String}=String[],
    task_conditioning::Bool=false
)
    total_loss = 0.0f0
    total_weight = Zygote.@ignore Ref(0.0)

    for traj in trajectories
        skip = Zygote.@ignore (isempty(traj.steps) || traj.terminal_reward <= 0.001)
        skip && continue

        log_pf = _edit_tb_compute_log_pf(model, params, states, traj;
            task_names=task_names,
            task_conditioning=task_conditioning)

        w = Zygote.@ignore Float64(traj.terminal_reward) ^ beta
        Zygote.@ignore (total_weight[] += w)

        total_loss = total_loss - Float32(w) * log_pf
    end

    n_div = Zygote.@ignore Float32(max(total_weight[], 1e-8))
    return total_loss / n_div
end

# ============================================================================
# TB-Style Loss (UPGRADE — for later if RWMLE succeeds)
# ============================================================================

"""
TB-style fine-tuning loss (forward-only TB variant).

L = Σ_k w_k · shifted_cosh(log Z + log P_F(τ_k) - log R(τ_k)) / Σ_j w_j

with log P_B = 0 (deterministic prefix backward).

NOT classical TB in the strong GFlowNet sense — child generation is stochastic,
so terminal states are not fully determined by policy choices alone. We accept
this as reward noise and use the TB loss form as a diversity regularizer.
"""
function compute_edit_tb_style_loss(
    model, params, states,
    log_Z::AbstractVector{Float32},
    trajectories::Vector{EditTBTrajectory};
    beta::Float64=4.0, threshold::Real=2.0,
    reward_weighted::Bool=true,
    task_names::Vector{String}=String[],
    task_conditioning::Bool=false
)
    total_loss = 0.0f0
    total_weight = Zygote.@ignore Ref(0.0)

    for traj in trajectories
        skip = Zygote.@ignore (isempty(traj.steps) || traj.terminal_reward <= 0.001)
        skip && continue

        log_pf = _edit_tb_compute_log_pf(model, params, states, traj;
            task_names=task_names,
            task_conditioning=task_conditioning)

        log_R = Zygote.@ignore Float32(log(max(traj.terminal_reward, 0.001)))
        delta = log_Z[1] + log_pf - log_R

        w = Zygote.@ignore (reward_weighted ? traj.terminal_reward ^ beta : 1.0)
        Zygote.@ignore (total_weight[] += w)
        total_loss = total_loss + Float32(w) * apply_tb_loss(delta, :shifted_cosh; threshold=Float64(threshold))
    end

    n_div = Zygote.@ignore Float32(
        reward_weighted ? max(total_weight[], 1e-8) : Float64(max(1, length(trajectories)))
    )
    return total_loss / n_div
end
