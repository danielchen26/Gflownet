# Loss Computation for GFlowNet Training
# Handles different training objectives and trajectory loss calculation

using Zygote
using Statistics

using ..GFlowNet: GFlowNetModel, Trajectory, TrainingConfig, TrainingObjective
using ..GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING, SUB_TRAJECTORY_BALANCE, DIRECT_FLOW_OBJECTIVE, TRAJECTORY_LIKELIHOOD_MAXIMIZATION, MULTI_OBJECTIVE_TB
using ..GFlowNet: AbstractState, AbstractAction
using ..GFlowNet: state_to_features, is_terminal_state, reward, is_applicable, apply_action
using ..GFlowNet: get_applicable_actions, is_valid_trajectory
using ..GFlowNet: forward_action_probabilities, compute_backward_probability
using ..GFlowNet: forward_transition_probability, backward_transition_probability
using ..GFlowNet: flow, flow_estimate, compute_flow_estimate
using ..GFlowNet: sub_trajectory_balance_loss_batch, direct_flow_loss_batch

# =============================================================================
# Loss Computation - Mathematically Correct Implementations
# =============================================================================

"""
    compute_trajectory_loss(model, trajectories, params, config)

Compute loss based on the specified training objective.

Supports:
- TRAJECTORY_BALANCE: P_F(τ) ∝ R(s_T)
- DETAILED_BALANCE: P_F(s→s') F(s) = P_B(s'→s) F(s')
- FLOW_MATCHING: F(s) = Σ_{s'} P_F(s'|s) * F(s')
"""
function compute_trajectory_loss(model::GFlowNetModel, trajectories::Vector{Trajectory},
                                params, config::TrainingConfig)

    if config.objective == TRAJECTORY_BALANCE
        # Filter valid trajectories (discrete validation - non-differentiable)
        valid_trajectories = Zygote.@ignore [traj for traj in trajectories if is_valid_trajectory(traj)]

        if isempty(valid_trajectories)
            return 0.0
        end

        # Compute losses using Zygote-safe operations
        losses = [compute_single_trajectory_loss(model, traj, params) for traj in valid_trajectories]

        # Filter out infinite losses
        finite_losses = filter(!isinf, losses)

        if isempty(finite_losses)
            return Inf
        end

        base_loss = mean(finite_losses)

        # Add entropy regularization if configured (AISTATS 2024: GFlowNets as Entropy-Regularized RL)
        # This encourages exploration and prevents mode collapse
        if config.entropy_weight > 0.0
            entropy_loss = compute_policy_entropy_loss(model, valid_trajectories, params)
            return base_loss + config.entropy_weight * entropy_loss
        end

        return base_loss

    elseif config.objective == DETAILED_BALANCE
        # For detailed balance, we need pairs of states
        # Extract state pairs from trajectories (done outside gradient computation)
        state_pairs = Zygote.@ignore begin
            pairs = Tuple{AbstractState, AbstractState}[]
            
            for traj in trajectories
                if !is_valid_trajectory(traj)
                    continue
                end
                
                # Extract consecutive state pairs from trajectory
                for i in 1:(length(traj.states)-1)
                    push!(pairs, (traj.states[i], traj.states[i+1]))
                end
            end
            
            pairs
        end
        
        if isempty(state_pairs)
            return 0.0
        end
        
        # Compute detailed balance loss for each pair using array comprehension (Zygote-safe)
        # Filter out invalid transitions using try-catch outside gradient computation
        valid_pairs = Zygote.@ignore begin
            valid = Tuple{AbstractState, AbstractState}[]
            for (source, target) in state_pairs
                # Check if transition is valid
                applicable_actions = get_applicable_actions(source, model.all_actions)
                can_transition = false
                for action in applicable_actions
                    if apply_action(action, source) == target
                        can_transition = true
                        break
                    end
                end
                if can_transition && !is_terminal_state(source)
                    push!(valid, (source, target))
                end
            end
            valid
        end
        
        if isempty(valid_pairs)
            return 0.0
        end
        
        # Now compute losses only for valid pairs using array comprehension
        # We need to compute the detailed balance loss with the current parameters
        losses = [
            begin
                source, target = pair
                
                # Compute probabilities with current parameters
                # Forward probability
                applicable_actions = get_applicable_actions(source, model.all_actions)
                valid_actions = [action for action in applicable_actions 
                               if apply_action(action, source) == target]
                
                if isempty(valid_actions)
                    Inf  # Skip this pair
                else
                    # Get forward probabilities
                    probs = forward_action_probabilities(
                        model.forward_policy, source, model.all_actions,
                        params.forward, model.states.forward
                    )
                    
                    forward_prob = 0.0
                    for (i, action) in enumerate(model.all_actions)
                        if action in valid_actions
                            forward_prob += probs[i]
                        end
                    end
                    
                    # Backward probability
                    backward_prob = if isnothing(model.backward_policy)
                        1.0
                    else
                        compute_backward_probability(
                            model.backward_policy, target, source,
                            params.backward, model.states.backward,
                            model.all_actions
                        )
                    end
                    
                    # Flows must be DIFFERENTIABLE for Detailed Balance to be
                    # trainable: DB's whole content is that F adjusts until
                    # P_F(s'|s)F(s) = P_B(s|s')F(s'). These were previously
                    # `Zygote.@ignore flow(model, ...)`, which left DB's :flow
                    # gradient norm at exactly 0.0 -- the objective could only
                    # move the policy against a frozen target it could never
                    # close against.
                    #
                    # TERMINAL BOUNDARY F(x) = R(x). Without it the reward never
                    # enters DB at all: switching to the learned estimator
                    # attached the gradient but made the loss reward-blind,
                    # measured as a delta of exactly 0.0 under a 100x reward
                    # change. DB's boundary condition IS what ties F to the task.
                    #
                    # A model with no flow estimator is still supported: DB then
                    # falls back to the recursive flow, exactly as before. There
                    # is no flow network to train in that case, so nothing is
                    # lost by it being non-differentiable, and the reward still
                    # enters because compute_recursive_flow returns R at
                    # terminals. Erroring here instead broke every existing
                    # DB caller that omitted the estimator.
                    has_flow_net = !isnothing(model.flow_estimator) && haskey(params, :flow)

                    flow_at(s) = if Zygote.@ignore(is_terminal_state(s))
                        Zygote.@ignore(max(reward(s), 1e-8))
                    elseif has_flow_net
                        flow_estimate(model.flow_estimator, s, params.flow, model.states.flow)
                    else
                        Zygote.@ignore(max(flow(model, s), 1e-8))
                    end

                    source_flow = flow_at(source)
                    target_flow = flow_at(target)
                    
                    # Compute detailed balance loss
                    left_side = log(max(forward_prob, 1e-8)) + log(max(source_flow, 1e-8))
                    right_side = log(max(backward_prob, 1e-8)) + log(max(target_flow, 1e-8))
                    (left_side - right_side)^2
                end
            end
            for pair in valid_pairs
        ]
        
        # Filter out infinite losses
        finite_losses = filter(!isinf, losses)

        if isempty(finite_losses)
            return Inf
        end

        base_loss = mean(finite_losses)

        # Add entropy regularization if configured
        if config.entropy_weight > 0.0
            valid_trajs = Zygote.@ignore [traj for traj in trajectories if is_valid_trajectory(traj)]
            if !isempty(valid_trajs)
                entropy_loss = compute_policy_entropy_loss(model, valid_trajs, params)
                return base_loss + config.entropy_weight * entropy_loss
            end
        end

        return base_loss

    elseif config.objective == FLOW_MATCHING
        # Flow conservation on STATE flows, in log space:
        #
        #     sum over parents p of  F(p) * P_F(s|p)   ==   F(s)
        #     with the boundary condition F(x) = R(x) at terminal x
        #
        # Three defects are fixed here, all measured:
        #  1. The policy was inside `Zygote.@ignore`, so the forward parameters
        #     received a gradient norm of 0.004535 against TB's 9.923 -- and that
        #     residual was only the entropy term, whose sign drives the sampler
        #     TOWARD uniform. Flow matching could not learn a reward-seeking
        #     policy at all. P_F is now inside the gradient.
        #  2. The comparison was `(estimated - expected)^2` in RAW flow space, so
        #     the loss scaled as O(R^2) and was dominated by reward magnitude.
        #     It is now a difference of logs.
        #  3. There was no in-flow over parents and no R(s) term, so reward never
        #     entered: a 100x reward change moved the loss by exactly 0.0.
        #
        # NOTE ON FAITHFULNESS: Bengio et al. 2021 parameterise EDGE flows
        # F(s,a). This estimator produces STATE flows, so what is implemented is
        # the state-flow form of the same conservation law. That is a valid
        # condition with the same fixed point, but it is not literally the
        # edge-flow objective of the paper, and the docs should not claim it is.
        states = Zygote.@ignore begin
            all_states = AbstractState[]
            for traj in trajectories
                is_valid_trajectory(traj) || continue
                # Terminal states are INCLUDED: F(x) = R(x) is where the reward
                # enters the objective.
                for state in traj.states
                    push!(all_states, state)
                end
            end
            unique(all_states)
        end

        if isempty(states)
            return 0.0
        end

        # A model with no flow estimator falls back to the recursive flow, so
        # existing callers that omitted the estimator keep working. There is no
        # flow network to train in that case, and the reward still enters
        # through the terminal boundary.
        fm_has_flow_net = !isnothing(model.flow_estimator) && haskey(params, :flow)

        # log F(s), applying the terminal boundary.
        log_flow_of(s) = if Zygote.@ignore(is_terminal_state(s))
            log(max(Zygote.@ignore(reward(s)), 1e-8))
        elseif fm_has_flow_net
            log(max(flow_estimate(model.flow_estimator, s, params.flow, model.states.flow), 1e-12))
        else
            log(max(Zygote.@ignore(flow(model, s)), 1e-12))
        end

        losses = [
            begin
                parents = Zygote.@ignore backward_parent_states(state, model.all_actions)
                if isempty(parents)
                    # The initial state has no parents; conservation is vacuous there.
                    0.0
                else
                    log_terms = map(parents) do p
                        # index of the action taking p -> state
                        idx = Zygote.@ignore begin
                            applicable = get_applicable_actions(p, model.all_actions)
                            findfirst(a -> a in applicable && apply_action(a, p) == state,
                                      model.all_actions)
                        end
                        if idx === nothing
                            -Inf
                        else
                            probs = forward_action_probabilities(
                                model.forward_policy, p, model.all_actions,
                                params.forward, model.states.forward
                            )
                            log_flow_of(p) + log(max(probs[idx], 1e-12))
                        end
                    end
                    finite_terms = filter(!isinf, log_terms)
                    if isempty(finite_terms)
                        0.0
                    else
                        (logsumexp(finite_terms) - log_flow_of(state))^2
                    end
                end
            end
            for state in states
        ]
        
        # Filter out infinite losses
        finite_losses = filter(!isinf, losses)

        if isempty(finite_losses)
            return Inf
        end

        base_loss = mean(finite_losses)

        # Add entropy regularization if configured
        if config.entropy_weight > 0.0
            valid_trajs = Zygote.@ignore [traj for traj in trajectories if is_valid_trajectory(traj)]
            if !isempty(valid_trajs)
                entropy_loss = compute_policy_entropy_loss(model, valid_trajs, params)
                return base_loss + config.entropy_weight * entropy_loss
            end
        end

        return base_loss

    elseif config.objective == SUB_TRAJECTORY_BALANCE
        # Validate that flow estimator exists (REQUIRED for SubTB)
        if isnothing(model.flow_estimator)
            throw(ArgumentError(
                "SUB_TRAJECTORY_BALANCE requires a flow estimator. " *
                "Create model with: include_flow_estimator=true"
            ))
        end

        # Get sub-trajectory length from config
        sub_length = config.sub_trajectory_length

        # Compute sub-trajectory balance loss with params for differentiability
        return sub_trajectory_balance_loss_batch(model, trajectories, params; sub_length=sub_length)
        
    elseif config.objective == DIRECT_FLOW_OBJECTIVE
        # DELIBERATELY UNAVAILABLE. This branch called
        # `direct_flow_loss_batch(model, valid_trajectories)` WITHOUT `params`, and
        # direct_flow_loss internally reads `model.parameters` and wraps log Z in
        # Zygote.@ignore. The whole expression was therefore a constant with
        # respect to the differentiation variable: Zygote returned nothing, and
        # train_step! short-circuited to `return Inf, 0.0`. Measured: it throws
        # during compilation. Training under it was a no-op that looked like
        # training.
        #
        # It is also not a published objective -- it is TB with a state-conditioned
        # Z(s0) from the flow network. Use TRAJECTORY_BALANCE instead, which is now
        # verified to sample proportionally to reward.
        throw(ArgumentError(
            "DIRECT_FLOW_OBJECTIVE is not implemented correctly and is disabled. " *
            "Its loss was constant with respect to the model parameters, so " *
            "training under it silently did nothing. Use TRAJECTORY_BALANCE."
        ))

    elseif config.objective == TRAJECTORY_LIKELIHOOD_MAXIMIZATION
        # TLM (ICLR 2025): Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization
        #
        # Key insight: Max-entropy backward policy is P_B(s|s') ∝ n(s)/n(s') where n(s) = #paths to s
        # Training backward policy via -log P_B(s|s') implicitly encodes path counts
        # This directly solves the extreme path asymmetry problem (e.g., 70:1)
        #
        # Loss: L_TLM = L_forward + λ * L_backward
        # where L_forward is standard TB loss and L_backward = -Σ log P_B(s_{i-1}|s_i)

        # Filter valid trajectories
        valid_trajectories = Zygote.@ignore [traj for traj in trajectories if is_valid_trajectory(traj)]

        if isempty(valid_trajectories)
            return 0.0
        end

        # Compute forward loss (standard TB loss)
        forward_losses = [compute_single_trajectory_loss(model, traj, params) for traj in valid_trajectories]
        finite_forward = filter(!isinf, forward_losses)
        forward_loss = isempty(finite_forward) ? Inf : mean(finite_forward)

        # Compute backward likelihood loss if backward policy exists
        backward_loss = if !isnothing(model.backward_policy) && haskey(params, :backward)
            compute_tlm_backward_loss(model, valid_trajectories, params)
        else
            0.0
        end

        # Combine losses with TLM backward weight
        total_loss = forward_loss + config.tlm_backward_weight * backward_loss

        # Add entropy regularization if configured (for forward policy)
        if config.entropy_weight > 0.0
            entropy_loss = compute_policy_entropy_loss(model, valid_trajectories, params)
            total_loss += config.entropy_weight * entropy_loss
        end

        # Add backward policy entropy if configured (encourages uniform backward exploration)
        if config.tlm_entropy_coeff > 0.0 && !isnothing(model.backward_policy)
            backward_entropy_loss = compute_backward_policy_entropy_loss(model, valid_trajectories, params)
            total_loss += config.tlm_entropy_coeff * backward_entropy_loss
        end

        return total_loss

    elseif config.objective == MULTI_OBJECTIVE_TB
        # MOGFN-PC (Gap 5, ICML 2023): Preference-conditioned trajectory balance
        #
        # L(τ, w) = (log Z(w) + Σ log P_F(aᵢ|sᵢ, w) - log R(s_T, w))²
        #
        # Each batch uses a freshly sampled preference vector w from Dirichlet(α).
        # The preference is embedded via shared encoder: w_embed = encode(w).
        # Forward policy sees [state_features; w_embed], Z is computed via Z network.

        # Filter valid trajectories
        valid_trajectories = Zygote.@ignore [traj for traj in trajectories if is_valid_trajectory(traj)]

        if isempty(valid_trajectories)
            return 0.0
        end

        # Check that model has MOGFN components
        if isnothing(model.preference_encoder) || isnothing(model.z_network)
            throw(ArgumentError(
                "MULTI_OBJECTIVE_TB requires preference_encoder and z_network. " *
                "Use create_mogfn_gflownet() or create_mogfn_molecular_gflownet()."
            ))
        end

        # Sample preference vector for this batch (non-differentiable)
        n_obj = config.mogfn_n_objectives
        w = Zygote.@ignore begin
            alpha = config.mogfn_dirichlet_alpha
            if alpha == 1.0
                gammas = [randexp() for _ in 1:n_obj]
            else
                gammas = Float64[_mogfn_sample_gamma(alpha) for _ in 1:n_obj]
            end
            gammas ./ sum(gammas)
        end
        w_f32 = Float32.(w)

        # Compute preference embedding (differentiable)
        w_embed, _ = model.preference_encoder(w_f32, params.preference, model.states.preference)

        # Compute losses for each trajectory with this preference
        losses = [compute_mogfn_single_trajectory_loss(model, traj, w, w_embed, params)
                  for traj in valid_trajectories]

        finite_losses = filter(!isinf, losses)
        if isempty(finite_losses)
            return Inf
        end

        base_loss = mean(finite_losses)

        # Add entropy regularization if configured
        if config.entropy_weight > 0.0
            entropy_loss = compute_mogfn_entropy_loss(model, valid_trajectories, w_embed, params)
            base_loss += config.entropy_weight * entropy_loss
        end

        return base_loss

    else
        throw(ArgumentError("Unsupported training objective: $(config.objective)"))
    end
end

"""
    compute_single_trajectory_loss(model, trajectory, params)

Compute loss for single trajectory with CORRECTED trajectory balance.
"""
function compute_single_trajectory_loss(model::GFlowNetModel, trajectory::Trajectory, params)

    # Compute log probability of trajectory
    log_prob_sum = 0.0

    # Backward log-probability over the SAME transitions. The published TB
    # objective is (log Z + sum log P_F - log R - sum log P_B)^2; this term was
    # missing, which is equivalent to asserting P_B == 1 for every edge, i.e. that
    # every state has exactly one parent. The grid and fragment DAGs are lattices,
    # not trees, so omitting it biased the optimum by the path count n(x):
    # verified on the 3x3 grid, the coded optimum was exactly
    # n(x)R(x)/sum_y n(y)R(y) to 1.11e-16, giving Z = 78 instead of 19 and
    # per-terminal sampling ratios from 0.2436 (n=1) to 1.4615 (n=6).
    log_backward_sum = 0.0

    for i in 1:(length(trajectory.states)-1)
        state = trajectory.states[i]
        action = trajectory.actions[i]

        # Get state features (non-differentiable: features are constant inputs, not trainable)
        features = Zygote.@ignore state_to_features(state)

        # Compute forward logits using proper Lux call (Zygote-safe)
        logits_vec, _ = model.forward_policy.model(features, params.forward, model.states.forward)

        # Get applicable actions on-demand (discrete logic - non-differentiable)
        applicable_actions = Zygote.@ignore get_applicable_actions(state, model.all_actions)

        if isempty(applicable_actions)
            return Inf  # Invalid trajectory
        end

        # Find action and applicable indices (discrete logic - non-differentiable)
        action_idx = Zygote.@ignore findfirst(a -> a == action, model.all_actions)
        applicable_indices = Zygote.@ignore [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]

        if isnothing(action_idx)
            return Inf  # Invalid action
        end

        if !(action_idx in applicable_indices)
            return Inf  # Action not applicable
        end

        # Compute log probability using numerically stable operations
        applicable_logits = logits_vec[applicable_indices]
        if isempty(applicable_logits)
            return Inf
        end

        # Use logsumexp for numerical stability
        log_probs = applicable_logits .- logsumexp(applicable_logits)

        # Find action position in applicable actions (discrete logic - non-differentiable)
        action_pos = Zygote.@ignore findfirst(==(action_idx), applicable_indices)

        if isnothing(action_pos)
            return Inf
        end

        log_prob_sum += log_probs[action_pos]

        # P_B(s_i | s_{i+1}) for this same transition.
        #
        # The previous code fell through with log P_B = 0, i.e. P_B == 1, and a comment
        # here called it "a stated assumption rather than an omission". Stating an
        # assumption is not checking it: P_B == 1 is a distribution only when every
        # state has ONE parent. Grid world's (2,2) has two.
        #
        # Measured consequence before this repair: TB converged to
        # sum_x n_paths(x) R(x) instead of sum_x R(x) -- 77.928 vs a true Z of 19.0 on
        # the 3x3 grid, and 22.0 vs 12.0 on the 2x2 -- so the sampler was biased toward
        # states reachable by more paths. That is the exact defect
        # test/theory/test_reward_proportionality.jl calls "the bug that was fixed"
        # while only ever exercising the analytic helpers in enumerate.jl, never this
        # loss. The regression test guarded a repair that had never been applied here.
        #
        # TB is valid for ANY fixed normalised P_B; uniform-over-parents is the
        # canonical choice and makes sum_tau P_B(tau|x) = 1, restoring Z = sum_x R(x).
        child = trajectory.states[i + 1]
        if !isnothing(model.backward_policy) && haskey(params, :backward)
            pb = compute_backward_probability(
                model.backward_policy, child, state,
                params.backward, model.states.backward, model.all_actions
            )
            log_backward_sum += log(max(pb, 1e-8))
        else
            n_parents = Zygote.@ignore length(backward_parent_states(child, model.all_actions))
            n_parents > 1 && (log_backward_sum += -log(n_parents))
        end
    end

    # Get terminal reward (domain-specific function - non-differentiable)
    terminal_state = trajectory.states[end]
    terminal_reward = Zygote.@ignore reward(terminal_state)

    # Ensure positive reward for GFlowNet
    if terminal_reward <= 0
        terminal_reward = 1e-8
    end

    # Trajectory Balance (Malkin et al. 2022):
    #   L = (log Z + sum log P_F(tau) - sum log P_B(tau) - log R(s_T))^2
    log_reward = log(terminal_reward)

    # Add log Z term if using LEARNABLE_ESTIMATION
    log_Z = if haskey(params, :log_Z)
        params.log_Z  # Use learnable Z parameter
    else
        0.0  # SIMPLE_ESTIMATION: Z = 1, so log Z = 0
    end

    trajectory_balance_error = log_Z + log_prob_sum - log_backward_sum - log_reward

    return trajectory_balance_error^2
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    logsumexp(x)

Numerically stable log-sum-exp operation.
"""
function logsumexp(x::AbstractVector)
    if isempty(x)
        return -Inf
    end
    max_x = maximum(x)
    if isinf(max_x)
        return max_x
    end
    return max_x + log(sum(exp.(x .- max_x)))
end

"""
    compute_policy_entropy_loss(model, trajectories, params)

Compute negative policy entropy loss for entropy regularization.

The entropy loss encourages exploration by penalizing low-entropy (deterministic) policies.
Returns negative entropy so that adding it to the loss increases exploration.

Mathematical foundation:
L_entropy = -H(π) = Σ_s Σ_a P_F(a|s) log P_F(a|s)
"""
function compute_policy_entropy_loss(model::GFlowNetModel, trajectories::Vector{Trajectory}, params)
    total_entropy = 0.0
    n_states = 0

    for traj in trajectories
        for i in 1:(length(traj.states) - 1)  # Skip terminal states
            state = traj.states[i]

            # Skip if terminal (no applicable actions)
            if Zygote.@ignore is_terminal_state(state)
                continue
            end

            # Get state features and applicable indices (non-differentiable)
            features = Zygote.@ignore state_to_features(state)
            applicable_indices = Zygote.@ignore begin
                applicable = get_applicable_actions(state, model.all_actions)
                isempty(applicable) ? Int[] : [idx for (idx, a) in enumerate(model.all_actions) if a in applicable]
            end

            if isempty(applicable_indices)
                continue
            end

            # Compute forward logits directly (differentiable — same pattern as TB loss)
            logits_vec, _ = model.forward_policy.model(features, params.forward, model.states.forward)

            # Softmax over applicable actions (differentiable)
            applicable_logits = logits_vec[applicable_indices]
            log_probs = applicable_logits .- logsumexp(applicable_logits)
            probs = exp.(log_probs)

            # Compute entropy: H = -Σ p * log(p) using log_probs for numerical stability
            entropy = 0.0
            for (p, lp) in zip(probs, log_probs)
                if p > 1e-10
                    entropy -= p * lp
                end
            end

            total_entropy += entropy
            n_states += 1
        end
    end

    # Return negative average entropy (minimize this to maximize entropy)
    return n_states > 0 ? -(total_entropy / n_states) : 0.0
end

# =============================================================================
# Importance-Weighted Loss for Off-Policy Learning (Phase 4)
# =============================================================================

"""
    compute_weighted_trajectory_loss(model, trajectories, weights, params, config)

Compute importance-weighted trajectory loss for off-policy learning.

This is essential when using experience replay, as trajectories from
the replay buffer were sampled under a different (older) policy.

Mathematical Foundation (JMLR 2023: GFlowNet Foundations):
    L_weighted = (1/Σw) × Σᵢ wᵢ × L(τᵢ)

where wᵢ are importance sampling weights that correct for the
distribution mismatch between behavior policy and current policy.

# Arguments
- `model::GFlowNetModel`: The GFlowNet model
- `trajectories::Vector{Trajectory}`: Trajectories to compute loss for
- `weights::Vector{Float64}`: Importance weights for each trajectory
- `params`: Current model parameters
- `config::TrainingConfig`: Training configuration

# Returns
Weighted loss value (Float64)
"""
function compute_weighted_trajectory_loss(model::GFlowNetModel,
                                         trajectories::Vector{Trajectory},
                                         weights::Vector{Float64},
                                         params, config::TrainingConfig)
    if isempty(trajectories)
        return 0.0
    end

    # Ensure weights match trajectories
    if length(weights) != length(trajectories)
        throw(ArgumentError("weights length ($(length(weights))) must match trajectories length ($(length(trajectories)))"))
    end

    # Filter valid trajectories with their weights
    valid_data = Zygote.@ignore begin
        [(traj, w) for (traj, w) in zip(trajectories, weights) if is_valid_trajectory(traj)]
    end

    if isempty(valid_data)
        return 0.0
    end

    valid_trajectories = [d[1] for d in valid_data]
    valid_weights = [d[2] for d in valid_data]

    # Compute individual losses (same as compute_trajectory_loss but for TB only currently)
    if config.objective == TRAJECTORY_BALANCE
        losses = [compute_single_trajectory_loss(model, traj, params) for traj in valid_trajectories]

        # Apply importance weights
        weighted_losses = losses .* valid_weights

        # Filter out infinite losses
        finite_mask = .!isinf.(weighted_losses)
        if !any(finite_mask)
            return Inf
        end

        # Normalize by sum of weights for valid samples
        base_loss = sum(weighted_losses[finite_mask]) / sum(valid_weights[finite_mask])

        # Add entropy regularization if configured
        if config.entropy_weight > 0.0
            entropy_loss = compute_policy_entropy_loss(model, valid_trajectories, params)
            return base_loss + config.entropy_weight * entropy_loss
        end

        return base_loss
    else
        # For other objectives, fall back to unweighted loss for now
        # (Full weighted support can be added as needed)
        return compute_trajectory_loss(model, trajectories, params, config)
    end
end

# =============================================================================
# TLM (Trajectory Likelihood Maximization) Loss Functions - ICLR 2025
# =============================================================================

"""
    compute_tlm_backward_loss(model, trajectories, params)

Compute the backward likelihood loss for TLM training.

# Mathematical Foundation (ICLR 2025)
The TLM backward loss maximizes the likelihood of backward transitions:
    L_backward = -(1/N) Σ_{τ} Σ_{i=1}^{T-1} log P_B(s_{i-1}|s_i)

This trains the backward policy to learn the path count structure:
- States with many paths leading to them get higher backward probability
- Max-entropy backward policy: P_B(s|s') = n(s)/n(s') where n(s) = #paths
- This implicitly compensates for path asymmetry

# Arguments
- `model::GFlowNetModel`: The GFlowNet model with backward policy
- `trajectories::Vector{Trajectory}`: Trajectories to compute loss for
- `params`: Model parameters including backward policy params

# Returns
Average negative log-likelihood of backward transitions (lower is better convergence)
"""
function compute_tlm_backward_loss(model::GFlowNetModel, trajectories::Vector{Trajectory}, params)
    total_log_prob = 0.0
    n_transitions = 0

    for traj in trajectories
        # Iterate through trajectory transitions
        for i in 2:length(traj.states)
            source_state = traj.states[i-1]  # s_{i-1}
            target_state = traj.states[i]     # s_i

            # Skip if target is terminal (no backward transition from terminal)
            if Zygote.@ignore is_terminal_state(source_state)
                continue
            end

            # Compute P_B(s_{i-1}|s_i) - probability of going backward from s_i to s_{i-1}
            backward_prob = compute_backward_probability(
                model.backward_policy, target_state, source_state,
                params.backward, model.states.backward,
                model.all_actions
            )

            # Clamp to avoid log(0)
            safe_prob = max(backward_prob, 1e-8)
            total_log_prob += log(safe_prob)
            n_transitions += 1
        end
    end

    if n_transitions == 0
        return 0.0
    end

    # Return negative average log-likelihood (minimize this to maximize likelihood)
    return -total_log_prob / n_transitions
end

"""
    compute_backward_policy_entropy_loss(model, trajectories, params)

Compute negative entropy loss for the backward policy.

# Mathematical Foundation
Encourages the backward policy to be exploratory:
    L_backward_entropy = -H(P_B) = Σ_s' Σ_s P_B(s|s') log P_B(s|s')

A higher entropy backward policy helps discover diverse backward paths,
which is important for learning the correct path count structure.

# Arguments
- `model::GFlowNetModel`: The GFlowNet model with backward policy
- `trajectories::Vector{Trajectory}`: Trajectories to compute entropy over
- `params`: Model parameters

# Returns
Negative average backward policy entropy (minimize to maximize entropy)
"""
function compute_backward_policy_entropy_loss(model::GFlowNetModel, trajectories::Vector{Trajectory}, params)
    total_entropy = 0.0
    n_states = 0

    for traj in trajectories
        # Consider each non-initial state as a potential backward source
        for i in 2:length(traj.states)
            target_state = traj.states[i]

            # Skip terminal states
            if Zygote.@ignore is_terminal_state(target_state)
                continue
            end

            # Get potential parent states (states that could transition to target)
            parent_states = Zygote.@ignore begin
                parents = eltype(traj.states)[]
                for state in traj.states[1:i-1]
                    if !is_terminal_state(state)
                        applicable = get_applicable_actions(state, model.all_actions)
                        for action in applicable
                            if apply_action(action, state) == target_state
                                push!(parents, state)
                                break
                            end
                        end
                    end
                end
                unique(parents)
            end

            if isempty(parent_states)
                continue
            end

            # Compute backward probabilities for all parents
            probs = [
                compute_backward_probability(
                    model.backward_policy, target_state, parent,
                    params.backward, model.states.backward,
                    model.all_actions
                )
                for parent in parent_states
            ]

            # Normalize probabilities
            prob_sum = sum(probs)
            if prob_sum > 1e-8
                normalized_probs = probs ./ prob_sum

                # Compute entropy: -Σ p log(p+ε)
                entropy = 0.0
                for p in normalized_probs
                    if p > 1e-10
                        entropy -= p * log(p)
                    end
                end

                total_entropy += entropy
                n_states += 1
            end
        end
    end

    # Return negative average entropy (minimize to maximize entropy)
    return n_states > 0 ? -(total_entropy / n_states) : 0.0
end

# =============================================================================
# MOGFN-PC Loss Functions (Gap 5, ICML 2023)
# =============================================================================

"""
    compute_mogfn_single_trajectory_loss(model, trajectory, w, w_embed, params)

Compute MOGFN preference-conditioned trajectory balance loss for a single trajectory.

L(τ, w) = (log Z(w) + Σ log P_F(aᵢ|sᵢ, w) - log R(s_T, w))²

The forward policy receives augmented features [state_features; w_embed].
Z(w) is computed via the z_network from the preference embedding.
R(s_T, w) is the linear scalarization of objectives with preference weights.
"""
function compute_mogfn_single_trajectory_loss(model::GFlowNetModel,
                                              trajectory::Trajectory,
                                              w::Vector{Float64},
                                              w_embed::AbstractVector,
                                              params)

    # Compute log Z(w) via Z network (differentiable)
    log_Z_vec, _ = model.z_network(w_embed, params.z_net, model.states.z_net)
    log_Z = log_Z_vec[1]  # Scalar

    # Compute log probability of trajectory under conditioned policy
    log_prob_sum = 0.0

    for i in 1:(length(trajectory.states)-1)
        state = trajectory.states[i]
        action = trajectory.actions[i]

        # Get state features and augment with preference embedding
        features = Zygote.@ignore state_to_features(state)
        augmented_features = vcat(features, w_embed)

        # Compute forward logits using augmented features (differentiable)
        logits_vec, _ = model.forward_policy.model(augmented_features, params.forward, model.states.forward)

        # Get applicable actions (non-differentiable)
        applicable_actions = Zygote.@ignore get_applicable_actions(state, model.all_actions)

        if isempty(applicable_actions)
            return Inf
        end

        # Find action and applicable indices
        action_idx = Zygote.@ignore findfirst(a -> a == action, model.all_actions)
        applicable_indices = Zygote.@ignore [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]

        if isnothing(action_idx) || !(action_idx in applicable_indices)
            return Inf
        end

        # Compute log probability using numerically stable operations
        applicable_logits = logits_vec[applicable_indices]
        if isempty(applicable_logits)
            return Inf
        end

        log_probs = applicable_logits .- logsumexp(applicable_logits)
        action_pos = Zygote.@ignore findfirst(==(action_idx), applicable_indices)

        if isnothing(action_pos)
            return Inf
        end

        log_prob_sum += log_probs[action_pos]
    end

    # Get terminal reward with preference scalarization (non-differentiable)
    terminal_state = trajectory.states[end]
    terminal_reward = Zygote.@ignore reward(terminal_state, w)

    if terminal_reward <= 0
        terminal_reward = 1e-8
    end

    log_reward = log(terminal_reward)

    # MOGFN Trajectory Balance Loss
    trajectory_balance_error = log_Z + log_prob_sum - log_reward
    return trajectory_balance_error^2
end

"""
    compute_mogfn_entropy_loss(model, trajectories, w_embed, params)

Compute policy entropy loss for MOGFN (preference-conditioned).
Same as standard entropy loss but with augmented features [state; w_embed].
"""
function compute_mogfn_entropy_loss(model::GFlowNetModel, trajectories::Vector{Trajectory},
                                    w_embed::AbstractVector, params)
    total_entropy = 0.0
    n_states = 0

    for traj in trajectories
        for i in 1:(length(traj.states) - 1)
            state = traj.states[i]

            if Zygote.@ignore is_terminal_state(state)
                continue
            end

            features = Zygote.@ignore state_to_features(state)
            augmented_features = vcat(features, w_embed)

            applicable_indices = Zygote.@ignore begin
                applicable = get_applicable_actions(state, model.all_actions)
                isempty(applicable) ? Int[] : [idx for (idx, a) in enumerate(model.all_actions) if a in applicable]
            end

            if isempty(applicable_indices)
                continue
            end

            logits_vec, _ = model.forward_policy.model(augmented_features, params.forward, model.states.forward)
            applicable_logits = logits_vec[applicable_indices]
            log_probs = applicable_logits .- logsumexp(applicable_logits)
            probs = exp.(log_probs)

            entropy = 0.0
            for (p, lp) in zip(probs, log_probs)
                if p > 1e-10
                    entropy -= p * lp
                end
            end

            total_entropy += entropy
            n_states += 1
        end
    end

    return n_states > 0 ? -(total_entropy / n_states) : 0.0
end

"""Helper: Sample from Gamma(alpha, 1) for non-unit Dirichlet (used in MOGFN loss)."""
function _mogfn_sample_gamma(alpha::Float64)::Float64
    if alpha >= 1.0
        d = alpha - 1.0/3.0
        c = 1.0 / sqrt(9.0 * d)
        while true
            x = randn()
            v = (1.0 + c * x)^3
            if v > 0.0
                u = rand()
                if u < 1.0 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
                    return d * v
                end
            end
        end
    else
        return _mogfn_sample_gamma(alpha + 1.0) * rand()^(1.0 / alpha)
    end
end
