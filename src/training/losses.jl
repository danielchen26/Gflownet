# Loss Computation for GFlowNet Training
# Handles different training objectives and trajectory loss calculation

using Zygote
using Statistics

using ..GFlowNet: GFlowNetModel, Trajectory, TrainingConfig, TrainingObjective
using ..GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING, SUB_TRAJECTORY_BALANCE, DIRECT_FLOW_OBJECTIVE, TRAJECTORY_LIKELIHOOD_MAXIMIZATION
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
                    
                    # Compute flows in a non-differentiable way to avoid cache issues
                    source_flow = Zygote.@ignore flow(model, source)
                    target_flow = Zygote.@ignore flow(model, target)
                    
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
        # For flow matching, we need non-terminal states from trajectories
        # Extract all non-terminal states
        states = Zygote.@ignore begin
            all_states = AbstractState[]
            
            for traj in trajectories
                if !is_valid_trajectory(traj)
                    continue
                end
                
                # Add all non-terminal states
                for state in traj.states[1:end-1]  # Exclude last state (terminal)
                    if !is_terminal_state(state)
                        push!(all_states, state)
                    end
                end
            end
            
            # Remove duplicates to avoid biasing training
            unique(all_states)
        end
        
        if isempty(states)
            return 0.0
        end
        
        # Compute flow matching loss for each state
        losses = [
            begin
                # Compute expected flow (wrap flow computation in Zygote.@ignore)
                expected_flow = Zygote.@ignore begin
                    applicable_actions = get_applicable_actions(state, model.all_actions)
                    if isempty(applicable_actions)
                        0.0
                    else
                        action_probs = forward_action_probabilities(
                            model.forward_policy, state, model.all_actions,
                            params.forward, model.states.forward
                        )
                        
                        flow_sum = 0.0
                        for (action_idx, action) in enumerate(model.all_actions)
                            if action in applicable_actions
                                next_state = apply_action(action, state)
                                transition_prob = action_probs[action_idx]
                                next_flow = flow(model, next_state)
                                flow_sum += transition_prob * next_flow
                            end
                        end
                        flow_sum
                    end
                end
                
                # Get flow estimate from neural network (this is differentiable)
                estimated_flow = flow_estimate(
                    model.flow_estimator, state,
                    params.flow, model.states.flow
                )
                
                # Flow matching loss
                (estimated_flow - expected_flow)^2
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
        # Filter valid trajectories
        valid_trajectories = Zygote.@ignore [traj for traj in trajectories if is_valid_trajectory(traj)]

        if isempty(valid_trajectories)
            return 0.0
        end

        # Compute direct flow loss
        return direct_flow_loss_batch(model, valid_trajectories)

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

    for i in 1:(length(trajectory.states)-1)
        state = trajectory.states[i]
        action = trajectory.actions[i]

        # Get state features
        features = state_to_features(state)

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
    end

    # Get terminal reward (domain-specific function - non-differentiable)
    terminal_state = trajectory.states[end]
    terminal_reward = Zygote.@ignore reward(terminal_state)

    # Ensure positive reward for GFlowNet
    if terminal_reward <= 0
        terminal_reward = 1e-8
    end

    # Trajectory Balance Loss with optional learnable Z parameter
    # Standard form: (log Z + log P_F(τ) - log R(s_T))²
    log_reward = log(terminal_reward)
    
    # Add log Z term if using LEARNABLE_ESTIMATION
    log_Z = if haskey(params, :log_Z)
        params.log_Z  # Use learnable Z parameter
    else
        0.0  # SIMPLE_ESTIMATION: Z = 1, so log Z = 0
    end
    
    trajectory_balance_error = log_Z + log_prob_sum - log_reward

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
            if is_terminal_state(state)
                continue
            end

            # Get applicable actions
            applicable_actions = Zygote.@ignore get_applicable_actions(state, model.all_actions)
            if isempty(applicable_actions)
                continue
            end

            # Get action probabilities
            probs = forward_action_probabilities(
                model.forward_policy, state, model.all_actions,
                params.forward, model.states.forward
            )

            # Compute entropy: -Σ p log(p+ε)
            entropy = 0.0
            for p in probs
                if p > 1e-10
                    entropy -= p * log(p)
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
