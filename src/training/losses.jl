# Loss Computation for GFlowNet Training
# Handles different training objectives and trajectory loss calculation

using Zygote
using Statistics

using ..GFlowNet: GFlowNetModel, Trajectory, TrainingConfig, TrainingObjective
using ..GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING, SUB_TRAJECTORY_BALANCE
using ..GFlowNet: AbstractState, AbstractAction
using ..GFlowNet: state_to_features, is_terminal_state, reward, is_applicable, apply_action
using ..GFlowNet: get_applicable_actions, is_valid_trajectory
using ..GFlowNet: forward_action_probabilities, compute_backward_probability
using ..GFlowNet: forward_transition_probability, backward_transition_probability
using ..GFlowNet: flow, flow_estimate
using ..GFlowNet: sub_trajectory_balance_loss_batch

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

        return mean(finite_losses)
        
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
        
        return mean(finite_losses)
        
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
        
        return mean(finite_losses)
        
    elseif config.objective == SUB_TRAJECTORY_BALANCE
        # Get sub-trajectory length from config
        sub_length = config.sub_trajectory_length
        
        # Compute sub-trajectory balance loss
        return sub_trajectory_balance_loss_batch(model, trajectories; sub_length=sub_length)
        
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