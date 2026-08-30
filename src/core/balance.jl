# Balance Conditions - Core GFlowNet Mathematical Equations
# Implementation of Trajectory Balance, Detailed Balance, and Flow Matching

using Zygote
using Statistics: mean

# =============================================================================
# Balance Condition Types and Enumerations
# =============================================================================

"""
    BalanceCondition

Enumeration of GFlowNet balance conditions.

# Mathematical Foundation
Different balance conditions define different ways to enforce flow conservation:
- Trajectory Balance (TB): ∏P_F(s'|s) * Z(s₀) = R(sₜ)
- Detailed Balance (DB): P_F(s'|s) * F(s) = P_B(s|s') * F(s')
- Flow Matching (FM): F(s) = Σ_{s'} P_F(s'|s) * F(s')
"""
@enum BalanceCondition begin
    TRAJECTORY_BALANCE_CONDITION
    DETAILED_BALANCE_CONDITION
    FLOW_MATCHING_CONDITION
end

"""
    TrajectoryBalanceVariant

Variants of trajectory balance formulation.

# Mathematical Foundation
Different ways to formulate the trajectory balance equation:
- `:standard`: Log ratio between forward and backward probabilities
- `:geometric_mean`: Geometric mean of forward trajectory probability and flow
- `:arithmetic_mean`: Arithmetic mean formulation (less common)
"""
@enum TrajectoryBalanceVariant begin
    STANDARD_TB
    GEOMETRIC_MEAN_TB
    ARITHMETIC_MEAN_TB
end

# =============================================================================
# Sub-Trajectory Balance - Mathematical Foundation
# =============================================================================

# Import necessary functions for on-demand computation
using ..GFlowNet: state_to_features, get_applicable_actions, apply_action, is_terminal_state

"""
    logsumexp_stb(x)

Numerically stable log-sum-exp for SubTB computation.
"""
function logsumexp_stb(x::AbstractVector)
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
    sub_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory, params;
                               sub_length::Int=5)::Float64

Compute sub-trajectory balance loss for a trajectory by considering all sub-trajectories.

# Mathematical Foundation
Sub-Trajectory Balance enforces flow conservation on partial trajectories:

For a sub-trajectory from state s_i to s_j:
∏_{k=i}^{j-1} P_F(s_{k+1}|s_k) * F(s_i) = F(s_j)

This provides more frequent learning signals compared to full trajectory balance.

# Arguments
- `model::GFlowNetModel`: Complete GFlowNet model with flow_estimator (REQUIRED)
- `trajectory::Trajectory`: Full trajectory to extract sub-trajectories from
- `params`: Model parameters for differentiable computation
- `sub_length::Int`: Maximum length of sub-trajectories to consider

# Returns
- `Float64`: Average sub-trajectory balance loss

# Requirements
- Model must have flow_estimator: !isnothing(model.flow_estimator)

# Mathematical Properties
- Provides local credit assignment
- More stable gradients than full trajectory balance
- Reduces variance in long trajectories
- Domain-agnostic: works with any state/action types implementing the GFlowNet interface
"""
function sub_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory, params;
                                   sub_length::Int=5)::Float64
    # Validate that flow estimator exists (REQUIRED for SubTB)
    if isnothing(model.flow_estimator)
        throw(ArgumentError(
            "SUB_TRAJECTORY_BALANCE requires a flow estimator. " *
            "Create model with: include_flow_estimator=true"
        ))
    end

    # A LEARNABLE partition function is equally REQUIRED, and for a mathematical
    # reason rather than a convenience one.
    #
    # SubTB anchors the flow at both ends: F(s_0) = Z and F(x) = R(x). Under
    # SIMPLE_ESTIMATION, Z is pinned to 1 (losses.jl records this), so the root
    # anchor asserts F(s_0) = 1 while the true Z on, say, the 3x3 grid is 19. The
    # objective is then UNSATISFIABLE, and the optimiser resolves the contradiction
    # by collapsing.
    #
    # Measured on the 3x3 grid, three unrelated seeds, 400 iterations:
    #   SIMPLE_ESTIMATION    loss froze at 226.214 for ALL THREE, sampling one
    #                        terminal state with probability 1.000, TV 0.9474
    #   LEARNABLE_ESTIMATION loss reached 0.000 and TV 0.0144 on one seed
    # The 226.214 plateau is identical to six significant figures across seeds, so it
    # is a fixed point, not slow convergence. Failing loudly beats training to a
    # confidently wrong sampler.
    if !haskey(params, :log_Z)
        throw(ArgumentError(
            "SUB_TRAJECTORY_BALANCE requires a learnable partition function. " *
            "Create the model with partition_function_method = LEARNABLE_ESTIMATION. " *
            "Under SIMPLE_ESTIMATION Z is fixed at 1, which contradicts the " *
            "F(s_0) = Z boundary condition and collapses the sampler."
        ))
    end

    n_states = length(trajectory.states)
    if n_states < 2
        return 0.0
    end

    # Pre-compute valid sub-trajectory indices OUTSIDE gradient (non-differentiable)
    # This avoids mutations inside the gradient computation
    valid_pairs = Zygote.@ignore begin
        pairs = Tuple{Int,Int}[]
        for start_idx in 1:n_states-1
            for end_idx in start_idx+1:min(start_idx+sub_length, n_states)
                if end_idx - start_idx >= 1  # At least 2 states
                    push!(pairs, (start_idx, end_idx))
                end
            end
        end

        # ALWAYS include the FULL trajectory (1, n_states), even when it is longer
        # than sub_length. This is what pins log Z, and omitting it is why SubTB
        # diverged.
        #
        # Only a sub-trajectory spanning s_0 -> terminal relates log F(s_0) = log Z to
        # log R(x); that pair IS the Trajectory Balance constraint, and SubTB(lambda)
        # in Madan et al. 2023 contains it, weighted lambda^(n-1). The loop above
        # caps end_idx at start_idx + sub_length, so for a 5-state trajectory with
        # sub_length = 3 it generates (1,2) (1,3) (1,4) ... and never (1,5). log Z
        # then appears only in residuals whose other endpoint is a FREE network
        # output, so nothing ties it to sum_x R(x) and the optimiser is free to run
        # it anywhere.
        #
        # Measured, seed 7, LEARNABLE_ESTIMATION: log_Z reached -8.517 against a true
        # log Z of +2.9444 -- off by 11.5 nats in the wrong direction -- while
        # max|log F| grew to 6.785, the policy saturated to an exact 50/50 split, the
        # gradient decayed to 2e-4 and the loss froze at 169.661. Three unrelated
        # seeds froze at the same value to six significant figures, which is a fixed
        # point, not slow convergence.
        if n_states >= 2 && !((1, n_states) in pairs)
            push!(pairs, (1, n_states))
        end

        pairs
    end

    n_pairs = Zygote.@ignore length(valid_pairs)
    if n_pairs == 0
        return 0.0
    end

    # Compute sum of losses directly (Zygote-safe)
    # Replace Inf/NaN with 0 using a conditional
    total_loss = sum(
        begin
            loss = _compute_sub_trajectory_loss_differentiable(
                model,
                trajectory.states[start_idx:end_idx],
                trajectory,
                start_idx,
                params
            )
            # Replace invalid losses with 0 (Zygote-safe conditional)
            ifelse(isnan(loss) || isinf(loss), 0.0, loss)
        end
        for (start_idx, end_idx) in valid_pairs
    )

    # Count valid losses outside gradient
    n_valid = Zygote.@ignore begin
        count = 0
        for (start_idx, end_idx) in valid_pairs
            loss = _compute_sub_trajectory_loss_differentiable(model, trajectory.states[start_idx:end_idx], trajectory, start_idx, params)
            if !isnan(loss) && !isinf(loss)
                count += 1
            end
        end
        count
    end

    return n_valid > 0 ? total_loss / n_valid : 0.0
end

# Legacy version for backward compatibility (non-differentiable)
function sub_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory;
                                   sub_length::Int=5)::Float64
    @warn "Using legacy non-differentiable SubTB. For training, use sub_trajectory_balance_loss(model, traj, params; ...)"

    if length(trajectory.states) < 2
        return 0.0
    end

    losses = Float64[]

    for start_idx in 1:length(trajectory.states)-1
        for end_idx in start_idx+1:min(start_idx+sub_length, length(trajectory.states))
            sub_states = trajectory.states[start_idx:end_idx]

            if length(sub_states) < 2
                continue
            end

            loss = _compute_sub_trajectory_loss_legacy(model, sub_states, trajectory, start_idx)
            if !isnan(loss) && !isinf(loss)
                push!(losses, loss)
            end
        end
    end

    return isempty(losses) ? 0.0 : mean(losses)
end

"""
    _compute_sub_trajectory_loss_differentiable(model, sub_states, full_trajectory, start_idx, params)

Compute DIFFERENTIABLE loss for a single sub-trajectory.
Uses on-demand forward probability computation and differentiable flow estimator.
"""
function _compute_sub_trajectory_loss_differentiable(
    model::GFlowNetModel,
    sub_states::Vector{<:AbstractState},
    full_trajectory::Trajectory,
    start_idx::Int,
    params
)::Float64

    # Accumulate log P_F and log P_B over the SAME transitions. The published
    # SubTB (Madan et al. 2023) is
    #     (log F(s_i) + sum log P_F - log F(s_j) - sum log P_B)^2
    # The backward term was absent, which is why the backward parameters had a
    # gradient norm of exactly 0 under this objective: P_B appeared nowhere in
    # the expression, so it could never be trained.
    log_forward_prob = 0.0
    log_backward_prob = 0.0

    for i in 1:length(sub_states)-1
        source_state = sub_states[i]
        target_state = sub_states[i+1]

        # Get state features and compute logits (DIFFERENTIABLE)
        features = state_to_features(source_state)
        logits, _ = model.forward_policy.model(features, params.forward, model.states.forward)

        # Get applicable actions (discrete logic - non-differentiable)
        applicable_actions = Zygote.@ignore get_applicable_actions(source_state, model.all_actions)
        applicable_indices = Zygote.@ignore [idx for (idx, a) in enumerate(model.all_actions) if a in applicable_actions]

        if isempty(applicable_indices)
            return Inf
        end

        # Find which action leads to target_state (discrete logic - non-differentiable)
        target_action_idx = Zygote.@ignore begin
            for (idx, action) in enumerate(model.all_actions)
                if action in applicable_actions && apply_action(action, source_state) == target_state
                    return idx
                end
            end
            return nothing
        end

        if isnothing(target_action_idx)
            return Inf
        end

        # Compute softmax log-probabilities (DIFFERENTIABLE)
        applicable_logits = logits[applicable_indices]
        log_probs = applicable_logits .- logsumexp_stb(applicable_logits)

        # Find position of target action in applicable actions
        action_pos = Zygote.@ignore findfirst(==(target_action_idx), applicable_indices)

        if isnothing(action_pos)
            return Inf
        end

        log_forward_prob += log_probs[action_pos]

        # P_B(source | target) for this same transition. Same repair as the TB path:
        # falling through with log P_B = 0 means P_B == 1 unnormalised, valid only if
        # every state has a unique parent. Uniform over parents otherwise.
        if isnothing(model.backward_policy)
            parents = backward_parent_states(target_state, model.all_actions)
            isempty(parents) || (log_backward_prob += -log(length(parents)))
        else
            log_backward_prob += log(max(
                compute_backward_probability(
                    model.backward_policy, target_state, source_state,
                    params.backward, model.states.backward, model.all_actions
                ), 1e-8))
        end
    end

    # Compute flow estimates using flow estimator (DIFFERENTIABLE!)
    start_features = state_to_features(sub_states[1])
    end_features = state_to_features(sub_states[end])

    # Reshape features for Lux (expects [features, batch])
    start_features_mat = reshape(convert(Array{Float32}, start_features), :, 1)
    end_features_mat = reshape(convert(Array{Float32}, end_features), :, 1)

    start_flow_vec, _ = model.flow_estimator.model(start_features_mat, params.flow, model.states.flow)
    end_flow_vec, _ = model.flow_estimator.model(end_features_mat, params.flow, model.states.flow)

    # TERMINAL BOUNDARY CONDITION: F(x) = R(x) for terminal x.
    #
    # This was missing, and it is why reward never entered the SubTB objective at
    # all: the raw flow-network output was used even when the endpoint was
    # terminal. Measured proof -- scaling R(3,3) from 10 to 1000 left the loss
    # bit-identical at 9.990423551228066, so a 100x reward change moved nothing
    # and any constant flow network minimised the objective.
    # BOTH boundary conditions, F(s_0) = Z and F(x) = R(x).
    #
    # The initial-state anchor was missing entirely: `log_Z` appeared exactly once
    # in this file, inside a comment. Its absence makes a COLLAPSED SAMPLER a global
    # optimum with loss exactly 0. Take a deterministic policy that always reaches a
    # single terminal x, with P_B = 1, and a flow network that is constant at R(x):
    # every non-terminal sub-trajectory has residual log F(s_i) - log F(s_j) = 0, and
    # the only terminal-anchored pair needs log F = log R(x), which holds. Nothing in
    # the objective penalises ignoring every other terminal state.
    #
    # Measured before adding this: SubTB collapsed to ONE terminal with probability
    # 1.000 and TV(p_hat, R/Z) = 0.9474, which is 1 - 1/19, i.e. it settled on a
    # reward-1.0 corner rather than the reward-10 one. Identical to four decimal
    # places at learning rates 0.005 and 0.0005, clip norms 1.0 and 0.1, and
    # sub-trajectory lengths 5 and 2 -- so it was structural, not an optimisation
    # scale problem.
    #
    # Anchoring F(s_0) = Z breaks that optimum, because a constant flow now also has
    # to equal Z at the root while equalling R(x) at the leaf. It also gives SubTB a
    # log_Z gradient, which the component-gradient table previously reported as dead.
    log_start_flow = if Zygote.@ignore(is_terminal_state(sub_states[1]))
        log(max(Zygote.@ignore(reward(sub_states[1])), 1e-8))
    elseif start_idx == 1 && haskey(params, :log_Z)
        params.log_Z
    else
        log(max(start_flow_vec[1], 1e-8))
    end
    log_end_flow = if Zygote.@ignore(is_terminal_state(sub_states[end]))
        log(max(Zygote.@ignore(reward(sub_states[end])), 1e-8))
    else
        log(max(end_flow_vec[1], 1e-8))
    end

    # SubTB loss: (log F(s_i) + sum log P_F - log F(s_j) - sum log P_B)^2
    error = log_start_flow + log_forward_prob - log_end_flow - log_backward_prob
    return error^2
end

"""
    _compute_sub_trajectory_loss_legacy(model, sub_states, full_trajectory, start_idx)

Legacy non-differentiable version for backward compatibility.
"""
function _compute_sub_trajectory_loss_legacy(model::GFlowNetModel, sub_states::Vector{<:AbstractState},
                                    full_trajectory::Trajectory, start_idx::Int)::Float64

    log_forward_prob = 0.0

    for i in 1:length(sub_states)-1
        source_state = sub_states[i]
        target_state = sub_states[i+1]

        action_idx = start_idx + i - 1
        if action_idx <= length(full_trajectory.actions)
            prob = forward_transition_probability(model, source_state, target_state)
            if prob <= 0
                return Inf
            end
            log_forward_prob += log(prob)
        end
    end

    start_flow = Zygote.@ignore flow(model, sub_states[1])
    end_flow = Zygote.@ignore flow(model, sub_states[end])

    if start_flow <= 0 || end_flow <= 0
        return 0.0
    end

    log_start_flow = log(start_flow)
    log_end_flow = log(end_flow)

    error = log_forward_prob + log_start_flow - log_end_flow
    return error^2
end

"""
    sub_trajectory_balance_loss_batch(model::GFlowNetModel, trajectories::Vector{Trajectory}, params;
                                     sub_length::Int=5)::Float64

Compute average sub-trajectory balance loss over a batch of trajectories.
DIFFERENTIABLE version that requires params.
"""
function sub_trajectory_balance_loss_batch(model::GFlowNetModel, trajectories::Vector{Trajectory}, params;
                                         sub_length::Int=5)::Float64
    if isempty(trajectories)
        return 0.0
    end

    # Validate flow estimator
    if isnothing(model.flow_estimator)
        throw(ArgumentError("SUB_TRAJECTORY_BALANCE requires include_flow_estimator=true"))
    end

    # Compute losses using comprehension (Zygote-safe)
    losses = [sub_trajectory_balance_loss(model, traj, params; sub_length=sub_length) for traj in trajectories]

    # Filter NaN/Inf outside gradient computation
    valid_indices = Zygote.@ignore [i for (i, l) in enumerate(losses) if !isnan(l) && !isinf(l)]

    if isempty(valid_indices)
        return 0.0
    end

    # Use indices to compute mean (Zygote-safe)
    return sum(losses[i] for i in valid_indices) / length(valid_indices)
end

# Legacy batch version for backward compatibility
function sub_trajectory_balance_loss_batch(model::GFlowNetModel, trajectories::Vector{Trajectory};
                                         sub_length::Int=5)::Float64
    
    if isempty(trajectories)
        return 0.0
    end
    
    losses = [sub_trajectory_balance_loss(model, traj; sub_length=sub_length) for traj in trajectories]
    valid_losses = filter(!isnan, losses)
    
    return isempty(valid_losses) ? 0.0 : mean(valid_losses)
end

# =============================================================================
# Trajectory Balance - Mathematical Foundation
# =============================================================================

"""
    trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory;
                           variant::TrajectoryBalanceVariant=STANDARD_TB)::Float64

Compute trajectory balance loss for a single trajectory.

# Mathematical Foundation
Trajectory Balance enforces that the product of forward probabilities
times the initial flow equals the terminal reward:

∏_{i=0}^{T-1} P_F(s_{i+1}|s_i) * Z(s_0) = R(s_T)

Taking logarithms for numerical stability:
log(Z(s_0)) + Σ_{i=0}^{T-1} log(P_F(s_{i+1}|s_i)) = log(R(s_T))

The loss is the squared difference:
L_TB(τ) = (log(Z(s_0)) + Σ log(P_F) - log(R(s_T)))²

# Arguments
- `model::GFlowNetModel`: Complete GFlowNet model
- `trajectory::Trajectory`: Trajectory τ = (s_0, a_0, s_1, ..., s_T)
- `variant::TrajectoryBalanceVariant`: TB formulation variant

# Returns
- `Float64`: Trajectory balance loss L_TB(τ)

# Mathematical Properties
- Non-negative: L_TB(τ) ≥ 0
- Zero when balance condition is perfectly satisfied
- Differentiable w.r.t. model parameters
"""
function trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory;
                                variant::TrajectoryBalanceVariant=STANDARD_TB)::Float64

    if isempty(trajectory.states)
        throw(ArgumentError("Cannot compute trajectory balance for empty trajectory"))
    end

    if length(trajectory.states) < 2
        throw(ArgumentError("Trajectory must have at least 2 states for balance computation"))
    end

    initial_state = trajectory.states[1]
    terminal_state = trajectory.states[end]

    # Validate terminal state
    if !is_terminal_state(terminal_state)
        throw(ArgumentError("Trajectory must end in terminal state for trajectory balance"))
    end

    if variant == STANDARD_TB
        return _standard_trajectory_balance_loss(model, trajectory)
    elseif variant == GEOMETRIC_MEAN_TB
        return _geometric_mean_trajectory_balance_loss(model, trajectory)
    elseif variant == ARITHMETIC_MEAN_TB
        return _arithmetic_mean_trajectory_balance_loss(model, trajectory)
    else
        throw(ArgumentError("Unknown trajectory balance variant: $variant"))
    end
end

"""
    _standard_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64

Standard trajectory balance formulation.

# Mathematical Foundation
General form: L_TB(τ) = (log(Z(s_0)) + Σ log(P_F(s_{i+1}|s_i)) - log(R(s_T)) - Σ log(P_B(s_i|s_{i+1})))²

When backward policy is deterministic or not available, P_B terms = 1 (log P_B = 0)
"""
function _standard_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64
    initial_state = trajectory.states[1]
    terminal_state = trajectory.states[end]

    # Compute log(Z) based on partition function method
    log_initial_flow = if isnothing(model.log_partition_function)
        # SIMPLE_ESTIMATION: Z(s_0) = 1, so log(Z) = 0
        # This is mathematically valid when the initial state is fixed
        0.0
    else
        # LEARNABLE_ESTIMATION: Use learnable parameter from parameters structure
        # The actual optimization happens on model.parameters.log_Z, but we use the model field for consistency
        model.log_partition_function
    end

    # Compute sum of log forward probabilities: Σ log(P_F(s_{i+1}|s_i))
    log_forward_prob_sum = 0.0

    for i in 1:(length(trajectory.states)-1)
        source_state = trajectory.states[i]
        target_state = trajectory.states[i+1]

        # Get transition probability
        transition_prob = forward_transition_probability(model, source_state, target_state)

        if transition_prob <= 0
            @warn "Non-positive transition probability: $transition_prob from $source_state to $target_state"
            transition_prob = 1e-8
        end

        log_forward_prob_sum += log(transition_prob)
    end

    # Compute sum of log backward probabilities: Σ log(P_B(s_i|s_{i+1}))
    #
    # With NO backward policy this used to leave the sum at 0.0, i.e. P_B == 1
    # unnormalised. The comment justifying it said "deterministic backward (following
    # unique parent)" -- and that precondition is exactly what fails: it holds only
    # when every state has ONE parent. Grid world's (2,2) has two, so P_B == 1 is not
    # a distribution and the residual's zero moves.
    #
    # Consequence, measured: TB converges to Z = sum_x n_paths(x) R(x) instead of
    # sum_x R(x) -- 77.928 against 78.0 on the 3x3 grid where Z is 19.0 -- and the
    # terminal law becomes n(x)R(x)/sum_y n(y)R(y). The sampler is biased toward
    # states reachable by more paths, which is the defect
    # test/theory/test_reward_proportionality.jl calls "the bug that was fixed" while
    # only ever exercising the analytic helpers, never this loss.
    #
    # The same file already had the right convention for DB/FM (line ~1003):
    # uniform over parents. TB now uses it too. TB is valid for ANY fixed normalised
    # P_B, and sum_tau P_B(tau|x) = 1 then restores Z = sum_x R(x).
    log_backward_prob_sum = 0.0

    for i in 1:(length(trajectory.states)-1)
        source_state = trajectory.states[i]
        target_state = trajectory.states[i+1]

        if isnothing(model.backward_policy)
            parents = backward_parent_states(target_state, model.all_actions)
            isempty(parents) && continue          # log P_B = log 1 = 0
            log_backward_prob_sum += -log(length(parents))
        else
            # Try to compute backward probability
            try
                backward_prob = backward_transition_probability(model, target_state, source_state)
                if backward_prob <= 0
                    @warn "Non-positive backward probability: $backward_prob from $target_state to $source_state"
                    backward_prob = 1e-8
                end
                log_backward_prob_sum += log(backward_prob)
            catch e
                # If backward probability can't be computed, assume deterministic (log P_B = 0)
                @debug "Could not compute backward probability, assuming deterministic: $e"
            end
        end
    end

    # Compute log terminal reward: log(R(s_T))
    terminal_reward = reward(terminal_state)
    if terminal_reward <= 0
        throw(ArgumentError("Terminal reward must be positive: got $terminal_reward"))
    end
    log_terminal_reward = log(terminal_reward)

    # General trajectory balance equation: 
    # log(Z) + Σ log(P_F) - log(R) - Σ log(P_B) = 0
    balance_error = log_initial_flow + log_forward_prob_sum - log_terminal_reward - log_backward_prob_sum

    # Return squared loss
    return balance_error^2
end

"""
    _geometric_mean_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64

Geometric mean formulation of trajectory balance.

# Mathematical Foundation
Alternative formulation using geometric mean of forward probabilities and flows.
"""
function _geometric_mean_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64
    # Compute standard TB loss but with geometric mean normalization
    standard_loss = _standard_trajectory_balance_loss(model, trajectory)
    trajectory_length = length(trajectory.states) - 1

    # Normalize by trajectory length (geometric mean effect)
    return standard_loss / max(trajectory_length, 1)
end

"""
    _arithmetic_mean_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64

Arithmetic mean formulation (less commonly used).
"""
function _arithmetic_mean_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64
    # This is a less common variant - implement if needed for research
    return _standard_trajectory_balance_loss(model, trajectory)
end

# =============================================================================
# Detailed Balance - Mathematical Foundation
# =============================================================================

"""
    detailed_balance_loss(model::GFlowNetModel, source_state, target_state)::Float64

Compute detailed balance loss for a state transition.

# Mathematical Foundation
Detailed Balance enforces that the flow through an edge is the same in both directions:

P_F(s'|s) * F(s) = P_B(s|s') * F(s')

Taking logarithms:
log(P_F(s'|s)) + log(F(s)) = log(P_B(s|s')) + log(F(s'))

The loss is the squared difference:
L_DB(s,s') = (log(P_F(s'|s)) + log(F(s)) - log(P_B(s|s')) - log(F(s')))²

# Arguments
- `model::GFlowNetModel`: Model with both forward and backward policies
- `source_state`: Source state s
- `target_state`: Target state s'

# Returns
- `Float64`: Detailed balance loss L_DB(s,s')

# Requirements
- Model must have backward policy: model.backward_policy ≠ nothing
- States must be connected: there exists an action a such that apply_action(a, source_state) = target_state
"""
function detailed_balance_loss(model::GFlowNetModel, source_state, target_state)::Float64

    # Validate that backward policy exists
    if isnothing(model.backward_policy)
        throw(ArgumentError("Detailed balance requires backward policy"))
    end

    # Validate states are connected by checking if any action can transition between them
    # This replaces the old DAG-based validation
    applicable_actions = get_applicable_actions(source_state, model.all_actions)
    can_transition = false
    for action in applicable_actions
        if apply_action(action, source_state) == target_state
            can_transition = true
            break
        end
    end
    
    if !can_transition
        throw(ArgumentError("Target state $target_state not reachable from source state $source_state"))
    end

    # Compute forward transition probability: P_F(s'|s)
    forward_prob = forward_transition_probability(model, source_state, target_state)
    if forward_prob <= 0
        @warn "Non-positive forward probability: $forward_prob"
        forward_prob = 1e-8
    end

    # Compute backward transition probability: P_B(s|s')
    backward_prob = backward_transition_probability(model, target_state, source_state)
    if backward_prob <= 0
        @warn "Non-positive backward probability: $backward_prob"
        backward_prob = 1e-8
    end

    # Compute flows using our flow computation functions
    source_flow = flow(model, source_state)
    target_flow = flow(model, target_state)
    
    # Ensure flows are positive
    if source_flow <= 0
        @warn "Non-positive source flow: $source_flow"
        source_flow = 1e-8
    end
    if target_flow <= 0
        @warn "Non-positive target flow: $target_flow"
        target_flow = 1e-8
    end

    # Detailed balance equation in log space:
    # log(P_F(s'|s)) + log(F(s)) = log(P_B(s|s')) + log(F(s'))
    left_side = log(forward_prob) + log(source_flow)
    right_side = log(backward_prob) + log(target_flow)

    balance_error = left_side - right_side

    return balance_error^2
end

"""
    detailed_balance_loss_batch(model::GFlowNetModel, state_pairs)::Float64

Compute detailed balance loss over a batch of state pairs.

# Arguments
- `model::GFlowNetModel`: Model with backward policy
- `state_pairs::Vector{Tuple{S,S}}`: Vector of (source, target) state pairs

# Returns
- `Float64`: Average detailed balance loss over all pairs
"""
function detailed_balance_loss_batch(model::GFlowNetModel, state_pairs)::Float64
    if isempty(state_pairs)
        return 0.0
    end

    total_loss = 0.0
    valid_pairs = 0

    for (source, target) in state_pairs
        try
            loss = detailed_balance_loss(model, source, target)
            total_loss += loss
            valid_pairs += 1
        catch e
            @warn "Failed to compute detailed balance for pair ($source, $target): $e"
        end
    end

    return valid_pairs > 0 ? total_loss / valid_pairs : 0.0
end

# =============================================================================
# Flow Matching - Mathematical Foundation
# =============================================================================

"""
    flow_matching_loss(model::GFlowNetModel, state)::Float64

Compute flow matching loss for a single state.

# Mathematical Foundation
Flow Matching enforces the flow conservation equation directly:

F(s) = Σ_{s'∈children(s)} P_F(s'|s) * F(s')

The loss is the squared difference between estimated flow and computed flow:
L_FM(s) = (Z(s) - Σ_{s'} P_F(s'|s) * F(s'))²

where Z(s) is the direct flow estimate and the sum represents recursive flow.

# Arguments
- `model::GFlowNetModel`: Model with flow estimator
- `state`: State to compute flow matching for

# Returns
- `Float64`: Flow matching loss L_FM(s)

# Requirements
- Model must have flow estimator: model.flow_estimator ≠ nothing
- State should not be terminal (terminal states satisfy conservation trivially)
"""
function flow_matching_loss(model::GFlowNetModel, state)::Float64

    # Validate that flow estimator exists
    if isnothing(model.flow_estimator)
        throw(ArgumentError("Flow matching requires flow estimator"))
    end

    # Skip terminal states (they satisfy conservation by definition)
    if is_terminal_state(state)
        return 0.0
    end

    # Get the flow estimate from the neural network Z(s)
    features = state_to_features(state)
    estimated_flow = flow_estimate(
        model.flow_estimator, state,
        model.parameters.flow, model.states.flow
    )
    
    # Compute the true flow using recursive computation
    # F(s) = Σ_{s'} P_F(s'|s) * F(s')
    applicable_actions = get_applicable_actions(state, model.all_actions)
    
    if isempty(applicable_actions)
        # No outgoing transitions, flow should be 0
        return (estimated_flow - 0.0)^2
    end
    
    # Compute expected flow by summing over all possible next states
    expected_flow = 0.0
    
    # Get forward policy probabilities for all actions
    action_probs = forward_action_probabilities(
        model.forward_policy, state, model.all_actions,
        model.parameters.forward, model.states.forward
    )
    
    # Sum over all applicable actions
    for (action_idx, action) in enumerate(model.all_actions)
        if action in applicable_actions
            # Get next state
            next_state = apply_action(action, state)
            
            # Get transition probability P_F(s'|s)
            transition_prob = action_probs[action_idx]
            
            # Get flow of next state F(s')
            next_flow = flow(model, next_state)
            
            # Add contribution to expected flow
            expected_flow += transition_prob * next_flow
        end
    end
    
    # Flow matching loss: (Z(s) - F(s))²
    return (estimated_flow - expected_flow)^2
end

"""
    flow_matching_loss_batch(model::GFlowNetModel, states)::Float64

Compute flow matching loss over a batch of states.

# Arguments
- `model::GFlowNetModel`: Model with flow estimator
- `states::Vector{S}`: Vector of states to compute flow matching for

# Returns
- `Float64`: Average flow matching loss over all states
"""
function flow_matching_loss_batch(model::GFlowNetModel, states)::Float64
    if isempty(states)
        return 0.0
    end

    total_loss = 0.0
    valid_states = 0

    for state in states
        # Skip terminal states
        if is_terminal_state(state)
            continue
        end

        try
            loss = flow_matching_loss(model, state)
            total_loss += loss
            valid_states += 1
        catch e
            @warn "Failed to compute flow matching for state $state: $e"
        end
    end

    return valid_states > 0 ? total_loss / valid_states : 0.0
end

# =============================================================================
# Unified Balance Loss Interface
# =============================================================================

"""
    compute_balance_loss(condition::BalanceCondition, model::GFlowNetModel, data)::Float64

Unified interface for computing different balance condition losses.

# Arguments
- `condition::BalanceCondition`: Which balance condition to use
- `model::GFlowNetModel`: Complete GFlowNet model
- `data`: Condition-specific data (trajectories, state pairs, or states)

# Returns
- `Float64`: Computed loss for the specified balance condition

# Data Requirements
- TRAJECTORY_BALANCE: `data` should be Vector{Trajectory} or single Trajectory
- DETAILED_BALANCE: `data` should be Vector{Tuple{State,State}} or single tuple
- FLOW_MATCHING: `data` should be Vector{State} or single State
"""
function compute_balance_loss(condition::BalanceCondition, model::GFlowNetModel, data)::Float64

    if condition == TRAJECTORY_BALANCE_CONDITION
        if data isa Trajectory
            return trajectory_balance_loss(model, data)
        elseif data isa Vector{<:Trajectory}
            return mean([trajectory_balance_loss(model, traj) for traj in data])
        else
            throw(ArgumentError("Trajectory balance requires Trajectory or Vector{Trajectory} data"))
        end

    elseif condition == DETAILED_BALANCE_CONDITION
        if data isa Tuple{<:AbstractState, <:AbstractState}
            return detailed_balance_loss(model, data[1], data[2])
        elseif data isa Vector{<:Tuple{<:AbstractState, <:AbstractState}}
            return mean([detailed_balance_loss(model, pair[1], pair[2]) for pair in data])
        else
            throw(ArgumentError("Detailed balance requires (state, state) tuple or vector of tuples"))
        end

    elseif condition == FLOW_MATCHING_CONDITION
        if data isa AbstractState
            return flow_matching_loss(model, data)
        elseif data isa Vector{<:AbstractState}
            return mean([flow_matching_loss(model, state) for state in data])
        else
            throw(ArgumentError("Flow matching requires AbstractState or Vector{AbstractState} data"))
        end

    else
        throw(ArgumentError("Unknown balance condition: $condition"))
    end
end

# =============================================================================
# Balance Validation and Analysis
# =============================================================================

"""
    validate_balance_conditions(model::GFlowNetModel;
                               n_trajectories::Int=10,
                               n_state_pairs::Int=20,
                               n_states::Int=20)

Comprehensive validation of all balance conditions.

# Arguments
- `model::GFlowNetModel`: Model to validate
- `n_trajectories::Int`: Number of trajectories to sample for TB validation
- `n_state_pairs::Int`: Number of state pairs for DB validation
- `n_states::Int`: Number of states for FM validation

# Returns
Named tuple with validation results for each balance condition.
"""
function validate_balance_conditions(model::GFlowNetModel;
                                   n_trajectories::Int=10,
                                   n_state_pairs::Int=20,
                                   n_states::Int=20)

    results = Dict{String, Any}()

    # Trajectory Balance validation
    try
        trajectories = [sample_trajectory(model) for _ in 1:n_trajectories]
        tb_losses = [trajectory_balance_loss(model, traj) for traj in trajectories]
        results["trajectory_balance"] = (
            mean_loss = mean(tb_losses),
            std_loss = std(tb_losses),
            max_loss = maximum(tb_losses),
            min_loss = minimum(tb_losses),
            n_samples = length(tb_losses)
        )
    catch e
        @warn "Trajectory balance validation failed: $e"
        results["trajectory_balance"] = nothing
    end

    # Detailed Balance: measure the residual of
    #     log P_F(s'|s) + log F(s)  ==  log P_B(s|s') + log F(s')
    # over sampled transitions. This returned `nothing` behind a TODO, so no
    # caller could ever detect a DB violation.
    results["detailed_balance"] = try
        residuals = Float64[]
        for _ in 1:n_state_pairs
            traj = sample_trajectory(model)
            is_valid_trajectory(traj) || continue
            for i in 1:length(traj.states)-1
                s, sp = traj.states[i], traj.states[i+1]
                probs = forward_action_probabilities(model.forward_policy, s,
                            model.all_actions, model.parameters.forward, model.states.forward)
                applicable = get_applicable_actions(s, model.all_actions)
                idx = findfirst(a -> a in applicable && apply_action(a, s) == sp,
                                model.all_actions)
                isnothing(idx) && continue
                pb = if isnothing(model.backward_policy)
                    parents = backward_parent_states(sp, model.all_actions)
                    isempty(parents) ? 1.0 : 1.0 / length(parents)
                else
                    compute_backward_probability(model.backward_policy, sp, s,
                        model.parameters.backward, model.states.backward, model.all_actions)
                end
                fs  = is_terminal_state(s)  ? reward(s)  : flow(model, s)
                fsp = is_terminal_state(sp) ? reward(sp) : flow(model, sp)
                push!(residuals, abs((log(max(probs[idx],1e-12)) + log(max(fs,1e-12))) -
                                     (log(max(pb,1e-12))        + log(max(fsp,1e-12)))))
            end
        end
        isempty(residuals) ? nothing :
            (mean_residual = mean(residuals), max_residual = maximum(residuals),
             n_samples = length(residuals))
    catch e
        @warn "Detailed balance validation failed: $e"
        nothing
    end

    # Flow Matching: residual of the conservation law
    #     sum over parents p of F(p) P_F(s|p)  ==  F(s)
    results["flow_matching"] = try
        residuals = Float64[]
        seen = Set{Any}()
        for _ in 1:n_states
            traj = sample_trajectory(model)
            is_valid_trajectory(traj) || continue
            for s in traj.states
                s in seen && continue
                push!(seen, s)
                parents = backward_parent_states(s, model.all_actions)
                isempty(parents) && continue
                inflow = 0.0
                for p in parents
                    probs = forward_action_probabilities(model.forward_policy, p,
                                model.all_actions, model.parameters.forward, model.states.forward)
                    applicable = get_applicable_actions(p, model.all_actions)
                    idx = findfirst(a -> a in applicable && apply_action(a, p) == s,
                                    model.all_actions)
                    isnothing(idx) && continue
                    fp = is_terminal_state(p) ? reward(p) : flow(model, p)
                    inflow += fp * probs[idx]
                end
                fs = is_terminal_state(s) ? reward(s) : flow(model, s)
                push!(residuals, abs(log(max(inflow,1e-12)) - log(max(fs,1e-12))))
            end
        end
        isempty(residuals) ? nothing :
            (mean_residual = mean(residuals), max_residual = maximum(residuals),
             n_samples = length(residuals))
    catch e
        @warn "Flow matching validation failed: $e"
        nothing
    end

    return NamedTuple(Symbol(k) => v for (k, v) in results)
end

# =============================================================================
# Balance Condition Utilities
# =============================================================================

"""
    balance_condition_requirements(condition::BalanceCondition)::Vector{String}

Get the model requirements for a specific balance condition.

# Returns
Vector of strings describing what model components are required.
"""
function balance_condition_requirements(condition::BalanceCondition)::Vector{String}
    if condition == TRAJECTORY_BALANCE_CONDITION
        return ["forward_policy", "flow_computation"]
    elseif condition == DETAILED_BALANCE_CONDITION
        return ["forward_policy", "backward_policy", "flow_computation"]
    elseif condition == FLOW_MATCHING_CONDITION
        return ["forward_policy", "flow_estimator", "flow_computation"]
    else
        return String[]
    end
end

"""
    check_balance_condition_compatibility(model::GFlowNetModel, condition::BalanceCondition)::Bool

Check if a model is compatible with a specific balance condition.

# Returns
`true` if the model has all required components for the balance condition.
"""
function check_balance_condition_compatibility(model::GFlowNetModel, condition::BalanceCondition)::Bool
    requirements = balance_condition_requirements(condition)

    for req in requirements
        if req == "forward_policy" && isnothing(model.forward_policy)
            return false
        elseif req == "backward_policy" && isnothing(model.backward_policy)
            return false
        elseif req == "flow_estimator" && isnothing(model.flow_estimator)
            return false
        end
    end

    return true
end

# =============================================================================
# Direct Flow Loss - Mathematical Foundation
# =============================================================================

"""
    compute_log_forward_probability(model::GFlowNetModel, trajectory::Trajectory)::Float64

Compute log forward probability of a trajectory.

# Mathematical Foundation
Computes log P_F(τ) = Σ_{t=0}^{T-1} log P_F(s_{t+1}|s_t)

# Arguments
- `model::GFlowNetModel`: GFlowNet model
- `trajectory::Trajectory`: Trajectory to compute probability for

# Returns
- `Float64`: Log forward probability
"""
function compute_log_forward_probability(model::GFlowNetModel, trajectory::Trajectory)::Float64
    if length(trajectory.states) < 2
        return 0.0
    end
    
    log_prob_sum = 0.0
    
    for i in 1:(length(trajectory.states)-1)
        source_state = trajectory.states[i]
        target_state = trajectory.states[i+1]
        
        # Compute transition probability
        trans_prob = forward_transition_probability(model, source_state, target_state)
        
        if trans_prob <= 0
            @warn "Non-positive transition probability: $trans_prob"
            return -Inf
        end
        
        log_prob_sum += log(trans_prob)
    end
    
    return log_prob_sum
end

"""
    direct_flow_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64

Compute direct flow loss using neural network flow estimation.

# Mathematical Foundation
Instead of computing flows recursively, train a neural network Z(s) to directly
estimate F(s). The loss ensures consistency with trajectory rewards:

L_DF(τ) = (log ∏ P_F(s_t+1|s_t) + log Z(s_0) - log R(s_T))²

This is similar to trajectory balance but uses Z(s) from the flow estimator
network instead of a learned scalar Z.

# Arguments
- `model::GFlowNetModel`: Model with flow estimator
- `trajectory::Trajectory`: Single trajectory τ = (s_0, s_1, ..., s_T)

# Returns
- `Float64`: Direct flow loss

# Requirements
- Model must have flow estimator: !isnothing(model.flow_estimator)
"""
function direct_flow_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64
    if isnothing(model.flow_estimator)
        throw(ArgumentError("Model must have flow estimator for DIRECT_FLOW_OBJECTIVE"))
    end
    
    # Compute log forward probability of trajectory
    log_pf = compute_log_forward_probability(model, trajectory)
    
    # Get initial state flow estimate from neural network
    initial_state = trajectory.states[1]
    initial_flow_estimate = Zygote.@ignore compute_flow_estimate(model, initial_state)
    log_z = log(max(initial_flow_estimate, 1e-8))
    
    # Get terminal reward
    terminal_state = trajectory.states[end]
    if !is_terminal_state(terminal_state)
        @warn "Trajectory does not end in terminal state for direct flow loss"
        return 0.0
    end
    
    reward_value = reward(terminal_state)
    if reward_value <= 0
        @warn "Non-positive reward for terminal state: $reward_value"
        reward_value = 1e-8
    end
    log_reward = log(reward_value)
    
    # Direct flow loss: (log P_F(τ) + log Z(s_0) - log R(s_T))²
    return (log_pf + log_z - log_reward)^2
end

"""
    direct_flow_loss_batch(model::GFlowNetModel, trajectories::Vector{Trajectory})::Float64

Compute average direct flow loss over a batch of trajectories.

# Arguments
- `model::GFlowNetModel`: Model with flow estimator
- `trajectories::Vector{Trajectory}`: Batch of trajectories

# Returns
- `Float64`: Average direct flow loss
"""
function direct_flow_loss_batch(model::GFlowNetModel, trajectories::Vector{Trajectory})::Float64
    if isempty(trajectories)
        return 0.0
    end
    
    total_loss = 0.0
    valid_trajectories = 0
    
    for trajectory in trajectories
        try
            loss = direct_flow_loss(model, trajectory)
            total_loss += loss
            valid_trajectories += 1
        catch e
            @debug "Failed to compute direct flow loss for trajectory: $e"
        end
    end
    
    return valid_trajectories > 0 ? total_loss / valid_trajectories : 0.0
end

# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, condition::BalanceCondition)
    condition_name = if condition == TRAJECTORY_BALANCE_CONDITION
        "Trajectory Balance"
    elseif condition == DETAILED_BALANCE_CONDITION
        "Detailed Balance"
    elseif condition == FLOW_MATCHING_CONDITION
        "Flow Matching"
    else
        "Unknown Balance Condition"
    end
    print(io, condition_name)
end

function Base.show(io::IO, variant::TrajectoryBalanceVariant)
    variant_name = if variant == STANDARD_TB
        "Standard TB"
    elseif variant == GEOMETRIC_MEAN_TB
        "Geometric Mean TB"
    elseif variant == ARITHMETIC_MEAN_TB
        "Arithmetic Mean TB"
    else
        "Unknown TB Variant"
    end
    print(io, variant_name)
end
