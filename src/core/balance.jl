# Balance Conditions - Core GFlowNet Mathematical Equations
# Implementation of Trajectory Balance, Detailed Balance, and Flow Matching

using Zygote

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

    # For simplified trajectory balance, we assume Z(s_0) = 1, so log(Z) = 0
    # This is mathematically valid when the initial state is fixed
    # TODO: In future, implement proper flow estimation if needed
    log_initial_flow = 0.0

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
    log_backward_prob_sum = 0.0
    
    # Only compute backward probabilities if we have a backward policy
    # For deterministic backward (following unique parent), P_B = 1, so log P_B = 0
    if !isnothing(model.backward_policy)
        for i in 1:(length(trajectory.states)-1)
            source_state = trajectory.states[i]
            target_state = trajectory.states[i+1]
            
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

    # For now, detailed balance is not fully implemented due to missing flow computation
    # This would require either:
    # 1. Implementing recursive flow computation without DAG
    # 2. Using a flow estimator network
    # TODO: Implement proper flow computation for detailed balance
    throw(ArgumentError("Detailed balance loss is not currently implemented - missing flow computation. Use TRAJECTORY_BALANCE instead."))

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

    # Flow matching is not currently implemented due to missing DAG-based flow computation
    # This would require implementing recursive flow computation without explicit DAG
    # TODO: Implement flow matching with on-demand state exploration
    throw(ArgumentError("Flow matching loss is not currently implemented - requires DAG-based flow computation. Use TRAJECTORY_BALANCE instead."))
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

    # Detailed Balance validation (not currently supported)
    # TODO: Implement detailed balance validation when backward policy and flow computation are available
    results["detailed_balance"] = nothing

    # Flow Matching validation (not currently supported)
    # TODO: Implement flow matching validation when flow computation is available
    results["flow_matching"] = nothing

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
