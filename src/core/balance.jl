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
L_TB(τ) = (log(Z(s_0)) + Σ_{i=0}^{T-1} log(P_F(s_{i+1}|s_i)) - log(R(s_T)))²
"""
function _standard_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64
    initial_state = trajectory.states[1]
    terminal_state = trajectory.states[end]

    # Compute log initial flow: log(Z(s_0))
    initial_flow = flow(model, initial_state)
    if initial_flow <= 0
        @warn "Non-positive initial flow detected: $initial_flow. Using small positive value."
        initial_flow = 1e-8
    end
    log_initial_flow = log(initial_flow)

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

    # Compute log terminal reward: log(R(s_T))
    terminal_reward = reward(terminal_state)
    if terminal_reward <= 0
        throw(ArgumentError("Terminal reward must be positive: got $terminal_reward"))
    end
    log_terminal_reward = log(terminal_reward)

    # Trajectory balance equation: log(Z) + Σ log(P_F) - log(R) = 0
    balance_error = log_initial_flow + log_forward_prob_sum - log_terminal_reward

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
- States must be connected: target_state ∈ get_next_states(dag, source_state)
"""
function detailed_balance_loss(model::GFlowNetModel, source_state, target_state)::Float64

    # Validate that backward policy exists
    if isnothing(model.backward_policy)
        throw(ArgumentError("Detailed balance requires backward policy"))
    end

    # Validate that states are connected
    next_states = get_next_states(model.dag, source_state)
    if target_state ∉ next_states
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

    # Compute flows: F(s) and F(s')
    source_flow = flow(model, source_state)
    target_flow = flow(model, target_state)

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

    # Get direct flow estimate: Z(s)
    estimated_flow = compute_flow_estimate(model, state)

    # Compute recursive flow: Σ_{s'} P_F(s'|s) * F(s')
    next_states = get_next_states(model.dag, state)

    if isempty(next_states)
        # Non-terminal state with no outgoing edges - should have zero flow
        return estimated_flow^2
    end

    recursive_flow = 0.0
    for next_state in next_states
        transition_prob = forward_transition_probability(model, state, next_state)
        next_flow = flow(model, next_state; method=RECURSIVE_FLOW)  # Use recursive to avoid circular dependency
        recursive_flow += transition_prob * next_flow
    end

    # Flow matching loss: (Z(s) - Σ P_F(s'|s) * F(s'))²
    flow_error = estimated_flow - recursive_flow

    return flow_error^2
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

    # Detailed Balance validation (if backward policy exists)
    if !isnothing(model.backward_policy)
        try
            # Sample random connected state pairs
            state_pairs = Tuple{eltype(model.dag.states), eltype(model.dag.states)}[]
            attempts = 0
            while length(state_pairs) < n_state_pairs && attempts < n_state_pairs * 10
                source = rand(model.dag.states)
                next_states = get_next_states(model.dag, source)
                if !isempty(next_states)
                    target = rand(next_states)
                    push!(state_pairs, (source, target))
                end
                attempts += 1
            end

            if !isempty(state_pairs)
                db_losses = [detailed_balance_loss(model, src, tgt) for (src, tgt) in state_pairs]
                results["detailed_balance"] = (
                    mean_loss = mean(db_losses),
                    std_loss = std(db_losses),
                    max_loss = maximum(db_losses),
                    min_loss = minimum(db_losses),
                    n_samples = length(db_losses)
                )
            else
                results["detailed_balance"] = nothing
            end
        catch e
            @warn "Detailed balance validation failed: $e"
            results["detailed_balance"] = nothing
        end
    else
        results["detailed_balance"] = nothing
    end

    # Flow Matching validation (if flow estimator exists)
    if !isnothing(model.flow_estimator)
        try
            # Sample random non-terminal states
            non_terminal_states = [s for s in model.dag.states if !is_terminal_state(s)]
            if !isempty(non_terminal_states)
                sample_states = rand(non_terminal_states, min(n_states, length(non_terminal_states)))
                fm_losses = [flow_matching_loss(model, state) for state in sample_states]
                results["flow_matching"] = (
                    mean_loss = mean(fm_losses),
                    std_loss = std(fm_losses),
                    max_loss = maximum(fm_losses),
                    min_loss = minimum(fm_losses),
                    n_samples = length(fm_losses)
                )
            else
                results["flow_matching"] = nothing
            end
        catch e
            @warn "Flow matching validation failed: $e"
            results["flow_matching"] = nothing
        end
    else
        results["flow_matching"] = nothing
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
