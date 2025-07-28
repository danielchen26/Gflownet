# Training Objectives and Loss Functions
# Mathematical foundations for GFlowNet training objectives and optimization

using Zygote
using Statistics: mean, std
using LinearAlgebra: norm

# =============================================================================
# Training Objective Types and Configuration
# =============================================================================

# Note: TrainingObjective enum is defined in training/configuration.jl
# We use the BalanceCondition enum for mathematical operations

"""
    ObjectiveConfig

Configuration for training objectives and loss computation.

# Fields
- `objective::TrainingObjective`: Primary training objective
- `trajectory_weight::Float64`: Weight for trajectory balance loss
- `detailed_weight::Float64`: Weight for detailed balance loss
- `flow_weight::Float64`: Weight for flow matching loss
- `regularization_weight::Float64`: Weight for regularization terms
- `entropy_weight::Float64`: Weight for policy entropy regularization
- `clip_gradients::Bool`: Whether to clip gradients for stability
- `gradient_clip_norm::Float64`: Maximum gradient norm for clipping

"""
struct ObjectiveConfig
    objective::TrainingObjective
    trajectory_weight::Float64
    detailed_weight::Float64
    flow_weight::Float64
    regularization_weight::Float64
    entropy_weight::Float64
    clip_gradients::Bool
    gradient_clip_norm::Float64


    function ObjectiveConfig(;
        objective::TrainingObjective=TRAJECTORY_BALANCE,
        trajectory_weight::Float64=1.0,
        detailed_weight::Float64=0.0,
        flow_weight::Float64=0.0,
        regularization_weight::Float64=1e-4,
        entropy_weight::Float64=0.0,
        clip_gradients::Bool=true,
        gradient_clip_norm::Float64=1.0,)
        # Validate weights
        if trajectory_weight < 0 || detailed_weight < 0 || flow_weight < 0
            throw(ArgumentError("Loss weights must be non-negative"))
        end

        if regularization_weight < 0 || entropy_weight < 0
            throw(ArgumentError("Regularization weights must be non-negative"))
        end

        if gradient_clip_norm <= 0
            throw(ArgumentError("Gradient clip norm must be positive"))
        end

        new(objective, trajectory_weight, detailed_weight, flow_weight,
            regularization_weight, entropy_weight, clip_gradients,
            gradient_clip_norm)
    end
end

# =============================================================================
# Trajectory Balance Objective - Mathematical Foundation
# =============================================================================

"""
    trajectory_balance_objective(model::GFlowNetModel, trajectories::Vector{Trajectory};
                                config::ObjectiveConfig = ObjectiveConfig())::Float64

Compute trajectory balance training objective for a batch of trajectories.

# Mathematical Foundation
The trajectory balance objective minimizes the squared difference between
the forward trajectory log-probability and the log-reward:

L_TB = 𝔼_{τ~π}[(log P_F(τ) + log Z(s₀) - log R(s_T))²]

where:
- P_F(τ) = ∏ P_F(s'|s) is the forward trajectory probability
- Z(s₀) is the flow from initial state
- R(s_T) is the terminal reward

# Arguments
- `model::GFlowNetModel`: Complete GFlowNet model
- `trajectories::Vector{Trajectory}`: Batch of training trajectories
- `config::ObjectiveConfig`: Training objective configuration

# Returns
- `Float64`: Trajectory balance loss value

# Mathematical Properties
- Differentiable w.r.t. all model parameters
- Unbiased estimator of true trajectory balance
- Converges to flow conservation at optimum
"""
function trajectory_balance_objective(model::GFlowNetModel, trajectories::Vector{Trajectory};
    config::ObjectiveConfig=ObjectiveConfig())::Float64
    if isempty(trajectories)
        return 0.0
    end

    # Compute trajectory balance losses using functional approach (Zygote-safe)
    function safe_loss(traj)
        try
            return trajectory_balance_loss(model, traj)
        catch e
            @warn "Failed to compute trajectory balance for trajectory: $e"
            return nothing
        end
    end

    all_losses = [safe_loss(traj) for traj in trajectories]
    losses = filter(!isnothing, all_losses)

    if isempty(losses)
        @warn "No valid trajectory balance losses computed"
        return 0.0
    end

    # Return mean loss
    return mean(losses)
end



# =============================================================================
# Detailed Balance Objective - Mathematical Foundation
# =============================================================================

"""
    detailed_balance_objective(model::GFlowNetModel, state_pairs::Vector{Tuple{S,S}};
                              config::ObjectiveConfig = ObjectiveConfig())::Float64 where {S<:AbstractState}

Compute detailed balance training objective for state pairs.

# Mathematical Foundation
The detailed balance objective enforces flow conservation on each edge:

L_DB = 𝔼_{(s,s')~E}[(log P_F(s'|s) + log F(s) - log P_B(s|s') - log F(s'))²]

where (s,s') are connected state pairs in the DAG.

# Arguments
- `model::GFlowNetModel`: Model with backward policy
- `state_pairs::Vector{Tuple{S,S}}`: Connected state pairs
- `config::ObjectiveConfig`: Training configuration

# Returns
- `Float64`: Detailed balance loss value

# Requirements
- Model must have backward policy
- State pairs must be connected in the DAG
"""
function detailed_balance_objective(model::GFlowNetModel, state_pairs::Vector{Tuple{S,S}};
    config::ObjectiveConfig=ObjectiveConfig())::Float64 where {S<:AbstractState}
    if isnothing(model.backward_policy)
        throw(ArgumentError("Detailed balance objective requires backward policy"))
    end

    if isempty(state_pairs)
        return 0.0
    end

    return detailed_balance_loss_batch(model, state_pairs)
end

# =============================================================================
# Flow Matching Objective - Mathematical Foundation
# =============================================================================

"""
    flow_matching_objective(model::GFlowNetModel, states::Vector{S};
                           config::ObjectiveConfig = ObjectiveConfig())::Float64 where {S<:AbstractState}

Compute flow matching training objective for states.

# Mathematical Foundation
The flow matching objective enforces flow conservation via direct estimation:

L_FM = 𝔼_{s~S}[(Z(s) - Σ_{s'} P_F(s'|s) * F(s'))²]

where Z(s) is the direct flow estimate and the sum is recursive flow.

# Arguments
- `model::GFlowNetModel`: Model with flow estimator
- `states::Vector{S}`: States to compute flow matching for
- `config::ObjectiveConfig`: Training configuration

# Returns
- `Float64`: Flow matching loss value

# Requirements
- Model must have flow estimator
- States should be non-terminal for meaningful loss
"""
function flow_matching_objective(model::GFlowNetModel, states::Vector{S};
    config::ObjectiveConfig=ObjectiveConfig())::Float64 where {S<:AbstractState}
    if isnothing(model.flow_estimator)
        throw(ArgumentError("Flow matching objective requires flow estimator"))
    end

    if isempty(states)
        return 0.0
    end

    return flow_matching_loss_batch(model, states)
end

# =============================================================================
# Combined Training Objective - Mathematical Foundation
# =============================================================================

"""
    combined_objective(model::GFlowNetModel, trajectories::Vector{Trajectory},
                      state_pairs::Vector{Tuple{S,S}}, states::Vector{S};
                      config::ObjectiveConfig = ObjectiveConfig())::Float64 where {S<:AbstractState}

Compute combined training objective using multiple balance conditions.

# Mathematical Foundation
Combines multiple objectives with configurable weights:

L_combined = α₁*L_TB + α₂*L_DB + α₃*L_FM + α₄*L_reg + α₅*L_entropy

where:
- L_TB is trajectory balance loss
- L_DB is detailed balance loss
- L_FM is flow matching loss
- L_reg is regularization term
- L_entropy is policy entropy regularization

# Arguments
- `model::GFlowNetModel`: Complete model
- `trajectories::Vector{Trajectory}`: Training trajectories
- `state_pairs::Vector{Tuple{S,S}}`: State pairs for detailed balance
- `states::Vector{S}`: States for flow matching
- `config::ObjectiveConfig`: Objective weights and configuration

# Returns
- `Float64`: Combined weighted loss
"""
function combined_objective(model::GFlowNetModel, trajectories::Vector{Trajectory},
    state_pairs::Vector{Tuple{S,S}}, states::Vector{S};
    config::ObjectiveConfig=ObjectiveConfig())::Float64 where {S<:AbstractState}
    total_loss = 0.0

    # Trajectory Balance component
    if config.trajectory_weight > 0 && !isempty(trajectories)
        tb_loss = trajectory_balance_objective(model, trajectories; config=config)
        total_loss += config.trajectory_weight * tb_loss
    end

    # Detailed Balance component
    if config.detailed_weight > 0 && !isempty(state_pairs) && !isnothing(model.backward_policy)
        db_loss = detailed_balance_objective(model, state_pairs; config=config)
        total_loss += config.detailed_weight * db_loss
    end

    # Flow Matching component
    if config.flow_weight > 0 && !isempty(states) && !isnothing(model.flow_estimator)
        fm_loss = flow_matching_objective(model, states; config=config)
        total_loss += config.flow_weight * fm_loss
    end

    # Regularization component
    if config.regularization_weight > 0
        reg_loss = parameter_regularization_loss(model)
        total_loss += config.regularization_weight * reg_loss
    end

    # Entropy regularization component
    if config.entropy_weight > 0 && !isempty(trajectories)
        entropy_loss = policy_entropy_loss(model, trajectories)
        total_loss += config.entropy_weight * entropy_loss
    end

    return total_loss
end

# =============================================================================
# Regularization Terms - Mathematical Foundation
# =============================================================================

"""
    parameter_regularization_loss(model::GFlowNetModel)::Float64

Compute L2 regularization loss for model parameters.

# Mathematical Foundation
L2 regularization penalizes large parameter values:

L_reg = (1/2) * Σᵢ θᵢ²

This helps prevent overfitting and improves generalization.

# Arguments
- `model::GFlowNetModel`: Model to regularize

# Returns
- `Float64`: L2 regularization loss
"""
function parameter_regularization_loss(model::GFlowNetModel)::Float64
    total_norm_squared = 0.0

    # Forward policy parameters
    forward_params = model.parameters.forward
    total_norm_squared += _compute_parameter_norm_squared(forward_params)

    # Backward policy parameters (if present)
    if !isnothing(model.backward_policy) && haskey(model.parameters, :backward)
        backward_params = model.parameters.backward
        total_norm_squared += _compute_parameter_norm_squared(backward_params)
    end

    # Flow estimator parameters (if present)
    if !isnothing(model.flow_estimator) && haskey(model.parameters, :flow)
        flow_params = model.parameters.flow
        total_norm_squared += _compute_parameter_norm_squared(flow_params)
    end

    return 0.5 * total_norm_squared
end

"""
    _compute_parameter_norm_squared(params)::Float64

Compute squared L2 norm of parameters recursively.
"""
function _compute_parameter_norm_squared(params)::Float64
    if params isa AbstractArray
        return sum(abs2, params)
    elseif params isa NamedTuple
        return sum(_compute_parameter_norm_squared(p) for p in params)
    elseif params isa ComponentArray
        return sum(abs2, params)
    else
        return 0.0
    end
end

"""
    policy_entropy_loss(model::GFlowNetModel, trajectories::Vector{Trajectory})::Float64

Compute policy entropy regularization loss.

# Mathematical Foundation
Entropy regularization encourages exploration by penalizing low-entropy policies:

L_entropy = -𝔼_{s~τ}[H(P_F(·|s))] = -𝔼_{s~τ}[Σₐ P_F(a|s) log P_F(a|s)]

Higher entropy corresponds to more exploratory behavior.

# Arguments
- `model::GFlowNetModel`: Model to compute entropy for
- `trajectories::Vector{Trajectory}`: Trajectories for state sampling

# Returns
- `Float64`: Negative entropy (loss to minimize)
"""
function policy_entropy_loss(model::GFlowNetModel, trajectories::Vector{Trajectory})::Float64
    if isempty(trajectories)
        return 0.0
    end

    total_entropy = 0.0
    n_states = 0

    for trajectory in trajectories
        for state in trajectory.states
            # Skip terminal states (no policy entropy)
            if is_terminal_state(state)
                continue
            end

            try
                # Get action probabilities
                probs = forward_action_probabilities(
                    model.forward_policy, state, model.dag.actions,
                    model.parameters.forward, model.states.forward
                )

                # Compute entropy: -Σ p log p
                entropy = 0.0
                for p in probs
                    if p > 0
                        entropy -= p * log(p)
                    end
                end

                total_entropy += entropy
                n_states += 1

            catch e
                @warn "Failed to compute entropy for state $state: $e"
                continue
            end
        end
    end

    # Return negative entropy (to minimize)
    return n_states > 0 ? -(total_entropy / n_states) : 0.0
end

# =============================================================================
# Unified Objective Interface - Mathematical Foundation
# =============================================================================

"""
    compute_training_objective(objective::TrainingObjective, model::GFlowNetModel, data...;
                              config::ObjectiveConfig = ObjectiveConfig())::Float64

Unified interface for computing any training objective.

# Arguments
- `objective::TrainingObjective`: Which objective to compute
- `model::GFlowNetModel`: Complete model
- `data...`: Objective-specific data (trajectories, state pairs, etc.)
- `config::ObjectiveConfig`: Training configuration

# Returns
- `Float64`: Computed objective value

# Data Requirements
- TRAJECTORY_BALANCE: trajectories::Vector{Trajectory}
- DETAILED_BALANCE: state_pairs::Vector{Tuple{State,State}}
- FLOW_MATCHING: states::Vector{State}
- COMBINED_OBJECTIVE: trajectories, state_pairs, states
"""
function compute_training_objective(objective, model::GFlowNetModel, data...;
    config::ObjectiveConfig=ObjectiveConfig())::Float64

    # Map training objectives to balance conditions
    if objective == GFlowNet.TRAJECTORY_BALANCE
        if length(data) >= 1 && data[1] isa Vector{<:Trajectory}
            return trajectory_balance_objective(model, data[1]; config=config)
        else
            throw(ArgumentError("Trajectory balance requires Vector{Trajectory} as first argument"))
        end

    elseif objective == GFlowNet.DETAILED_BALANCE
        if length(data) >= 1 && data[1] isa Vector{<:Tuple{<:AbstractState,<:AbstractState}}
            return detailed_balance_objective(model, data[1]; config=config)
        else
            throw(ArgumentError("Detailed balance requires Vector{Tuple{State,State}} as first argument"))
        end

    elseif objective == GFlowNet.FLOW_MATCHING
        if length(data) >= 1 && data[1] isa Vector{<:AbstractState}
            return flow_matching_objective(model, data[1]; config=config)
        else
            throw(ArgumentError("Flow matching requires Vector{State} as first argument"))
        end

    elseif objective == GFlowNet.COMBINED_OBJECTIVES_TRAINING
        if length(data) >= 3
            trajectories = data[1]
            state_pairs = data[2]
            states = data[3]
            return combined_objective(model, trajectories, state_pairs, states; config=config)
        else
            throw(ArgumentError("Combined objective requires trajectories, state_pairs, and states"))
        end

    else
        throw(ArgumentError("Unknown training objective: $objective"))
    end
end

# =============================================================================
# Gradient Computation and Optimization Utilities
# =============================================================================

"""
    compute_gradients(objective_fn, model::GFlowNetModel)

Compute gradients of objective function w.r.t. model parameters.

# Mathematical Foundation
Computes ∇_θ L(θ) where L is the loss function and θ are model parameters.
Uses automatic differentiation via Zygote.jl for efficiency and accuracy.

# Arguments
- `objective_fn`: Function that computes objective given model
- `model::GFlowNetModel`: Model to compute gradients for

# Returns
- Gradients w.r.t. model parameters in same structure as parameters
"""
function compute_gradients(objective_fn, model::GFlowNetModel)
    # Use Zygote to compute gradients w.r.t. parameters
    gradients = Zygote.gradient(model.parameters) do params
        # Create temporary model with new parameters for gradient computation
        temp_model = GFlowNetModel(
            model.dag, model.forward_policy, params, model.states;
            backward_policy=model.backward_policy,
            flow_estimator=model.flow_estimator
        )
        objective_fn(temp_model)
    end

    return gradients[1]  # Return parameter gradients
end

"""
    clip_gradients!(gradients, max_norm::Float64)

Clip gradients by global norm to prevent exploding gradients.

# Mathematical Foundation
Scales gradients to have maximum norm:
g_clipped = g * min(1, max_norm / ||g||)

This helps maintain training stability.

# Arguments
- `gradients`: Gradient structure to modify in-place
- `max_norm::Float64`: Maximum allowed gradient norm

# Returns
- `Float64`: Original gradient norm before clipping
"""
function clip_gradients!(gradients, max_norm::Float64)
    # Compute global gradient norm
    grad_norm = _compute_gradient_norm(gradients)

    if grad_norm > max_norm
        # Scale gradients to max_norm
        scale_factor = max_norm / grad_norm
        _scale_gradients!(gradients, scale_factor)
    end

    return grad_norm
end

"""
    _compute_gradient_norm(gradients)::Float64

Compute L2 norm of gradients recursively.
"""
function _compute_gradient_norm(gradients)::Float64
    if gradients isa AbstractArray
        return norm(gradients)
    elseif gradients isa NamedTuple
        return sqrt(sum(_compute_gradient_norm(g)^2 for g in gradients))
    elseif gradients isa ComponentArray
        return norm(gradients)
    else
        return 0.0
    end
end

"""
    _scale_gradients!(gradients, scale::Float64)

Scale gradients by a constant factor in-place.
"""
function _scale_gradients!(gradients, scale::Float64)
    if gradients isa AbstractArray
        gradients .*= scale
    elseif gradients isa NamedTuple
        for g in gradients
            _scale_gradients!(g, scale)
        end
    elseif gradients isa ComponentArray
        gradients .*= scale
    end
end

# =============================================================================
# Objective Analysis and Diagnostics
# =============================================================================

"""
    analyze_objective_components(model::GFlowNetModel, trajectories::Vector{Trajectory};
                                config::ObjectiveConfig = ObjectiveConfig())

Analyze individual components of the training objective.

# Returns
Named tuple with detailed breakdown of objective components.
"""
function analyze_objective_components(model::GFlowNetModel, trajectories::Vector{Trajectory};
    config::ObjectiveConfig=ObjectiveConfig())

    results = Dict{String,Any}()

    # Trajectory balance analysis
    try
        tb_losses = [trajectory_balance_loss(model, traj) for traj in trajectories]
        results["trajectory_balance"] = (
            mean=mean(tb_losses),
            std=std(tb_losses),
            min=minimum(tb_losses),
            max=maximum(tb_losses),
            individual_losses=tb_losses
        )
    catch e
        @warn "Trajectory balance analysis failed: $e"
        results["trajectory_balance"] = nothing
    end

    # Parameter regularization
    try
        reg_loss = parameter_regularization_loss(model)
        results["regularization"] = reg_loss
    catch e
        @warn "Regularization analysis failed: $e"
        results["regularization"] = nothing
    end

    # Policy entropy
    try
        entropy_loss = policy_entropy_loss(model, trajectories)
        results["entropy"] = entropy_loss
    catch e
        @warn "Entropy analysis failed: $e"
        results["entropy"] = nothing
    end

    # Flow conservation validation
    try
        conservation_score = validate_flow_consistency(model)
        results["flow_conservation"] = conservation_score
    catch e
        @warn "Flow conservation analysis failed: $e"
        results["flow_conservation"] = nothing
    end

    return NamedTuple(Symbol(k) => v for (k, v) in results)
end

# =============================================================================
# Performance Optimization
# =============================================================================

"""
    precompile_objectives()

Precompile common objective computations for better runtime performance.
"""
function precompile_objectives()
    # This would contain precompilation directives for common use cases
    # Left as placeholder for future optimization
    nothing
end

# =============================================================================
# Display Methods
# =============================================================================

# Note: Display methods for objectives are defined in their respective modules
# to avoid method overwriting conflicts

function Base.show(io::IO, config::ObjectiveConfig)
    print(io, "ObjectiveConfig($(config.objective), TB=$(config.trajectory_weight), DB=$(config.detailed_weight), FM=$(config.flow_weight))")
end
