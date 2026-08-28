# Validation Utilities
# Consolidated validation functions for GFlowNet components

using Statistics
using Zygote

"""
    validate_numerical_array(array, name::String; allow_empty=false, max_magnitude=1e6)

General validation for numerical arrays used throughout GFlowNet.

# Arguments
- `array`: Array to validate
- `name`: Descriptive name for error messages
- `allow_empty`: Whether to allow empty arrays (default: false)
- `max_magnitude`: Maximum allowed magnitude before warning (default: 1e6)

# Throws
- `ArgumentError` if array contains invalid values
"""
function validate_numerical_array(array, name::String; allow_empty=false, max_magnitude=1e6)
    if !allow_empty && isempty(array)
        throw(ArgumentError("$name cannot be empty"))
    end

    if any(isnan.(array))
        nan_count = count(isnan.(array))
        throw(ArgumentError("$name contains $nan_count NaN values. This will cause numerical instability."))
    end

    if any(isinf.(array))
        inf_count = count(isinf.(array))
        throw(ArgumentError("$name contains $inf_count infinite values. This will cause numerical issues."))
    end

    # Check for extremely large values that could cause numerical issues
    if !isempty(array)
        max_val = maximum(abs.(array))
        if max_val > max_magnitude
            @warn "$name contains very large values (max: $max_val). This may cause numerical instability."
        end
    end
end

"""
    validate_reward(reward_value, state_description::String="state")

Validate reward values for GFlowNet training.

# Arguments
- `reward_value`: Reward to validate
- `state_description`: Description of state for error messages

# Throws
- `ArgumentError` if reward is invalid for GFlowNet training
"""
function validate_reward(reward_value, state_description::String="state")
    if isnan(reward_value)
        throw(ArgumentError("Reward for $state_description is NaN. GFlowNets require finite rewards."))
    end

    if isinf(reward_value)
        throw(ArgumentError("Reward for $state_description is infinite. GFlowNets require finite rewards."))
    end

    if reward_value <= 0.0
        throw(ArgumentError("Reward for $state_description is $reward_value. GFlowNets require strictly positive rewards. Consider adding a small positive constant (e.g., 0.01)."))
    end

    if reward_value > 1e10
        @warn "Very large reward ($reward_value) for $state_description. This may cause numerical instability."
    end

    return reward_value
end

"""
    validate_state_features(features, name::String)

Validate state features for neural network input.

# Arguments
- `features`: Feature array to validate
- `name`: Descriptive name for error messages

# Throws
- `ArgumentError` if features are invalid
"""
function validate_state_features(features, name::String)
    validate_numerical_array(features, name; allow_empty=false, max_magnitude=1e6)
end

"""
    validate_neural_network_input(input, name::String)

Validate neural network input for numerical issues.

# Arguments
- `input`: Input array to validate
- `name`: Descriptive name for error messages

# Throws
- `ArgumentError` if input contains invalid values
"""
function validate_neural_network_input(input, name::String)
    validate_numerical_array(input, name; allow_empty=false, max_magnitude=1e6)
end

"""
    validate_neural_network_output(output, name::String)

Validate neural network output for numerical issues.

# Arguments
- `output`: Output array to validate
- `name`: Descriptive name for error messages

# Throws
- `ArgumentError` if output contains invalid values
"""
function validate_neural_network_output(output, name::String)
    validate_numerical_array(output, name; allow_empty=false, max_magnitude=Inf)  # No magnitude warning for output
end

"""
    validate_model_parameters(params, name::String)

Validate model parameters for numerical issues.

# Arguments
- `params`: Parameters to validate (NamedTuple or ComponentArray)
- `name`: Descriptive name for error messages
"""
function validate_model_parameters(params, name::String)
    if isnothing(params)
        throw(ArgumentError("$name cannot be nothing"))
    end

    # Recursively validate parameter values
    _validate_params_recursive(params, name)
end

"""
    _validate_params_recursive(params, name::String)

Helper function to recursively validate nested parameter structures.
"""
function _validate_params_recursive(params, name::String)
    if params === nothing
        # Skip validation for nothing values (e.g., unused backward policy)
        return
    elseif params isa AbstractArray
        # Filter out nothing values before validation
        non_nothing_params = filter(x -> x !== nothing, params)
        if !isempty(non_nothing_params)
            validate_numerical_array(non_nothing_params, name; allow_empty=true, max_magnitude=1e6)
        end
    elseif params isa NamedTuple
        for (key, value) in pairs(params)
            _validate_params_recursive(value, "$name.$key")
        end
    elseif params isa ComponentArray
        # ComponentArrays need special handling - validate each component separately
        for (key, value) in pairs(params)
            if value !== nothing
                _validate_params_recursive(value, "$name.$key")
            end
        end
    end
end

"""
    validate_policy_output(features, expected_length::Int)

Validate features for policy network input.

# Arguments
- `features`: Feature array to validate
- `expected_length`: Expected length of features array

# Throws
- `ArgumentError` if features are invalid or wrong length
"""
function validate_policy_output(features, expected_length::Int)
    validate_numerical_array(features, "policy features"; allow_empty=false, max_magnitude=1e6)

    if length(features) != expected_length
        throw(ArgumentError("Policy features have length $(length(features)) but expected $expected_length"))
    end
end

"""
    validate_state_for_policy(state::AbstractState)

Validate that a state is suitable for policy operations.

# Arguments
- `state`: State to validate

# Throws
- `ArgumentError` if state is invalid for policy operations
"""
function validate_state_for_policy(state::AbstractState)
    # Validate that we can extract features from the state
    try
        features = state_to_features(state)
        validate_state_features(features, "state features for policy")
    catch e
        throw(ArgumentError("Invalid state for policy operations: $e"))
    end

    # Additional validation could be added here as needed
end

# =============================================================================
# Learnable Partition Function (Z) Validation
# =============================================================================

"""
    validate_z_learning(model::GFlowNetModel, config::TrainingConfig)

Validate that Z learning is configured and working correctly.

# Returns
- `Bool` - True if Z learning validation passes
"""
function validate_z_learning(model::GFlowNetModel, config::TrainingConfig)
    if config.partition_function_method != LEARNABLE_ESTIMATION
        @info "Validation skipped: Not using LEARNABLE_ESTIMATION"
        return true
    end
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Model has log_partition_function field set
    if !isnothing(model.log_partition_function)
        @info "✅ Model has log_partition_function field initialized"
        checks_passed += 1
    else
        @warn "❌ Model log_partition_function field is nothing"
    end
    
    # Check 2: Parameters contain log_Z
    if haskey(model.parameters, :log_Z)
        @info "✅ Parameters contain log_Z parameter"
        checks_passed += 1
    else
        @warn "❌ Parameters missing log_Z parameter"
    end
    
    # Check 3: log_Z is synchronized with model field
    if haskey(model.parameters, :log_Z) && !isnothing(model.log_partition_function)
        if abs(model.parameters.log_Z - model.log_partition_function) < 1e-6
            @info "✅ log_Z parameter synchronized with model field"
            checks_passed += 1
        else
            @warn "❌ log_Z parameter ($(model.parameters.log_Z)) not synchronized with model field ($(model.log_partition_function))"
        end
    else
        @warn "❌ Cannot check synchronization - missing parameters"
    end
    
    # Check 4: Optimizer state includes log_Z
    if hasfield(typeof(model.optimizer), :log_Z) || 
       (hasfield(typeof(model.optimizer), :tree) && haskey(model.optimizer.tree, :log_Z))
        @info "✅ Optimizer configured for log_Z parameter"
        checks_passed += 1
    else
        @warn "❌ Optimizer may not be configured for log_Z parameter"
    end
    
    success_rate = checks_passed / total_checks
    
    if success_rate >= 1.0
        @info "🎉 All Z learning validation checks passed!"
        return true
    elseif success_rate >= 0.75
        @warn "⚠️  Most Z learning checks passed ($checks_passed/$total_checks), but some issues detected"
        return true
    else
        @error "❌ Z learning validation failed ($checks_passed/$total_checks checks passed)"
        return false
    end
end

"""
    validate_z_gradients(model::GFlowNetModel, trajectories::Vector{Trajectory})

Validate that the Z parameter is receiving gradients during training.

# Returns
- `Float64` - Gradient magnitude for log_Z parameter (NaN if no gradient)
"""
function validate_z_gradients(model::GFlowNetModel, trajectories::Vector{Trajectory})
    if !haskey(model.parameters, :log_Z)
        @warn "Cannot validate Z gradients: log_Z parameter not found"
        return NaN
    end
    
    # Compute gradient w.r.t. log_Z parameter only
    loss_function = ps -> begin 
        # Create temporary model with new parameters
        temp_model = GFlowNetModel(
            model.initial_state,
            model.all_actions,
            model.forward_policy,
            model.backward_policy,
            model.flow_estimator,
            # Float64(...) is REQUIRED: the struct field is Union{Nothing,Float64}
            # (types.jl:160) while ps.log_Z follows the network parameter eltype,
            # which is Float32. Passing it straight through threw
            # MethodError: no method matching GFlowNetModel(..., ::Float32, ...).
            # The field is a mirror of the parameter, so widening it is unnecessary;
            # converting here keeps the struct concretely typed.
            haskey(ps, :log_Z) ? Float64(ps.log_Z) : nothing,
            ps,
            model.optimizer,
            model.states
        )
        
        # Compute loss using the same function as training
        compute_trajectory_loss(temp_model, trajectories, ps, TrainingConfig())
    end
    
    try
        _, grads = Zygote.withgradient(loss_function, model.parameters)
        
        if grads[1] !== nothing && haskey(grads[1], :log_Z)
            z_gradient = grads[1].log_Z
            gradient_magnitude = abs(z_gradient)
            
            if isfinite(gradient_magnitude) && gradient_magnitude > 1e-10
                @info "✅ Z parameter receiving gradients (magnitude: $gradient_magnitude)"
                return gradient_magnitude
            else
                @warn "⚠️  Z gradient is very small or zero: $z_gradient"
                return gradient_magnitude
            end
        else
            @warn "❌ No gradient computed for log_Z parameter"
            return NaN
        end
    catch e
        @error "Error computing Z gradients" exception=e
        return NaN
    end
end

"""
    validate_z_mathematical_properties(model::GFlowNetModel, trajectories::Vector{Trajectory})

Validate mathematical properties of learned Z parameter.

# Returns
- `Bool` - True if mathematical validation passes
"""
function validate_z_mathematical_properties(model::GFlowNetModel, trajectories::Vector{Trajectory})
    if !haskey(model.parameters, :log_Z)
        @info "Skipping mathematical validation: not using learnable Z"
        return true
    end
    
    checks_passed = 0
    total_checks = 3
    
    log_Z = model.parameters.log_Z
    Z = exp(log_Z)
    
    # Check 1: Z should be positive (log Z can be any real number)
    if isfinite(log_Z) && Z > 0
        @info "✅ Z is positive and finite (Z = $Z, log Z = $log_Z)"
        checks_passed += 1
    else
        @warn "❌ Z has invalid value (Z = $Z, log Z = $log_Z)"
    end
    
    # Check 2: Z should be reasonable magnitude (not too large or small)
    if 1e-6 < Z < 1e6
        @info "✅ Z has reasonable magnitude"
        checks_passed += 1
    else
        @warn "⚠️  Z has extreme magnitude (Z = $Z) - may indicate training issues"
    end
    
    # Check 3: Trajectory balance equation should be approximately satisfied
    if !isempty(trajectories)
        try
            # Check balance for a few trajectories
            balance_errors = []
            for traj in trajectories[1:min(5, length(trajectories))]
                if length(traj.states) >= 2
                    # Compute log P_F(τ) + log Z - log R(s_T)
                    log_prob = compute_trajectory_log_probability(model, traj)
                    terminal_reward = reward(traj.states[end])
                    balance_error = log_Z + log_prob - log(max(terminal_reward, 1e-8))
                    push!(balance_errors, abs(balance_error))
                end
            end
            
            if !isempty(balance_errors)
                mean_error = mean(balance_errors)
                if mean_error < 5.0  # Allow some tolerance during training
                    @info "✅ Trajectory balance approximately satisfied (mean error: $mean_error)"
                    checks_passed += 1
                else
                    @warn "⚠️  Large trajectory balance errors (mean error: $mean_error)"
                end
            end
        catch e
            @warn "Could not validate trajectory balance" exception=e
        end
    end
    
    success_rate = checks_passed / total_checks
    
    if success_rate >= 0.8
        @info "✅ Z mathematical properties validation passed"
        return true
    else
        @warn "⚠️  Some Z mathematical property checks failed ($checks_passed/$total_checks)"
        return false
    end
end

"""
    compute_trajectory_log_probability(model::GFlowNetModel, trajectory::Trajectory)

Helper function to compute log probability of a trajectory.

# Returns
- `Float64` - Log probability of trajectory under forward policy
"""
function compute_trajectory_log_probability(model::GFlowNetModel, trajectory::Trajectory)
    # Use the same computation as in training for consistency
    log_prob_sum = 0.0
    
    for i in 1:(length(trajectory.states)-1)
        state = trajectory.states[i]
        action = trajectory.actions[i]
        
        # Get state features
        features = state_to_features(state)
        
        # Get forward policy logits using proper Lux interface
        logits_vec, _ = model.forward_policy.model(features, model.parameters.forward, model.states.forward)
        logits = logits_vec  # For compatibility
        
        # Get applicable actions and their indices
        applicable_actions = get_applicable_actions(state, model.all_actions)
        if isempty(applicable_actions)
            return -Inf
        end
        
        # Find action index in all_actions
        action_idx = findfirst(==(action), model.all_actions)
        if isnothing(action_idx)
            return -Inf
        end
        
        # Get indices of applicable actions
        applicable_indices = [findfirst(==(a), model.all_actions) for a in applicable_actions]
        applicable_logits = logits[applicable_indices]
        
        # Compute log probabilities using logsumexp for stability
        log_probs = applicable_logits .- logsumexp(applicable_logits)
        
        # Find action position in applicable actions
        action_pos = findfirst(==(action), applicable_actions)
        if isnothing(action_pos)
            return -Inf
        end
        
        log_prob_sum += log_probs[action_pos]
    end
    
    return log_prob_sum
end

# logsumexp is defined in core/interface.jl
