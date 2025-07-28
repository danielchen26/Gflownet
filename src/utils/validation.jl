# Validation Utilities
# Consolidated validation functions for GFlowNet components

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
