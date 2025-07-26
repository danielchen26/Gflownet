# Core Abstract Interfaces for GFlowNet
# This file defines the required interfaces that must be implemented by concrete types

"""
    AbstractState

Abstract base type for all states in a GFlowNet.
Concrete implementations must define specific state representations for their domain.
"""
abstract type AbstractState end

"""
    AbstractAction

Abstract base type for all actions in a GFlowNet.
Concrete implementations must define specific actions for their domain.
"""
abstract type AbstractAction end

"""
    AbstractGFlowNetObjective

Abstract base type for all GFlowNet training objectives.
"""
abstract type AbstractGFlowNetObjective end

"""
    AbstractPolicy

Abstract base type for all policy types (forward, backward).
"""
abstract type AbstractPolicy end

"""
    AbstractPartitionFunctionEstimator

Abstract base type for partition function estimation methods.
"""
abstract type AbstractPartitionFunctionEstimator end

# =============================================================================
# Required Interface Methods for States
# =============================================================================

"""
    is_terminal_state(state::AbstractState) -> Bool

Check if a state is terminal (no outgoing transitions possible).
This function MUST be implemented for all concrete state types.

# Arguments
- `state`: State to check

# Returns
- `true` if state is terminal, `false` otherwise

# Example
```julia
function is_terminal_state(state::MyState)
    return state.size >= MAX_SIZE
end
```
"""
function is_terminal_state(state::AbstractState)
    error("is_terminal_state not implemented for state type $(typeof(state)). " *
          "Please implement this method for your domain-specific types.")
end

"""
    state_to_features(state::AbstractState) -> Vector{Float32}

Convert a state to a feature vector for neural network input.
This function MUST be implemented for all concrete state types.

# Arguments
- `state`: State to convert

# Returns
- Feature vector as Vector{Float32}

# Example
```julia
function state_to_features(state::MyState)
    return Float32[state.x, state.y, state.value]
end
```
"""
function state_to_features(state::AbstractState)
    error("state_to_features not implemented for state type $(typeof(state)). " *
          "Please implement this method for your domain-specific types.")
end

"""
    function base_reward(state::AbstractState) -> Float64

    Compute the base reward for a state. For GFlowNets, rewards must be positive.
    This function MUST be implemented for all concrete state types.

    Note: Use `reward(state, env_data)` from training/rewards.jl for full reward computation.

    # Arguments
    - `state`: State to compute reward for

    # Returns
    - Positive reward value

    # Example
    ```julia
    function base_reward(state::MyState)
        if is_terminal_state(state)
            return exp(-state.energy)  # Always positive
        else
            return 0.0
        end
    end
    ```
"""
function base_reward(state::AbstractState)
    error("base_reward not implemented for state type $(typeof(state)). " *
          "Please implement this method for your domain-specific types.")
end

"""
    Base.hash(state::AbstractState, h::UInt) -> UInt

Compute hash for state. Required for using states as dictionary keys.
This function SHOULD be implemented for all concrete state types.

# Arguments
- `state`: State to hash
- `h`: Hash seed

# Returns
- Hash value

# Example
```julia
function Base.hash(state::MyState, h::UInt)
    return hash((state.x, state.y, state.value), h)
end
```
"""
function Base.hash(state::AbstractState, h::UInt)
    return hash(typeof(state), h)  # Default implementation
end

"""
    Base.:(==)(state1::AbstractState, state2::AbstractState) -> Bool

Check equality between states. Required for state comparison.
This function SHOULD be implemented for all concrete state types.

# Arguments
- `state1`, `state2`: States to compare

# Returns
- `true` if states are equal, `false` otherwise

# Example
```julia
function Base.:(==)(s1::MyState, s2::MyState)
    return s1.x == s2.x && s1.y == s2.y && s1.value == s2.value
end
```
"""
function Base.:(==)(state1::AbstractState, state2::AbstractState)
    return false  # Default: different types are not equal
end

function Base.:(==)(state1::T, state2::T) where {T<:AbstractState}
    # Default implementation for same type - should be overridden
    return hash(state1) == hash(state2)
end

# =============================================================================
# Required Interface Methods for Actions
# =============================================================================

"""
    is_applicable(action::AbstractAction, state::AbstractState) -> Bool

Check if an action can be applied to a state.
This function MUST be implemented for all concrete action types.

# Arguments
- `action`: Action to check
- `state`: State to check applicability for

# Returns
- `true` if action is applicable, `false` otherwise

# Example
```julia
function is_applicable(action::MyAction, state::MyState)
    return state.size + action.increment <= MAX_SIZE
end
```
"""
function is_applicable(action::AbstractAction, state::AbstractState)
    error("is_applicable not implemented for action type $(typeof(action)) and state type $(typeof(state)). " *
          "Please implement this method for your domain-specific types.")
end

"""
    apply_action(action::AbstractAction, state::AbstractState) -> AbstractState

Apply an action to a state to produce a new state.
This function MUST be implemented for all concrete action types.

# Arguments
- `action`: Action to apply
- `state`: State to apply action to

# Returns
- New state after applying action

# Example
```julia
function apply_action(action::MyAction, state::MyState)
    return MyState(state.x + action.dx, state.y + action.dy, state.value)
end
```
"""
function apply_action(action::AbstractAction, state::AbstractState)
    error("apply_action not implemented for action type $(typeof(action)) and state type $(typeof(state)). " *
          "Please implement this method for your domain-specific types.")
end

"""
    Base.hash(action::AbstractAction, h::UInt) -> UInt

Compute hash for action. Required for using actions as dictionary keys.
This function SHOULD be implemented for all concrete action types.

# Arguments
- `action`: Action to hash
- `h`: Hash seed

# Returns
- Hash value

# Example
```julia
function Base.hash(action::MyAction, h::UInt)
    return hash((action.dx, action.dy, action.type), h)
end
```
"""
function Base.hash(action::AbstractAction, h::UInt)
    return hash(typeof(action), h)  # Default implementation
end

"""
    Base.:(==)(action1::AbstractAction, action2::AbstractAction) -> Bool

Check equality between actions. Required for action comparison.
This function SHOULD be implemented for all concrete action types.

# Arguments
- `action1`, `action2`: Actions to compare

# Returns
- `true` if actions are equal, `false` otherwise

# Example
```julia
function Base.:(==)(a1::MyAction, a2::MyAction)
    return a1.dx == a2.dx && a1.dy == a2.dy && a1.type == a2.type
end
```
"""
function Base.:(==)(action1::AbstractAction, action2::AbstractAction)
    return false  # Default: different types are not equal
end

function Base.:(==)(action1::T, action2::T) where {T<:AbstractAction}
    # Default implementation for same type - should be overridden
    return hash(action1) == hash(action2)
end

# =============================================================================
# DAG Interface Methods
# =============================================================================

# DAG Interface methods will be defined in dag.jl after DirectedAcyclicGraph is defined

# Default implementations will be defined in types.jl after SimpleState and SimpleAction are defined

# =============================================================================
# Validation Helpers
# =============================================================================

"""
    validate_state_interface(StateType::Type{<:AbstractState})

Validate that a state type properly implements the required interface.

# Arguments
- `StateType`: The state type to validate

# Throws
- `MethodError` if required methods are not implemented
"""
function validate_state_interface(StateType::Type{<:AbstractState})
    required_methods = [
        is_terminal_state,
        state_to_features,
        reward
    ]

    # Create a dummy instance for testing (this may fail for some types)
    try
        dummy_state = StateType()  # Assumes zero-argument constructor

        for method in required_methods
            if !hasmethod(method, (StateType,))
                throw(MethodError(method, (StateType,)))
            end
        end

        @info "State type $StateType implements required interface"
    catch e
        @warn "Could not fully validate state type $StateType" exception = e
    end
end

"""
    validate_action_interface(ActionType::Type{<:AbstractAction}, StateType::Type{<:AbstractState})

Validate that an action type properly implements the required interface.

# Arguments
- `ActionType`: The action type to validate
- `StateType`: The state type it should work with

# Throws
- `MethodError` if required methods are not implemented
"""
function validate_action_interface(ActionType::Type{<:AbstractAction}, StateType::Type{<:AbstractState})
    required_methods = [
        is_applicable,
        apply_action
    ]

    try
        # This validation is limited without actual instances
        for method in required_methods
            if !hasmethod(method, (ActionType, StateType))
                throw(MethodError(method, (ActionType, StateType)))
            end
        end

        @info "Action type $ActionType implements required interface for $StateType"
    catch e
        @warn "Could not fully validate action type $ActionType with $StateType" exception = e
    end
end
