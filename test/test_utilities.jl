# Test Utilities for GFlowNet
# This file contains basic state and action types ONLY for testing purposes
# These types should NOT be used in production code or examples

using GFlowNet: AbstractState, AbstractAction, is_terminal_state, state_to_features, reward, is_applicable, apply_action

"""
    SimpleState <: AbstractState

A simple state type ONLY for testing purposes.
Contains a vector of integers representing the state data.

**Note**: This is a test utility and should not be used in production code.

# Fields
- `data`: Vector of integers representing the state
"""
struct SimpleState <: AbstractState
    data::Vector{Int}
end

"""
    SimpleAction <: AbstractAction

A simple action type ONLY for testing purposes.
Contains an integer value representing the action.

**Note**: This is a test utility and should not be used in production code.

# Fields
- `value`: Integer value representing the action
"""
struct SimpleAction <: AbstractAction
    value::Int
end

# =============================================================================
# Hash and Equality Implementations (for testing)
# =============================================================================

Base.:(==)(s1::SimpleState, s2::SimpleState) = s1.data == s2.data
Base.hash(s::SimpleState, h::UInt) = hash(s.data, h)

Base.:(==)(a1::SimpleAction, a2::SimpleAction) = a1.value == a2.value
Base.hash(a::SimpleAction, h::UInt) = hash(a.value, h)

# =============================================================================
# Interface Implementations (for testing only)
# =============================================================================

"""
    is_terminal_state(state::SimpleState) -> Bool

Check if a SimpleState is terminal (for testing).
By default, SimpleState with data [-1] is terminal (sink state).
"""
function is_terminal_state(state::SimpleState)
    return length(state.data) == 1 && state.data[1] == -1
end

"""
    state_to_features(state::SimpleState) -> Vector{Float32}

Convert SimpleState to feature vector (for testing).
Simply converts the integer data to Float32.
"""
function state_to_features(state::SimpleState)
    return Float32.(state.data)
end

"""
    reward(state::SimpleState) -> Float64

Compute reward for SimpleState (for testing).
Returns 1.0 for terminal states, 0.0 otherwise.
"""
function reward(state::SimpleState)
    if is_terminal_state(state)
        return 1.0  # Simple reward for reaching terminal state
    else
        return 0.0
    end
end

"""
    is_applicable(action::SimpleAction, state::SimpleState) -> Bool

Check if a SimpleAction can be applied to a SimpleState (for testing).
Simple rule: action is applicable if its value is not already in state data.
"""
function is_applicable(action::SimpleAction, state::SimpleState)
    # Simple rule: action is applicable if its value is not already in state data
    return action.value ∉ state.data
end

"""
    apply_action(action::SimpleAction, state::SimpleState) -> SimpleState

Apply a SimpleAction to a SimpleState (for testing).
Simple rule: add action value to state data.
"""
function apply_action(action::SimpleAction, state::SimpleState)
    # Simple rule: add action value to state data
    new_data = vcat(state.data, [action.value])
    return SimpleState(new_data)
end

# =============================================================================
# Test Helper Functions
# =============================================================================

"""
    create_test_initial_state() -> SimpleState

Create a simple initial state for testing.
"""
function create_test_initial_state()
    return SimpleState([0])
end

"""
    create_test_terminal_state() -> SimpleState

Create a simple terminal state for testing.
"""
function create_test_terminal_state()
    return SimpleState([-1])
end

"""
    create_test_actions(max_value::Int = 5) -> Vector{SimpleAction}

Create a set of simple actions for testing.
"""
function create_test_actions(max_value::Int=5)
    return [SimpleAction(i) for i in 1:max_value]
end

"""
    create_test_dag()

Create a simple DAG for testing purposes using SimpleState and SimpleAction.
"""
function create_test_dag()
    initial_state = create_test_initial_state()
    terminal_states = [create_test_terminal_state()]
    terminal_sink = create_test_terminal_state()
    actions = create_test_actions(3)

    return create_dag(initial_state, terminal_states, terminal_sink, actions)
end

# =============================================================================
# Display Methods (for testing)
# =============================================================================

function Base.show(io::IO, state::SimpleState)
    print(io, "SimpleState($(state.data))")
end

function Base.show(io::IO, action::SimpleAction)
    print(io, "SimpleAction($(action.value))")
end
