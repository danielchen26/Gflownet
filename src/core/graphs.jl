# Graph Operations - Minimal On-Demand Approach
# Simple functions for computing DAG operations without explicit construction

# =============================================================================
# On-Demand DAG Operations - Core Functions Only
# =============================================================================

"""
    get_applicable_actions(state::AbstractState, all_actions::Vector{<:AbstractAction})

Get actions applicable from a state using on-demand computation.

This is the core function that replaces complex DAG caching with robust computation.
No cache misses, no object identity issues.

# Mathematical Foundation
Returns A(s) = {a ∈ A : is_applicable(a, s)}

# Arguments
- `state::AbstractState` - Current state
- `all_actions::Vector{<:AbstractAction}` - Complete action space

# Returns
- `Vector{<:AbstractAction}` - Actions applicable from state
"""
function get_applicable_actions(state::AbstractState, all_actions::Vector{<:AbstractAction})
    return [action for action in all_actions if is_applicable(action, state)]
end

"""
    compute_next_state(action::AbstractAction, state::AbstractState)

Compute next state from current state and action.

Standard wrapper around apply_action for consistency.

# Arguments
- `action::AbstractAction` - Action to apply
- `state::AbstractState` - Current state

# Returns
- `AbstractState` - Next state after applying action
"""
function compute_next_state(action::AbstractAction, state::AbstractState)
    return apply_action(action, state)
end

"""
    is_valid_transition(action::AbstractAction, state::AbstractState)

Check if a state-action transition is valid.

Combines applicability check with basic validation.

# Arguments
- `action::AbstractAction` - Action to check
- `state::AbstractState` - Current state

# Returns
- `Bool` - True if transition is valid
"""
function is_valid_transition(action::AbstractAction, state::AbstractState)
    return !is_terminal_state(state) && is_applicable(action, state)
end

# =============================================================================
# Optional Analysis Functions - For Debugging Only
# =============================================================================

"""
    explore_state_space(initial_state::AbstractState, actions::Vector{<:AbstractAction}; max_states::Int=1000)

Explore state space starting from initial state for analysis purposes.

This is used only for debugging and analysis, not for training.
Training uses on-demand computation instead.

# Arguments
- `initial_state::AbstractState` - Starting state
- `actions::Vector{<:AbstractAction}` - Available actions
- `max_states::Int` - Maximum states to explore

# Returns
- `Set{AbstractState}` - Set of discovered states
"""
function explore_state_space(initial_state::AbstractState, actions::Vector{<:AbstractAction}; max_states::Int=1000)
    visited = Set{AbstractState}([initial_state])
    queue = [initial_state]
    queue_idx = 1

    while queue_idx <= length(queue) && length(visited) < max_states
        current_state = queue[queue_idx]
        queue_idx += 1

        if is_terminal_state(current_state)
            continue
        end

        applicable_actions = get_applicable_actions(current_state, actions)

        for action in applicable_actions
            try
                next_state = compute_next_state(action, current_state)
                if next_state ∉ visited
                    push!(visited, next_state)
                    push!(queue, next_state)
                end
            catch e
                # Skip invalid transitions
                continue
            end
        end
    end

    return visited
end

"""
    count_reachable_states(initial_state::AbstractState, actions::Vector{<:AbstractAction}; max_states::Int=1000)

Count the number of reachable states for analysis.

# Returns
- `Int` - Number of reachable states
"""
function count_reachable_states(initial_state::AbstractState, actions::Vector{<:AbstractAction}; max_states::Int=1000)
    return length(explore_state_space(initial_state, actions; max_states=max_states))
end

"""
    analyze_state_space(initial_state::AbstractState, actions::Vector{<:AbstractAction}; max_states::Int=1000)

Analyze state space properties for debugging.

# Returns
- `NamedTuple` with analysis results
"""
function analyze_state_space(initial_state::AbstractState, actions::Vector{<:AbstractAction}; max_states::Int=1000)
    states = explore_state_space(initial_state, actions; max_states=max_states)
    terminal_states = filter(is_terminal_state, states)

    return (
        total_states = length(states),
        terminal_states = length(terminal_states),
        non_terminal_states = length(states) - length(terminal_states),
        actions_count = length(actions),
        exploration_complete = length(states) < max_states
    )
end

# =============================================================================
# Exports - Minimal Interface
# =============================================================================

# Core functions (used during training)
export get_applicable_actions, compute_next_state, is_valid_transition

# Analysis functions (for debugging only)
export explore_state_space, count_reachable_states, analyze_state_space

# =============================================================================
# Architecture Notes
# =============================================================================

"""
This simplified approach eliminates the complex DAG construction that was causing
training errors. Key improvements:

1. ✅ NO EXPLICIT DAG CONSTRUCTION
   - No complex data structures to maintain
   - No cache misses or object identity issues
   - Simpler and more robust

2. ✅ ON-DEMAND COMPUTATION
   - Applicable actions computed when needed
   - State transitions computed when needed
   - No pre-computation overhead

3. ✅ MINIMAL INTERFACE
   - Only essential functions for training
   - Analysis functions separate and optional
   - Clear separation of concerns

4. ✅ MATHEMATICAL EQUIVALENCE
   - Same GFlowNet mathematical properties
   - DAG structure defined implicitly through domain functions
   - All training objectives work unchanged

The mathematical DAG still exists conceptually - it's just computed on-demand
rather than stored explicitly. This approach is:
- More robust (no cache issues)
- Simpler to understand and debug
- Compatible with all GFlowNet algorithms
- Eliminates the source of training errors
"""
