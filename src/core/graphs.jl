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
# DAG Functions for Flow Computation
# =============================================================================

"""
    get_next_states(dag, state::AbstractState)

Get all states reachable from the given state in one step.

This function is required for flow computation but implemented using on-demand approach.
For compatibility with flow functions that expect DAG interface.

# Arguments  
- `dag` - Ignored, kept for API compatibility
- `state::AbstractState` - Current state

# Returns
- `Vector{AbstractState}` - Next states reachable in one step

# Implementation Note
Uses get_applicable_actions + apply_action instead of explicit DAG traversal.
"""
function get_next_states(dag, state::AbstractState)
    # Extract all_actions from the dag parameter
    # dag should be a GFlowNetModel in practice
    if hasfield(typeof(dag), :all_actions)
        all_actions = dag.all_actions
    else
        throw(ArgumentError("DAG parameter must have all_actions field (expected GFlowNetModel)"))
    end
    
    applicable_actions = get_applicable_actions(state, all_actions)
    next_states = AbstractState[]
    
    for action in applicable_actions
        try
            next_state = compute_next_state(action, state)
            push!(next_states, next_state)
        catch e
            # Skip invalid transitions
            continue
        end
    end
    
    return next_states
end

"""
    get_previous_states(dag, state::AbstractState)

Get all states that can reach the given state in one step.

This is computationally expensive as it requires reverse search.
Used for backward policy and detailed balance computations.

# Arguments
- `dag` - Expected to be GFlowNetModel with all_actions and initial_state
- `state::AbstractState` - Target state

# Returns  
- `Vector{AbstractState}` - Previous states that can reach target state

# Implementation Note
Performs exhaustive search through state space - expensive but necessary for accuracy.
"""
function get_previous_states(dag, state::AbstractState)
    if hasfield(typeof(dag), :all_actions) && hasfield(typeof(dag), :initial_state)
        all_actions = dag.all_actions
        initial_state = dag.initial_state
    else
        throw(ArgumentError("DAG parameter must have all_actions and initial_state fields"))
    end
    
    # Explore state space to find all states that can transition to target
    previous_states = AbstractState[]
    explored_states = explore_state_space(initial_state, all_actions; max_states=10000)
    
    for candidate_state in explored_states
        if is_terminal_state(candidate_state)
            continue
        end
        
        applicable_actions = get_applicable_actions(candidate_state, all_actions)
        for action in applicable_actions
            try
                next_state = compute_next_state(action, candidate_state)
                if next_state == state
                    push!(previous_states, candidate_state)
                    break  # Found one transition, don't need to check other actions
                end
            catch e
                continue
            end
        end
    end
    
    return unique(previous_states)
end

"""
    get_root_state(dag)

Get the root (initial) state of the DAG.

# Arguments
- `dag` - Expected to be GFlowNetModel with initial_state field

# Returns
- `AbstractState` - The root/initial state
"""
function get_root_state(dag)
    if hasfield(typeof(dag), :initial_state)
        return dag.initial_state
    else
        throw(ArgumentError("DAG parameter must have initial_state field (expected GFlowNetModel)"))
    end
end

# =============================================================================
# Exports - Minimal Interface
# =============================================================================

# Core functions (used during training)
export get_applicable_actions, compute_next_state, is_valid_transition

# DAG compatibility functions (for flow computation)
export get_next_states, get_previous_states, get_root_state

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
