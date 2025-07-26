using Graphs

# Uses SimpleDiGraph for proper directed graph representation
# This ensures that edge directions are preserved, which is crucial for GFlowNets
# where forward and backward transitions have different semantics

"""
    create_dag(initial_state::S, terminal_states::Vector{S},
               terminal_sink::S, actions::Vector{A}) where {S <: AbstractState, A <: AbstractAction}

Create a DirectedAcyclicGraph from given states and actions.

This function constructs a graph representation of the state space for a GFlowNet.
The type parameters S and A allow for specialized implementations with concrete state
and action types, ensuring type stability and better performance.

# Arguments
- `initial_state::S`: The starting state for the GFlowNet
- `terminal_states::Vector{S}`: States representing completed objects
- `terminal_sink::S`: Special sink state that connects to all terminal states
- `actions::Vector{A}`: All possible actions in the GFlowNet

# Returns
- `DirectedAcyclicGraph{S,A}`: The constructed DAG with properly typed nodes and edges

# Example
```julia
# Using generic state and action types (defined in your domain)
initial = YourState(initial_data, false)
terminals = YourState[]
sink = YourState(terminal_data, true)
actions = [YourAction1(), YourAction2(), TerminateAction()]

dag = create_dag(initial, terminals, sink, actions)
```
"""
function create_dag(initial_state::S, terminal_states::Vector{S},
    terminal_sink::S, actions::Vector{A}) where {S<:AbstractState,A<:AbstractAction}

    # Create an initial DAG with just the states
    all_states = [initial_state; terminal_states; [terminal_sink]]
    unique_states = unique(all_states)

    state_to_idx = Dict(state => i for (i, state) in enumerate(unique_states))

    # Initialize an empty directed graph
    graph = SimpleDiGraph(length(unique_states))

    # Generate all possible transitions
    for action in actions
        for state in unique_states
            if is_applicable(action, state)
                next_state = apply_action(action, state)
                if next_state in unique_states
                    add_edge!(graph, state_to_idx[state], state_to_idx[next_state])
                end
            end
        end
    end

    # Add edges from terminal states to sink
    for term_state in terminal_states
        add_edge!(graph, state_to_idx[term_state], state_to_idx[terminal_sink])
    end

    # Comprehensive DAG validation
    validate_dag_construction(graph, unique_states, actions, initial_state, terminal_states, terminal_sink)

    # Build action cache for efficient lookups
    S_type = typeof(initial_state)
    A_type = eltype(actions)  # Use the abstract action type
    action_cache = Dict{S_type,Vector{A_type}}()

    for state in unique_states
        applicable_actions = A_type[action for action in actions if is_applicable(action, state)]
        action_cache[state] = applicable_actions
    end

    # Create the DAG structure
    return DirectedAcyclicGraph(
        graph,
        collect(unique_states),
        actions,
        state_to_idx,
        initial_state,
        terminal_states,
        terminal_sink,
        action_cache
    )
end

# =============================================================================
# =============================================================================
# Interface Methods
# =============================================================================

"""
    is_applicable(action::AbstractAction, state::AbstractState)

Check if an action can be applied to a given state.
This is a core interface method that must be implemented for each domain.

# Arguments
- `action::AbstractAction`: The action to check
- `state::AbstractState`: The state to check against

# Returns
- `Bool`: true if the action can be applied, false otherwise

# Implementation Note
This is a generic fallback that throws an error. Domain-specific implementations
should override this method for their concrete action and state types.

Implementation moved to src/core/interfaces.jl to avoid method overwriting.
"""

"""
    apply_action(action::AbstractAction, state::AbstractState)

Apply an action to a state and return the resulting new state.
This is a core interface method that must be implemented for each domain.

# Arguments
- `action::AbstractAction`: The action to apply
- `state::AbstractState`: The state to apply the action to

# Returns
- `AbstractState`: The new state after applying the action

# Implementation Note
This is a generic fallback that throws an error. Domain-specific implementations
should override this method for their concrete action and state types.

Implementation moved to src/core/interfaces.jl to avoid method overwriting.
"""

"""
    get_possible_actions(dag::DirectedAcyclicGraph, state::AbstractState)

Get all possible actions that can be applied from a given state.
This function uses the cached applicable actions for optimal performance.
Uses action cache for O(1) lookup instead of O(A) filtering.

# Arguments
- `dag::DirectedAcyclicGraph`: The DAG containing all possible actions
- `state::AbstractState`: The state to get actions for

# Returns
- `Vector{AbstractAction}`: Vector of applicable actions
"""
function get_possible_actions(dag::DirectedAcyclicGraph, state::AbstractState)
    # Use cached applicable actions instead of filtering all actions
    return get(dag.action_cache, state, eltype(dag.actions)[])
end

"""
    get_applicable_actions(dag::DirectedAcyclicGraph, state::AbstractState)

Get all actions applicable to a given state from the DAG. This is an alias for get_possible_actions.

# Arguments
- `dag::DirectedAcyclicGraph`: The DAG containing all possible actions
- `state::AbstractState`: The state to get actions for

# Returns
- `Vector{AbstractAction}`: Vector of applicable actions
"""
function get_applicable_actions(dag::DirectedAcyclicGraph, state::AbstractState)
    return get_possible_actions(dag, state)
end

"""
    is_terminal_state(state::AbstractState)

Check if a state is a terminal state.
This is a core interface method that must be implemented for each domain.

# Arguments
- `state::AbstractState`: The state to check

# Returns
- `Bool`: true if the state is terminal, false otherwise

# Implementation Note
This is a generic fallback that throws an error. Domain-specific implementations
should override this method for their concrete state types.
"""


"""
    get_next_states(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}

Get all possible next states from the current state in the DAG.
Returns a vector of states that can be reached by applying valid actions.

# Arguments
- `dag`: The directed acyclic graph
- `state`: The current state

# Returns
- Vector of states that can be reached from the current state
"""
function get_next_states(dag::DirectedAcyclicGraph, state::S) where {S<:AbstractState}
    if !(state in dag.states)
        return AbstractState[]
    end

    state_idx = dag.state_to_idx[state]
    neighbors = outneighbors(dag.graph, state_idx)  # Works correctly with SimpleDiGraph
    return [dag.states[i] for i in neighbors]
end

"""
    get_previous_states(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}

Get all possible previous states from the current state in the DAG.
Returns a vector of states that can lead to the current state.

# Arguments
- `dag`: The directed acyclic graph
- `state`: The current state

# Returns
- Vector of states that can lead to the current state
"""
function get_previous_states(dag::DirectedAcyclicGraph, state::S) where {S<:AbstractState}
    if !(state in dag.states)
        return AbstractState[]
    end

    state_idx = dag.state_to_idx[state]
    neighbors = inneighbors(dag.graph, state_idx)  # Works correctly with SimpleDiGraph
    return [dag.states[i] for i in neighbors]
end

"""
    get_incoming_edges(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}

Get all incoming edges to the given state, returning tuples of (source_state, action).
This is useful for implementing backward policies and for flow matching training.
Uses efficient action filtering instead of checking all actions.

# Arguments
- `dag`: The directed acyclic graph
- `state`: The target state

# Returns
- Vector of (source_state, action) tuples representing incoming edges
"""
function get_incoming_edges(dag::DirectedAcyclicGraph, state::S) where {S<:AbstractState}
    prev_states = get_previous_states(dag, state)
    edges = Tuple{AbstractState,AbstractAction}[]

    # Use cached applicable actions for maximum efficiency
    for prev_state in prev_states
        # Use pre-computed applicable actions from cache
        applicable_actions = get(dag.action_cache, prev_state, eltype(dag.actions)[])

        for action in applicable_actions
            if apply_action(action, prev_state) == state
                push!(edges, (prev_state, action))
                break  # Found the action that connects these states, move to next prev_state
            end
        end
    end

    return edges
end

"""
    get_outgoing_edges(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}

Get all outgoing edges from the given state, returning tuples of (action, target_state).
This is useful for implementing forward policies and exploring the state space.
Uses efficient action filtering and early termination.

# Arguments
- `dag`: The directed acyclic graph
- `state`: The source state

# Returns
- Vector of (action, target_state) tuples representing outgoing edges
"""
function get_outgoing_edges(dag::DirectedAcyclicGraph, state::S) where {S<:AbstractState}
    next_states = get_next_states(dag, state)
    edges = Tuple{AbstractAction,AbstractState}[]

    # Use cached applicable actions and Set for O(1) lookups
    applicable_actions = get(dag.action_cache, state, eltype(dag.actions)[])
    next_states_set = Set(next_states)  # O(1) lookup instead of O(n) with Vector

    for action in applicable_actions
        next_state = apply_action(action, state)
        if next_state in next_states_set
            push!(edges, (action, next_state))
        end
    end

    return edges
end

"""
    validate_dag_construction(graph, states, actions, initial_state, terminal_states, terminal_sink)

Comprehensive validation for DAG construction to ensure mathematical correctness.
Validates all aspects of DAG structure for GFlowNet requirements.

# Arguments
- `graph`: The constructed directed graph
- `states`: Vector of all states
- `actions`: Vector of all actions
- `initial_state`: Starting state
- `terminal_states`: Terminal states
- `terminal_sink`: Sink state

# Throws
- `ArgumentError` if DAG is invalid
"""
function validate_dag_construction(graph, states, actions, initial_state, terminal_states, terminal_sink)
    # Check for cycles in the directed graph
    if is_cyclic(graph)
        throw(ArgumentError("The constructed graph contains cycles, which is not allowed for a DAG"))
    end

    # Validate states
    if isempty(states)
        throw(ArgumentError("DAG must contain at least one state"))
    end

    if !(initial_state in states)
        throw(ArgumentError("Initial state must be in the states vector"))
    end

    if !(terminal_sink in states)
        throw(ArgumentError("Terminal sink must be in the states vector"))
    end

    for term_state in terminal_states
        if !(term_state in states)
            throw(ArgumentError("Terminal state $term_state must be in the states vector"))
        end
    end

    # Validate actions
    if isempty(actions)
        @warn "DAG has no actions - this may indicate an incomplete specification"
    end

    # Check connectivity
    if nv(graph) != length(states)
        throw(ArgumentError("Graph vertex count ($(nv(graph))) doesn't match state count ($(length(states)))"))
    end

    # Validate initial state has outgoing edges (unless it's also terminal)
    if initial_state ∉ terminal_states
        initial_idx = findfirst(s -> s == initial_state, states)
        if !isnothing(initial_idx) && isempty(outneighbors(graph, initial_idx))
            throw(ArgumentError("Initial state has no outgoing edges"))
        end
    end

    # Validate terminal states have incoming edges (unless they're also initial)
    for term_state in terminal_states
        if term_state != initial_state
            term_idx = findfirst(s -> s == term_state, states)
            if !isnothing(term_idx) && isempty(inneighbors(graph, term_idx))
                @warn "Terminal state $term_state has no incoming edges - may be unreachable"
            end
        end
    end

    # Check for isolated vertices (states with no connections)
    isolated_count = 0
    for i in 1:nv(graph)
        if isempty(inneighbors(graph, i)) && isempty(outneighbors(graph, i))
            isolated_count += 1
        end
    end

    if isolated_count > 0
        @warn "DAG contains $isolated_count isolated vertices (states with no connections)"
    end

    @debug "DAG validation passed: $(length(states)) states, $(ne(graph)) edges, $(length(actions)) actions"
end

"""
    is_applicable(action::AbstractAction, state::AbstractState)

Check if an action is applicable to a state. Should be implemented by concrete types.
This function is part of the required interface for all action types.
"""
function is_applicable end

"""
    apply_action(action::AbstractAction, state::AbstractState)

Apply an action to a state, returning the new state. Should be implemented by concrete types.
This function is part of the required interface for all action types.
"""
function apply_action end

# =============================================================================
# SimpleState/SimpleAction Implementation
# =============================================================================





# =============================================================================
# Basic DAG Manipulation Functions
# =============================================================================

"""
    add_state!(dag::DirectedAcyclicGraph, state)

Add a state to the DAG if it doesn't already exist.
This is a simple utility function for basic DAG construction.
Updates action cache when adding new states.

# Arguments
- `dag`: The directed acyclic graph to modify
- `state`: The state to add

# Returns
- Index of the state in the DAG
"""
function add_state!(dag::DirectedAcyclicGraph, state)
    if state in dag.states
        return dag.state_to_idx[state]
    end

    # Add state to the graph
    add_vertex!(dag.graph)
    push!(dag.states, state)

    # Update mapping
    idx = length(dag.states)
    dag.state_to_idx[state] = idx

    # Update action cache for the new state
    applicable_actions = [action for action in dag.actions if is_applicable(action, state)]
    dag.action_cache[state] = applicable_actions

    return idx
end

"""
    add_action!(dag::DirectedAcyclicGraph, action)

Add an action to the DAG if it doesn't already exist.
This is a simple utility function for basic DAG construction.
Updates action cache when adding new actions.

# Arguments
- `dag`: The directed acyclic graph to modify
- `action`: The action to add
"""
function add_action!(dag::DirectedAcyclicGraph, action)
    if !(action in dag.actions)
        push!(dag.actions, action)

        # Update action cache for all states when new action is added
        for state in dag.states
            if is_applicable(action, state)
                if !haskey(dag.action_cache, state)
                    dag.action_cache[state] = eltype(dag.actions)[]
                end
                push!(dag.action_cache[state], action)
            end
        end
    end
end
