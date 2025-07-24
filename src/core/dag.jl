using Graphs

# Don't import from GFlowNet since we're inside the module
# using .GFlowNet: AbstractState, AbstractAction, DirectedAcyclicGraph

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
initial = MoleculeState(MoleculeData([], []), false)
terminals = MoleculeState[]
sink = MoleculeState(MoleculeData([:SINK], []), true)
actions = [AddAtomAction(:C, (1.0, 1.0, 1.0)), TerminateMoleculeAction()]

dag = create_dag(initial, terminals, sink, actions)
```
"""
function create_dag(initial_state::S, terminal_states::Vector{S}, 
                   terminal_sink::S, actions::Vector{A}) where {S <: AbstractState, A <: AbstractAction}
    
    # Create an initial DAG with just the states
    all_states = [initial_state; terminal_states; [terminal_sink]]
    unique_states = unique(all_states)
    
    state_to_idx = Dict(state => i for (i, state) in enumerate(unique_states))
    
    # Initialize an empty graph
    graph = SimpleGraph(length(unique_states))
    
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
    
    # Check for cycles in the graph
    if is_cyclic(graph)
        error("The constructed graph contains cycles, which is not allowed for a DAG")
    end
    
    # Create the DAG structure
    return DirectedAcyclicGraph(
        graph,
        collect(unique_states),
        actions,
        state_to_idx,
        initial_state,
        terminal_states,
        terminal_sink
    )
end

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
function get_next_states(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}
    if !(state in dag.states)
        return AbstractState[]
    end
    
    state_idx = dag.state_to_idx[state]
    neighbors = outneighbors(dag.graph, state_idx)
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
function get_previous_states(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}
    if !(state in dag.states)
        return AbstractState[]
    end
    
    state_idx = dag.state_to_idx[state]
    neighbors = inneighbors(dag.graph, state_idx)
    return [dag.states[i] for i in neighbors]
end

"""
    get_incoming_edges(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}

Get all incoming edges to the given state, returning tuples of (source_state, action).
This is useful for implementing backward policies and for flow matching training.

# Arguments
- `dag`: The directed acyclic graph
- `state`: The target state

# Returns
- Vector of (source_state, action) tuples representing incoming edges
"""
function get_incoming_edges(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}
    prev_states = get_previous_states(dag, state)
    edges = Tuple{AbstractState, AbstractAction}[]
    
    for prev_state in prev_states
        for action in dag.actions
            if is_applicable(action, prev_state) && apply_action(action, prev_state) == state
                push!(edges, (prev_state, action))
            end
        end
    end
    
    return edges
end

"""
    get_outgoing_edges(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}

Get all outgoing edges from the given state, returning tuples of (action, target_state).
This is useful for implementing forward policies and exploring the state space.

# Arguments
- `dag`: The directed acyclic graph
- `state`: The source state

# Returns
- Vector of (action, target_state) tuples representing outgoing edges
"""
function get_outgoing_edges(dag::DirectedAcyclicGraph, state::S) where {S <: AbstractState}
    next_states = get_next_states(dag, state)
    edges = Tuple{AbstractAction, AbstractState}[]
    
    for action in dag.actions
        if is_applicable(action, state)
            next_state = apply_action(action, state)
            if next_state in next_states
                push!(edges, (action, next_state))
            end
        end
    end
    
    return edges
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

"""
    is_applicable(action::SimpleAction, state::SimpleState)

Check if a SimpleAction is applicable to a SimpleState.

For SimpleAction, we implement a simple rule-based system:
- Action value 0: Always applicable (represents "do nothing" or "terminate")
- Action value 1: Applicable if state data sum < 10 (represents "increment")
- Action value 2: Applicable if state data sum > 0 (represents "decrement")
- Action value -1: Always applicable (represents "terminate to sink")
"""
function is_applicable(action::SimpleAction, state::SimpleState)
    data_sum = sum(state.data)

    if action.value == 0
        return true  # "Do nothing" always applicable
    elseif action.value == 1
        return data_sum < 10  # "Increment" only if sum < 10
    elseif action.value == 2
        return data_sum > 0   # "Decrement" only if sum > 0
    elseif action.value == -1
        return true  # "Terminate" always applicable
    else
        return false  # Unknown action
    end
end

"""
    apply_action(action::SimpleAction, state::SimpleState)

Apply a SimpleAction to a SimpleState, returning the new state.

Action semantics:
- Action value 0: Return same state (no change)
- Action value 1: Increment first element by 1
- Action value 2: Decrement first element by 1
- Action value -1: Return terminal sink state ([-1])
"""
function apply_action(action::SimpleAction, state::SimpleState)
    if action.value == 0
        # No change
        return SimpleState(copy(state.data))
    elseif action.value == 1
        # Increment first element
        new_data = copy(state.data)
        if !isempty(new_data)
            new_data[1] += 1
        else
            new_data = [1]
        end
        return SimpleState(new_data)
    elseif action.value == 2
        # Decrement first element
        new_data = copy(state.data)
        if !isempty(new_data)
            new_data[1] -= 1
        else
            new_data = [-1]
        end
        return SimpleState(new_data)
    elseif action.value == -1
        # Terminate to sink
        return SimpleState([-1])
    else
        error("Unknown SimpleAction value: $(action.value)")
    end
end

# =============================================================================
# Basic DAG Manipulation Functions
# =============================================================================

"""
    add_state!(dag::DirectedAcyclicGraph, state)

Add a state to the DAG if it doesn't already exist.
This is a simple utility function for basic DAG construction.

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

    return idx
end

"""
    add_action!(dag::DirectedAcyclicGraph, action)

Add an action to the DAG if it doesn't already exist.
This is a simple utility function for basic DAG construction.

# Arguments
- `dag`: The directed acyclic graph to modify
- `action`: The action to add
"""
function add_action!(dag::DirectedAcyclicGraph, action)
    if !(action in dag.actions)
        push!(dag.actions, action)
    end
end