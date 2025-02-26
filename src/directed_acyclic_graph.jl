using Graphs

# Don't import from GFlowNet since we're inside the module
# using .GFlowNet: AbstractState, AbstractAction, DirectedAcyclicGraph

"""
    create_dag(initial_state::S, terminal_states::Vector{S}, 
               terminal_sink::S, actions::Vector{A}) where {S<:AbstractState, A<:AbstractAction}

Create a DirectedAcyclicGraph from given states and actions.
"""
function create_dag(initial_state::S, terminal_states::Vector{S}, 
                   terminal_sink::S, actions::Vector{A}) where {S<:AbstractState, A<:AbstractAction}
    
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
    
    # Check for cycles
    if has_cycle(graph)
        error("The graph contains cycles and is not a valid DAG.")
    end
    
    return DirectedAcyclicGraph(graph, unique_states, actions, state_to_idx, 
                               initial_state, terminal_states, terminal_sink)
end

"""
    get_next_states(dag::DirectedAcyclicGraph, state::AbstractState)

Get all possible next states from the current state.
"""
function get_next_states(dag::DirectedAcyclicGraph, state::AbstractState)
    if !haskey(dag.state_to_idx, state)
        return []
    end
    
    state_idx = dag.state_to_idx[state]
    next_indices = outneighbors(dag.graph, state_idx)
    
    return [dag.states[idx] for idx in next_indices]
end

"""
    get_previous_states(dag::DirectedAcyclicGraph, state::AbstractState)

Get all possible previous states that lead to the current state.
"""
function get_previous_states(dag::DirectedAcyclicGraph, state::AbstractState)
    if !haskey(dag.state_to_idx, state)
        return []
    end
    
    state_idx = dag.state_to_idx[state]
    prev_indices = inneighbors(dag.graph, state_idx)
    
    return [dag.states[idx] for idx in prev_indices]
end

"""
    get_incoming_edges(dag::DirectedAcyclicGraph, state::AbstractState)

Get all incoming edges (source state, action) pairs for the given state.
"""
function get_incoming_edges(dag::DirectedAcyclicGraph, state::AbstractState)
    prev_states = get_previous_states(dag, state)
    
    edges = []
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
    get_outgoing_edges(dag::DirectedAcyclicGraph, state::AbstractState)

Get all outgoing edges (target state, action) pairs for the given state.
"""
function get_outgoing_edges(dag::DirectedAcyclicGraph, state::AbstractState)
    next_states = get_next_states(dag, state)
    
    edges = []
    for next_state in next_states
        for action in dag.actions
            if is_applicable(action, state) && apply_action(action, state) == next_state
                push!(edges, (next_state, action))
            end
        end
    end
    
    return edges
end

"""
    is_applicable(action::AbstractAction, state::AbstractState)

Check if an action is applicable to a state. Should be implemented by concrete types.
"""
function is_applicable end

"""
    apply_action(action::AbstractAction, state::AbstractState)

Apply an action to a state, returning the new state. Should be implemented by concrete types.
"""
function apply_action end 