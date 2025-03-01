# Directed Acyclic Graphs

This page documents the functionality related to Directed Acyclic Graphs (DAGs) in GFlowNet.jl. DAGs are the underlying data structure that represent state spaces and transitions in GFlowNets.

## Core DAG Functions

* `create_dag()`: Creates a new directed acyclic graph

## State and Edge Operations

* `add_node!(dag, node)`: Adds a node to the DAG
* `add_edge!(dag, source, target)`: Adds an edge to the DAG
* `remove_edge!(dag, source, target)`: Removes an edge from the DAG
* `has_edge(dag, source, target)`: Checks if an edge exists
* `get_nodes(dag)`: Returns all nodes in the DAG
* `get_edges(dag)`: Returns all edges in the DAG

## Graph Operations for Causal Discovery

* `is_acyclic(dag)`: Checks if the graph is acyclic
* `has_cycle(dag)`: Checks if the graph contains a cycle
* `add_edge!(dag, source, target)`: Adds an edge to the DAG while verifying acyclicity
* `remove_edge!(dag, source, target)`: Removes an edge from the DAG

## DirectedAcyclicGraph Structure

The `DirectedAcyclicGraph` type is the core data structure used to represent state spaces and transitions in GFlowNets. It encapsulates states, actions, and the transitions between states.

## Creating a DAG

DAGs can be created either explicitly by enumerating all states and transitions, or procedurally by defining rules for state transitions:

### Explicit Creation

```julia
function create_small_grid_dag()
    # Define states
    states = [
        GridState(1, 1, false),
        GridState(1, 2, false),
        GridState(2, 1, false),
        GridState(2, 2, false),
        GridState(1, 1, true),
        GridState(1, 2, true),
        GridState(2, 1, true),
        GridState(2, 2, true),
    ]
    
    # Define the initial state
    initial_state = states[1]  # (1,1) non-terminal
    
    # Define terminal states
    terminal_states = states[5:8]  # All terminal states
    
    # Define actions
    actions = [
        MoveRightAction(),
        MoveDownAction(),
        TerminateAction()
    ]
    
    # Create a dictionary mapping states to indices
    state_to_idx = Dict(state => i for (i, state) in enumerate(states))
    
    # Create an empty graph
    graph = SimpleGraph(length(states))
    
    # Add edges for MoveRight
    add_edge!(graph, state_to_idx[states[1]], state_to_idx[states[2]])  # (1,1) -> (1,2)
    add_edge!(graph, state_to_idx[states[3]], state_to_idx[states[4]])  # (2,1) -> (2,2)
    
    # Add edges for MoveDown
    add_edge!(graph, state_to_idx[states[1]], state_to_idx[states[3]])  # (1,1) -> (2,1)
    add_edge!(graph, state_to_idx[states[2]], state_to_idx[states[4]])  # (1,2) -> (2,2)
    
    # Add edges for Terminate
    for i in 1:4
        add_edge!(graph, state_to_idx[states[i]], state_to_idx[states[i+4]])
    end
    
    # Create a special terminal sink state
    terminal_sink = GridState(0, 0, true)
    push!(states, terminal_sink)
    state_to_idx[terminal_sink] = length(states)
    
    # Create the DAG
    return DirectedAcyclicGraph{GridState, typeof(actions[1])}(
        graph,
        states,
        actions,
        state_to_idx,
        initial_state,
        terminal_states,
        terminal_sink
    )
end
```

### Procedural Creation

```julia
function create_grid_dag(grid_size::Int)
    states = Vector{GridState}()
    actions = [
        MoveRightAction(),
        MoveLeftAction(),
        MoveUpAction(),
        MoveDownAction(),
        TerminateAction()
    ]
    
    # Create non-terminal states
    for x in 1:grid_size
        for y in 1:grid_size
            push!(states, GridState(x, y, false))
        end
    end
    
    # Create terminal states
    terminal_states = Vector{GridState}()
    for x in 1:grid_size
        for y in 1:grid_size
            terminal_state = GridState(x, y, true)
            push!(states, terminal_state)
            push!(terminal_states, terminal_state)
        end
    end
    
    # Set initial state
    initial_state = states[1]  # (1,1) non-terminal
    
    # Create state index mapping
    state_to_idx = Dict(state => i for (i, state) in enumerate(states))
    
    # Create graph
    graph = SimpleGraph(length(states))
    
    # Add edges for movement actions
    for state in states
        if state.is_terminal
            continue
        end
        
        x, y = state.x, state.y
        
        # Add edge for MoveRight
        if x < grid_size
            next_state = states[findfirst(s -> 
                s.x == x + 1 && s.y == y && !s.is_terminal, states)]
            add_edge!(graph, state_to_idx[state], state_to_idx[next_state])
        end
        
        # Add edge for MoveLeft
        if x > 1
            next_state = states[findfirst(s -> 
                s.x == x - 1 && s.y == y && !s.is_terminal, states)]
            add_edge!(graph, state_to_idx[state], state_to_idx[next_state])
        end
        
        # Add edge for MoveUp
        if y > 1
            next_state = states[findfirst(s -> 
                s.x == x && s.y == y - 1 && !s.is_terminal, states)]
            add_edge!(graph, state_to_idx[state], state_to_idx[next_state])
        end
        
        # Add edge for MoveDown
        if y < grid_size
            next_state = states[findfirst(s -> 
                s.x == x && s.y == y + 1 && !s.is_terminal, states)]
            add_edge!(graph, state_to_idx[state], state_to_idx[next_state])
        end
        
        # Add edge for Terminate
        terminal_state = states[findfirst(s -> 
            s.x == x && s.y == y && s.is_terminal, states)]
        add_edge!(graph, state_to_idx[state], state_to_idx[terminal_state])
    end
    
    # Create a special terminal sink state
    terminal_sink = GridState(0, 0, true)
    push!(states, terminal_sink)
    state_to_idx[terminal_sink] = length(states)
    
    # Create the DAG
    return DirectedAcyclicGraph{GridState, typeof(actions[1])}(
        graph,
        states,
        actions,
        state_to_idx,
        initial_state,
        terminal_states,
        terminal_sink
    )
end
```

## DAGs for Causal Discovery

In causal discovery applications, DAGs represent causal relationships between variables. The acyclicity constraint ensures there are no feedback loops in the causal structure.

```julia
function is_valid_causal_dag(adjacency_matrix::Matrix{Int})
    # Check if the graph is acyclic
    if !is_acyclic(adjacency_matrix)
        return false
    end
    
    # Additional validity checks can be added here
    
    return true
end

function add_causal_edge!(state::DAGState, from_node::Int, to_node::Int)
    # Create a copy of the adjacency matrix
    new_adjacency = copy(state.adjacency_matrix)
    
    # Add the edge
    new_adjacency[from_node, to_node] = 1
    
    # Check if adding the edge creates a cycle
    if is_acyclic(new_adjacency)
        state.adjacency_matrix = new_adjacency
        return true
    end
    
    return false
end
```

## Topological Sorting

Topological sorting is useful for many operations on DAGs, including inference in Bayesian networks:

```julia
function topological_sort(adjacency_matrix::Matrix{Int})
    n = size(adjacency_matrix, 1)
    visited = falses(n)
    temp = falses(n)
    order = Int[]
    
    function visit(node)
        if temp[node]
            # Graph has a cycle
            return false
        end
        
        if !visited[node]
            temp[node] = true
            
            # Visit children
            for child in 1:n
                if adjacency_matrix[node, child] == 1
                    if !visit(child)
                        return false
                    end
                end
            end
            
            temp[node] = false
            visited[node] = true
            pushfirst!(order, node)
        end
        
        return true
    end
    
    for node in 1:n
        if !visited[node]
            if !visit(node)
                return Int[] # Return empty array if graph has a cycle
            end
        end
    end
    
    return order
end
``` 