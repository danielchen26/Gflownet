# Generic DAG Construction System
# Provides robust, flexible DAG construction strategies for any GFlowNet domain

using Graphs
using DataStructures: Queue, Stack

"""
    AbstractDAGBuilder

Abstract base type for DAG construction strategies.
All DAG builders should implement the `build_dag` method.
"""
abstract type AbstractDAGBuilder end

"""
    DAGBuilderConfig

Configuration for DAG construction with sensible defaults.

# Fields
- `max_states::Int`: Maximum number of states to generate (default: 10000)
- `max_depth::Int`: Maximum depth for exploration (default: 100)
- `enable_pruning::Bool`: Enable state space pruning (default: true)
- `validate_construction::Bool`: Enable comprehensive validation (default: true)
- `exploration_strategy::Symbol`: :bfs or :dfs (default: :bfs)
- `cycle_detection_method::Symbol`: :strict or :optimistic (default: :strict)
"""
struct DAGBuilderConfig
    max_states::Int
    max_depth::Int
    enable_pruning::Bool
    validate_construction::Bool
    exploration_strategy::Symbol
    cycle_detection_method::Symbol

    function DAGBuilderConfig(;
        max_states::Int=10000,
        max_depth::Int=100,
        enable_pruning::Bool=true,
        validate_construction::Bool=true,
        exploration_strategy::Symbol=:bfs,
        cycle_detection_method::Symbol=:strict
    )
        @assert max_states > 0 "max_states must be positive"
        @assert max_depth > 0 "max_depth must be positive"
        @assert exploration_strategy in [:bfs, :dfs] "exploration_strategy must be :bfs or :dfs"
        @assert cycle_detection_method in [:strict, :optimistic] "cycle_detection_method must be :strict or :optimistic"

        new(max_states, max_depth, enable_pruning, validate_construction,
            exploration_strategy, cycle_detection_method)
    end
end

# =============================================================================
# Exploration-Based DAG Builder (RECOMMENDED for most use cases)
# =============================================================================

"""
    ExplorationDAGBuilder{S,A}

Generic DAG builder that explores the state space from an initial state.
This is the RECOMMENDED approach for most domains as it:
- Automatically discovers reachable states
- Prevents cycles by construction
- Handles infinite/large state spaces gracefully
- Provides proper topological ordering

# Type Parameters
- `S <: AbstractState`: State type
- `A <: AbstractAction`: Action type

# Usage
```julia
builder = ExplorationDAGBuilder(GridState, GridAction)
dag = build_dag(builder, initial_state, actions, config)
```
"""
struct ExplorationDAGBuilder{S<:AbstractState,A<:AbstractAction} <: AbstractDAGBuilder
    state_type::Type{S}
    action_type::Type{A}
end

"""
    build_dag(builder::ExplorationDAGBuilder, initial_state, actions, config=DAGBuilderConfig())

Build a DAG by exploring the state space from the initial state.
This method guarantees cycle-free construction and proper topological ordering.

# Arguments
- `builder`: ExplorationDAGBuilder instance
- `initial_state`: Starting state for exploration
- `actions`: Vector of all possible actions
- `config`: Configuration for construction (optional)

# Returns
- `DirectedAcyclicGraph`: Constructed DAG with all reachable states
"""
function build_dag(builder::ExplorationDAGBuilder{S,A},
    initial_state::S,
    actions::Vector{A},
    config::DAGBuilderConfig=DAGBuilderConfig()) where {S,A}

    # Initialize exploration data structures
    visited = Set{S}()
    states = S[]
    edges = Vector{Tuple{Int,Int}}()  # (from_idx, to_idx)
    state_to_idx = Dict{S,Int}()

    # Initialize exploration queue/stack based on strategy
    frontier = config.exploration_strategy == :bfs ? Queue{Tuple{S,Int}}() : Stack{Tuple{S,Int}}()

    # Add initial state
    push!(visited, initial_state)
    push!(states, initial_state)
    state_to_idx[initial_state] = 1
    enqueue!(frontier, (initial_state, 0))  # (state, depth)

    terminal_states = S[]

    println("🔍 Exploring state space with $(config.exploration_strategy) strategy...")

    # Main exploration loop
    while !isempty(frontier) && length(states) < config.max_states
        current_state, depth = dequeue!(frontier)
        current_idx = state_to_idx[current_state]

        # Check if this is a terminal state
        if is_terminal_state(current_state)
            push!(terminal_states, current_state)
            continue  # Don't explore from terminal states
        end

        # Depth limit check
        if depth >= config.max_depth
            @warn "Reached maximum depth $(config.max_depth) at state $current_state"
            continue
        end

        # Explore all applicable actions
        for action in actions
            if is_applicable(action, current_state)
                next_state = apply_action(action, current_state)

                # Check if we've seen this state before
                if !(next_state in visited)
                    # Add new state
                    push!(visited, next_state)
                    push!(states, next_state)
                    next_idx = length(states)
                    state_to_idx[next_state] = next_idx

                    # Add to frontier for further exploration
                    enqueue!(frontier, (next_state, depth + 1))

                    # Add edge
                    push!(edges, (current_idx, next_idx))
                else
                    # State already exists, just add edge
                    next_idx = state_to_idx[next_state]

                    # Cycle detection
                    if config.cycle_detection_method == :strict
                        # Check if adding this edge would create a cycle
                        if !_would_create_cycle(edges, current_idx, next_idx, length(states))
                            push!(edges, (current_idx, next_idx))
                        else
                            @debug "Skipping edge $current_idx -> $next_idx to prevent cycle"
                        end
                    else
                        # Optimistic: allow edge and check later
                        push!(edges, (current_idx, next_idx))
                    end
                end
            end
        end
    end

    # Create special terminal sink if needed
    terminal_sink = _create_terminal_sink(S, terminal_states)
    if terminal_sink !== nothing
        push!(states, terminal_sink)
        sink_idx = length(states)
        state_to_idx[terminal_sink] = sink_idx

        # Connect all terminal states to sink
        for term_state in terminal_states
            term_idx = state_to_idx[term_state]
            push!(edges, (term_idx, sink_idx))
        end
    end

    println("✅ Exploration complete: $(length(states)) states, $(length(edges)) edges")

    # Build the actual graph
    dag = _build_graph_from_exploration(states, edges, actions, initial_state,
        terminal_states, terminal_sink, config)

    return dag
end

# =============================================================================
# Explicit DAG Builder (for pre-computed state spaces)
# =============================================================================

"""
    ExplicitDAGBuilder{S,A}

DAG builder for cases where all states are pre-computed.
Use this when you have explicit control over the state space.

# Usage
```julia
builder = ExplicitDAGBuilder(GridState, GridAction)
dag = build_dag(builder, all_states, actions, initial_state, terminal_states, terminal_sink)
```
"""
struct ExplicitDAGBuilder{S<:AbstractState,A<:AbstractAction} <: AbstractDAGBuilder
    state_type::Type{S}
    action_type::Type{A}
end

"""
    build_dag(builder::ExplicitDAGBuilder, all_states, actions, initial_state, terminal_states, terminal_sink)

Build a DAG from explicitly provided states.
This method provides full control but requires manual state enumeration.
"""
function build_dag(builder::ExplicitDAGBuilder{S,A},
    all_states::Vector{S},
    actions::Vector{A},
    initial_state::S,
    terminal_states::Vector{S},
    terminal_sink::S,
    config::DAGBuilderConfig=DAGBuilderConfig()) where {S,A}

    # This follows the original create_dag logic but with better validation
    unique_states = unique(all_states)
    state_to_idx = Dict(state => i for (i, state) in enumerate(unique_states))

    # Initialize graph
    graph = SimpleDiGraph(length(unique_states))

    # Generate all possible transitions
    for action in actions
        for state in unique_states
            if is_applicable(action, state)
                next_state = apply_action(action, state)
                if next_state in unique_states
                    from_idx = state_to_idx[state]
                    to_idx = state_to_idx[next_state]

                    if config.cycle_detection_method == :strict
                        # Check for cycles before adding edge
                        if !_would_create_cycle_in_graph(graph, from_idx, to_idx)
                            add_edge!(graph, from_idx, to_idx)
                        end
                    else
                        add_edge!(graph, from_idx, to_idx)
                    end
                end
            end
        end
    end

    # Add edges from terminal states to sink
    if terminal_sink in unique_states
        sink_idx = state_to_idx[terminal_sink]
        for term_state in terminal_states
            if term_state in unique_states
                term_idx = state_to_idx[term_state]
                add_edge!(graph, term_idx, sink_idx)
            end
        end
    end

    # Validation
    if config.validate_construction
        validate_dag_construction(graph, unique_states, actions, initial_state, terminal_states, terminal_sink)
    end

    # Build action cache
    action_cache = _build_action_cache(unique_states, actions)

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
# Helper Functions
# =============================================================================

"""
    _would_create_cycle(edges, from_idx, to_idx, num_states)

Check if adding an edge would create a cycle using DFS.
"""
function _would_create_cycle(edges::Vector{Tuple{Int,Int}}, from_idx::Int, to_idx::Int, num_states::Int)
    # Build temporary adjacency list
    adj = [Int[] for _ in 1:num_states]
    for (f, t) in edges
        push!(adj[f], t)
    end

    # Check if there's already a path from to_idx to from_idx
    return _has_path_dfs(adj, to_idx, from_idx, Set{Int}())
end

"""
    _would_create_cycle_in_graph(graph, from_idx, to_idx)

Check if adding an edge to an existing graph would create a cycle.
"""
function _would_create_cycle_in_graph(graph::SimpleDiGraph, from_idx::Int, to_idx::Int)
    # Use Graphs.jl's built-in cycle detection
    # Check if there's a path from to_idx to from_idx
    return has_path(graph, to_idx, from_idx)
end

"""
    _has_path_dfs(adj, start, target, visited)

DFS to check if there's a path from start to target.
"""
function _has_path_dfs(adj::Vector{Vector{Int}}, start::Int, target::Int, visited::Set{Int})
    if start == target
        return true
    end

    if start in visited
        return false
    end

    push!(visited, start)

    for neighbor in adj[start]
        if _has_path_dfs(adj, neighbor, target, visited)
            return true
        end
    end

    return false
end

"""
    _create_terminal_sink(state_type, terminal_states)

Create a special terminal sink state if needed.
Returns nothing if the domain doesn't support automatic sink creation.
"""
function _create_terminal_sink(::Type{S}, terminal_states::Vector{S}) where {S}
    # This is domain-specific logic that could be overridden
    # For now, return nothing to indicate no automatic sink creation
    return nothing
end

"""
    _build_graph_from_exploration(states, edges, actions, initial_state, terminal_states, terminal_sink, config)

Build the final DirectedAcyclicGraph from exploration results.
"""
function _build_graph_from_exploration(states::Vector{S},
    edges::Vector{Tuple{Int,Int}},
    actions::Vector{A},
    initial_state::S,
    terminal_states::Vector{S},
    terminal_sink::Union{S,Nothing},
    config::DAGBuilderConfig) where {S,A}

    # Create state-to-index mapping
    state_to_idx = Dict(state => i for (i, state) in enumerate(states))

    # Build graph
    graph = SimpleDiGraph(length(states))
    for (from_idx, to_idx) in edges
        add_edge!(graph, from_idx, to_idx)
    end

    # Final cycle check if using optimistic method
    if config.cycle_detection_method == :optimistic && is_cyclic(graph)
        error("Constructed graph contains cycles. Consider using :strict cycle detection.")
    end

    # Validation
    if config.validate_construction && terminal_sink !== nothing
        validate_dag_construction(graph, states, actions, initial_state, terminal_states, terminal_sink)
    end

    # Build action cache
    action_cache = _build_action_cache(states, actions)

    return DirectedAcyclicGraph(
        graph,
        states,
        actions,
        state_to_idx,
        initial_state,
        terminal_states,
        terminal_sink !== nothing ? terminal_sink : initial_state,  # fallback
        action_cache
    )
end

"""
    _build_action_cache(states, actions)

Build cache of applicable actions for each state.
"""
function _build_action_cache(states::Vector{S}, actions::Vector{A}) where {S,A}
    cache = Dict{S,Vector{A}}()
    for state in states
        applicable_actions = A[action for action in actions if is_applicable(action, state)]
        cache[state] = applicable_actions
    end
    return cache
end

# =============================================================================
# High-Level API Functions
# =============================================================================

"""
    create_dag_with_exploration(initial_state::S, actions::Vector{A}, config=DAGBuilderConfig()) where {S,A}

High-level function to create a DAG using exploration-based construction.
This is the RECOMMENDED approach for most use cases.

# Example
```julia
# Simple usage with defaults
dag = create_dag_with_exploration(initial_state, actions)

# With custom configuration
config = DAGBuilderConfig(max_states=5000, exploration_strategy=:dfs)
dag = create_dag_with_exploration(initial_state, actions, config)
```
"""
function create_dag_with_exploration(initial_state::S, actions::Vector{A},
    config::DAGBuilderConfig=DAGBuilderConfig()) where {S,A}
    builder = ExplorationDAGBuilder(S, A)
    return build_dag(builder, initial_state, actions, config)
end

"""
    create_dag_from_states(all_states::Vector{S}, actions::Vector{A}, initial_state::S,
                          terminal_states::Vector{S}, terminal_sink::S,
                          config=DAGBuilderConfig()) where {S,A}

High-level function to create a DAG from explicitly provided states.
Use this when you have full control over the state space.
"""
function create_dag_from_states(all_states::Vector{S}, actions::Vector{A},
    initial_state::S, terminal_states::Vector{S},
    terminal_sink::S, config::DAGBuilderConfig=DAGBuilderConfig()) where {S,A}
    builder = ExplicitDAGBuilder(S, A)
    return build_dag(builder, all_states, actions, initial_state, terminal_states, terminal_sink, config)
end

# =============================================================================
# Analysis and Optimization Utilities
# =============================================================================

"""
    analyze_dag(dag::DirectedAcyclicGraph)

Analyze DAG properties and return metrics.
"""
function analyze_dag(dag::DirectedAcyclicGraph)
    graph = dag.graph
    n_states = nv(graph)
    n_edges = ne(graph)

    # Compute basic metrics
    avg_degree = n_edges / n_states
    max_in_degree = maximum(indegree(graph, v) for v in vertices(graph))
    max_out_degree = maximum(outdegree(graph, v) for v in vertices(graph))

    # Topological properties
    longest_path = _compute_longest_path(graph)

    return (
        n_states=n_states,
        n_edges=n_edges,
        avg_degree=avg_degree,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        longest_path=longest_path,
        is_connected=is_connected(graph),
        is_acyclic=!is_cyclic(graph)
    )
end

"""
    _compute_longest_path(graph)

Compute the length of the longest path in the DAG.
"""
function _compute_longest_path(graph::SimpleDiGraph)
    # Use topological sort and dynamic programming
    try
        topo_order = topological_sort_by_dfs(graph)
        distances = zeros(Int, nv(graph))

        for v in topo_order
            for u in inneighbors(graph, v)
                distances[v] = max(distances[v], distances[u] + 1)
            end
        end

        return maximum(distances)
    catch
        return -1  # Graph is not acyclic
    end
end

"""
    optimize_dag(dag::DirectedAcyclicGraph)

Apply optimizations to reduce DAG size while preserving essential structure.
"""
function optimize_dag(dag::DirectedAcyclicGraph)
    # This could include:
    # - Removing unreachable states
    # - Merging equivalent states
    # - Pruning low-probability paths
    # For now, return the original DAG
    @warn "DAG optimization not yet implemented"
    return dag
end
