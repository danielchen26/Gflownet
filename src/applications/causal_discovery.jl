using ..GFlowNet: AbstractState, AbstractAction, state_to_features, is_applicable, apply_action, reward
using LinearAlgebra
using Statistics

"""
    DAGData

Data structure for causal DAG information. Used with composition pattern
to create domain-specific states.

# Fields
- `adjacency_matrix`: Matrix representing directed edges
- `node_names`: Vector of variable names
"""
struct DAGData
    adjacency_matrix::Matrix{Int}
    node_names::Vector{String}
end

"""
    DAGState <: AbstractState

State representation for causal DAGs in GFlowNets.
"""
struct DAGState <: AbstractState
    adjacency_matrix::Matrix{Bool}  # Binary adjacency matrix
    node_names::Vector{String}      # Names of variables
    is_terminal::Bool
end

"""
    Base.:(==)(a::DAGState, b::DAGState)
    Base.hash(state::DAGState, h::UInt)

Structural equality and hashing for DAG states.

REQUIRED, and the absence of these was not cosmetic. Julia's default `==` on a
struct is `===`, which compares fields by identity, and `adjacency_matrix` is a
`Matrix` -- so two states holding equal but distinct matrices compared UNEQUAL.
Every consumer that asks "is this the same state" therefore answered no. In
particular `is_valid_backward_transition` (policies.jl:560) decides parenthood by
`apply_action(action, parent) == child`, so it rejected genuinely valid
transitions and `backward_parent_states` came back EMPTY no matter what
`find_parent_for_action` returned. Measured before these methods existed, on the
3-node initial state s0 and a = AddEdgeAction(1, 2):

    apply_action(a, s0) == apply_action(a, s0)              false
    is_valid_backward_transition(s0, apply_action(a, s0))   false
    length(backward_parent_states(apply_action(a, s0), …))  0

`node_names` is part of the identity because `state_to_features` is not the whole
state: two DAGs over different variables are different states.
"""
Base.:(==)(a::DAGState, b::DAGState) =
    a.is_terminal == b.is_terminal &&
    a.adjacency_matrix == b.adjacency_matrix &&
    a.node_names == b.node_names

Base.hash(state::DAGState, h::UInt) =
    hash(state.is_terminal, hash(state.node_names, hash(state.adjacency_matrix, h)))

"""
    DAGAction <: AbstractAction

Action representation for causal DAG building.
"""
abstract type DAGAction <: AbstractAction end

"""
    AddEdgeAction <: DAGAction

Action to add a directed edge to the DAG.
"""
struct AddEdgeAction <: DAGAction
    from_node::Int
    to_node::Int
end

"""
    RemoveEdgeAction <: DAGAction

Action to remove an existing edge from the DAG.
"""
struct RemoveEdgeAction <: DAGAction
    from_node::Int
    to_node::Int
end

"""
    TerminateDAGAction <: DAGAction

Action to terminate DAG construction.
"""
struct TerminateDAGAction <: DAGAction end

# Implementation of required interface functions

"""
    is_applicable(action::AddEdgeAction, state::DAGState)

Check if adding an edge is valid.
"""
function is_applicable(action::AddEdgeAction, state::DAGState)
    # Cannot modify terminal states
    if state.is_terminal
        return false
    end

    n_nodes = size(state.adjacency_matrix, 1)

    # Check if node indices are valid
    if action.from_node < 1 || action.from_node > n_nodes ||
       action.to_node < 1 || action.to_node > n_nodes
        return false
    end

    # Cannot add self-loops
    if action.from_node == action.to_node
        return false
    end

    # Cannot add edge if it already exists
    if state.adjacency_matrix[action.from_node, action.to_node]
        return false
    end

    # Check if adding this edge would create a cycle
    temp_matrix = copy(state.adjacency_matrix)
    temp_matrix[action.from_node, action.to_node] = true

    return !has_cycle(temp_matrix)
end

"""
    is_applicable(action::RemoveEdgeAction, state::DAGState)

Check if removing an edge is valid.
"""
function is_applicable(action::RemoveEdgeAction, state::DAGState)
    # Cannot modify terminal states
    if state.is_terminal
        return false
    end

    n_nodes = size(state.adjacency_matrix, 1)

    # Check if node indices are valid
    if action.from_node < 1 || action.from_node > n_nodes ||
       action.to_node < 1 || action.to_node > n_nodes
        return false
    end

    # Can only remove existing edges
    return state.adjacency_matrix[action.from_node, action.to_node]
end

"""
    is_applicable(action::TerminateDAGAction, state::DAGState)

Check if termination is valid.
"""
function is_applicable(action::TerminateDAGAction, state::DAGState)
    # Can terminate if not already terminated
    return !state.is_terminal
end

"""
    apply_action(action::AddEdgeAction, state::DAGState)

Apply the action to add an edge.
"""
function apply_action(action::AddEdgeAction, state::DAGState)
    new_matrix = copy(state.adjacency_matrix)
    new_matrix[action.from_node, action.to_node] = true

    return DAGState(new_matrix, copy(state.node_names), false)
end

"""
    apply_action(action::RemoveEdgeAction, state::DAGState)

Apply the action to remove an edge.
"""
function apply_action(action::RemoveEdgeAction, state::DAGState)
    new_matrix = copy(state.adjacency_matrix)
    new_matrix[action.from_node, action.to_node] = false

    return DAGState(new_matrix, copy(state.node_names), false)
end

"""
    apply_action(action::TerminateDAGAction, state::DAGState)

Apply the action to terminate DAG construction.
"""
function apply_action(action::TerminateDAGAction, state::DAGState)
    return DAGState(copy(state.adjacency_matrix), copy(state.node_names), true)
end

"""
    GFlowNet.is_terminal_state(state::DAGState)

Whether this DAG is a terminal state.

REQUIRED interface method, and it was missing, so any use of this domain failed --
examples/causal_discovery/causal_discovery.jl had to define it locally to run at all.

Note the QUALIFIED name. This file's `using ..GFlowNet: ...` list does not import
`is_terminal_state`, so an unqualified definition would create a NEW function local
to this module instead of adding a method to GFlowNet's, and the package would never
call it -- a silent no-op. `grid_world.jl:141` is the working precedent.
"""
GFlowNet.is_terminal_state(state::DAGState)::Bool = state.is_terminal

"""
    state_to_features(state::DAGState)

Convert a DAG state to a feature vector.
"""
function state_to_features(state::DAGState)
    # Flatten the adjacency matrix
    flat_adjacency = vec(state.adjacency_matrix)

    # Add some graph statistics:
    # - Number of edges
    # - Sparsity
    # - Maximum indegree
    # - Maximum outdegree
    n_nodes = size(state.adjacency_matrix, 1)
    n_edges = sum(state.adjacency_matrix)
    sparsity = n_edges / (n_nodes * (n_nodes - 1))

    in_degrees = sum(state.adjacency_matrix, dims=1)[:]
    out_degrees = sum(state.adjacency_matrix, dims=2)[:]

    max_in_degree = maximum(in_degrees)
    max_out_degree = maximum(out_degrees)

    # Create feature vector
    features = [
        flat_adjacency;
        n_edges;
        sparsity;
        max_in_degree;
        max_out_degree;
        Int(state.is_terminal)
    ]

    return Float32.(features)
end

"""
    has_cycle(adjacency_matrix::Matrix{Bool})

Check if a directed graph has a cycle.
"""
function has_cycle(adjacency_matrix::Matrix{Bool})
    n = size(adjacency_matrix, 1)
    visited = falses(n)
    rec_stack = falses(n)

    for i in 1:n
        if !visited[i]
            if is_cyclic_util(adjacency_matrix, i, visited, rec_stack)
                return true
            end
        end
    end

    return false
end

"""
    is_cyclic_util(adjacency_matrix::Matrix{Bool}, v::Int, visited::Vector{Bool}, rec_stack::Vector{Bool})

Utility function for cycle detection using DFS.
"""
function is_cyclic_util(adjacency_matrix::Matrix{Bool}, v::Int, visited::AbstractVector{Bool}, rec_stack::AbstractVector{Bool})
    visited[v] = true
    rec_stack[v] = true

    # Visit all neighbors
    for i in 1:size(adjacency_matrix, 1)
        if adjacency_matrix[v, i]
            if !visited[i]
                if is_cyclic_util(adjacency_matrix, i, visited, rec_stack)
                    return true
                end
            elseif rec_stack[i]
                return true
            end
        end
    end

    rec_stack[v] = false
    return false
end

"""
    base_reward(state::DAGState)

Calculate the base reward for a DAG state based on structural properties.
This implements the required interface method.
"""
function base_reward(state::DAGState)
    if !state.is_terminal
        return 0.0
    end

    # Without data, we can use structural properties
    # Here we use a simple sparsity-based reward
    n_nodes = size(state.adjacency_matrix, 1)
    n_edges = sum(state.adjacency_matrix)

    # Penalize dense graphs
    sparsity_factor = exp(-n_edges / n_nodes)

    return sparsity_factor
end

"""
    reward(state::DAGState, data::Matrix{Float64})

Compute reward for a DAG state based on how well it explains the data.
This uses Bayesian Information Criterion (BIC) as a scoring function.
"""
function reward(state::DAGState, data::Matrix{Float64})
    if !state.is_terminal
        return 0.0
    end

    # Compute the Bayesian Information Criterion (BIC) score
    # This is a simplified version - in practice, you would want to use
    # proper Bayesian network scoring methods

    n_samples, n_vars = size(data)

    # Compute log-likelihood (simplified)
    # Assume Gaussian variables with DAG structure

    # First, standardize the data
    data_std = (data .- mean(data, dims=1)) ./ std(data, dims=1)

    # Compute residual variances based on the DAG structure
    residual_vars = ones(n_vars)
    log_likelihood = 0.0

    # For each variable, compute its residual variance given its parents
    for i in 1:n_vars
        # Find parents of node i
        parents = findall(state.adjacency_matrix[:, i])

        if isempty(parents)
            # No parents, use marginal variance (already standardized to 1)
            continue
        else
            # Fit a linear regression from parents to child
            X = data_std[:, parents]
            y = data_std[:, i]

            # Add intercept
            X = hcat(ones(n_samples), X)

            # Solve for regression coefficients
            # Using normal equations: β = (X^T X)^{-1} X^T y
            beta = pinv(X' * X) * X' * y

            # Compute residual variance
            residuals = y - X * beta
            residual_vars[i] = var(residuals)

            # Update log-likelihood
            log_likelihood -= 0.5 * n_samples * log(2π * residual_vars[i])
            log_likelihood -= 0.5 * sum(residuals .^ 2) / residual_vars[i]
        end
    end

    # Compute BIC
    # BIC = log-likelihood - (number of parameters) * log(n_samples) / 2
    n_params = sum(state.adjacency_matrix) # Number of edges
    bic = log_likelihood - n_params * log(n_samples) / 2

    # Return exp(BIC) to ensure positive rewards for GFlowNet
    return exp(bic / n_samples)  # Scale by n_samples to avoid numerical issues
end

"""
    reward(state::DAGState)

Compute reward for a DAG state using structural properties when no data is available.
This is the full reward interface for compatibility.
"""
function reward(state::DAGState)
    return base_reward(state)
end

"""
    create_dag_actions(n_nodes::Int; allow_edge_removal::Bool = false)

Create the DAG building action set: one `AddEdgeAction` per ordered node pair
plus `TerminateDAGAction`, and `RemoveEdgeAction` only when asked for.

ACYCLIC BY DEFAULT, which is what a GFlowNet requires of its state graph, and
the reason `allow_edge_removal` is opt-in rather than always on. Add-then-remove
is a 2-cycle: `apply_action(RemoveEdgeAction(1, 2), apply_action(AddEdgeAction(1, 2), s))`
returns `s`. Enumerating the 3-node space with removals included therefore has no
topological order and no finite path count -- the enumeration in
test/applications/causal_discovery/test_dag_parents.jl refuses it with
"state graph is not acyclic: (1 edge, non-terminal) -> (0 edges, non-terminal)".
With no path count there is no Z either: `sum_x n(x) R(x)` diverges, so nothing
the trainer converges to can be checked against a ground truth.

This mirrors `create_grid_world_gflownet`'s `allow_all_moves` (grid_world.jl:264):
`MoveLeft`/`MoveDown` exist and have backward parents, but the default action set
is "only up and right moves to prevent cycles", and every verified grid number
(Z = 19.0 on the 3x3) was measured on that acyclic set.

Measured cost of the old default on the 3-node instance, 1000 iterations at
batch 32, lr 0.005, z_learning_rate_multiplier 10, seed 20260828: learned
Z 30.88 against an enumerated Z_true of 13.667470 -- exactly the add-only
path-count-biased Z_PB1 = 30.864859, because Trajectory Balance can only satisfy
itself on a cyclic graph by driving every removal probability to zero, which
leaves the add-only paths and their multiplicity behind.
"""
function create_dag_actions(n_nodes::Int; allow_edge_removal::Bool = false)
    actions = DAGAction[]

    for i in 1:n_nodes
        for j in 1:n_nodes
            if i != j
                push!(actions, AddEdgeAction(i, j))
                allow_edge_removal && push!(actions, RemoveEdgeAction(i, j))
            end
        end
    end

    # Add terminate action
    push!(actions, TerminateDAGAction())

    return actions
end

"""
    create_initial_dag_state(n_nodes::Int, node_names::Vector{String}=String[])

Create the initial state for DAG building.
"""
function create_initial_dag_state(n_nodes::Int, node_names::Vector{String}=String[])
    # Empty adjacency matrix
    adjacency_matrix = falses(n_nodes, n_nodes)

    # Create default node names if not provided
    if isempty(node_names)
        node_names = ["X$i" for i in 1:n_nodes]
    end

    # Ensure we have exactly n_nodes names
    if length(node_names) != n_nodes
        throw(ArgumentError("Number of node names must match n_nodes"))
    end

    return DAGState(adjacency_matrix, node_names, false)
end

# =============================================================================
# Backward Sampling Support
# =============================================================================
#
# `backward_parent_states` (policies.jl:540) builds the parent set by asking
# `find_parent_for_action` once per action in the model's action set, and the
# DEFAULT for that hook (interface.jl:809) returns `nothing`. Only grid_world.jl
# and molecular_generation.jl overrode it, so this domain's parent set was empty,
# `n_parents > 1` in the Trajectory Balance loss (losses.jl:567) never fired, and
# log P_B stayed identically 0 -- P_B == 1, which is a distribution only where
# every state has a unique parent. A DAG with k edges has k of them.
#
# Note the QUALIFIED names, for the reason spelled out above
# `GFlowNet.is_terminal_state`: this file's `using ..GFlowNet: ...` list does not
# import `find_parent_for_action`.

"""
    GFlowNet.find_parent_for_action(target_state::DAGState, action::AddEdgeAction)

Inverse of `apply_action(::AddEdgeAction, ::DAGState)`: the parent is `target`
with that one edge deleted, and it exists only if `target` actually has the edge.

A DAG is the same state whichever order its edges arrived in, so a `target` with
k edges has exactly k parents this way -- one per edge. Each is reachable: every
subgraph of an acyclic graph is acyclic, so `is_applicable(AddEdgeAction(i, j), parent)`
holds for the deleted edge, which is what `backward_parent_states` re-checks.

`nothing` for a terminal `target`, because `apply_action(::AddEdgeAction, ·)`
always returns `is_terminal = false`; only `TerminateDAGAction` reaches a terminal
state.
"""
function GFlowNet.find_parent_for_action(target_state::DAGState, action::AddEdgeAction)
    target_state.is_terminal && return nothing

    n_nodes = size(target_state.adjacency_matrix, 1)
    (1 <= action.from_node <= n_nodes && 1 <= action.to_node <= n_nodes) || return nothing
    action.from_node == action.to_node && return nothing

    # The edge has to be present, or this action did not produce `target`.
    target_state.adjacency_matrix[action.from_node, action.to_node] || return nothing

    parent_matrix = copy(target_state.adjacency_matrix)
    parent_matrix[action.from_node, action.to_node] = false

    return DAGState(parent_matrix, copy(target_state.node_names), false)
end

"""
    GFlowNet.find_parent_for_action(target_state::DAGState, action::RemoveEdgeAction)

Inverse of `apply_action(::RemoveEdgeAction, ::DAGState)`: the parent is `target`
with that one edge added back, and it exists only if `target` lacks the edge.

Consulted ONLY when the caller opted into `create_dag_actions(n; allow_edge_removal = true)`;
the default action set contains no `RemoveEdgeAction`, so this method never runs
there. It is defined for the same reason grid_world.jl defines parents for
`MoveLeft`/`MoveDown` (grid_world.jl:466, :488), which its own default action set
also omits: an action that can occur must have its inverse enumerated, or P_B is
normalised over an incomplete parent set.

The acyclicity guard is NOT redundant. `is_applicable(::RemoveEdgeAction, ·)`
checks only that the edge is present, so `backward_parent_states` would accept a
cyclic parent -- yet `is_applicable(::AddEdgeAction, ·)` refuses every
cycle-creating edge, so no cyclic graph is reachable forward from
`create_initial_dag_state`. Handing one back would put P_B mass on a state no
trajectory can occupy.
"""
function GFlowNet.find_parent_for_action(target_state::DAGState, action::RemoveEdgeAction)
    target_state.is_terminal && return nothing

    n_nodes = size(target_state.adjacency_matrix, 1)
    (1 <= action.from_node <= n_nodes && 1 <= action.to_node <= n_nodes) || return nothing
    action.from_node == action.to_node && return nothing

    # The edge has to be absent, or this action did not produce `target`.
    target_state.adjacency_matrix[action.from_node, action.to_node] && return nothing

    parent_matrix = copy(target_state.adjacency_matrix)
    parent_matrix[action.from_node, action.to_node] = true
    has_cycle(parent_matrix) && return nothing

    return DAGState(parent_matrix, copy(target_state.node_names), false)
end

"""
    GFlowNet.find_parent_for_action(target_state::DAGState, action::TerminateDAGAction)

Inverse of `apply_action(::TerminateDAGAction, ::DAGState)`: the same graph, not
yet terminal. `nothing` for a non-terminal `target`, since terminating is the only
way into a terminal state and it never leaves one non-terminal. Mirrors
grid_world.jl:505.
"""
function GFlowNet.find_parent_for_action(target_state::DAGState, action::TerminateDAGAction)
    target_state.is_terminal || return nothing
    return DAGState(copy(target_state.adjacency_matrix), copy(target_state.node_names), false)
end
