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
    reward(state::DAGState, data::Matrix{Float64})

Calculate the reward for a DAG state based on how well it fits the observed data.
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

Default reward function when no data is provided.
"""
function reward(state::DAGState)
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
    create_dag_actions(n_nodes::Int)

Create a set of possible DAG building actions.
"""
function create_dag_actions(n_nodes::Int)
    actions = DAGAction[]

    # Add edge actions
    for i in 1:n_nodes
        for j in 1:n_nodes
            if i != j
                push!(actions, AddEdgeAction(i, j))
                push!(actions, RemoveEdgeAction(i, j))
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
