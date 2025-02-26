# Causal Discovery with GFlowNets

Causal discovery is the task of identifying causal relationships between variables from observational data. It is a fundamental problem in fields ranging from economics and social sciences to biology and medicine. This task is particularly well-suited for GFlowNets because of the vast, discrete search space and the need to discover diverse, high-scoring causal structures.

## Why Use GFlowNets for Causal Discovery?

GFlowNets are particularly well-suited for causal discovery because:

1. The space of possible causal structures (directed acyclic graphs or DAGs) is vast and discrete
2. Causal structures can be built incrementally by adding edges
3. The scoring functions for evaluating causal models are often complex and non-differentiable
4. We want to discover diverse, high-scoring causal structures, not just a single "best" graph

In our implementation, we focus on learning the structure of a Bayesian Network (a DAG) that best explains observed data, using a score-based approach.

## Mathematical Framework

### Causal Structures as DAGs

A causal structure is represented as a Directed Acyclic Graph (DAG) $G = (V, E)$ where:
- $V$ is the set of nodes (variables)
- $E$ is the set of directed edges (causal relationships)

The acyclicity constraint ensures that no variable can be a cause of itself, either directly or indirectly.

### Scoring Function and Reward

The quality of a causal structure is assessed using a scoring function that measures how well the DAG explains the observed data. Common scoring functions include:

1. **Bayesian Information Criterion (BIC):**
   $$\text{BIC}(G|D) = \log P(D|G, \hat{\theta}) - \frac{d}{2} \log n$$
   where $D$ is the data, $\hat{\theta}$ are the maximum likelihood parameters, $d$ is the number of parameters, and $n$ is the sample size.

2. **Bayesian Dirichlet equivalent uniform (BDeu) score:**
   $$\text{BDeu}(G|D) = \log P(D|G) = \log \int P(D|G, \theta) P(\theta|G) d\theta$$
   which integrates over all possible parameters $\theta$.

The reward function $R(G)$ is derived from these scores, often with a monotonic transformation to ensure positivity:
$$R(G) = \exp(\alpha \cdot \text{Score}(G|D))$$
where $\alpha$ is a temperature parameter controlling the concentration of the target distribution.

## Implementation Details

```julia
struct DAGState <: GFlowNet.AbstractState
    adjacency_matrix::Matrix{Int}  # Binary adjacency matrix
    node_names::Vector{String}     # Names of variables
    is_terminal::Bool              # Whether DAG construction is complete
end

struct AddEdgeAction <: GFlowNet.AbstractAction
    from_node::String
    to_node::String
end

struct RemoveEdgeAction <: GFlowNet.AbstractAction
    from_node::String
    to_node::String
end

struct TerminateDAGAction <: GFlowNet.AbstractAction end
```

### Acyclicity Constraint

To maintain acyclicity, we check before applying an action that would add an edge:

```julia
function has_cycle(adjacency_matrix::Matrix{Int})
    n = size(adjacency_matrix, 1)
    visited = zeros(Bool, n)
    rec_stack = zeros(Bool, n)
    
    function dfs_has_cycle(node)
        visited[node] = true
        rec_stack[node] = true
        
        for neighbor in 1:n
            if adjacency_matrix[node, neighbor] == 1
                if !visited[neighbor]
                    if dfs_has_cycle(neighbor)
                        return true
                    end
                elseif rec_stack[neighbor]
                    return true
                end
            end
        end
        
        rec_stack[node] = false
        return false
    end
    
    for node in 1:n
        if !visited[node]
            if dfs_has_cycle(node)
                return true
            end
        end
    end
    
    return false
end
```

## Using the Causal Discovery Example

To run the causal discovery example:

```julia
julia examples/causal_discovery/causal_discovery.jl
```

The example will:
1. Generate synthetic data from a known causal structure
2. Train a GFlowNet to discover causal structures that explain the data
3. Visualize the discovered causal graphs
4. Compare the discovered graphs with the ground truth

## Applications

Causal discovery GFlowNets have potential applications in:

1. **Genomics:** Discovering gene regulatory networks from expression data
2. **Epidemiology:** Identifying risk factors and causal pathways for diseases
3. **Economics:** Understanding causal relationships between economic variables
4. **Neuroscience:** Mapping functional connections in the brain
5. **Climate science:** Disentangling causal factors in climate systems

## Advantages Over Other Causal Discovery Methods

GFlowNets offer several advantages compared to other causal discovery approaches:

1. **Constraint-based methods (e.g., PC algorithm):**
   - GFlowNets incorporate the quality of the fit to data, not just conditional independence tests
   - They handle uncertainty better and provide multiple plausible causal explanations

2. **Score-based methods (e.g., greedy equivalence search):**
   - GFlowNets explore the space more thoroughly, avoiding local optima
   - They provide a distribution over structures rather than a single point estimate

3. **Continuous optimization methods (e.g., NOTEARS):**
   - GFlowNets work directly with discrete graph structures, avoiding relaxation approximations
   - They naturally incorporate acyclicity constraints without penalty terms

## Handling Markov Equivalence Classes

Two DAGs are Markov equivalent if they encode the same conditional independence relationships. The set of Markov equivalent DAGs forms an equivalence class, represented by a Completed Partially Directed Acyclic Graph (CPDAG).

GFlowNets can be adapted to sample from the space of equivalence classes rather than individual DAGs:

$$P(E) \propto \sum_{G \in E} R(G)$$

where $E$ is an equivalence class.

## Further Reading

For a more detailed explanation of causal discovery with GFlowNets, see the comprehensive documentation in the examples directory.
