# Causal Discovery with GFlowNets

This document provides a comprehensive explanation of the Causal Discovery example implemented using Generative Flow Networks (GFlowNets).

## Conceptual Overview

Causal discovery is the task of identifying causal relationships between variables from observational data. It is a fundamental problem in fields ranging from economics and social sciences to biology and medicine. This task is particularly well-suited for GFlowNets because:

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

### State Space

The state space consists of all valid DAGs over a fixed set of variables. Each state $s$ in this space represents a particular DAG structure, which may be complete or partially constructed.

### Action Space

The action space for causal discovery includes:
- $\mathcal{A}_{\text{add\_edge}}$: Add a directed edge from one variable to another
- $\mathcal{A}_{\text{remove\_edge}}$: Remove an existing directed edge
- $\mathcal{A}_{\text{terminate}}$: Declare the DAG as complete

Each action must maintain the acyclicity constraint.

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

## GFlowNet for Causal Discovery

The GFlowNet learns a policy that samples causal structures with probability proportional to their score:
$$P(G) \propto R(G)$$

This approach has several advantages:
1. It can discover multiple plausible causal explanations, not just the highest-scoring one
2. It naturally handles the uncertainty inherent in causal discovery
3. It avoids getting stuck in local optima that plague greedy search methods

## Implementation Details

### DAG State Representation

```julia
struct DAGState <: GFlowNet.AbstractState
    adjacency_matrix::Matrix{Int}  # Binary adjacency matrix
    node_names::Vector{String}     # Names of variables
    is_terminal::Bool              # Whether DAG construction is complete
end
```

### Action Types

```julia
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

### Reward Calculation

The reward function evaluates how well the DAG explains the data:

```julia
function reward(state::DAGState, data::Matrix{Float64})
    if !state.is_terminal
        return 0.0
    end
    
    # Compute score (BIC or BDeu)
    score = compute_score(state.adjacency_matrix, data)
    
    # Transform score to positive reward
    return exp(score)
end
```

## Mathematical Derivation: Score-Based Causal Discovery

### Bayesian Approach to Causal Discovery

In a Bayesian framework, we want to compute the posterior probability of a DAG given data:

$$P(G|D) = \frac{P(D|G)P(G)}{P(D)}$$

The marginal likelihood $P(D|G)$ requires integrating over all possible parameters:

$$P(D|G) = \int_{\Theta_G} P(D|G, \theta_G)P(\theta_G|G)d\theta_G$$

For linear Gaussian models, this can be computed in closed form. For categorical variables, the BDeu score provides a closed-form solution under certain assumptions.

### Equivalence Classes and Markov Equivalence

Two DAGs are Markov equivalent if they encode the same conditional independence relationships. The set of Markov equivalent DAGs forms an equivalence class, represented by a Completed Partially Directed Acyclic Graph (CPDAG).

GFlowNets can be adapted to sample from the space of equivalence classes rather than individual DAGs:

$$P(E) \propto \sum_{G \in E} R(G)$$

where $E$ is an equivalence class.

### Incorporating Prior Knowledge

We can incorporate prior knowledge about causal relationships through the prior distribution $P(G)$:

$$P(G|D) \propto P(D|G)P(G)$$

In the GFlowNet framework, this modifies the reward function:

$$R(G) = \exp(\alpha \cdot \text{Score}(G|D)) \cdot P(G)$$

## Intuitive Explanation

To understand causal discovery with GFlowNets intuitively:

1. Imagine trying to figure out which factors influence a set of variables (e.g., which genes regulate others in a biological network).

2. There are many possible causal structures (DAGs) that could explain the observed data, but some fit better than others.

3. A traditional approach might use greedy search to find the highest-scoring structure, but this can miss important alternatives.

4. GFlowNets instead learn to sample causal structures with probability proportional to how well they explain the data.

5. This is like having multiple scientific hypotheses that all explain the observations well, rather than committing to a single explanation that might be wrong.

The advantage of the GFlowNet approach is that it:
- Captures uncertainty in causal discovery
- Finds diverse, high-scoring structures
- Avoids getting trapped in local optima
- Can incorporate prior knowledge naturally

## Relation to Other Causal Discovery Methods

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

## Applications of Causal Discovery GFlowNets

Causal discovery GFlowNets have potential applications in:

1. **Genomics:** Discovering gene regulatory networks from expression data

2. **Epidemiology:** Identifying risk factors and causal pathways for diseases

3. **Economics:** Understanding causal relationships between economic variables

4. **Neuroscience:** Mapping functional connections in the brain

5. **Climate science:** Disentangling causal factors in climate systems

## Advanced Concepts: Evaluating Causal Discovery Performance

### Structural Hamming Distance

To evaluate how close a discovered DAG is to the true causal structure, we use the Structural Hamming Distance (SHD):

$$\text{SHD}(G_1, G_2) = \text{number of edge additions, deletions, or reversals to transform } G_1 \text{ into } G_2$$

Lower SHD indicates better recovery of the true causal structure.

### Intervention Distribution Shift

For practical applications, we care about how well the discovered causal structure predicts the effects of interventions. This can be measured by the KL divergence between the true and predicted intervention distributions:

$$\text{IDS}(G_{\text{true}}, G_{\text{disc}}) = \sum_{X_i \in V} D_{KL}(P(V|do(X_i); G_{\text{true}}) || P(V|do(X_i); G_{\text{disc}}))$$

where $do(X_i)$ represents an intervention on variable $X_i$.

## Conclusion

The Causal Discovery example demonstrates the power of GFlowNets for tackling one of the most challenging problems in data science: inferring causal relationships from observational data. By sampling causal structures with probability proportional to their score, GFlowNets provide a comprehensive view of the causal uncertainty landscape, enabling more robust and reliable causal inference.

This approach aligns with the philosophy that in complex domains with limited data, it's often better to consider multiple plausible explanations rather than committing to a single model. GFlowNets provide a principled way to explore the space of causal hypotheses, potentially leading to new scientific discoveries and improved decision-making under uncertainty. 