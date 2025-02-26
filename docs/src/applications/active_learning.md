# Active Learning with GFlowNets

Active learning is a paradigm in machine learning where an algorithm intelligently selects the most informative data points to label or experiments to perform, optimizing the learning process with minimal resources. This approach is particularly valuable in domains where data acquisition is expensive, time-consuming, or resource-intensive, such as scientific experimentation, clinical trials, or sensor deployment.

## Why Use GFlowNets for Active Learning?

GFlowNets are particularly well-suited for active learning tasks because:

1. The space of possible experiment combinations is vast and discrete
2. The value of an experiment depends on the entire selected set, not just individual experiments
3. We want to balance exploration (diverse experiments) and exploitation (informative experiments)
4. The reward function can incorporate multiple objectives like information gain and cost

In our implementation, we focus on selecting a subset of experiments from a larger pool, where each experiment has associated features and an unknown value that we want to discover efficiently.

## Mathematical Framework

### Experimental Design State Space

The state space consists of all possible subsets of experiments that can be selected from a pool of $n$ available experiments. Each state $s$ in this space represents a collection of selected experiments, with a maximum of $k$ experiments allowed.

Formally, the state space is:
$$\mathcal{S} = \{s \subseteq \{1, 2, \ldots, n\} : |s| \leq k\}$$

### Action Space

The action space for active learning includes:
- $\mathcal{A}_{\text{select}}$: Select a specific experiment from the available pool
- $\mathcal{A}_{\text{terminate}}$: Declare the experiment selection as complete

### Information-Theoretic Reward Function

The key challenge in active learning is defining an appropriate reward function that captures the information value of experiments. Common approaches include:

1. **Expected Information Gain:**
   $$R_{\text{info}}(s) = H(Y) - \mathbb{E}_{y_s}[H(Y | Y_s = y_s)]$$
   where $H$ is the entropy and $Y_s$ represents the outcomes of experiments in set $s$.

2. **Mutual Information with Model Parameters:**
   $$R_{\text{MI}}(s) = I(Y_s; \Theta) = H(\Theta) - \mathbb{E}_{y_s}[H(\Theta | Y_s = y_s)]$$
   where $\Theta$ represents model parameters.

3. **Bayesian Experimental Design:**
   $$R_{\text{BED}}(s) = \mathbb{E}_{y_s}[D_{KL}(p(\theta | y_s, D) || p(\theta | D))]$$
   which measures the expected KL divergence between posterior and prior distributions.

## Implementation Details

```julia
struct ExperimentState <: GFlowNet.AbstractState
    experiments::Vector{Int}  # Indices of selected experiments
    max_experiments::Int      # Maximum number of experiments allowed
    is_terminal::Bool         # Whether experiment selection is complete
end

struct SelectExperimentAction <: GFlowNet.AbstractAction
    experiment_idx::Int  # Index of experiment to select
end

struct TerminateExperimentSelectionAction <: GFlowNet.AbstractAction end
```

### Feature Representation

```julia
function state_to_features(state::ExperimentState, experiment_features::Matrix{Float64})
    # Combine features of selected experiments
    n_experiments, feature_dim = size(experiment_features)
    
    # Create a binary mask of selected experiments
    selection_mask = zeros(Float32, n_experiments)
    selection_mask[state.experiments] .= 1.0
    
    # Concatenate the selection mask with other features
    features = vcat(
        selection_mask,
        [length(state.experiments) / state.max_experiments],  # Fill ratio
        [state.is_terminal ? 1.0 : 0.0]  # Terminal state flag
    )
    
    return features
end
```

## Using the Active Learning Example

To run the active learning example:

```julia
julia examples/active_learning/active_learning.jl
```

The example will:
1. Create a synthetic dataset of experiments with features and values
2. Train a GFlowNet to select informative and diverse experiments
3. Visualize the selected experiments in feature space
4. Compare the performance to random selection and greedy strategies

## Applications

Active learning GFlowNets have potential applications in:

1. **Drug Discovery:** Selecting which compounds to synthesize and test
2. **Materials Science:** Determining which material compositions to experiment with
3. **Environmental Monitoring:** Deciding where to place sensors for maximum information
4. **Clinical Trials:** Designing efficient experimental protocols with minimal patient burden
5. **Robotics:** Selecting informative actions for more efficient learning

## Further Reading

For a more detailed explanation of active learning with GFlowNets, see the comprehensive documentation in the examples directory.
