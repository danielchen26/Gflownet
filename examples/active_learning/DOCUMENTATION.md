# Active Learning with GFlowNets

This document provides a comprehensive explanation of the Active Learning example implemented using Generative Flow Networks (GFlowNets).

## Conceptual Overview

Active learning is a paradigm in machine learning where an algorithm intelligently selects the most informative data points to label or experiments to perform, optimizing the learning process with minimal resources. This approach is particularly valuable in domains where data acquisition is expensive, time-consuming, or resource-intensive, such as scientific experimentation, clinical trials, or sensor deployment.

This task is well-suited for GFlowNets because:

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

In practice, these information-theoretic measures are often approximated for computational efficiency.

### Balancing Exploration and Exploitation

A good active learning strategy must balance:
- **Exploitation:** Selecting experiments likely to yield high-value information
- **Exploration:** Ensuring diversity in the selected experiments

GFlowNets naturally handle this trade-off by sampling experiment sets with probability proportional to their reward, which incorporates both factors.

## Implementation Details

### Experiment State Representation

```julia
struct ExperimentState <: GFlowNet.AbstractState
    experiments::Vector{Int}  # Indices of selected experiments
    max_experiments::Int      # Maximum number of experiments allowed
    is_terminal::Bool         # Whether experiment selection is complete
end
```

### Action Types

```julia
struct SelectExperimentAction <: GFlowNet.AbstractAction
    experiment_idx::Int  # Index of experiment to select
end

struct TerminateExperimentSelectionAction <: GFlowNet.AbstractAction end
```

### Feature Representation

Each experiment has associated features, and we convert the state (selected experiments) to a feature vector for the neural network:

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

### Reward Calculation

The reward function evaluates the information value and diversity of the selected experiments:

```julia
function reward(state::ExperimentState, experiment_features::Matrix{Float64}, experiment_values::Vector{Float64})
    if !state.is_terminal || isempty(state.experiments)
        return 0.0
    end
    
    # Calculate information value (e.g., sum of values)
    total_value = sum(experiment_values[state.experiments])
    
    # Calculate diversity penalty
    diversity = calculate_diversity(state.experiments, experiment_features)
    
    # Combined reward with diversity bonus
    alpha = 0.8  # Weight for value vs. diversity
    reward = alpha * total_value + (1 - alpha) * diversity
    
    # Experimental cost penalty (fewer experiments is better if values are similar)
    cost_penalty = 0.95^length(state.experiments)
    
    return reward * cost_penalty
end
```

## Mathematical Derivation: Batch Active Learning

### Expected Information Gain

In active learning, we want to select experiments that maximize expected information gain. For a set of experiments $s$ with features $X_s$ and unknown values $Y_s$, the expected information gain is:

$$\text{EIG}(s) = H(Y) - \mathbb{E}_{y_s}[H(Y | Y_s = y_s)]$$

where $H$ denotes the entropy. This can be interpreted as the expected reduction in uncertainty about the target values after observing the outcomes of the selected experiments.

### Batch Selection with Submodularity

When selecting multiple experiments simultaneously (batch selection), we need to account for redundancy between experiments. This is captured by the submodularity property:

$$\text{Gain}(e | s) \geq \text{Gain}(e | s')$$

for all $s \subseteq s'$ and $e \notin s'$, where $\text{Gain}(e | s)$ is the marginal gain of adding experiment $e$ to set $s$.

Maximizing a submodular function is NP-hard, but greedy algorithms achieve a $(1-1/e)$ approximation. GFlowNets offer an alternative approach by sampling diverse, high-value experiment sets.

### Diversity Metrics

To ensure diversity in selected experiments, we use metrics like:

1. **Feature-based diversity:** Measures the coverage of the feature space
   $$\text{Div}_{\text{feature}}(s) = \sum_{i,j \in s, i < j} \|x_i - x_j\|_2$$

2. **Determinant-based diversity:** Uses the determinant of the kernel matrix
   $$\text{Div}_{\text{det}}(s) = \det(K_{s,s})$$
   where $K_{s,s}$ is the kernel matrix of selected experiments.

3. **Coverage-based diversity:** Measures how well the selected experiments cover the entire pool
   $$\text{Div}_{\text{coverage}}(s) = \sum_{j=1}^n \max_{i \in s} \text{sim}(i, j)$$
   where $\text{sim}(i, j)$ is a similarity measure between experiments $i$ and $j$.

## Principal Component Analysis for Feature Visualization

To visualize high-dimensional experiment features, we use Principal Component Analysis (PCA):

1. Given the feature matrix $X \in \mathbb{R}^{n \times d}$, compute the covariance matrix $\Sigma = \frac{1}{n} X^T X$

2. Find the eigenvectors and eigenvalues of $\Sigma$: $\Sigma v_i = \lambda_i v_i$

3. Select the top $k$ eigenvectors (principal components) corresponding to the largest eigenvalues

4. Project the data onto these components: $X_{\text{proj}} = X V_k$

This allows us to visualize the distribution of experiments in a lower-dimensional space and analyze the diversity of selected experiments.

## Intuitive Explanation

To understand active learning with GFlowNets intuitively:

1. Imagine you're a scientist with limited resources who needs to select a small set of experiments from a large pool of possibilities.

2. Each experiment has different properties (features) and potential information value, but you don't know the outcomes in advance.

3. You want to select experiments that:
   - Are likely to yield valuable information
   - Are diverse and cover different aspects of the phenomenon
   - Are complementary, avoiding redundant information

4. A traditional approach might select experiments one by one, greedily choosing the most informative at each step.

5. GFlowNets instead learn to sample sets of experiments, with probability proportional to their combined information value and diversity.

The advantage of the GFlowNet approach is that it:
- Considers the value of the entire set, not just individual experiments
- Balances exploitation (high expected value) and exploration (diversity)
- Provides multiple alternative experiment designs, not just a single solution

## Relation to Other Active Learning Methods

GFlowNets offer several advantages compared to other active learning approaches:

1. **Uncertainty Sampling:**
   - Traditional: Selects points with highest model uncertainty (e.g., entropy, margin)
   - GFlowNets: Consider entire sets and their joint information value, not just individual points

2. **Query-by-Committee:**
   - Traditional: Selects points where an ensemble of models disagrees
   - GFlowNets: Learn a distribution over informative experiment sets rather than using fixed heuristics

3. **Expected Model Change:**
   - Traditional: Selects points expected to cause largest model parameter updates
   - GFlowNets: Naturally incorporate expected information gain through the reward function

4. **Greedy Batch Selection:**
   - Traditional: Sequentially selects experiments, potentially missing global optima
   - GFlowNets: Consider complete experiment sets, better handling interdependencies

## Applications of Active Learning GFlowNets

Active learning GFlowNets have potential applications in:

1. **Drug Discovery:** Selecting which compounds to synthesize and test

2. **Materials Science:** Determining which material compositions to experiment with

3. **Environmental Monitoring:** Deciding where to place sensors for maximum information

4. **Clinical Trials:** Designing efficient experimental protocols with minimal patient burden

5. **Robotics:** Selecting informative actions for more efficient learning

## Advanced Concepts: Batch Acquisition Functions

In batch active learning, acquisition functions evaluate the quality of a set of experiments. Common acquisition functions include:

1. **Batch Expected Improvement (qEI):**
   $$\text{qEI}(s) = \mathbb{E}_s[\max(0, \min(y_s) - y_{\text{best}})]$$
   Where $y_{\text{best}}$ is the best observed value so far.

2. **Determinantal Point Process (DPP):**
   $$P(s) \propto \det(L_s)$$
   Where $L_s$ is a kernel matrix capturing both quality and diversity.

3. **Upper Confidence Bound (UCB):**
   $$\text{UCB}(s) = \sum_{i \in s} (\mu_i + \beta \sigma_i)$$
   Where $\mu_i$ and $\sigma_i$ are the predicted mean and standard deviation.

GFlowNets can be viewed as learning a sampling policy over sets, where the probability of sampling a set is proportional to its acquisition value.

## Conclusion

The Active Learning example demonstrates how GFlowNets can be used for experimental design and selecting informative experiments. By learning to sample experiment sets with probability proportional to their information value and diversity, GFlowNets provide a powerful framework for active learning that balances exploration and exploitation.

This approach is particularly valuable in scientific domains where experiments are costly or time-consuming, allowing researchers to make the most of limited resources and accelerate the discovery process. The ability to generate diverse, high-value experiment designs makes GFlowNets a promising tool for scientific exploration and knowledge discovery. 