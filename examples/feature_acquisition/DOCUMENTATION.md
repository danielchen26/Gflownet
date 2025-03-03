# Feature Acquisition with GFlowNets

This document provides a comprehensive explanation of the Feature Acquisition example implemented using Generative Flow Networks (GFlowNets).

## Conceptual Overview

### Problem Domain and Motivation

Feature acquisition is a critical challenge in experimental design where we need to make strategic decisions about which measurements or tests to perform, particularly when these measurements are expensive, time-consuming, or resource-intensive. 

**Goal**: The primary objective is to identify high-value experiments while minimizing the total cost of measurements. This involves:
1. Deciding which features to measure for each experiment
2. Determining when to stop measuring features
3. Maximizing the information gained per measurement cost
4. Efficiently identifying the most promising experiments

### Real-World Application: Drug Discovery Example

Consider a pharmaceutical company developing new drug candidates:
- **Experiments**: Different drug compounds (e.g., 100 potential candidates)
- **Features**: Various tests that can be performed on each compound
  - Binding affinity tests (~$5,000 per test)
  - Toxicity assays (~$10,000 per test)
  - Solubility measurements (~$2,000 per test)
  - Metabolic stability tests (~$8,000 per test)
  - Bioavailability assessments (~$15,000 per test)

**Challenge**: With 100 compounds and 5 possible tests, running all tests on all compounds would cost $4 million. The company needs to:
- Identify the most promising drug candidates
- Minimize testing costs
- Make informed decisions about which tests to run
- Know when to stop testing a particular compound

**GFlowNet Solution**: Our feature acquisition system learns to:
1. Start with inexpensive, informative tests
2. Selectively perform additional tests on promising compounds
3. Skip unnecessary tests on less promising candidates
4. Automatically determine when enough information has been gathered

This pharmaceutical example illustrates why efficient feature acquisition is crucial:
- Each test is expensive
- Budget constraints prevent testing everything
- Early identification of promising candidates saves resources
- Some features may be more informative than others
- The value of a test may depend on previous test results

### Why GFlowNets for Feature Acquisition

Feature acquisition is well-suited for GFlowNets because:

1. The space of possible feature measurement combinations is vast and discrete
2. The value of measuring a feature depends on previously measured features
3. We want to balance exploration (diverse measurements) and exploitation (informative features)
4. The reward function can incorporate multiple objectives like information gain and measurement cost

In our implementation, we focus on selecting which features to measure for each experiment from a larger pool, where each experiment has multiple potential features and an unknown true value that we want to discover efficiently.

## Example Setup and Problem Formulation

### Synthetic Data Generation

In this example, we create a synthetic experimental setup with:
```julia
num_experiments = 10  # Number of experiments
num_features = 10     # Features per experiment
max_steps = 5        # Maximum measurements allowed
```

The synthetic data is generated as follows:

1. **Feature Matrix Generation**:
```julia
# Generate random features for each experiment
features = randn(feature_dim, n_experiments)

# Generate feature weights (importance of each feature)
weights = normalize(abs.(randn(feature_dim)))

# Calculate true experiment values
values = normalize(features' * weights + randn(n_experiments) * noise_level)
```

2. **Ground Truth Structure**:
- Each experiment has an intrinsic value determined by its features
- Some features are more informative than others (controlled by weights)
- Added noise represents experimental uncertainty

### Problem Structure

The feature acquisition task is structured as:

1. **Initial State**:
- No features measured
- Full measurement budget available
- All experiment values unknown

2. **Per Step**:
- Choose an experiment and feature to measure
- Pay a fixed cost (default: 0.1) per measurement
- Update knowledge about experiment values

3. **Terminal Condition**:
- Budget exhausted, or
- Agent decides to terminate
- Final reward based on best identified experiment

### Example Trajectory

Here's an example of how a trajectory might unfold:

```julia
# Initial state: No measurements
state = [
    0 0 0 0 0 0 0 0 0 0  # Experiment 1
    0 0 0 0 0 0 0 0 0 0  # Experiment 2
    ⋮
]

# Step 1: Measure feature 3 of experiment 1
state[1,3] = 1  # Cost: 0.1

# Step 2: Measure feature 7 of experiment 1
state[1,7] = 1  # Cost: 0.1

# Step 3: Measure feature 3 of experiment 4
state[4,3] = 1  # Cost: 0.1

# Terminal: Three measurements made
# Final reward = max_value_observed - total_cost
```

### Configuration Parameters

The example can be customized with several parameters:

```julia
# Data generation
num_features = 10        # Number of features per experiment
num_experiments = 10     # Number of experiments to generate
noise_level = 0.1       # Noise in experiment values

# Training
n_iterations = 100      # Training iterations
batch_size = 16        # Trajectories per batch
learning_rate = 0.001  # Optimizer learning rate

# Reward structure
cost_per_measurement = 0.1  # Cost for each feature measurement
value_weight = 1.0         # Weight for experiment value
cost_weight = 0.5         # Weight for measurement cost
```

### Evaluation Metrics

We evaluate the performance using:

1. **Value Discovery**:
   - Best experiment value found vs. true best value
   - Number of measurements needed to find good experiments

2. **Efficiency Metrics**:
   - Cost per unit value gained
   - Feature selection efficiency (informative vs. uninformative features)
   - Budget utilization

3. **Strategy Analysis**:
   - Feature selection patterns
   - Experiment prioritization
   - Termination decisions

## Mathematical Framework

### Feature Acquisition State Space

The state space consists of all possible combinations of measured features across experiments. Each state $s$ represents a binary matrix $O \in \{0,1\}^{N \times M}$ where:
- $N$ is the number of experiments
- $M$ is the number of features per experiment
- $O_{ij} = 1$ if feature $j$ has been measured for experiment $i$

Formally, the state space is:
$$\mathcal{S} = \{O \in \{0,1\}^{N \times M} : \sum_{i,j} O_{ij} \leq B\}$$
where $B$ is the measurement budget.

### Action Space

The action space for feature acquisition includes:
- $\mathcal{A}_{\text{measure}}$: Measure feature $j$ of experiment $i$
- $\mathcal{A}_{\text{terminate}}$: Declare the feature acquisition process complete

### Reward Function

The reward function balances information gain against measurement costs:

$$R(s) = \max_{i \in \{1,\ldots,N\}} v_i \cdot f(O_i) - c \sum_{i,j} O_{ij}$$

where:
- $v_i$ is the true value of experiment $i$
- $f(O_i)$ is a function that determines how well we can estimate $v_i$ from measured features
- $c$ is the cost per measurement
- $O_i$ is the vector of measured features for experiment $i$

## Implementation Details

### State Representation

```julia
struct FeatureAcquisitionState <: GFlowNet.AbstractState
    observed_features::Matrix{Bool}  # Which features have been measured
    measurements_remaining::Int      # Remaining measurement budget
    is_terminal::Bool               # Whether acquisition is complete
end
```

### Action Types

```julia
struct MeasureFeatureAction <: GFlowNet.AbstractAction
    experiment_idx::Int  # Which experiment to measure
    feature_idx::Int    # Which feature to measure
end

struct TerminateAction <: GFlowNet.AbstractAction end
```

### Custom Reward Type (Version 2)

```julia
struct FeatureAcquisitionReward <: GFlowNet.RewardFunction
    experiment_values::Vector{Float64}  # True values of experiments
    cost_per_measurement::Float64       # Cost of each measurement
    value_weight::Float64              # Weight for experiment value
    cost_weight::Float64              # Weight for measurement cost
end
```

### Feature Representation

The state is converted to a feature vector for the neural network:

```julia
function state_to_features(state::FeatureAcquisitionState)
    # Flatten the observed features matrix
    features = vec(Float32.(state.observed_features))
    
    # Add metadata features
    metadata = [
        state.measurements_remaining / total_budget,
        state.is_terminal ? 1.0 : 0.0
    ]
    
    return vcat(features, metadata)
end
```

### Reward Calculation

```julia
function calculate_reward(
    state::FeatureAcquisitionState,
    reward_fn::FeatureAcquisitionReward
)
    if !state.is_terminal
        return 0.0
    end
    
    # Calculate best experiment value
    best_value = 0.0
    for i in 1:size(state.observed_features, 1)
        if any(state.observed_features[i,:])
            best_value = max(best_value, reward_fn.experiment_values[i])
        end
    end
    
    # Calculate measurement cost
    total_cost = sum(state.observed_features) * reward_fn.cost_per_measurement
    
    # Combine value and cost with weights
    reward = reward_fn.value_weight * best_value - 
             reward_fn.cost_weight * total_cost
             
    return max(0.0, reward)  # Ensure non-negative reward
end
```

## Training Process

The training process involves:

1. **Initialization**:
   - Generate synthetic experiment data
   - Initialize neural network policy
   - Set up reward function and parameters

2. **Training Loop**:
   ```julia
   for iteration in 1:n_iterations
       # Sample trajectories
       trajectories = sample_trajectories(model, n_trajectories)
       
       # Calculate rewards and losses
       rewards = [calculate_reward(t.final_state, reward_fn) 
                 for t in trajectories]
       loss = gflownet_loss(trajectories, rewards)
       
       # Update model parameters
       update!(model, loss)
       
       # Log metrics
       log_training_metrics(iteration, loss, rewards)
   end
   ```

3. **Evaluation**:
   - Sample evaluation trajectories
   - Analyze feature selection patterns
   - Compare with ground truth values

## Visualization and Analysis

The implementation includes several visualization tools:

1. **Training Metrics**:
   - Loss convergence
   - Mean and max rewards
   - Feature selection frequency

2. **Strategy Analysis**:
   - Feature selection heatmaps
   - Value vs. cost trade-off
   - Ground truth comparison

3. **Reports**:
   - HTML summary
   - Markdown documentation
   - Performance metrics

## Applications

Feature acquisition with GFlowNets has potential applications in:

1. **Scientific Experimentation**:
   - Selecting which measurements to make in complex experiments
   - Optimizing experimental protocols
   - Reducing experimental costs

2. **Medical Diagnosis**:
   - Choosing which tests to run
   - Minimizing patient discomfort
   - Optimizing diagnostic accuracy

3. **Sensor Networks**:
   - Deciding which sensors to activate
   - Managing power consumption
   - Maximizing information gain

4. **Quality Control**:
   - Selecting which product features to test
   - Minimizing testing costs
   - Ensuring product quality

## Version Differences

### Version 1 (Original)

- Basic reward structure
- Simple state representation
- Limited configuration options

### Version 2 (Enhanced)

- Custom reward types
- Improved state representation
- More configuration options
- Better exploration strategy
- Enhanced visualization

## Best Practices

1. **Parameter Tuning**:
   - Start with small number of experiments/features
   - Adjust cost per measurement based on domain
   - Balance value and cost weights

2. **Training Tips**:
   - Monitor loss convergence
   - Check reward distribution
   - Validate feature selection patterns

3. **Evaluation**:
   - Compare with random selection
   - Analyze cost-effectiveness
   - Verify diversity of solutions

## References

1. Bengio, Y., et al. (2021). "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation"
2. Related work on experimental design and feature selection
3. GFlowNet documentation and tutorials

## Future Work

Potential improvements include:

1. **Adaptive Costs**:
   - Variable measurement costs
   - Context-dependent costs
   - Budget-aware strategies

2. **Multi-objective Optimization**:
   - Balance multiple reward components
   - Pareto-optimal strategies
   - Constraint satisfaction

3. **Real-world Applications**:
   - Integration with laboratory systems
   - Domain-specific constraints
   - Online learning capabilities 