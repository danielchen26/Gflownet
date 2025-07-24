# Feature Acquisition with GFlowNets

This document provides a comprehensive explanation of the Feature Acquisition example implemented using Generative Flow Networks (GFlowNets), updated to reflect the modern training interface and latest capabilities.

## Quick Start

To run the feature acquisition example:

```bash
# From the project root
julia --project=. examples/feature_acquisition/main.jl

# Or from the feature_acquisition directory
julia --project=../.. main.jl
```

For comprehensive documentation on GFlowNets training and API, see [`docs/GFLOWNET_DOCUMENTATION.md`](../../docs/GFLOWNET_DOCUMENTATION.md).

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

### Why GFlowNets for Feature Acquisition

Feature acquisition is well-suited for GFlowNets because:

1. **Discrete Action Space**: The space of possible feature measurement combinations is vast and discrete
2. **Sequential Decision Making**: The value of measuring a feature depends on previously measured features
3. **Exploration vs. Exploitation**: We want to balance exploration (diverse measurements) and exploitation (informative features)
4. **Multi-objective Optimization**: The reward function can incorporate multiple objectives like information gain and measurement cost
5. **Diverse Solutions**: GFlowNets naturally provide diverse high-reward strategies

## Modern Training Interface

This example uses the modern GFlowNet training interface introduced in the latest version. Key components include:

### Training Configuration

```julia
# Modern training configuration
config = TrainingConfig(
    objective = :trajectory_balance,        # or :flow_consistency, :sub_trajectory_balance
    z_estimation = :adaptive,              # Partition function estimation
    n_iterations = 1000,
    batch_size = 32,
    learning_rate = 0.001,
    log_frequency = 50
)

# Train the model
model, metrics = train_gflownet(
    dag, policy, reward_fn, config;
    validation_fn = evaluate_policy
)
```

### Available Training Objectives

The example supports all modern GFlowNet training objectives:

1. **Trajectory Balance (TB)**:
   - `:trajectory_balance_simplified`: Standard TB without backward probabilities
   - `:trajectory_balance_general`: Full TB with backward probability terms

2. **Sub-Trajectory Balance (STB)**:
   - `:sub_trajectory_balance`: Standard STB
   - `:sub_trajectory_balance_hierarchical`: Hierarchical decomposition
   - `:sub_trajectory_balance_adaptive`: Adaptive sub-trajectory selection

3. **Flow Consistency (Unified)**:
   - `:flow_consistency`: Combines detailed balance and flow matching
   - Modes: `:edge_level`, `:state_level`, `:mixed`

### Partition Function (Z) Estimation Methods

The example demonstrates various Z estimation approaches:

1. **Simple Sum**: `z_estimation = :simple_sum`
2. **Learnable Parameter**: `z_estimation = :learnable`
3. **Sampling-based**: `z_estimation = :sampling`
4. **Adaptive Switching**: `z_estimation = :adaptive` (recommended)

## Example Setup and Problem Formulation

### Synthetic Data Generation

The example creates a synthetic experimental setup with configurable parameters:

```julia
# Default configuration
num_experiments = 10    # Number of experiments
num_features = 10      # Features per experiment
max_measurements = 5   # Maximum measurements allowed
feature_dim = 8        # Dimensionality of feature vectors
noise_level = 0.1      # Noise in experiment values
```

**Data Generation Process**:

1. **Feature Matrix Generation**:
```julia
# Generate random feature vectors for each experiment
features = randn(feature_dim, num_experiments)

# Generate feature importance weights
weights = normalize(abs.(randn(feature_dim)))

# Calculate true experiment values with noise
values = normalize(features' * weights + 0.1 * randn(num_experiments))
```

2. **Ground Truth Structure**:
- Each experiment has an intrinsic value determined by its feature vector
- Feature importance varies (controlled by weights)
- Gaussian noise represents experimental uncertainty

### Problem Structure

The feature acquisition task follows this structure:

1. **Initial State**:
   - No features measured: `observed_features = zeros(Bool, num_experiments, num_features)`
   - Full measurement budget available
   - All experiment values unknown

2. **Actions per Step**:
   - **Measure Feature**: Choose experiment `i` and feature `j` to measure
   - **Terminate**: End the acquisition process and receive reward

3. **Constraints**:
   - Cannot exceed measurement budget
   - Cannot re-measure already observed features
   - Must terminate before budget exhaustion

4. **Terminal Reward**:
   - Based on best identified experiment value minus measurement costs

### State Representation

```julia
struct FeatureAcquisitionState <: GFlowNet.AbstractState
    observed_features::Matrix{Bool}     # N×M matrix of measured features
    measurements_remaining::Int         # Remaining budget
    is_terminal::Bool                  # Terminal state flag
end

# Convert to neural network features
function state_to_features(state::FeatureAcquisitionState)
    # Flatten observation matrix
    obs_features = vec(Float32.(state.observed_features))
    
    # Add contextual information
    context = Float32[
        state.measurements_remaining / max_measurements,
        state.is_terminal ? 1.0 : 0.0,
        sum(state.observed_features) / (num_experiments * num_features)
    ]
    
    return vcat(obs_features, context)
end
```

### Action Space

```julia
# Measure a specific feature
struct MeasureFeatureAction <: GFlowNet.AbstractAction
    experiment_idx::Int
    feature_idx::Int
end

# End acquisition process
struct TerminateAction <: GFlowNet.AbstractAction end

# Generate valid actions for current state
function get_valid_actions(state::FeatureAcquisitionState)
    actions = GFlowNet.AbstractAction[]
    
    # Add measurement actions for unobserved features
    if state.measurements_remaining > 0 && !state.is_terminal
        for i in 1:size(state.observed_features, 1)
            for j in 1:size(state.observed_features, 2)
                if !state.observed_features[i, j]
                    push!(actions, MeasureFeatureAction(i, j))
                end
            end
        end
    end
    
    # Add termination action if measurements have been made
    if sum(state.observed_features) > 0 && !state.is_terminal
        push!(actions, TerminateAction())
    end
    
    return actions
end
```

## Mathematical Framework

### State Space Formulation

The state space consists of all valid measurement configurations:

$$\mathcal{S} = \{O \in \{0,1\}^{N \times M} : \sum_{i,j} O_{ij} \leq B, \text{terminal flag}\}$$

where:
- $N$ = number of experiments
- $M$ = number of features per experiment  
- $B$ = measurement budget
- $O_{ij} = 1$ if feature $j$ measured for experiment $i$

### Reward Function Design

The reward function balances discovery value against measurement costs:

$$R(s) = \mathbb{I}[\text{terminal}] \cdot \left( w_v \max_{i} v_i \cdot \mathbb{I}[\text{observed}_i] - w_c \sum_{i,j} O_{ij} \cdot c \right)$$

where:
- $v_i$ = true value of experiment $i$
- $\mathbb{I}[\text{observed}_i]$ = 1 if any feature of experiment $i$ is measured
- $w_v, w_c$ = value and cost weights
- $c$ = cost per measurement

### Modern Reward Implementation

```julia
struct FeatureAcquisitionReward <: GFlowNet.RewardFunction
    experiment_values::Vector{Float64}    # True experiment values
    cost_per_measurement::Float64         # Cost per feature measurement
    value_weight::Float64                # Importance of discovery value
    cost_weight::Float64                 # Importance of cost minimization
    
    function FeatureAcquisitionReward(values, cost=0.1, w_v=1.0, w_c=0.5)
        new(values, cost, w_v, w_c)
    end
end

function (reward_fn::FeatureAcquisitionReward)(state::FeatureAcquisitionState)
    if !state.is_terminal
        return 0.0
    end
    
    # Find best experiment with at least one measured feature
    best_value = 0.0
    for i in 1:size(state.observed_features, 1)
        if any(state.observed_features[i, :])
            best_value = max(best_value, reward_fn.experiment_values[i])
        end
    end
    
    # Calculate total measurement cost
    total_measurements = sum(state.observed_features)
    total_cost = total_measurements * reward_fn.cost_per_measurement
    
    # Weighted combination
    reward = reward_fn.value_weight * best_value - reward_fn.cost_weight * total_cost
    
    return max(0.0, reward)  # Ensure non-negative
end
```

## Training Process and Configuration

### Complete Training Example

```julia
using GFlowNet
using Random, Plots, DataFrames, CSV

function run_feature_acquisition_training()
    # 1. Generate synthetic data
    Random.seed!(42)
    num_experiments, num_features = 10, 10
    max_measurements = 5
    
    features = randn(8, num_experiments)
    weights = normalize(abs.(randn(8)))
    experiment_values = normalize(features' * weights + 0.1 * randn(num_experiments))
    
    # 2. Create DAG and components
    dag = FeatureAcquisitionDAG(num_experiments, num_features, max_measurements)
    reward_fn = FeatureAcquisitionReward(experiment_values, 0.1, 1.0, 0.5)
    
    # 3. Initialize policy network
    state_dim = num_experiments * num_features + 3  # +3 for context
    hidden_dim = 64
    policy = GFlowNet.create_policy(dag, state_dim, hidden_dim)
    
    # 4. Configure training
    config = TrainingConfig(
        objective = :trajectory_balance_simplified,
        z_estimation = :adaptive,
        n_iterations = 1000,
        batch_size = 32,
        learning_rate = 0.001,
        log_frequency = 50,
        validation_frequency = 100
    )
    
    # 5. Train model
    model, metrics = train_gflownet(
        dag, policy, reward_fn, config;
        validation_fn = (model, iter) -> evaluate_feature_acquisition(model, dag, reward_fn, 100)
    )
    
    return model, metrics, dag, reward_fn
end
```

### Training Objectives Comparison

You can easily compare different training objectives:

```julia
# Compare trajectory balance variants
configs = [
    TrainingConfig(objective = :trajectory_balance_simplified, n_iterations = 500),
    TrainingConfig(objective = :trajectory_balance_general, n_iterations = 500),
    TrainingConfig(objective = :sub_trajectory_balance, n_iterations = 500),
    TrainingConfig(objective = :flow_consistency, n_iterations = 500)
]

results = []
for (i, config) in enumerate(configs)
    println("Training with $(config.objective)...")
    model, metrics = train_gflownet(dag, policy, reward_fn, config)
    push!(results, (objective = config.objective, model = model, metrics = metrics))
end
```

### Partition Function Analysis

Different Z estimation methods can be compared:

```julia
z_methods = [:simple_sum, :learnable, :sampling, :adaptive]
z_results = []

for method in z_methods
    config = TrainingConfig(
        objective = :trajectory_balance_simplified,
        z_estimation = method,
        n_iterations = 500
    )
    
    model, metrics = train_gflownet(dag, policy, reward_fn, config)
    push!(z_results, (method = method, final_loss = metrics.losses[end]))
end
```

## Advanced Features and Extensions

### Handling Partially Observed Features

For scenarios where some features are already known:

```julia
struct PartialFeatureAcquisitionState <: GFlowNet.AbstractState
    observed_features::Matrix{Bool}      # Current observations
    initial_features::Matrix{Bool}       # Pre-existing observations
    measurements_remaining::Int          # Remaining budget
    is_terminal::Bool                   # Terminal flag
end

function create_initial_state_with_prior(
    num_experiments::Int, 
    num_features::Int, 
    max_measurements::Int,
    prior_ratio::Float64 = 0.2
)
    # Randomly select features to observe initially
    initial_obs = zeros(Bool, num_experiments, num_features)
    n_prior = floor(Int, num_experiments * num_features * prior_ratio)
    
    indices = randperm(num_experiments * num_features)[1:n_prior]
    for idx in indices
        i, j = divrem(idx - 1, num_features) .+ (1, 1)
        initial_obs[i, j] = true
    end
    
    return PartialFeatureAcquisitionState(
        copy(initial_obs),
        initial_obs,
        max_measurements - sum(initial_obs),
        false
    )
end
```

### Multi-Objective Reward Functions

Advanced reward functions can incorporate multiple objectives:

```julia
struct MultiObjectiveFeatureReward <: GFlowNet.RewardFunction
    experiment_values::Vector{Float64}
    feature_costs::Matrix{Float64}       # Variable costs per feature
    diversity_weight::Float64            # Encourage diverse measurements
    efficiency_weight::Float64           # Encourage efficient strategies
    
    function MultiObjectiveFeatureReward(values, costs, w_div=0.1, w_eff=0.2)
        new(values, costs, w_div, w_eff)
    end
end

function (reward_fn::MultiObjectiveFeatureReward)(state::FeatureAcquisitionState)
    if !state.is_terminal
        return 0.0
    end
    
    # Value component
    best_value = maximum(
        i -> any(state.observed_features[i, :]) ? reward_fn.experiment_values[i] : 0.0,
        1:size(state.observed_features, 1)
    )
    
    # Cost component (variable per feature)
    total_cost = sum(
        state.observed_features[i, j] * reward_fn.feature_costs[i, j]
        for i in 1:size(state.observed_features, 1)
        for j in 1:size(state.observed_features, 2)
    )
    
    # Diversity component (entropy of measurements across experiments)
    exp_counts = [sum(state.observed_features[i, :]) for i in 1:size(state.observed_features, 1)]
    diversity = -sum(p * log(p + 1e-8) for p in normalize(exp_counts .+ 1e-8))
    
    # Efficiency component (information per unit cost)
    efficiency = best_value / (total_cost + 1e-8)
    
    return best_value - total_cost + 
           reward_fn.diversity_weight * diversity + 
           reward_fn.efficiency_weight * efficiency
end
```

### Adaptive Budget Allocation

Dynamic budget allocation based on intermediate results:

```julia
struct AdaptiveBudgetDAG <: GFlowNet.AbstractDAG
    base_budget::Int
    bonus_threshold::Float64    # Value threshold for budget bonus
    bonus_amount::Int          # Additional measurements if threshold met
    
    # ... implementation details
end

function get_dynamic_budget(state, interim_value)
    base = state.base_budget
    if interim_value > state.bonus_threshold
        return base + state.bonus_amount
    end
    return base
end
```

## Evaluation and Analysis

### Comprehensive Evaluation Function

```julia
function evaluate_feature_acquisition(model, dag, reward_fn, n_samples=1000)
    trajectories = [sample_trajectory(model, dag) for _ in 1:n_samples]
    rewards = [reward_fn(t.final_state) for t in trajectories]
    
    # Performance metrics
    metrics = Dict(
        :mean_reward => mean(rewards),
        :max_reward => maximum(rewards),
        :std_reward => std(rewards),
        :success_rate => mean(rewards .> 0),
        :avg_measurements => mean(sum(t.final_state.observed_features) for t in trajectories),
        :feature_diversity => calculate_feature_diversity(trajectories),
        :experiment_coverage => calculate_experiment_coverage(trajectories)
    )
    
    return metrics
end

function calculate_feature_diversity(trajectories)
    # Measure how diverse the feature selection strategies are
    all_patterns = [vec(t.final_state.observed_features) for t in trajectories]
    pattern_counts = Dict()
    
    for pattern in all_patterns
        key = join(string.(Int.(pattern)))
        pattern_counts[key] = get(pattern_counts, key, 0) + 1
    end
    
    # Shannon entropy of patterns
    probs = collect(values(pattern_counts)) ./ length(trajectories)
    return -sum(p * log(p) for p in probs if p > 0)
end

function calculate_experiment_coverage(trajectories)
    # Measure how well we cover different experiments
    exp_counts = zeros(Int, size(trajectories[1].final_state.observed_features, 1))
    
    for traj in trajectories
        for i in 1:size(traj.final_state.observed_features, 1)
            if any(traj.final_state.observed_features[i, :])
                exp_counts[i] += 1
            end
        end
    end
    
    return exp_counts ./ length(trajectories)
end
```

### Visualization and Reporting

```julia
function generate_analysis_report(model, dag, reward_fn, experiment_values)
    # Sample evaluation trajectories
    n_eval = 1000
    trajectories = [sample_trajectory(model, dag) for _ in 1:n_eval]
    
    # Generate visualizations
    p1 = plot_reward_distribution(trajectories, reward_fn)
    p2 = plot_feature_selection_heatmap(trajectories)
    p3 = plot_value_vs_cost(trajectories, reward_fn, experiment_values)
    p4 = plot_strategy_comparison(trajectories, experiment_values)
    
    # Combine plots
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
    
    # Generate summary statistics
    metrics = evaluate_feature_acquisition(model, dag, reward_fn, n_eval)
    
    # Create report
    report = """
    # Feature Acquisition Analysis Report
    
    ## Performance Summary
    - Mean Reward: $(round(metrics[:mean_reward], digits=3))
    - Max Reward: $(round(metrics[:max_reward], digits=3))
    - Success Rate: $(round(metrics[:success_rate] * 100, digits=1))%
    - Average Measurements: $(round(metrics[:avg_measurements], digits=1))
    
    ## Strategy Analysis
    - Feature Diversity: $(round(metrics[:feature_diversity], digits=3))
    - Experiment Coverage: $(round(mean(metrics[:experiment_coverage]), digits=3))
    
    ## Ground Truth Comparison
    - Best Possible Value: $(round(maximum(experiment_values), digits=3))
    - Achievement Rate: $(round(metrics[:max_reward] / maximum(experiment_values) * 100, digits=1))%
    """
    
    return final_plot, report, metrics
end
```

## Applications and Use Cases

### 1. Scientific Experimentation

**Scenario**: Laboratory research with expensive assays
- **State**: Which experiments and assays have been performed
- **Actions**: Select next experiment-assay combination or stop
- **Reward**: Scientific value discovered minus experimental costs

```julia
# Example: Protein interaction studies
struct ProteinExperimentReward <: GFlowNet.RewardFunction
    interaction_strengths::Matrix{Float64}  # True protein-protein interactions
    assay_costs::Vector{Float64}           # Cost per assay type
    discovery_bonus::Float64               # Bonus for novel discoveries
end
```

### 2. Medical Diagnosis

**Scenario**: Optimal test ordering for patient diagnosis
- **State**: Which diagnostic tests have been performed
- **Actions**: Order additional tests or make diagnosis
- **Reward**: Diagnostic accuracy minus testing costs and patient burden

```julia
# Example: Multi-stage diagnostic protocol
struct DiagnosticReward <: GFlowNet.RewardFunction
    disease_probabilities::Vector{Float64}  # Prior disease probabilities
    test_sensitivities::Matrix{Float64}     # Test performance characteristics
    test_costs::Vector{Float64}             # Financial and time costs
    patient_discomfort::Vector{Float64}     # Patient burden per test
end
```

### 3. Quality Control

**Scenario**: Manufacturing quality assessment
- **State**: Which product features have been tested
- **Actions**: Test additional features or release product
- **Reward**: Quality assurance level minus testing costs

```julia
# Example: Pharmaceutical quality control
struct QualityControlReward <: GFlowNet.RewardFunction
    critical_attributes::Vector{String}     # Critical quality attributes
    test_reliabilities::Vector{Float64}     # Test reliability scores
    failure_costs::Float64                  # Cost of quality failure
    testing_throughput::Vector{Float64}     # Testing time requirements
end
```

## Version History and Migration

### Current Structure (Post-Modernization)

```
examples/feature_acquisition/
├── main.jl                    # Modern entry point using new training interface
├── DOCUMENTATION.md           # This comprehensive documentation
├── README.md                 # Quick start guide
├── archive/                  # Archived versions
│   ├── v1/                   # Original implementation
│   │   ├── main_v1.jl
│   │   └── v1_results/
│   ├── v2/                   # Enhanced version
│   │   ├── main_v2.jl
│   │   └── v2_results/
│   └── v3/                   # Experimental version
│       ├── main_v3.jl
│       └── v3_results/
└── Project.toml              # Dependencies
```

### Migration from Legacy Versions

If migrating from older versions:

1. **From V1/V2**: Replace manual training loops with `TrainingConfig` and `train_gflownet()`
2. **Objective Selection**: Choose from modern objectives (`:trajectory_balance_simplified`, `:flow_consistency`, etc.)
3. **Z Estimation**: Use `:adaptive` for automatic partition function estimation
4. **Reward Functions**: Implement `GFlowNet.RewardFunction` interface
5. **Validation**: Use built-in validation frequency instead of manual evaluation

### Breaking Changes

- **Training Interface**: Manual loss calculation replaced with declarative configuration
- **Reward Functions**: Must inherit from `GFlowNet.RewardFunction`
- **State Representation**: Enhanced state-to-features conversion required
- **Action Handling**: Improved action masking and validation

## Best Practices and Guidelines

### Parameter Tuning

1. **Start Small**: Begin with 5-10 experiments and features
2. **Balance Weights**: Typical ratios: `value_weight=1.0, cost_weight=0.3-0.7`
3. **Budget Setting**: Usually 20-50% of total possible measurements
4. **Learning Rate**: Start with 0.001, adjust based on loss convergence

### Training Tips

1. **Monitor Convergence**: Watch both loss and reward metrics
2. **Validation Frequency**: Evaluate every 50-100 iterations
3. **Batch Size**: 16-64 works well for most problem sizes
4. **Early Stopping**: Stop if validation rewards plateau

### Evaluation Standards

1. **Baseline Comparison**: Always compare against random selection
2. **Ground Truth Analysis**: Measure how often true best experiments are found
3. **Strategy Diversity**: Ensure the model learns varied approaches
4. **Cost-Effectiveness**: Analyze reward per unit cost spent

### Common Pitfalls

1. **Reward Scale**: Ensure rewards are positive and well-scaled
2. **Action Masking**: Properly handle invalid actions
3. **Terminal Conditions**: Clear criteria for when to stop
4. **Feature Normalization**: Normalize inputs to neural networks

## Integration with Main Codebase

This example demonstrates integration with the core GFlowNet framework:

- **Follows Modern Patterns**: Uses latest training interface and conventions
- **Comprehensive Coverage**: Demonstrates all major training objectives
- **Extensible Design**: Easy to adapt for domain-specific requirements
- **Documentation Standards**: Aligns with consolidated documentation approach

For more details on the GFlowNet framework, training objectives, and API reference, see:
- [`docs/GFLOWNET_DOCUMENTATION.md`](../../docs/GFLOWNET_DOCUMENTATION.md) - Comprehensive framework documentation
- [`examples/README.md`](../README.md) - Overview of all examples
- [Cursor Rules](../../.cursor/rules/) - Development standards and patterns

## Future Enhancements

### Planned Improvements

1. **Hierarchical Feature Selection**: Multi-level feature organization
2. **Online Learning**: Adaptation during deployment
3. **Constraint Handling**: Hard constraints on feature combinations
4. **Multi-Agent Scenarios**: Collaborative feature acquisition

### Research Directions

1. **Adaptive Budgeting**: Dynamic budget allocation based on intermediate results
2. **Transfer Learning**: Leveraging knowledge across similar domains
3. **Uncertainty Quantification**: Confidence intervals for feature value estimates
4. **Real-time Optimization**: Integration with laboratory automation systems

---

*This documentation reflects the modernized feature acquisition example using the latest GFlowNet training interface and capabilities. For questions or contributions, please refer to the main project documentation.* 