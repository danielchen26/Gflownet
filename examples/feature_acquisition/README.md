# Feature Acquisition Example

This example demonstrates how GFlowNets can be used for strategic feature acquisition in experimental design, optimizing which features to measure while balancing information gain and measurement costs.

## Features
- Simulates synthetic experimental data with multiple features
- Implements a feature acquisition state space
- Trains a GFlowNet to identify optimal feature measurement strategies
- Visualizes feature selection patterns and their effectiveness
- Analyzes the trade-off between information gain and measurement costs

## Running the Example
From the project root directory:
```julia
# Run version 1 (original implementation)
julia examples/feature_acquisition/run_v1.jl

# Run version 2 (enhanced implementation)
julia examples/feature_acquisition/run_v2.jl
```

## Output Files
The example generates several visualization files in the respective results directories:
- Training metrics and loss curves
- Strategy comparison plots
- Feature selection heatmaps
- Comprehensive HTML and Markdown reports

## Implementation Details
The example uses:
- Custom reward types for improved reward calculation
- Strategic feature selection with cost considerations
- Efficient state and action representations
- Comprehensive visualization and analysis tools

## Problem Setting

- **State**: Which features have been measured for each experiment
- **Actions**: Measure feature j of experiment i, or run an experiment
- **Reward**: Value of the best experiment performed minus the measurement cost

The GFlowNet learns to efficiently allocate the measurement budget to focus on the most informative features, particularly for high-value experiments.

## Implementation Versions

This example provides two implementations:

### Original Implementation (feature_acquisition.jl)

The original implementation demonstrates the core feature acquisition concept but may have issues with DAG creation and reward handling.

### Version 2 Implementation (feature_acquisition_v2.jl)

Version 2 addresses key challenges with:

1. **Improved DAG Creation**: Uses a step counter in the state representation to ensure that the DAG is truly acyclic. This prevents common issues with cycle detection in the GFlowNet state transition graph.

2. **Type-based Reward System**: Implements a concrete reward type `FeatureExperimentReward` that extends GFlowNet's `RewardFunction` type, providing a clean integration with the GFlowNet framework.

3. **Better Error Handling**: Includes comprehensive error handling and logging to help identify and troubleshoot issues.

## How It Works

1. We start with a partially observed feature matrix
2. The GFlowNet sequentially decides which feature to measure next
3. After each measurement, we update our understanding of the experiments
4. The process continues until the agent decides to terminate
5. Success is measured by how well we can identify high-value experiments with the limited measurements

## Visualization

The implementation includes basic visualization of training progress and results.

## Key Parameters

- `num_features`: Number of features per experiment
- `num_experiments`: Number of experiments available
- `cost_per_measurement`: Cost penalty for each feature observation
- `n_iterations`: Number of training iterations
- `n_trajectories`: Number of trajectories sampled per batch

## Acknowledgments

This example is part of the GFlowNet framework, which models generative processes as non-deterministic policies in MDPs. 