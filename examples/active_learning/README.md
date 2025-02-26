# Active Learning Example

This example demonstrates how GFlowNets can be used for experimental design and selecting informative experiments.

## Features
- Simulates synthetic experiment data
- Implements an experiment selection state space
- Trains a GFlowNet to identify informative experiment combinations
- Visualizes the selected experiments and their properties
- Analyzes the diversity of selected experiments using PCA

## Running the Example
From the project root directory:
```julia
julia examples/active_learning/active_learning.jl
```

## Output Files
The example generates several visualization files:
- `active_learning_loss.png`: Training loss curve
- `active_learning_best_selection.png`: Best experiment selection
- `active_learning_values.png`: Experiment values with selected ones highlighted
- `active_learning_rewards.png`: Distribution of rewards

## Implementation Details
The example uses:
- PCA (Principal Component Analysis) to visualize experiment features
- Diversity metrics to evaluate the quality of selected experiments
- Reward functions that balance informativeness and experimental cost 