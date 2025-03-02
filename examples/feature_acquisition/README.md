# Feature Acquisition with GFlowNets

This example demonstrates how to use GFlowNets for active feature acquisition in experimental design. It addresses a more realistic scenario where:

1. You have 50 potential experiments
2. Each experiment has 10 features, but initially only ~20% of features are observed
3. You have a limited budget (25 measurements) to reveal additional features
4. You want to strategically decide which features to measure to maximize information gain

This scenario simulates real-world experimental workflows where measuring every feature for every experiment would be prohibitively expensive or time-consuming.

## Problem Setting

- **State**: Which features have been measured for each experiment
- **Actions**: Measure feature j of experiment i
- **Reward**: Information gain (correlation between predicted and true experiment values) minus measurement cost

The GFlowNet learns to efficiently allocate the measurement budget to focus on the most informative features, particularly for high-value experiments.

## How It Works

1. We start with a partially observed feature matrix (80% of features are unknown)
2. The GFlowNet sequentially decides which feature to measure next
3. After each measurement, we update our understanding of the experiments
4. The process continues until the measurement budget is exhausted
5. Success is measured by how well we can identify high-value experiments with the limited measurements

## Running the Example

1. Navigate to this directory:
   ```
   cd examples/feature_acquisition
   ```

2. Activate the project environment:
   ```
   julia --project=.
   ```

3. Install dependencies (first time only):
   ```julia
   using Pkg
   Pkg.instantiate()
   ```

4. Run the example:
   ```
   julia feature_acquisition.jl
   ```

## Output

The example generates:

1. Terminal output showing training progress and final evaluation
2. `feature_acquisition_results.png` - Visualization of which features were measured
3. `feature_acquisition_training.png` - Training curves (loss, reward, information gain)
4. `feature_acquisition_training.csv` - Raw training metrics

## Visualization Explanation

The results visualization includes:

1. **Feature Acquisition Heatmap**: Shows which features were measured for each experiment
   - Experiments are sorted by true value (top = highest value)
   - Each row is an experiment, each column is a feature dimension
   - Bright cells indicate measured features

2. **Top Experiment Coverage**: Bar chart showing what percentage of features were measured for the top 10 experiments by value

This helps visualize how well the GFlowNet learned to focus measurements on high-value experiments. 