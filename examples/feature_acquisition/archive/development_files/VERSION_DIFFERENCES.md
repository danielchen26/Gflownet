# Feature Acquisition: Version Differences

This document outlines the key differences between version 2 and version 3 of the feature acquisition implementation.

## Key Differences Between Version 3 and Version 2

### 1. Continuous Feature Values
- **Version 2**: Features were treated as binary (observed or not observed)
- **Version 3**: Each feature has an actual continuous value stored in the `feature_values` field of the state, allowing for a more realistic representation

### 2. State Representation
- **Version 2**:
  ```julia
  mutable struct FeatureAcquisitionState <: GFlowNet.AbstractState
      observed_features::Matrix{Bool}
      measurements_remaining::Int
      is_terminal::Bool
  end
  ```
  
- **Version 3**:
  ```julia
  struct PartialFeatureState <: GFlowNet.AbstractState
      observed_features::Vector{Bool}  # Which features have been measured
      feature_values::Vector{Float64}  # Values of measured features
      initial_features::Vector{Bool}   # Which features were initially observed
      measurements_remaining::Int      # Number of measurements remaining
      is_terminal::Bool               # Whether this is a terminal state
  end
  ```

### 3. Partial Observation Handling
- **Version 3** explicitly tracks initially observed features via the `initial_features` field, allowing the system to calculate value gained from only new measurements
- **Version 3** adds the `observation_ratio` parameter to control what percentage of features are initially observed

### 4. Enhanced Metrics
- **Version 2** mainly tracked rewards
- **Version 3** adds two important additional metrics:
  - `value_discovery`: Ratio of found best value to true best value
  - `measurement_efficiency`: How efficiently measurements are used to discover value

### 5. More Sophisticated Reward Calculation
- **Version 3** uses a more nuanced reward function that considers:
  - The actual values of features
  - The difference between initial and final values
  - More detailed cost calculation based on new measurements only

### 6. Improved Visualization
- **Version 3** has more comprehensive plotting of metrics (loss, rewards, value discovery, measurement efficiency)
- **Version 3** generates a detailed Markdown report with results

### 7. State Vector Representation
- **Version 3** has a more sophisticated state representation that includes feature values, not just binary observations
- This allows the model to learn from the actual experimental values

## What Version 3 is Testing Additionally

### 1. Feature Value Learning
- Testing if the model can learn to select features based on their actual values rather than just binary presence/absence
- This is more realistic for scientific experiments where measurements have continuous values

### 2. Partial Information Scenarios
- Testing how well the model performs when starting with some initial observations
- This better represents real-world scenarios where some data is already available

### 3. Efficiency Metrics
- Testing how well the model optimizes the trade-off between information gain and measurement cost
- The measurement efficiency metric specifically tests if the model learns to make high-value measurements

### 4. Transfer Learning Potential
- By tracking which features are initially observed versus newly measured, v3 sets the foundation for transfer learning across experiments

### 5. Adaptability to Prior Knowledge
- Tests if the model can effectively build upon existing knowledge rather than starting from scratch

## Real-World Applications

The enhancements in version 3 make it more applicable to real-world scenarios where:
- Experiments have continuous outcomes, not just binary results
- Prior information exists before making decisions
- Resources are limited, requiring efficient measurement strategies
- The value of information needs to be balanced against the cost of acquisition

### Example: Drug Discovery
In a pharmaceutical context, version 3 better models the scenario where:
- Some preliminary tests have already been conducted (initial observations)
- Each test yields a numerical value (e.g., binding affinity) rather than a yes/no result
- The goal is to discover the best drug candidate while minimizing expensive tests
- The efficiency of testing strategy is explicitly measured

## Implementation Insights

The progression from version 2 to version 3 demonstrates how GFlowNets can be adapted to increasingly realistic and complex decision-making scenarios in experimental design and feature selection. 