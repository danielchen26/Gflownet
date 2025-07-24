# Feature Acquisition Example

This example demonstrates how GFlowNets can be used for strategic feature acquisition in experimental design, optimizing which features to measure while balancing information gain and measurement costs.

## 🚀 Quick Start

**Run the consolidated modern implementation:**
```bash
# From the project root
julia --project=. examples/feature_acquisition/main.jl

# Or from the feature_acquisition directory
julia --project=../.. main.jl
```

This uses the latest modernized implementation with:
- **Objective**: `ADAPTIVE_SUB_TB` (adaptive sub-trajectory balance)
- **Z Method**: `ADAPTIVE_ESTIMATION` (automatic partition function estimation)
- **Interface**: Modern `TrainingConfig` + `train_gflownet()` pattern
- **Integration**: Full compatibility with current GFlowNet core package

## 📁 Clean Directory Structure

### Current Files
- **`main.jl`** - Modern, consolidated implementation using latest patterns
- **`DOCUMENTATION.md`** - Comprehensive technical documentation  
- **`README.md`** - This quick start guide
- **`Project.toml`** / **`Manifest.toml`** - Dependencies

### Archived Materials
- **`archive/v1/`** - Original implementation with legacy training interface
- **`archive/v2/`** - Enhanced version with improved rewards
- **`archive/v3/`** - Advanced version (basis for current main.jl)
- **`archive/development_files/`** - Development utilities, duplicates, and analysis tools
- **`archive/legacy_figs/`** - Historical visualization outputs

## 🔬 Problem Description

**Scenario**: You have N experiments, each with M potential features to measure. Each measurement has a cost, but reveals information about the experiment's value. The goal is to strategically decide which features to measure to maximize information gain while minimizing costs.

**Real-world applications**:
- **Drug Discovery**: Deciding which assays to run on potential drug compounds
- **Medical Diagnosis**: Selecting optimal sequence of diagnostic tests
- **Quality Control**: Choosing which product features to test
- **Scientific Research**: Optimizing experimental protocols

## 🎯 Key Features

### Modern Training Interface
- Uses `GFlowNet.TrainingConfig` for declarative configuration
- Supports all latest training objectives (TB, STB, Flow Consistency)
- Automatic partition function estimation with adaptive switching
- Built-in validation and early stopping

### Problem Modeling
- **State Space**: Binary matrix tracking which features have been measured
- **Action Space**: Measure specific feature or terminate acquisition
- **Reward Function**: Balances discovery value against measurement costs
- **DAG Structure**: Represents all possible measurement sequences

### Advanced Capabilities
- **Multi-experiment**: Handles multiple experiments simultaneously
- **Budget Constraints**: Respects measurement budget limits
- **Adaptive Strategies**: Learns when to stop measuring
- **Comprehensive Evaluation**: Built-in metrics and visualization

## 🛠 Implementation Highlights

### State Representation
```julia
# State tracks measurements across all experiments
state.observed_features  # NUM_EXPERIMENTS × NUM_FEATURES matrix
state.measurements_remaining  # Budget constraint
state.is_terminal  # Termination flag
```

### Modern Configuration
```julia
config = GFlowNet.TrainingConfig(
    objective = GFlowNet.ADAPTIVE_SUB_TB,
    partition_function_method = GFlowNet.ADAPTIVE_ESTIMATION,
    batch_size = 32,
    learning_rate = 0.001,
    n_iterations = 1000,
    validation_frequency = 50
)
```

### Reward Function
```julia
# Balances value discovery vs. measurement costs
reward = value_weight * best_experiment_value - 
         cost_weight * total_measurement_cost
```

## 📊 Expected Output

When you run the example, you'll see:

1. **Synthetic Data Generation**: Creates realistic experimental setup
2. **Training Configuration**: Shows modern interface usage
3. **Integration Testing**: Verifies compatibility with core package
4. **Training Simulation**: Demonstrates the training process
5. **Strategy Analysis**: Analyzes learned measurement strategies
6. **Visualizations**: Training progress and performance plots

## 🔧 Customization

### Problem Size
```julia
const NUM_EXPERIMENTS = 10    # Number of experiments
const NUM_FEATURES = 8        # Features per experiment
const MAX_MEASUREMENTS = 5    # Budget constraint
```

### Training Configuration
```julia
# Try different objectives
objective = GFlowNet.TRAJECTORY_BALANCE  # Standard TB
objective = GFlowNet.SUB_TRAJECTORY_BALANCE  # Sub-trajectory balance
objective = GFlowNet.FLOW_CONSISTENCY  # Flow consistency

# Try different Z estimation methods
partition_function_method = GFlowNet.SIMPLE_ESTIMATION
partition_function_method = GFlowNet.LEARNABLE_PARAMETER
partition_function_method = GFlowNet.SAMPLING_BASED
```

### Reward Function
```julia
reward_fn = FeatureAcquisitionReward(
    experiment_values,
    cost_per_measurement = 0.1,  # Adjust cost
    value_weight = 1.0,         # Importance of discovery
    cost_weight = 0.5           # Importance of cost minimization
)
```

## 🧪 Integration Testing

The example includes built-in integration testing to verify compatibility with the core GFlowNet package:

- **Enum Availability**: Checks all training objectives and Z estimation methods
- **TrainingConfig**: Verifies configuration creation works
- **DAG Operations**: Tests state transitions and action handling
- **Reward Functions**: Validates reward calculation

## 📚 Further Reading

- **[`DOCUMENTATION.md`](DOCUMENTATION.md)** - Comprehensive technical details, mathematical framework, and advanced features
- **[`docs/GFLOWNET_DOCUMENTATION.md`](../../docs/GFLOWNET_DOCUMENTATION.md)** - Main framework documentation
- **[`examples/README.md`](../README.md)** - Overview of all examples

## 🤝 Contributing

This example follows the modern GFlowNet development patterns:

- **Training System**: Uses `TrainingConfig` and `train_gflownet()`
- **Code Organization**: Clean separation of concerns
- **Documentation**: Comprehensive and up-to-date
- **Testing**: Built-in integration verification

For development guidelines, see the [Cursor rules](../../.cursor/rules/) and main project documentation.

---

*This example demonstrates the power and flexibility of GFlowNets for strategic decision-making problems. The feature acquisition domain showcases how GFlowNets can learn to balance multiple objectives (information vs. cost) while handling complex, sequential decision processes.* 