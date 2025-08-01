# GFlowNet.jl Manual Overview

*Comprehensive manual for the modern GFlowNet.jl framework with cutting-edge 2024 research capabilities.*

## 📋 **Table of Contents**

### Core Documentation
1. [Training System](training_system.md) - Modern training infrastructure
2. [Training Objectives](objectives.md) - All available objectives and when to use them
3. [Backward Policy](backward_policy.md) - Full trajectory balance implementation
4. [Developer Guide](developer_guide.md) - Implementing new domains
5. [Migration Guide](migration.md) - Upgrading from legacy code

### Quick Links
- [Getting Started Guide](../guide/getting_started.md)
- [Mathematical Theory](../theory/mathematical_background.md)
- [API Reference](../api/core_types.md)
- [Example Walkthroughs](../guide/examples.md)

## 🚀 **Quick Start Example**

```julia
using GFlowNet

# Create your domain model with optional backward policy
model = create_grid_world_gflownet(
    grid_size=5,
    include_backward=false  # Set to true for full trajectory balance
)

# Modern training configuration
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    partition_function_method=SIMPLE_ESTIMATION,
    batch_size=32,
    n_iterations=1000
)

# Train using modern interface
history = train_gflownet(model, config; verbose=true)

# Sample trajectories
trajectories = [sample_trajectory(model) for _ in 1:10]
```

## 📚 **What's in This Manual**

### [Training System](training_system.md)
- TrainingConfig structure and parameters
- Core training functions and workflow
- Automatic configuration for different domains
- Performance optimization tips

### [Training Objectives](objectives.md)
- Trajectory Balance (standard and variants)
- Sub-Trajectory Balance (credit assignment)
- Hierarchical and Adaptive variants
- Flow Consistency objectives
- When to use each objective

### [Backward Policy](backward_policy.md)
- Complete trajectory balance implementation
- Joint state representation approach
- When and how to use backward policies
- Performance considerations

### [Developer Guide](developer_guide.md)
- Step-by-step domain implementation
- Required interface functions
- Testing your implementation
- Advanced customizations
- Best practices

### [Migration Guide](migration.md)
- Upgrading from legacy implementations
- API changes and compatibility
- Performance improvements
- Common migration patterns

## 🎯 **Key Features**

- **Modern Training Interface**: Configuration-based training with automatic optimization
- **Flexible Objectives**: Multiple training objectives for different use cases
- **Backward Policy Support**: Optional full trajectory balance implementation
- **Domain Agnostic**: Easy to implement new domains with clear interfaces
- **Performance Optimized**: Efficient implementation with Lux.jl and Zygote.jl
- **Comprehensive Testing**: Extensive test suite and validation tools

## 🔗 **See Also**

- [Theoretical Foundations](../theory/mathematical_background.md) - Mathematical details
- [Architecture Analysis](../internals/architecture.md) - Implementation decisions
- [Examples](../guide/examples.md) - Working examples for various domains
- [API Reference](../api/core_types.md) - Detailed API documentation