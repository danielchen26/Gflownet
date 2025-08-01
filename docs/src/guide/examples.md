# Examples Guide

This guide walks through the available examples in GFlowNet.jl, from simple to complex.

## Overview

| Example | Difficulty | Key Concepts | Location |
|---------|------------|--------------|----------|
| Grid World | Beginner | Basic GFlowNet | `examples/grid_world/` |
| Supply Chain | Intermediate | Complex states | `examples/supply_chain_optimization/` |
| Molecular Design | Advanced | Graph construction | `examples/molecule_design/` |
| Active Learning | Advanced | Sequential decisions | `examples/active_learning/` |
| Causal Discovery | Research | DAG learning | `examples/causal_discovery/` |

## Grid World (Beginner)

### What It Does
Navigate a 2D grid from top-left to bottom-right, with higher rewards for longer paths.

### Key Features
- Simple state space (x, y coordinates)
- 4 directional actions + terminate
- Clear reward structure
- Easy to visualize

### Running the Example
```bash
cd examples/grid_world
julia --project=. grid_world.jl
```

### Code Highlights
```julia
# Simple state representation
struct GridState <: AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

# Clear reward function
function reward(state::GridState)
    if !state.is_terminal
        return 0.0
    end
    # Reward based on distance from origin
    return Float64(state.x + state.y)
end
```

### Learning Points
- Basic GFlowNet implementation
- State and action design
- Reward shaping
- Trajectory sampling

### Try Modifying
- Change grid size
- Add obstacles
- Modify reward function
- Test with backward policy

## Supply Chain Optimization (Intermediate)

### What It Does
Optimize pharmaceutical supply chain networks with facilities, routes, and drug distribution.

### Key Features
- Complex state with multiple components
- Business constraints
- Multi-objective rewards
- Real-world application

### Running the Example
```bash
cd examples/supply_chain_optimization
julia --project=. ultimate_connected_gflownet.jl
```

### Code Highlights
```julia
# Complex state representation
struct SupplyChainState <: AbstractState
    network::SupplyChainNetwork
    month::Int
    is_terminal::Bool
end

# Multiple action types
abstract type SupplyChainAction <: AbstractAction end
struct ProduceAction <: SupplyChainAction
    facility_id::Int
    drug_id::Int
    quantity::Int
end
```

### Learning Points
- Complex domain modeling
- Multiple action types
- Business logic integration
- Performance metrics

### Interesting Aspects
- Monthly planning horizon
- Inventory management
- Route optimization
- Service level tracking

## Best Practices from Examples

### 1. State Design
From Grid World:
```julia
# Keep states simple and immutable
struct State <: AbstractState
    data::DataType
    is_terminal::Bool
end
```

### 2. Action Applicability
From Supply Chain:
```julia
function is_applicable(action::ProduceAction, state::SupplyChainState)
    # Check multiple conditions
    facility = get_facility(state.network, action.facility_id)
    return facility.has_capacity && 
           has_ingredients(facility, action.drug_id) &&
           !state.is_terminal
end
```

### 3. Reward Design
From Molecular Design:
```julia
function reward(state::MoleculeState)
    if !state.is_terminal
        return 0.0
    end
    
    # Multi-objective reward
    qed = calculate_qed(state.molecule)
    logp = calculate_logp(state.molecule)
    
    # Combine objectives
    return exp(qed + 0.5 * logp)
end
```

### 4. Feature Extraction
From Active Learning:
```julia
function state_to_features(state::ActiveLearningState)
    features = Float32[]
    
    # Current state features
    append!(features, get_data_features(state))
    
    # Historical features
    append!(features, get_history_features(state))
    
    # Normalize
    return normalize_features(features)
end
```

## Running Examples

### Prerequisites
Each example has its own Project.toml:
```bash
cd examples/example_name
julia --project=.
```

First time setup:
```julia
using Pkg
Pkg.instantiate()
```

### Common Structure
All examples follow similar pattern:
1. Define domain types
2. Implement required interface
3. Create high-level constructor
4. Configure training
5. Run and analyze

### Expected Output
```
🚀 Starting GFlowNet training...
   Configuration:
     - Objective: TRAJECTORY_BALANCE
     - Iterations: 1000
     - Batch size: 32
   
   Iteration 100:
     - Loss: 2.345
     - Time: 0.23s
     
✅ Training completed!
   Results:
     - Mean reward: 15.2
     - Unique solutions: 85
```

## Analyzing Results

### Trajectory Analysis
```julia
# Sample trajectories
trajectories = [sample_trajectory(model) for _ in 1:1000]

# Analyze diversity
unique_terminals = unique([t.states[end] for t in trajectories])
println("Unique solutions: $(length(unique_terminals))")

# Reward distribution
rewards = [reward(t.states[end]) for t in trajectories]
println("Mean reward: $(mean(rewards))")
println("Max reward: $(maximum(rewards))")
```

### Visualization
Many examples include visualization:
```julia
# Grid world visualization
visualize_trajectories(trajectories, grid_size)

# Supply chain metrics
plot_service_levels(results)
plot_inventory_levels(results)
```

## Creating Your Own Example

### Template Structure
```
examples/my_domain/
├── Project.toml          # Dependencies
├── my_domain.jl         # Main implementation
├── README.md            # Documentation
└── results/            # Output directory
```

### Project.toml Template
```toml
[deps]
GFlowNet = "..."
Random = "..."
Statistics = "..."
# Domain-specific deps

[compat]
julia = "1.9"
```

### Implementation Template
```julia
# Load GFlowNet
push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using GFlowNet

# Define domain types
include("types.jl")

# Implement interface
include("interface.jl")

# Create and train model
model = create_my_domain_gflownet()
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 1000
)

history = train_gflownet(model, config; verbose=true)

# Analyze results
include("analysis.jl")
```

## Debugging Tips

### Common Issues

1. **Zero Rewards**
   - Check terminal state logic
   - Verify reward calculation
   - Ensure positive rewards

2. **No Valid Actions**
   - Debug `is_applicable`
   - Check state initialization
   - Verify action space

3. **Constant Trajectories**
   - Increase exploration
   - Check reward scaling
   - Verify state diversity

### Debug Mode
```julia
# Enable verbose sampling
config = SamplingConfig(strategy = STOCHASTIC_SAMPLING)
trajectory = sample_trajectory(model; config = config, verbose = true)
```

## Performance Tips

### From Examples

1. **Batch Size**: Grid world uses 16, supply chain uses 32
2. **Learning Rate**: Start with 0.01, reduce if unstable
3. **Hidden Dimension**: 64 for simple, 128 for complex domains
4. **Iterations**: 100-1000 for simple, 1000-10000 for complex

### Profiling
```julia
using Profile
@profile train_gflownet(model, config)
Profile.print()
```

## Next Steps

1. **Run all examples** to understand patterns
2. **Modify an example** to learn the API
3. **Create your own** following the template
4. **Share results** with the community

## See Also
- [Getting Started](getting_started.md) - Installation and basics
- [Developer Guide](../manual/developer_guide.md) - Creating new domains
- [API Reference](../api/core_types.md) - Detailed documentation