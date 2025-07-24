# GFlowNet.jl - Complete Documentation

*Comprehensive documentation for the modern GFlowNet.jl framework with cutting-edge 2024 research capabilities.*

## 📋 **Table of Contents**

1. [Getting Started](#getting-started)
2. [Modern Training System](#modern-training-system)
3. [Training Objectives](#training-objectives)
4. [Partition Function Learning](#partition-function-learning)
5. [Flow Consistency](#flow-consistency)
6. [Developer Guide](#developer-guide)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Migration Guide](#migration-guide)

---

## 🚀 **Getting Started**

### **Installation**
```bash
git clone https://github.com/yourusername/GFlowNet.jl.git
cd GFlowNet.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### **Basic Usage**
```julia
using GFlowNet

# Create your domain model (see examples for details)
model = create_grid_world_model()

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

---

## 🧠 **Modern Training System**

### **TrainingConfig Structure**
```julia
struct TrainingConfig
    objective::TrainingObjective
    partition_function_method::PartitionFunctionMethod
    batch_size::Int
    learning_rate::Float64
    n_iterations::Int
    partition_update_frequency::Int
    validation_frequency::Int
    early_stopping_patience::Int
    sub_trajectory_config::Dict{Symbol, Any}
end
```

### **Core Training Function**
```julia
history = train_gflownet(model, config; verbose=true, validation_data=nothing)
```

**Returns training history with:**
- `:losses` - Training loss over iterations
- `:partition_function_estimates` - Z evolution
- `:mean_rewards` - Average rewards (if available)
- `:max_rewards` - Maximum rewards (if available)

### **Automatic Configuration**
```julia
# Domain-specific optimal configurations
config = create_modern_training_config(:grid_world)        # Simple deterministic
config = create_modern_training_config(:active_learning)   # Sequential decisions
config = create_modern_training_config(:feature_acquisition) # Non-deterministic
config = create_modern_training_config(:causal_discovery) # Graph structures  
config = create_modern_training_config(:molecular_design) # Multi-scale
```

---

## 🎯 **Training Objectives**

### **1. Trajectory Balance (Standard)**
```julia
objective = TRAJECTORY_BALANCE
```
**Use for**: Simple deterministic paths where each state has unique parent
**Formulation**: `Z * P_F(τ) = R(s_τ)`
**Best for**: Grid worlds, simple sequential construction

### **2. General Trajectory Balance (Complete)**
```julia
objective = GENERAL_TRAJECTORY_BALANCE
```
**Use for**: Non-deterministic paths with multiple ways to reach states
**Formulation**: `Z * P_F(τ) = R(s_τ) * P_B(τ)`
**Best for**: Graph generation, causal discovery

### **3. Sub-Trajectory Balance (Credit Assignment)**
```julia
objective = SUB_TRAJECTORY_BALANCE
sub_trajectory_config = Dict(
    :min_length => 2,
    :max_length => 5,
    :n_subtrajectories => 3
)
```
**Use for**: Sequential decisions needing better credit assignment
**Best for**: Active learning, experiment design

### **4. Hierarchical Sub-Trajectory Balance (Multi-Scale)**
```julia
objective = HIERARCHICAL_SUB_TB
sub_trajectory_config = Dict(
    :scales => [2, 4, 8, 16],
    :n_subtrajectories => 5
)
```
**Use for**: Multi-scale structures with hierarchical components
**Best for**: Molecular design, hierarchical construction

### **5. Adaptive Sub-Trajectory Balance (Intelligent)**
```julia
objective = ADAPTIVE_SUB_TB
sub_trajectory_config = Dict(
    :difficulty_threshold => 0.05,
    :n_subtrajectories => 8
)
```
**Use for**: Sparse important decisions with varying difficulty
**Best for**: Feature acquisition, complex decision making

### **6. Flow Consistency (Local Balance)**
```julia
objective = FLOW_CONSISTENCY
flow_mode = EDGE_LEVEL  # or STATE_LEVEL or MIXED_LEVEL
```
**Use for**: Local structure learning and flow conservation
**Mathematical breakthrough**: Unifies detailed balance + flow matching

---

## 🧮 **Partition Function Learning**

### **1. Simple Estimation**
```julia
partition_function_method = SIMPLE_ESTIMATION
```
**Approach**: Sum of all terminal rewards
**Best for**: Enumerable terminal states
**When to use**: Small, well-defined state spaces

### **2. Learnable Parameter**
```julia
partition_function_method = LEARNABLE_PARAMETER
```
**Approach**: Gradient-based optimization of log(Z)
**Best for**: Complex but smooth spaces
**When to use**: When Z has smooth dependence on policy

### **3. Sampling-Based**
```julia
partition_function_method = SAMPLING_BASED
```
**Approach**: Policy sampling with exponential moving average
**Best for**: Dynamic or non-stationary spaces
**When to use**: Complex spaces where sampling is feasible

### **4. Adaptive Estimation**
```julia
partition_function_method = ADAPTIVE_ESTIMATION
```
**Approach**: Automatic method switching during training
**Best for**: Unknown space characteristics
**When to use**: When unsure about optimal Z estimation method

---

## ⚖️ **Flow Consistency**

### **Mathematical Unification**
Flow consistency unifies two classical approaches:

**Edge-Level (Detailed Balance)**:
```
F(s) * P_F(s → s') = F(s') * P_B(s' → s)
```

**State-Level (Flow Matching)**:
```
∑incoming_flow = ∑outgoing_flow
```

### **Implementation**
```julia
# Choose consistency mode
loss = flow_consistency_loss(
    model, 
    trajectories, 
    mode=EDGE_LEVEL    # EDGE_LEVEL, STATE_LEVEL, or MIXED_LEVEL
)
```

### **When to Use Each Mode**
- **EDGE_LEVEL**: Replace detailed balance, focus on individual transitions
- **STATE_LEVEL**: Replace flow matching, focus on state conservation  
- **MIXED_LEVEL**: Combine both for maximum constraint enforcement

---

## 👨‍💻 **Developer Guide**

### **Implementing a New Domain**

#### **1. Define Domain Data Structure**
```julia
struct MyDomainData
    field1::Type1
    field2::Type2
    # Domain-specific fields
end
```

#### **2. Create State and Action Types**
```julia
struct MyDomainState <: AbstractState
    data::MyDomainData
    is_terminal::Bool
end

abstract type MyDomainAction <: AbstractAction end
struct ActionType1 <: MyDomainAction
    # Action parameters
end
```

#### **3. Implement Required Interface**
```julia
# Check action applicability
function GFlowNet.is_applicable(action::ActionType1, state::MyDomainState)
    # Return true/false
end

# Apply action to state
function GFlowNet.apply_action(action::ActionType1, state::MyDomainState)
    # Return new state
end

# Convert state to neural network features
function GFlowNet.state_to_features(state::MyDomainState)
    # Return feature vector
end

# Calculate reward for terminal states
function GFlowNet.reward(state::MyDomainState)
    # Return non-negative reward
end
```

#### **4. Add Utility Functions**
```julia
# State equality and hashing
function Base.==(a::MyDomainState, b::MyDomainState)
    # Compare states
end

function Base.hash(state::MyDomainState, h::UInt)
    # Hash state for dictionaries
end

# Visualization
function visualize(state::MyDomainState)
    # Domain-specific visualization
end
```

### **Complete Implementation Steps**

#### **5. Helper Functions**
```julia
function create_initial_state()
    return MyDomainState(MyDomainData(...), false)
end

function create_model(params...)
    # Create DAG, policies, etc.
    initial_state = create_initial_state()
    terminal_states = [MyDomainState(..., true)]
    terminal_sink = MyDomainState(..., true)
    actions = [ActionType1(...), ActionType2(...)]
    return create_dag(initial_state, terminal_states, terminal_sink, actions)
end

function visualize(state::MyDomainState)
    # Domain-specific visualization
    println("State: $(state.data)")
end
```

#### **6. Additional Required Methods**
```julia
# Action equality and hashing (also required)
function Base.==(a::ActionType1, b::ActionType1)
    # Compare actions
end

function Base.hash(action::ActionType1, h::UInt)
    # Hash action for dictionaries
end
```

### **Testing Your Implementation**

#### **1. Basic State/Action Tests**
```julia
# Create an initial state
initial_state = create_initial_state()

# Create some actions
actions = [ActionType1(...), ActionType2(...)]

# Test if actions are applicable
for action in actions
    if is_applicable(action, initial_state)
        new_state = apply_action(action, initial_state)
        println("Applied $(typeof(action))")
        println("New state: $new_state")
    end
end
```

#### **2. DAG Construction Test**
```julia
# Create initial and terminal states
initial_state = create_initial_state()
terminal_states = [MyDomainState(..., true)]
terminal_sink = MyDomainState(..., true)

# Create actions
actions = [ActionType1(...), ActionType2(...)]

# Test DAG creation
dag = create_dag(initial_state, terminal_states, terminal_sink, actions)
println("DAG created with $(length(dag.states)) states")
```

#### **3. Feature Extraction Test**
```julia
# Test feature extraction produces sensible vectors
state = create_initial_state()
features = state_to_features(state)
println("Feature vector: $features")
println("Feature dimension: $(length(features))")
```

#### **4. Reward Function Test**
```julia
# Test reward function behavior
terminal_state = MyDomainState(..., true)
reward_value = reward(terminal_state)
println("Terminal reward: $reward_value")
```

### **Advanced Customizations**

#### **Custom Policy Networks**
```julia
# 1. Define model structure
struct MyCustomModel
    layers::Vector{Any}
    params::NamedTuple
end

# 2. Define forward pass
function (model::MyCustomModel)(features)
    # Compute and return policy distribution
    logits = model.layers[1](features)
    return softmax(logits)
end

# 3. Use with ForwardPolicy
policy = ForwardPolicy(MyCustomModel(...))
```

#### **Custom Training Objectives**
```julia
# 1. Define new objective type
struct MyCustomObjective <: AbstractGFlowNetObjective
    weight::Float64
    custom_param::Float64
end

# 2. Implement training logic
function compute_loss(obj::MyCustomObjective, model, trajectories)
    # Implement custom loss computation
    total_loss = 0.0
    for trajectory in trajectories
        # Custom loss calculation
        loss = custom_loss_function(trajectory, obj.custom_param)
        total_loss += obj.weight * loss
    end
    return total_loss / length(trajectories)
end
```

#### **Multiple Reward Components**
```julia
function reward(state::MyDomainState)
    if !state.is_terminal
        return 0.0
    end
    
    # Component 1: Task-specific reward
    task_reward = calculate_task_reward(state)
    
    # Component 2: Efficiency bonus
    efficiency_bonus = calculate_efficiency(state)
    
    # Component 3: Diversity incentive
    diversity_reward = calculate_diversity(state)
    
    # Combine rewards (can use addition, multiplication, or weighted sum)
    return task_reward + 0.1 * efficiency_bonus + 0.05 * diversity_reward
end
```

### **Best Practices**
1. **Immutable States**: Make states immutable to avoid bugs from shared references
2. **Copy Data**: When applying actions, create new copies of data structures  
3. **Type Stability**: Ensure all methods maintain type stability for performance
4. **Avoid Cycles**: The GFlowNet state space must be a directed acyclic graph
5. **Terminal States**: Clearly define the criteria for terminal states
6. **Meaningful Rewards**: Design reward functions that capture desired properties
7. **Efficient Features**: Keep `state_to_features` efficient as it's called frequently

### **Additional Resources for Developers**
- **Complete Examples**: See `examples/` directory for working implementations
- **Source Code Reference**: Study existing domain implementations:
  - `src/applications/molecular_design.jl` - Molecular construction
  - `src/applications/causal_discovery.jl` - Graph structure learning  
  - `src/applications/active_learning.jl` - Experiment selection
- **Modern Training**: All examples use modern `TrainingConfig` patterns
- **Testing Framework**: Run `julia --project test/runtests.jl` for test examples

---

## 📚 **API Reference**

### **Core Training Functions**
```julia
# Main training interface
train_gflownet(model, config; verbose=true, validation_data=nothing)

# Configuration creation
TrainingConfig(; objective, partition_function_method, ...)
create_modern_training_config(problem_type::Symbol)

# Model creation utilities
create_grid_world_model()
create_dag(initial_state, terminal_states, terminal_sink, actions)
```

### **Training Objectives**
```julia
# Available objectives
TRAJECTORY_BALANCE
GENERAL_TRAJECTORY_BALANCE  
SUB_TRAJECTORY_BALANCE
HIERARCHICAL_SUB_TB
ADAPTIVE_SUB_TB
FLOW_CONSISTENCY

# Objective-specific functions
trajectory_balance_loss(model, trajectories)
general_trajectory_balance_loss(model, trajectories)
sub_trajectory_balance_loss(model, trajectories, config)
flow_consistency_loss(model, trajectories, mode)
```

### **Partition Function Methods**
```julia
# Available methods
SIMPLE_ESTIMATION
LEARNABLE_PARAMETER
SAMPLING_BASED
ADAPTIVE_ESTIMATION

# Z estimation functions
estimate_partition_function_simple(model)
update_learnable_partition_function!(model, gradients)
estimate_partition_function_sampling(model, n_samples)
```

### **Legacy Functions (Deprecated)**
```julia
# Still available but use modern interface instead
compute_loss_and_grad(model, trajectories)  # Use train_gflownet()
apply_optimizer!(model, gradients)          # Use train_gflownet()
```

---

## 🎮 **Examples**

### **Grid World (Beginner)**
```julia
cd examples/grid_world && julia grid_world.jl
```
- **Objective**: `TRAJECTORY_BALANCE`
- **Z Method**: `SIMPLE_ESTIMATION`
- **Learning**: Basic GFlowNet concepts

### **Active Learning (Intermediate)**
```julia
cd examples/active_learning && julia active_learning.jl
```
- **Objective**: `SUB_TRAJECTORY_BALANCE`
- **Z Method**: `ADAPTIVE_ESTIMATION`
- **Learning**: Sequential decision making

### **Feature Acquisition (Advanced)**
```julia
cd examples/feature_acquisition && julia main_v3.jl
```
- **Objective**: `ADAPTIVE_SUB_TB`
- **Z Method**: `ADAPTIVE_ESTIMATION`
- **Learning**: Non-deterministic handling

### **Causal Discovery (Research)**
```julia
cd examples/causal_discovery && julia causal_discovery.jl
```
- **Objective**: `GENERAL_TRAJECTORY_BALANCE`
- **Z Method**: `SAMPLING_BASED`
- **Learning**: Complete theoretical formulation

### **Molecular Design (Specialized)**
```julia
cd examples/molecule_design && julia molecule_example.jl
```
- **Objective**: `HIERARCHICAL_SUB_TB`
- **Z Method**: `SIMPLE_ESTIMATION`
- **Learning**: Multi-scale structures

---

## 🔄 **Migration Guide**

### **From Legacy to Modern**

#### **OLD (Legacy)**
```julia
# Manual training loop
for iter in 1:n_iterations
    trajectories = [sample_trajectory(model) for _ in 1:batch_size]
    loss, grad = compute_loss_and_grad(model, trajectories)
    apply_optimizer!(model, grad)
    
    if iter % 10 == 0
        model.partition_function = estimate_partition_function(model)
    end
end
```

#### **NEW (Modern)**
```julia
# Configuration-based training
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    partition_function_method=SIMPLE_ESTIMATION,
    batch_size=32,
    n_iterations=1000,
    partition_update_frequency=10
)

history = train_gflownet(model, config; verbose=true)
```

### **Migration Steps**
1. **Replace training loops** with `TrainingConfig` + `train_gflownet()`
2. **Select appropriate objective** for your domain
3. **Choose Z estimation method** based on space characteristics
4. **Add error handling** with try-catch blocks
5. **Use training history** for metrics and visualization

### **Compatibility Notes**
- ✅ **Backward Compatible**: All legacy functions still work
- ✅ **Gradual Migration**: Can migrate piece by piece
- ✅ **Same Results**: Modern interface produces same/better results
- ⚠️ **Deprecation Warnings**: Legacy functions show warnings

---

## 🎯 **Domain-Specific Recommendations**

| **Domain Type** | **Optimal Configuration** | **Reasoning** |
|-----------------|---------------------------|---------------|
| **Simple Grid Worlds** | `TRAJECTORY_BALANCE` + `SIMPLE_ESTIMATION` | Deterministic, enumerable |
| **Sequential Experiments** | `SUB_TRAJECTORY_BALANCE` + `ADAPTIVE_ESTIMATION` | Credit assignment crucial |
| **Non-Deterministic Paths** | `ADAPTIVE_SUB_TB` + `ADAPTIVE_ESTIMATION` | Complex decision structure |
| **Graph Generation** | `GENERAL_TRAJECTORY_BALANCE` + `SAMPLING_BASED` | Multiple construction paths |
| **Hierarchical Structures** | `HIERARCHICAL_SUB_TB` + `SIMPLE_ESTIMATION` | Multi-scale organization |

---

## 📈 **Performance Tips**

### **Training Optimization**
- Use **appropriate batch sizes** (16-64 typically optimal)
- Choose **domain-specific objectives** for better convergence
- Enable **early stopping** to prevent overfitting
- Monitor **Z estimation quality** for training stability

### **Implementation Optimization**
- Keep `state_to_features()` **efficient** (called frequently)
- Use **immutable states** to avoid copying overhead
- Implement **efficient equality/hashing** for states and actions
- Profile **reward computation** if complex

### **Memory Management**
- **Batch size** affects memory usage quadratically
- **Large DAGs** may require streaming processing
- **Feature vectors** should be reasonably sized
- Monitor **gradient accumulation** in long training

---

## 🔬 **Research Extensions**

### **Adding New Objectives**
```julia
struct MyCustomObjective <: AbstractGFlowNetObjective
    weight::Float64
    custom_param::Float64
end

function compute_objective_loss(obj::MyCustomObjective, model, trajectories)
    # Implement custom loss computation
end
```

### **Custom Z Estimation**
```julia
struct MyCustomZMethod <: AbstractPartitionFunctionMethod
    custom_param::Float64
end

function estimate_partition_function(method::MyCustomZMethod, model)
    # Implement custom Z estimation
end
```

### **Research Applications**
- **Multi-Objective Optimization**: Combine multiple objectives
- **Hierarchical Planning**: Use hierarchical sub-trajectory balance
- **Continuous Spaces**: Extend to continuous action/state spaces
- **Meta-Learning**: Learn optimal configurations across domains

---

## 📚 **References**

1. **Trajectory Balance**: Malkin, N., et al. (2022). "Trajectory balance: Improved credit assignment in GFlowNets." *NeurIPS 2022*.

2. **GFlowNet Foundations**: Bengio, Y., et al. (2021). "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation." *NeurIPS 2021*.

3. **Sub-Trajectory Balance**: Pan, L., et al. (2023). "Better Training of GFlowNets with Local Credit and Incomplete Trajectories." *ICML 2023*.

4. **Flow Matching**: Lahlou, S., et al. (2023). "A Theory of Continuous Generative Flow Networks." *ICML 2023*.

5. **General Trajectory Balance**: Zhang, D., et al. (2023). "Generative Flow Networks for Discrete Probabilistic Modeling." *ICLR 2023*.

---

*This documentation covers the complete modern GFlowNet.jl framework. For the latest updates and community discussions, visit the GitHub repository.* 🚀 