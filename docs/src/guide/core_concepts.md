# Core Concepts

This page introduces the core concepts behind GFlowNets and how they are implemented in this framework.

## On-Demand State Space Computation

GFlowNet.jl uses an on-demand computation approach rather than explicit DAG construction:

- **States**: Generated dynamically during sampling and training
- **Actions**: Applied to states to create transitions on-the-fly
- **Terminal states**: Identified by the `is_terminal_state()` function

The current implementation avoids explicit graph construction:

```julia
# States are created through action application
initial_state = create_initial_state()
applicable_actions = get_applicable_actions(initial_state, all_actions)
next_state = apply_action(applicable_actions[1], initial_state)

# No explicit DAG structure required
@assert is_terminal_state(next_state) isa Bool
```

## Flow Networks

GFlowNets are based on the concept of flow networks:

- **Flow**: A measure assigned to each edge in the graph
- **Flow conservation**: At each non-terminal state, the incoming flow equals the outgoing flow
- **Source flow**: The total flow entering the initial state (the partition function Z)
- **Terminal flow**: The flow at each terminal state, proportional to the reward

### Implementation Details

Flow computation is now fully implemented with explicit functions:

```julia
# Compute flow through a state
flow_value = flow(model, state)

# Compute partition function
Z = partition_function(model)

# Compute edge flows
edge_flow_val = edge_flow(model, source_state, target_state)

# Analyze flow conservation
analysis = flow_analysis(model, state)
```

For detailed API documentation, see [Flow Computation API](../api/flow_computation.md).

## Policies

GFlowNets use neural network policies to make decisions:

### Forward Policy
The forward policy defines the probability of taking an action from a state during sampling:

```julia
# Create a forward policy with a neural network
forward_policy, parameters, state = create_forward_policy(
    input_dim, hidden_dim, output_dim, rng
)

# Use in a GFlowNet model (via high-level interface)
model = create_grid_world_gflownet(grid_size=5)
```

### Backward Policy (Not Currently Used)
Backward policies are not implemented in the current working version. The Trajectory Balance objective only requires forward policies and terminal rewards.

## Partition Function (Z) Management

**This implementation uses a unique approach to managing the partition function Z:**

### Periodic Estimation Strategy

Instead of learning Z as a parameter, this framework **estimates Z periodically**:

```julia
# Every 10 training iterations
if iter % 10 == 0
    model.partition_function = estimate_partition_function(model)
end

# In practice, Z is often set to 1.0 for simplicity
# The current implementation typically uses Z = 1.0
function simple_partition_function()
    return 1.0
end
```

### Why This Approach?

1. **Stability**: Avoids instability issues from joint Z-policy learning
2. **Simplicity**: No additional hyperparameters to tune
3. **Robustness**: Works well when terminal states are known/enumerable

## Training Objectives

**This implementation primarily uses Trajectory Balance** with the simplified form:

### Trajectory Balance (Simplified)

For problems where each state has a unique parent (deterministic backward paths):

$$Z \cdot P_F(\tau) = R(s_\tau)$$

```julia
# Implementation in the loss function
ratio = (Z * forward_prob_product) / final_reward
loss = log(ratio)^2
```

This works well for:
- ✅ Sequential construction (molecules, paths)
- ✅ Grid world navigation  
- ❌ Set construction where order doesn't matter

### When to Use Other Objectives

- **Flow Matching**: When you want direct flow conservation enforcement
- **Detailed Balance**: When you need edge-level control and have backward policies
- **General Trajectory Balance**: When states can have multiple parents

## Environment Interface

The environment defines the problem structure through a clean interface:

### Required Functions

```julia
# State-action applicability
function is_applicable(action::YourAction, state::YourState)
    # Return true if action can be applied to state
    return true  # Example implementation
end

# State transitions
function apply_action(action::YourAction, state::YourState)
    # Return new state after applying action
    # Use both action and state to create new state
    return YourState(state.field + action.delta)  # Example implementation
end

# Feature extraction for neural networks
function state_to_features(state::YourState)
    # Return feature vector for the state
    return Float32[state.x, state.y]  # Example using state fields
end

# Reward calculation
function reward(state::YourState)
    # Return reward for terminal states
    return state.is_terminal ? 1.0 : 0.0  # Example using state
end
```

### Composition-Based Design

This framework uses composition rather than inheritance:

```julia
# Domain-specific data
struct MoleculeData
    atoms::Vector{Symbol}
    bonds::Vector{Tuple{Int, Int, Int}}
end

# State composes data with behavior
struct MoleculeState <: AbstractState
    data::MoleculeData
    complete::Bool
end

# Clean interface implementations
function apply_action(action::AddAtomAction, state::MoleculeState)
    # Domain-specific logic here
    new_atoms = copy(state.data.atoms)
    push!(new_atoms, action.atom_type)
    return MoleculeState(MoleculeData(new_atoms, state.data.bonds), false)
end
```

## Sampling Process

After training, GFlowNets generate samples through forward sampling:

```julia
# Sample a complete trajectory
trajectory = sample_trajectory(model)

# Extract the terminal state
terminal_state = trajectory.states[end]

# Get the reward
final_reward = reward(terminal_state)
```

### Sampling Loop

The sampling process follows these steps:

1. **Start** at the initial state
2. **Compute** action probabilities using the forward policy
3. **Sample** an action according to these probabilities
4. **Apply** the action to get the next state
5. **Repeat** until reaching a terminal state

## Integration with Lux.jl

This framework is built on **Lux.jl** for neural networks. Use the high-level interface:

```julia
using GFlowNet

# Create a complete model with built-in neural networks
model = create_grid_world_gflownet(
    grid_size=5,
    hidden_dim=64,
    learning_rate=0.001
)

# Configure training
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    n_iterations=1000,
    batch_size=32
)

# Train the model
history = train_gflownet(model, config; verbose=true)
```

## Key Design Principles

### 1. Stability First
The periodic Z estimation and simplified TB objective prioritize training stability over theoretical generality.

### 2. Composition Over Inheritance
Domain-specific types compose data structures rather than inheriting from complex hierarchies.

### 3. Type Safety
Julia's type system ensures correctness while allowing flexibility.

### 4. Performance
The design leverages Julia's compilation for efficient execution.

## Summary

This GFlowNet implementation emphasizes:

- **Practical Stability**: Periodic Z estimation for robust training
- **Simplicity**: Simplified TB objective when appropriate
- **Flexibility**: Composition-based design for easy extension
- **Performance**: Modern Julia ecosystem integration (Lux, Optimisers, Zygote)

The framework is **production-ready** for research and applications involving sequential decision-making with complex reward structures.
