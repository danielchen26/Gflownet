# Core Concepts

This page introduces the core concepts behind GFlowNets and how they are implemented in this framework.

## Directed Acyclic Graphs (DAGs)

At the heart of GFlowNets is a directed acyclic graph (DAG) that represents the construction process:

- **States**: Nodes in the graph, representing partial or complete objects
- **Actions**: Edges in the graph, representing transitions between states
- **Terminal states**: Special states that represent fully constructed objects

In this implementation, DAGs are created using the composition pattern:

```julia
# Create a DAG for your domain
dag = create_dag(initial_state, terminal_states, terminal_sink, actions)

# The DAG contains all necessary information
@assert dag.initial_state isa YourStateType
@assert all(s -> s isa YourStateType, dag.terminal_states)
@assert all(a -> a isa YourActionType, dag.actions)
```

## Flow Networks

GFlowNets are based on the concept of flow networks:

- **Flow**: A measure assigned to each edge in the graph
- **Flow conservation**: At each non-terminal state, the incoming flow equals the outgoing flow
- **Source flow**: The total flow entering the initial state (the partition function Z)
- **Terminal flow**: The flow at each terminal state, proportional to the reward

### Implementation Details

In this framework, flows are computed using:

```julia
# Flow through a state
flow_value = flow(model, state)

# Flow along an edge
edge_flow_value = edge_flow(model, source_state, target_state)

# Forward transition probability
prob = forward_transition_prob(model, source_state, target_state)
```

## Policies

GFlowNets use neural network policies to make decisions:

### Forward Policy
The forward policy defines the probability of taking an action from a state during sampling:

```julia
# Create a forward policy with a neural network
forward_policy, parameters, state = create_forward_policy(
    input_dim, hidden_dim, output_dim, rng
)

# Use in a GFlowNet model
model = GFlowNetModel(dag, forward_policy, ...)
```

### Backward Policy (Optional)
The backward policy defines the probability of taking a reverse action from a state during training. **In this implementation, backward policies are optional** because we use the simplified Trajectory Balance objective.

## Partition Function (Z) Management

**This implementation uses a unique approach to managing the partition function Z:**

### Periodic Estimation Strategy

Instead of learning Z as a parameter, this framework **estimates Z periodically**:

```julia
# Every 10 training iterations
if iter % 10 == 0
    model.partition_function = estimate_partition_function(model)
end

# Simple estimation: sum of all terminal rewards
function estimate_partition_function(model::GFlowNetModel)
    return sum(reward(state) for state in model.dag.terminal_states)
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
end

# State transitions
function apply_action(action::YourAction, state::YourState)
    # Return new state after applying action
end

# Feature extraction for neural networks
function state_to_features(state::YourState)
    # Return feature vector for the state
end

# Reward calculation
function reward(state::YourState)
    # Return reward for terminal states
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

This framework is built on **Lux.jl** for neural networks:

```julia
using Lux, Random, Optimisers

# Create neural network components
rng = Random.default_rng()

# Forward policy network
nn_model = Chain(
    Dense(input_dim => 128, relu),
    Dense(128 => 128, relu), 
    Dense(128 => output_dim)
)

ps, st = Lux.setup(rng, nn_model)
forward_policy = ForwardPolicy(nn_model)

# Complete model
model = GFlowNetModel(
    dag, forward_policy, nothing, nothing, nothing,
    [TrajectoryBalanceObjective(1.0)],
    Optimisers.Adam(0.001), 
    (forward=ps, backward=nothing, flow=nothing),
    (forward=st, backward=nothing, flow=nothing)
)
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
