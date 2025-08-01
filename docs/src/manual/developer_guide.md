# Developer Guide

This guide walks through implementing a new domain for GFlowNet.jl step by step.

## Overview

To implement a new domain, you need to:
1. Define your domain's data structures
2. Create state and action types
3. Implement the required interface functions
4. Add utility functions
5. Create a high-level constructor
6. Test your implementation

## Step 1: Define Domain Data Structure

First, define the core data that represents your domain:

```julia
struct MyDomainData
    field1::Type1
    field2::Type2
    # Domain-specific fields
end
```

### Example: Supply Chain
```julia
struct SupplyChainNetwork
    facilities::Vector{Facility}
    routes::Vector{TransportRoute}
    drugs::Vector{Drug}
    patients::Vector{PatientRegion}
end
```

## Step 2: Create State and Action Types

### State Type
Your state must inherit from `AbstractState` and include `is_terminal::Bool`:

```julia
struct MyDomainState <: AbstractState
    data::MyDomainData
    is_terminal::Bool
end
```

### Action Types
Define an abstract action type and concrete actions:

```julia
abstract type MyDomainAction <: AbstractAction end

struct ActionType1 <: MyDomainAction
    parameter1::Type1
end

struct ActionType2 <: MyDomainAction
    parameter2::Type2
end

struct TerminateAction <: MyDomainAction end
```

## Step 3: Implement Required Interface

You MUST implement these 5 functions:

### 1. State to Features
Convert state to neural network input:

```julia
function GFlowNet.state_to_features(state::MyDomainState)::Vector{Float32}
    features = Float32[]
    
    # Extract features from your state
    push!(features, extract_feature_1(state.data))
    push!(features, extract_feature_2(state.data))
    
    return features
end
```

**Requirements:**
- Must return `Vector{Float32}`
- Fixed dimension for all states
- Include all relevant information

### 2. Terminal State Check
```julia
function GFlowNet.is_terminal_state(state::MyDomainState)::Bool
    return state.is_terminal
end
```

### 3. Reward Function
```julia
function GFlowNet.reward(state::MyDomainState)::Float64
    if !is_terminal_state(state)
        return 0.0  # Non-terminal states have zero reward
    end
    
    # Calculate reward for terminal state
    reward_value = calculate_domain_specific_reward(state.data)
    
    # Ensure positive reward (GFlowNet requirement)
    return max(reward_value, 1e-8)
end
```

**Requirements:**
- Must return positive values for terminal states
- Non-terminal states should return 0.0

### 4. Action Applicability
```julia
function GFlowNet.is_applicable(action::ActionType1, state::MyDomainState)::Bool
    # Check if action can be applied to state
    if state.is_terminal
        return false
    end
    
    # Domain-specific applicability logic
    return check_action_validity(action, state.data)
end
```

### 5. Apply Action
```julia
function GFlowNet.apply_action(action::ActionType1, state::MyDomainState)::MyDomainState
    # Create new state (never mutate!)
    new_data = apply_action_to_data(action, state.data)
    is_terminal = should_terminate(new_data)
    
    return MyDomainState(new_data, is_terminal)
end
```

**Critical**: Never mutate states! Always create new instances.

## Step 4: Add Utility Functions

### Equality and Hashing
Required for states and actions:

```julia
# State equality
function Base.==(a::MyDomainState, b::MyDomainState)
    return a.data == b.data && a.is_terminal == b.is_terminal
end

# State hashing
function Base.hash(state::MyDomainState, h::UInt)
    return hash((state.data, state.is_terminal), h)
end

# Action equality
function Base.==(a::ActionType1, b::ActionType1)
    return a.parameter1 == b.parameter1
end

# Action hashing  
function Base.hash(action::ActionType1, h::UInt)
    return hash(action.parameter1, h)
end
```

### Display Methods (Optional)
```julia
function Base.show(io::IO, state::MyDomainState)
    print(io, "MyDomainState($(state.data), terminal=$(state.is_terminal))")
end
```

## Step 5: Create High-Level Constructor

Create a convenient function to set up your domain:

```julia
function create_my_domain_gflownet(;
    param1 = default1,
    param2 = default2,
    hidden_dim = 64,
    learning_rate = 0.01,
    include_backward = false
)
    # Create initial state
    initial_data = create_initial_data(param1, param2)
    initial_state = MyDomainState(initial_data, false)
    
    # Create all possible actions
    all_actions = create_all_actions(param1, param2)
    
    # Get state dimension
    state_dim = length(state_to_features(initial_state))
    
    # Create model using core function
    return create_gflownet(
        initial_state,
        all_actions;
        state_dim = state_dim,
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        include_backward = include_backward
    )
end
```

## Step 6: Test Your Implementation

### Basic Functionality Test
```julia
# Create model
model = create_my_domain_gflownet()

# Test state features
initial_state = model.initial_state
features = state_to_features(initial_state)
@assert length(features) > 0
@assert eltype(features) == Float32

# Test actions
for action in model.all_actions
    if is_applicable(action, initial_state)
        new_state = apply_action(action, initial_state)
        @assert new_state != initial_state  # States are immutable
    end
end
```

### Training Test
```julia
# Configure training
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 10,
    batch_size = 4
)

# Train model
history = train_gflownet(model, config)
@assert length(history.losses) == 10
```

### Trajectory Sampling Test
```julia
# Sample trajectories
trajectories = [sample_trajectory(model) for _ in 1:10]

for traj in trajectories
    # Check validity
    @assert length(traj.states) >= 2
    @assert is_terminal_state(traj.states[end])
    
    # Check rewards
    terminal_reward = reward(traj.states[end])
    @assert terminal_reward > 0
end
```

## Best Practices

### 1. Immutable States
Always create new state instances:

```julia
# ❌ WRONG - Mutation breaks Zygote
function apply_action(action, state)
    state.data.field += action.delta  # Mutation!
    return state
end

# ✅ CORRECT - Create new instance
function apply_action(action, state)
    new_data = MyData(
        field = state.data.field + action.delta
    )
    return MyDomainState(new_data, false)
end
```

### 2. Efficient Features
Keep feature extraction fast:

```julia
# Cache expensive computations
const FEATURE_CACHE = Dict{MyDomainState, Vector{Float32}}()

function state_to_features(state::MyDomainState)
    if haskey(FEATURE_CACHE, state)
        return FEATURE_CACHE[state]
    end
    
    features = compute_features(state)
    FEATURE_CACHE[state] = features
    return features
end
```

### 3. Meaningful Rewards
Design rewards that capture your objectives:

```julia
function reward(state::MyDomainState)
    if !is_terminal_state(state)
        return 0.0
    end
    
    # Multi-objective reward
    quality = compute_quality(state.data)
    diversity = compute_diversity(state.data)
    efficiency = compute_efficiency(state.data)
    
    # Weighted combination
    total_reward = (
        0.5 * quality +
        0.3 * diversity +
        0.2 * efficiency
    )
    
    return max(total_reward, 1e-8)
end
```

### 4. Action Space Design
Keep action space manageable:

```julia
# ❌ Too many actions
all_actions = [Action(i, j) for i in 1:100 for j in 1:100]  # 10,000 actions!

# ✅ Hierarchical or filtered actions
function get_applicable_actions(state, all_actions)
    # Filter based on state
    return filter(a -> is_feasible(a, state), all_actions)
end
```

## Common Pitfalls

### 1. Forgetting Terminal States
Always provide a way to terminate:

```julia
struct TerminateAction <: MyDomainAction end

function is_applicable(::TerminateAction, state::MyDomainState)
    # Define when termination is allowed
    return !state.is_terminal && can_terminate(state)
end
```

### 2. Non-Positive Rewards
GFlowNets require positive rewards:

```julia
# ❌ Can be negative
reward = quality - cost

# ✅ Always positive
reward = max(quality - cost, 1e-8)
# or
reward = exp(quality - cost)  # Always positive
```

### 3. Inconsistent Features
Ensure feature dimension is constant:

```julia
# ❌ Variable length
features = [f for f in extract_features(state) if f > 0]

# ✅ Fixed length
features = zeros(Float32, feature_dim)
features[1:n] = extract_features(state)
```

## Advanced Topics

### Custom Neural Networks
Override the default architecture:

```julia
function create_custom_forward_policy(state_dim, action_dim, hidden_dim)
    # Custom architecture with skip connections
    net = Chain(
        Dense(state_dim => hidden_dim, relu),
        SkipConnection(
            Dense(hidden_dim => hidden_dim, relu),
            +
        ),
        Dense(hidden_dim => action_dim)
    )
    return net
end
```

### Domain-Specific Sampling
Implement custom sampling strategies:

```julia
function sample_trajectory_with_constraints(model, constraints)
    # Custom sampling logic
    # ...
end
```

### Multi-Agent Domains
Handle multiple decision makers:

```julia
struct MultiAgentState <: AbstractState
    agent_states::Vector{AgentState}
    current_agent::Int
    is_terminal::Bool
end
```

## Example Implementations

Study these existing implementations:
- `src/applications/grid_world.jl` - Simple spatial navigation
- `src/applications/molecular_design.jl` - Graph construction
- `src/applications/supply_chain_optimization.jl` - Complex business logic
- `src/applications/causal_discovery.jl` - DAG learning

## Getting Help

1. Check existing implementations for patterns
2. Run tests to verify interface compliance
3. Use `@assert` liberally during development
4. Profile performance with large state spaces

## See Also
- [Core Concepts](../guide/core_concepts.md) - Understanding GFlowNets
- [API Reference](../api/core_types.md) - Detailed type documentation
- [Testing Guide](../internals/testing.md) - Writing tests
- [Examples](../guide/examples.md) - Working examples