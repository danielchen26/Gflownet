# Migration Guide

This guide helps you migrate from legacy GFlowNet implementations to the modern framework.

## Overview of Changes

### Major Changes
1. **Configuration-based training** replaces manual training loops
2. **On-demand computation** replaces explicit DAG construction
3. **ComponentArrays + Lux.jl** replaces older neural network libraries
4. **Unified interface** with `create_*_gflownet()` functions
5. **Optional backward policy** for full trajectory balance

### What's Removed
- Explicit DAG construction
- Manual gradient computation
- Direct optimizer manipulation
- Flow computation functions (temporarily)

## Migration Examples

### Old Training Loop → Modern Training

#### OLD (Manual Loop)
```julia
# Manual training loop
for iter in 1:n_iterations
    # Sample trajectories manually
    trajectories = Trajectory[]
    for _ in 1:batch_size
        traj = sample_trajectory(model)
        push!(trajectories, traj)
    end
    
    # Compute loss and gradients manually
    loss, grad = compute_loss_and_grad(model, trajectories)
    
    # Apply optimizer manually
    apply_optimizer!(model, grad)
    
    # Update partition function periodically
    if iter % 10 == 0
        model.partition_function = estimate_partition_function(model)
    end
    
    # Manual logging
    if iter % 100 == 0
        println("Iteration $iter: loss = $loss")
    end
end
```

#### NEW (Configuration-Based)
```julia
# Configuration-based training
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    partition_function_method = SIMPLE_ESTIMATION,
    batch_size = 32,
    n_iterations = 1000,
    partition_update_frequency = 10,
    validation_frequency = 100
)

# Single function call
history = train_gflownet(model, config; verbose=true)
```

### Old Model Creation → Modern Interface

#### OLD (Manual Setup)
```julia
# Create DAG manually
initial_state = GridState(1, 1)
terminal_states = [GridState(5, 5)]
dag = DirectedAcyclicGraph()
add_state!(dag, initial_state)
explore_dag!(dag, initial_state, actions)

# Create policies manually
forward_net = create_mlp(state_dim, action_dim, hidden_dim)
forward_policy = ForwardPolicy(forward_net)

# Assemble model manually
model = GFlowNetModel(
    dag = dag,
    forward_policy = forward_policy,
    partition_function = 1.0
)
```

#### NEW (High-Level Interface)
```julia
# Single function creates everything
model = create_grid_world_gflownet(
    grid_size = 5,
    hidden_dim = 64,
    learning_rate = 0.01,
    include_backward = false  # Optional backward policy
)
```

### Old State/Action Interface → Modern Interface

#### OLD (Explicit DAG Methods)
```julia
# Get next states from DAG
next_states = get_next_states(model.dag, current_state)

# Get applicable actions from DAG
actions = get_actions_to_states(model.dag, current_state, next_states)

# Check if state exists in DAG
if haskey(model.dag.states, state)
    # ...
end
```

#### NEW (On-Demand Computation)
```julia
# Get applicable actions directly
applicable_actions = get_applicable_actions(current_state, model.all_actions)

# Apply action to get next state
next_state = apply_action(action, current_state)

# No need to check DAG - compute on demand
if is_applicable(action, state)
    # ...
end
```

### Old Neural Network → Modern Lux.jl

#### OLD (Flux or Custom)
```julia
# Flux-style model
forward_net = Chain(
    Dense(state_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, action_dim)
)

# Manual parameter handling
params = Flux.params(forward_net)
grads = gradient(params) do
    loss(forward_net, data)
end
```

#### NEW (Lux.jl + ComponentArrays)
```julia
# Lux model with explicit state
forward_net = Lux.Chain(
    Lux.Dense(state_dim => hidden_dim, tanh),
    Lux.Dense(hidden_dim => hidden_dim, tanh),
    Lux.Dense(hidden_dim => action_dim)
)

# Automatic parameter/state handling
ps, st = Lux.setup(rng, forward_net)
model.parameters = ComponentArray(forward=ps)
model.states = (forward=st,)
```

## Step-by-Step Migration

### Step 1: Update Dependencies

```toml
# Project.toml
[deps]
GFlowNet = "..."
Lux = "..."
ComponentArrays = "..."
Optimisers = "..."
Zygote = "..."
```

### Step 2: Update State/Action Types

Ensure your types follow the modern interface:

```julia
# State must have is_terminal field
struct MyState <: AbstractState
    data::MyData
    is_terminal::Bool  # Required!
end

# Actions must inherit from AbstractAction
abstract type MyAction <: AbstractAction end
```

### Step 3: Implement Required Functions

The 5 required functions:
```julia
# 1. State to features
GFlowNet.state_to_features(::MyState)::Vector{Float32}

# 2. Terminal check
GFlowNet.is_terminal_state(::MyState)::Bool

# 3. Reward (positive!)
GFlowNet.reward(::MyState)::Float64

# 4. Action applicability
GFlowNet.is_applicable(::MyAction, ::MyState)::Bool

# 5. Apply action (pure function!)
GFlowNet.apply_action(::MyAction, ::MyState)::MyState
```

### Step 4: Create High-Level Constructor

```julia
function create_my_domain_gflownet(; kwargs...)
    # Initialize
    initial_state = ...
    all_actions = ...
    state_dim = length(state_to_features(initial_state))
    
    # Use standard creator
    return create_gflownet(
        initial_state,
        all_actions;
        state_dim = state_dim,
        kwargs...
    )
end
```

### Step 5: Update Training Code

Replace manual loops with configuration:

```julia
# Create model
model = create_my_domain_gflownet()

# Configure training
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 1000,
    batch_size = 32
)

# Train
history = train_gflownet(model, config; verbose=true)
```

## Common Migration Issues

### Issue 1: Missing DAG Functions

**Error**: `get_next_states not defined`

**Solution**: Use on-demand computation:
```julia
# Instead of: next_states = get_next_states(dag, state)
applicable_actions = get_applicable_actions(state, all_actions)
next_states = [apply_action(a, state) for a in applicable_actions]
```

### Issue 2: Partition Function Updates

**Error**: `partition_function field not found`

**Solution**: The framework handles this automatically:
```julia
# Remove manual updates like:
# model.partition_function = estimate_partition_function(model)

# The framework assumes Z=1 for fixed initial states
```

### Issue 3: Gradient Computation

**Error**: `compute_loss_and_grad not defined`

**Solution**: Use the modern training interface:
```julia
# Remove manual gradient code
# The framework handles gradients internally
history = train_gflownet(model, config)
```

### Issue 4: State Mutations

**Error**: `Mutating arrays is not supported`

**Solution**: Create new states:
```julia
# ❌ OLD: Mutate state
state.position[1] += 1

# ✅ NEW: Create new state
new_position = copy(state.position)
new_position[1] += 1
new_state = MyState(new_position, false)
```

## Feature Comparison

| Feature | Legacy | Modern |
|---------|---------|---------|
| DAG Construction | Explicit | On-demand |
| Training Loop | Manual | Configuration-based |
| Neural Networks | Flux/custom | Lux.jl |
| Parameters | Manual handling | ComponentArrays |
| Gradients | Manual computation | Automatic (Zygote) |
| Partition Function | Manual updates | Assumes Z=1 |
| Backward Policy | Not supported | Optional |

## Performance Improvements

The modern framework is generally faster:
- **On-demand computation** avoids storing large DAGs
- **Lux.jl** provides better AD performance
- **ComponentArrays** enable efficient parameter updates
- **Batch operations** are properly vectorized

## Compatibility Layer

For gradual migration, you can create compatibility wrappers:

```julia
# Compatibility wrapper for old code
function get_next_states_compat(model, state)
    actions = get_applicable_actions(state, model.all_actions)
    return [apply_action(a, state) for a in actions]
end

# Alias for old function names
const get_possible_actions = get_applicable_actions
```

## Testing Migration

Ensure your migrated code works:

```julia
# Test 1: Model creation
model = create_my_domain_gflownet()
@assert !isnothing(model)

# Test 2: Trajectory sampling
traj = sample_trajectory(model)
@assert length(traj.states) >= 2

# Test 3: Training
config = TrainingConfig(n_iterations=10, batch_size=4)
history = train_gflownet(model, config)
@assert length(history.losses) == 10

# Test 4: Results
trajectories = [sample_trajectory(model) for _ in 1:100]
rewards = [reward(t.states[end]) for t in trajectories]
@assert all(r > 0 for r in rewards)
```

## Getting Help

1. **Check examples**: See migrated implementations in `examples/`
2. **Read the source**: Core implementation in `src/core/`
3. **Run tests**: Ensure interface compliance
4. **Use debug mode**: Set `verbose=true` during training

## See Also
- [Training System](training_system.md) - Modern training interface
- [Developer Guide](developer_guide.md) - Implementing new domains
- [Architecture Analysis](../internals/architecture.md) - Why changes were made
- [Examples](../guide/examples.md) - Migrated examples