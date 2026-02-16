# GFlowNet Core Concepts

Mathematical foundations and key concepts in GFlowNet.jl.

## Overview

GFlowNets (Generative Flow Networks) learn to sample objects proportionally to a reward function by treating sampling as a flow problem on a directed acyclic graph (DAG).

**Core Idea**: Learn a policy that induces flow conservation, such that the probability of generating an object is proportional to its reward.

## Flow Conservation

The fundamental principle of GFlowNets:

**For non-terminal states**:
```
F(s) = Σ_{s'} F(s → s')
```
Outgoing flow equals total incoming flow.

**For terminal states**:
```
F(s_T) = R(s_T)
```
Flow equals reward (must be positive).

## Key Components

### 1. State Space

- **States (s)**: Points in the search space
- **Initial State (s₀)**: Starting point of trajectories
- **Terminal States (s_T)**: End points with positive rewards
- **DAG Structure**: States form a directed acyclic graph

### 2. Actions and Transitions

- **Actions (a)**: Transformations between states
- **Forward Policy (P_F)**: Probability of taking action a in state s
- **Backward Policy (P_B)**: Probability of transitioning from s' to s (optional)
- **Applicability**: Not all actions applicable in all states

### 3. Flows and Rewards

- **Flow (F)**: Amount of "probability mass" flowing through a state
- **Partition Function (Z)**: Normalization constant
- **Reward (R)**: Positive value assigned to terminal states

## Training Objectives

GFlowNet.jl implements 6 training objectives:

### TRAJECTORY_BALANCE (Default)

**Objective**: `Z · ∏ P_F(s'|s) = R(s_T)`

**When to use**:
- General-purpose applications
- Dense reward signals
- Relatively short trajectories

**Requirements**: Forward policy only

**Pros**: Simple, well-understood, works for most cases
**Cons**: O(T) learning signal, credit assignment challenges

### DETAILED_BALANCE

**Objective**: `P_F(s'|s) · F(s) = P_B(s|s') · F(s')`

**When to use**:
- Sparse or delayed rewards
- Complex state-action dependencies
- Need better credit assignment

**Requirements**: Backward policy (`include_backward=true`)

**Pros**: Local balance constraints, better credit assignment
**Cons**: Requires backward policy, more computation

### SUB_TRAJECTORY_BALANCE

**Objective**: Trajectory balance on sub-trajectories

**When to use**:
- Long trajectories (> 20-50 steps)
- Sparse critical decisions
- Want more learning signals

**Requirements**: None (domain-agnostic)

**Pros**: O(T²) learning signals, better for long horizons
**Cons**: More computation, hyperparameter tuning needed

### FLOW_MATCHING

**Objective**: Minimize `(Z(s) - F(s))²`

**When to use**:
- Large state spaces
- Avoid recursive flow computation
- Direct flow estimation preferred

**Requirements**: Flow estimator network (`include_flow_estimator=true`)

**Pros**: Direct flow estimation, faster than recursive
**Cons**: Approximation error, requires additional network

### DIRECT_FLOW_OBJECTIVE

**Objective**: Neural network directly estimates F(s)

**When to use**:
- Very large state spaces
- Computational efficiency critical
- Accept flow approximation

**Requirements**: Flow estimator network

**Pros**: Efficient, scales to large spaces
**Cons**: Less theoretically grounded than recursive flow

### COMBINED_OBJECTIVES

**Objective**: Weighted sum of multiple objectives

**When to use**:
- Advanced training scenarios
- Multiple constraints needed
- Research and experimentation

**Requirements**: Depends on component objectives

## Partition Function Methods

The partition function Z normalizes the distribution. Four methods available:

### SIMPLE_ESTIMATION (Default)

**Z = 1** (fixed constant)

**Pros**: Fast, no overhead
**Cons**: Inaccurate for multi-modal distributions
**Use when**: Quick experiments, single-mode problems

### LEARNABLE_ESTIMATION ⭐ **RECOMMENDED**

**Z as trainable parameter**

**Pros**:
- Theoretically correct
- Improves exploration (~42% better)
- Minimal computational overhead

**Cons**: One additional parameter

**Usage**:
```julia
model = create_gflownet(
    initial_state, actions;
    partition_function_method = LEARNABLE_ESTIMATION
)
```

### SAMPLING_ESTIMATION

**Z estimated from samples**

**Pros**: More accurate than fixed Z
**Cons**: Higher computational cost, variance

### ADAPTIVE_ESTIMATION

**Z adapts during training**

**Pros**: Balances accuracy and computation
**Cons**: More complex, hyperparameter sensitive

## Mathematical Properties

### Flow Balance

At equilibrium, GFlowNets satisfy:
```
P(τ) ∝ R(s_T)
```
Probability of trajectory τ proportional to terminal reward.

### Detailed Balance Property

For DETAILED_BALANCE objective:
```
P_F(s → s') · F(s) = P_B(s' → s) · F(s')
```
Local balance at every edge in the DAG.

### Partition Function

The normalization constant:
```
Z = Σ_{s_T ∈ S_T} R(s_T)
```
Sum of rewards over all terminal states.

## Reward Requirements

**CRITICAL**: GFlowNet mathematics require:

1. **Positive rewards**: `R(s_T) > 0` for all terminal states
2. **Zero for non-terminals**: `R(s) = 0` if `!is_terminal(s)`
3. **Finite values**: No `Inf` or `NaN`

```julia
function reward(state)
    !is_terminal(state) && return 0.0f0

    # MUST be positive
    return Float32(max(compute_reward(state), 1e-8))
end
```

## Backward Policy

Required for DETAILED_BALANCE and FLOW_MATCHING objectives.

**Forward Policy**: `P_F(s' | s, a)` - probability of reaching s' from s via action a

**Backward Policy**: `P_B(s | s')` - probability of coming from s given you're in s'

**Joint Representation**:
```julia
# Concatenate state features
features = [state_to_features(s); state_to_features(s')]
P_B(s|s') = softmax(backward_network(features))
```

**Validation**:
```julia
# Backward probabilities must normalize
is_valid, total_prob, parents = validate_backward_policy_normalization(model, state)
@assert total_prob ≈ 1.0
```

## DAG Construction

GFlowNet.jl uses **on-demand DAG construction**:

1. States discovered during trajectory sampling
2. Edges created as actions are applied
3. Caching for frequently-visited states
4. Lazy evaluation (memory efficient)

**Not pre-computed**: No explicit graph construction needed by users.

## Trajectory Sampling

```julia
# Sample trajectory from initial state
trajectory = sample_trajectory(model)

# Trajectory structure
trajectory.states    # Vector of states [s₀, s₁, ..., s_T]
trajectory.actions   # Vector of actions [a₀, a₁, ..., a_{T-1}]
trajectory.rewards   # Reward at terminal state
```

**Sampling process**:
1. Start from initial state
2. Apply forward policy to select action
3. Transition to next state
4. Repeat until terminal state reached

## Loss Functions

### Trajectory Balance Loss

```julia
# MSE-based loss
loss = (log(Z) + Σ log P_F(a_i|s_i) - log R(s_T))²
```

### Detailed Balance Loss

```julia
# Sum over edges
loss = Σ_{(s,s')} (log P_F(s'|s) + log F(s) - log P_B(s|s') - log F(s'))²
```

### Flow Matching Loss

```julia
# Per-state flow estimation
loss = Σ_s (Z(s) - F(s))²
```

## Type System

### Abstract Base Types

```julia
abstract type AbstractState end
abstract type AbstractAction end
```

### Required Interface

Every domain must implement:

```julia
# Convert state to neural network features
state_to_features(state::YourState) -> Vector{Float32}

# Check action applicability
is_applicable(action::YourAction, state::YourState) -> Bool

# Apply action (pure functional!)
apply_action(action::YourAction, state::YourState) -> YourState

# Terminal state check
is_terminal_state(state::YourState) -> Bool

# Reward computation (positive for terminals!)
reward(state::YourState) -> Float32
```

## Neural Network Policies

GFlowNet.jl uses Lux.jl for neural networks:

**Forward Policy**:
```julia
features = state_to_features(state)
logits = forward_policy_network(features, parameters, states)
action_probs = softmax(logits)
```

**Backward Policy** (if included):
```julia
joint_features = [state_to_features(s); state_to_features(s')]
logits = backward_policy_network(joint_features, parameters, states)
state_probs = softmax(logits)
```

**Flow Estimator** (if included):
```julia
features = state_to_features(state)
flow_value = flow_estimator_network(features, parameters, states)
```

## Common Patterns

### Single-Mode Reward

```julia
# Simple reward function
function reward(state::GridState)
    !state.is_terminal && return 0.0f0

    # Higher reward for specific positions
    if (state.x == target_x) && (state.y == target_y)
        return 10.0f0
    else
        return 1.0f0
    end
end
```

### Multi-Modal Reward

```julia
# Multiple high-reward regions
function reward(state::GridState)
    !state.is_terminal && return 0.0f0

    # Different peaks
    for (pos, reward_value) in reward_positions
        if (state.x, state.y) == pos
            return Float32(reward_value)
        end
    end

    return 1.0f0  # Default terminal reward
end
```

### Sparse Rewards

When rewards are very sparse, consider:
1. Use DETAILED_BALANCE for better credit assignment
2. Use SUB_TRAJECTORY_BALANCE for more learning signals
3. Use LEARNABLE_ESTIMATION for partition function
4. Shape rewards if domain allows

## Best Practices

1. **Always use positive rewards** - Mathematical requirement
2. **Start with TRAJECTORY_BALANCE** - Simplest, works for most cases
3. **Use LEARNABLE_ESTIMATION for Z** - Better than fixed Z = 1
4. **Consider DETAILED_BALANCE for sparse rewards** - Better credit assignment
5. **Test Zygote compatibility** - Ensure pure functional `apply_action`
6. **Validate backward policy if used** - Check normalization

## References

- [Architecture](architecture.md) - System design
- [Project Structure](project_structure.md) - File organization
- Original GFlowNet papers for mathematical foundations

## Further Reading

For implementation details, see:
- [src/core/balance.jl](../../src/core/balance.jl) - Loss functions
- [src/training/objectives.jl](../../src/training/objectives.jl) - Objective enumeration
- [src/training/losses.jl](../../src/training/losses.jl) - Loss computation
