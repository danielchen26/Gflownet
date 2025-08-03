# Multi-Start GFlowNets Implementation Plan

## Overview

Extend GFlowNet.jl to support multiple initial states, each with its own partition function Z(s₀). This is crucial for domains where:
- Different starting configurations lead to different solution spaces
- We want to sample from a mixture of GFlowNets
- The initial state distribution is part of the learning problem

## Mathematical Foundation

For multi-start GFlowNets:
- Initial states: S₀ = {s₀¹, s₀², ..., s₀ᵏ}
- Each initial state has its own partition function: Z(s₀ⁱ)
- Initial state distribution: P(s₀ⁱ) = Z(s₀ⁱ) / Σⱼ Z(s₀ʲ)
- Trajectory probability: P(τ | s₀ⁱ) = ∏ P_F(aₜ|sₜ)

## Design Approach

### 1. Extended Model Structure
```julia
mutable struct MultiStartGFlowNetModel
    initial_states::Vector{<:AbstractState}  # Multiple initial states
    all_actions::Vector{<:AbstractAction}
    forward_policy::ForwardPolicy
    backward_policy::Union{Nothing,BackwardPolicy}
    flow_estimator::Union{Nothing,FlowEstimator}
    log_partition_functions::Vector{Float64}  # Per-initial-state log Z values
    parameters::ComponentArray
    optimizer
    states::NamedTuple
end
```

### 2. Backward Compatible API
Keep existing single-start API working:
```julia
# Single initial state (current API)
model = create_gflownet(initial_state, actions, ...)

# Multiple initial states (new API)
model = create_multi_start_gflownet(initial_states, actions, ...)
```

### 3. Training Modifications

#### Trajectory Sampling
```julia
function sample_trajectory(model::MultiStartGFlowNetModel)
    # Sample initial state based on partition functions
    log_probs = model.log_partition_functions
    probs = softmax(log_probs)
    idx = sample_categorical(probs)
    initial_state = model.initial_states[idx]
    
    # Continue with standard sampling
    return sample_trajectory_from_state(model, initial_state)
end
```

#### Loss Computation
For trajectory balance with multiple starts:
```julia
L_TB(τ) = (log Z(s₀ⁱ) + Σ log P_F - log R(s_T))²
```
where s₀ⁱ is the initial state of trajectory τ.

### 4. Partition Function Learning

Each Z(s₀ⁱ) is learned independently:
```julia
parameters = ComponentArray(
    forward = forward_ps,
    flow = flow_ps,
    log_Z = log_partition_functions  # Vector of k values
)
```

## Implementation Steps

### Phase 1: Core Infrastructure
1. Create `MultiStartGFlowNetModel` type
2. Extend `create_gflownet` to handle vector of initial states
3. Modify parameter structure for multiple log Z values

### Phase 2: Training System
1. Update `sample_trajectory` to sample initial states
2. Modify loss computation to use correct Z(s₀ⁱ)
3. Ensure gradients flow to correct log Z parameters

### Phase 3: Utilities
1. Add analysis tools for initial state distribution
2. Visualization of per-initial-state flows
3. Convergence diagnostics per initial state

### Phase 4: Testing
1. Unit tests for multi-start functionality
2. Integration tests with existing objectives
3. Example demonstrating benefits

## Use Cases

### 1. Multi-Modal Distributions
When the target distribution has multiple modes, different initial states can specialize in different regions.

### 2. Hierarchical Generation
Start from different levels of abstraction (e.g., molecule scaffolds).

### 3. Transfer Learning
Pre-trained models for different initial configurations.

## Challenges and Solutions

### Challenge 1: Initial State Selection
**Solution**: Learn the initial state distribution jointly with policies.

### Challenge 2: Imbalanced Exploration
**Solution**: Regularization to encourage balanced usage of initial states.

### Challenge 3: Convergence Monitoring
**Solution**: Track metrics per initial state.

## Example Usage

```julia
# Create model with multiple initial states
initial_states = [
    GridState(1, 1, false),
    GridState(5, 5, false),
    GridState(3, 3, false)
]

model = create_multi_start_gflownet(
    initial_states,
    all_actions,
    state_dim = 3,
    hidden_dim = 64
)

# Training automatically handles initial state selection
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 1000
)

history = train_gflownet(model, config)

# Analyze initial state usage
initial_state_probs = softmax(model.log_partition_functions)
println("Initial state distribution: ", initial_state_probs)
```