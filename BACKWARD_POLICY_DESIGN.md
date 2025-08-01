# Backward Policy Implementation Design

## Overview

This document outlines how to implement a complete trajectory balance with backward policy in GFlowNet.jl without requiring an explicit DAG.

## Mathematical Foundation

The full trajectory balance condition requires:
```
log Z(s₀) + Σ log P_F(s_{i+1}|s_i) - log R(s_T) - Σ log P_B(s_i|s_{i+1}) = 0
```

## Implementation Approach: Learned Backward Policy

### 1. Backward Policy Network Architecture

```julia
struct BackwardPolicy
    # Network that takes (source_state, target_state) and outputs P_B(source|target)
    joint_network::Chain
    
    # Alternative: Network that takes target_state and outputs distribution
    # conditional_network::Chain
end
```

### 2. Two Possible Architectures

#### Option A: Joint State Representation
```julia
function backward_probability_joint(policy::BackwardPolicy, 
                                  source_state::S, 
                                  target_state::S) where S
    # Concatenate features of both states
    source_features = state_to_features(source_state)
    target_features = state_to_features(target_state)
    joint_features = vcat(source_features, target_features)
    
    # Network outputs single probability value
    prob = policy.joint_network(joint_features)
    return sigmoid(prob)  # Ensure valid probability
end
```

**Pros:**
- Simple architecture
- Fixed output dimension
- No need to enumerate parents

**Cons:**
- Must be called for each possible parent
- Doesn't guarantee normalization

#### Option B: Conditional Distribution
```julia
function backward_probability_conditional(policy::BackwardPolicy,
                                        source_state::S,
                                        target_state::S,
                                        model::GFlowNetModel) where S
    # Get all possible parent states dynamically
    parent_states = get_possible_parents(target_state, model.all_actions)
    
    # Network takes target state and outputs distribution over parents
    target_features = state_to_features(target_state)
    logits = policy.conditional_network(target_features)
    
    # Match logits to parent states
    # This requires consistent ordering!
    source_idx = findfirst(s -> s == source_state, parent_states)
    
    probs = softmax(logits[1:length(parent_states)])
    return probs[source_idx]
end
```

**Pros:**
- Guarantees proper probability distribution
- More similar to forward policy

**Cons:**
- Requires parent enumeration
- Variable output size challenges

### 3. Hybrid Approach (Recommended)

```julia
function backward_transition_probability(model::GFlowNetModel, 
                                       target_state::S, 
                                       source_state::S) where S
    if isnothing(model.backward_policy)
        # Default: uniform over possible parents
        parents = get_possible_parents(target_state, model.all_actions)
        return 1.0 / length(parents)
    end
    
    # Use learned backward policy
    # Check if transition is valid first
    if !is_valid_transition(source_state, target_state, model.all_actions)
        return 0.0
    end
    
    # Compute backward probability using neural network
    return compute_backward_prob(model.backward_policy, source_state, target_state)
end
```

## Implementation Steps

### Step 1: Add Backward Policy to Model
```julia
@kwdef struct GFlowNetModel
    initial_state::AbstractState
    all_actions::Vector{<:AbstractAction}
    forward_policy::ForwardPolicy
    backward_policy::Union{Nothing,BackwardPolicy}  # Add this
    flow_estimator::Union{Nothing,FlowEstimator}
    parameters::ComponentArray
    optimizer
    states::NamedTuple
end
```

### Step 2: Create Backward Policy Network
```julia
function create_backward_policy(state_dim::Int, hidden_dim::Int, rng)
    # For joint representation approach
    input_dim = state_dim * 2  # source + target features
    
    backward_net = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => 1)  # Single probability output
    )
    
    ps, st = Lux.setup(rng, backward_net)
    return BackwardPolicy(backward_net), ps, st
end
```

### Step 3: Helper Functions
```julia
# Find possible parent states (for validation/initialization)
function get_possible_parents(target_state::S, all_actions::Vector{A}) where {S,A}
    parents = S[]
    
    # This is domain-specific and challenging!
    # For grid world example:
    for action in all_actions
        # Try inverse action
        if action isa MoveAction
            # Inverse of moving right is moving left, etc.
            inverse_dir = opposite_direction(action.direction)
            potential_parent = apply_action(MoveAction(inverse_dir), target_state)
            
            # Verify this actually leads to target
            if apply_action(action, potential_parent) == target_state
                push!(parents, potential_parent)
            end
        end
    end
    
    return unique(parents)
end

# Check if a transition is valid
function is_valid_transition(source::S, target::S, all_actions::Vector{A}) where {S,A}
    applicable = get_applicable_actions(source, all_actions)
    for action in applicable
        if apply_action(action, source) == target
            return true
        end
    end
    return false
end
```

### Step 4: Training Considerations

1. **Data Collection**: During trajectory sampling, store both forward and backward transitions
2. **Backward Policy Loss**: Can train separately or jointly with forward policy
3. **Consistency Regularization**: Ensure P_F and P_B are consistent

```julia
# Additional loss term for backward policy training
function backward_policy_loss(model, trajectory)
    loss = 0.0
    
    for i in 1:(length(trajectory.states)-1)
        source = trajectory.states[i]
        target = trajectory.states[i+1]
        
        # Backward policy should assign high probability to actual parent
        p_b = backward_transition_probability(model, target, source)
        loss += -log(p_b)
    end
    
    return loss / (length(trajectory.states) - 1)
end
```

## Efficiency Comparison

### Explicit DAG
- **Memory**: O(|S|²) for storing edges
- **Time**: O(1) parent lookup
- **Flexibility**: Limited to finite spaces

### Dynamic Discovery
- **Memory**: O(|A|) for actions only
- **Time**: O(|A|) per parent lookup
- **Flexibility**: Works with infinite spaces

### Learned Backward Policy
- **Memory**: O(θ) for network parameters
- **Time**: O(1) for probability computation
- **Flexibility**: Most general, but requires training

## Recommendation

For GFlowNet.jl, implement the **learned backward policy** approach:

1. It aligns with the existing forward policy design
2. No need for explicit parent enumeration
3. Works with continuous/infinite spaces
4. Most flexible for future extensions

The key insight is that we don't need to know ALL parents - we just need to compute P_B(s|s') for the specific transitions in our trajectory.