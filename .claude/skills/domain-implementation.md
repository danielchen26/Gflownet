---
name: domain-implementation
description: Step-by-step workflow for implementing new GFlowNet domains with required interfaces and high-level API usage
---

<EXTREMELY-IMPORTANT>
When implementing a new GFlowNet domain, you MUST follow this structured workflow.

**Never manually define neural networks** - always use the high-level API (`create_gflownet()`).

This skill creates a TodoWrite checklist to ensure all required interface functions are implemented.
</EXTREMELY-IMPORTANT>

## When to Use This Skill

Invoke this skill when:
- 🎯 Implementing a new domain application for GFlowNet
- 🎯 Creating custom state and action spaces
- 🎯 Designing reward functions for specific problems
- 🎯 Translating real-world problems into GFlowNet formulations

## Domain Implementation Workflow

### Step 1: Define State and Action Types

Create a TodoWrite checklist with these tasks:

**Task 1.1: Define State Type**
```julia
# ✅ REQUIRED: Concrete state type extending AbstractState
struct YourDomainState <: GFlowNet.AbstractState
    # Domain-specific fields
    field1::Type1
    field2::Type2
    is_terminal::Bool  # REQUIRED: Terminal state indicator
end
```

**Requirements**:
- Must extend `GFlowNet.AbstractState`
- Must have `is_terminal::Bool` field
- Use concrete types (not `Any` or abstract types)
- Keep it immutable (no mutable struct unless necessary)

**Task 1.2: Define Action Types**
```julia
# ✅ REQUIRED: Action hierarchy
abstract type YourDomainAction <: GFlowNet.AbstractAction end

# Concrete action types
struct SpecificAction1 <: YourDomainAction
    # Optional: action parameters
end

struct SpecificAction2 <: YourDomainAction
    # Optional: action parameters
end

# ✅ BEST PRACTICE: Define action constants for zero-parameter actions
const ACTION_1 = SpecificAction1()
const ACTION_2 = SpecificAction2()
const TERMINATE = TerminateAction()

# ✅ REQUIRED: List all available actions
const ALL_ACTIONS = [ACTION_1, ACTION_2, TERMINATE]
```

**Requirements**:
- Must extend `GFlowNet.AbstractAction`
- Define concrete action types
- Create constants for zero-parameter actions
- Include a termination action

### Step 2: Implement Required Interface Functions

**CRITICAL**: All 5 interface functions must be implemented.

**Task 2.1: Implement `state_to_features`**
```julia
"""
Convert state to Float32 feature vector for neural network input.

REQUIREMENTS:
- Must return Vector{Float32}
- Must be deterministic (same state → same features)
- Feature dimension must be consistent across all states
- Must be Zygote-compatible (no mutations!)
"""
function GFlowNet.state_to_features(state::YourDomainState)
    # Extract relevant features
    features = Float32[
        Float32(state.field1),
        Float32(state.field2),
        # ... more features
    ]
    return features
end
```

**Validation**:
```julia
# Test feature dimension consistency
@assert length(state_to_features(state1)) == length(state_to_features(state2))
@assert eltype(state_to_features(state)) == Float32
```

**Task 2.2: Implement `is_applicable`**
```julia
"""
Check if an action can be applied to a state.

REQUIREMENTS:
- Must return Bool
- Must be fast (called frequently)
- Should prevent invalid state transitions
- Terminal states should reject non-termination actions
"""
function GFlowNet.is_applicable(action::YourDomainAction, state::YourDomainState)
    # Terminal states can't take actions
    state.is_terminal && return false

    # Domain-specific applicability logic
    if isa(action, SpecificAction1)
        return check_action1_applicable(state)
    elseif isa(action, SpecificAction2)
        return check_action2_applicable(state)
    else
        return true  # Termination always applicable
    end
end
```

**Validation**:
```julia
# Test basic properties
@assert !is_applicable(ACTION_1, terminal_state)  # Terminal rejects actions
@assert is_applicable(TERMINATE, any_state)       # Can always terminate
```

**Task 2.3: Implement `apply_action`** ⚠️ ZYGOTE-CRITICAL
```julia
"""
Apply action to state and return new state.

CRITICAL ZYGOTE REQUIREMENTS:
- ❌ NO MUTATIONS: No +=, -=, push!, append!, setindex!
- ✅ PURE FUNCTIONS: Return new state, don't modify input
- ✅ CONDITIONAL EXPRESSIONS: Use ternary operators
- Must be deterministic

This function is called during gradient computation!
"""
function GFlowNet.apply_action(action::YourDomainAction, state::YourDomainState)
    # ✅ CORRECT: Pure functional approach
    new_field1 = isa(action, SpecificAction1) ?
        compute_new_field1(state.field1, action) : state.field1

    new_field2 = isa(action, SpecificAction2) ?
        compute_new_field2(state.field2, action) : state.field2

    # Determine if new state is terminal
    is_terminal = isa(action, TerminateAction) || check_termination_condition(new_field1, new_field2)

    # Return new state (immutable construction)
    return YourDomainState(new_field1, new_field2, is_terminal)
end

# ❌ WRONG EXAMPLE (causes Zygote errors):
# function apply_action(action, state)
#     state.field1 += 1  # ← MUTATION! Will break Zygote!
#     return state
# end
```

**Zygote Validation**:
```julia
# Test AD compatibility
using Zygote
@test_nowarn gradient(x -> sum(state_to_features(apply_action(ACTION_1, state))), 1.0)
```

**Task 2.4: Implement `is_terminal_state`**
```julia
"""
Check if a state is terminal.

REQUIREMENTS:
- Must return Bool
- Must be consistent with state.is_terminal field
- Terminal states should have positive rewards
"""
GFlowNet.is_terminal_state(state::YourDomainState) = state.is_terminal
```

**Task 2.5: Implement `reward`**
```julia
"""
Compute reward for a state.

CRITICAL MATHEMATICAL REQUIREMENT:
- Terminal states MUST return POSITIVE rewards (R > 0)
- Non-terminal states MUST return 0
- Type must be Float32

GFlowNet mathematics require R(s_T) > 0 for flow balance.
"""
function GFlowNet.reward(state::YourDomainState)
    # Non-terminal states have zero reward
    !state.is_terminal && return 0.0f0

    # Terminal state reward (MUST BE POSITIVE!)
    reward_value = compute_domain_reward(state)

    # Ensure positivity
    return Float32(max(reward_value, 1e-8))
end
```

**Reward Validation**:
```julia
# Test reward properties
@assert reward(non_terminal_state) == 0.0f0
@assert reward(terminal_state) > 0.0f0
@assert isa(reward(state), Float32)
```

### Step 3: Create Model Using High-Level API

**NEVER manually define neural networks!** Use the high-level interface.

**Option A: Generic High-Level API (Recommended)**
```julia
function create_your_domain_model(;
    state_dim::Int = 10,
    hidden_dim::Int = 64,
    learning_rate::Float64 = 0.01,
    include_backward::Bool = false,
    include_flow_estimator::Bool = false
)
    # 1. Define initial state
    initial_state = YourDomainState(...)

    # 2. Define all actions
    all_actions = ALL_ACTIONS  # Your action constants

    # 3. Create model using high-level API
    model = GFlowNet.create_gflownet(
        initial_state,
        all_actions;
        state_dim = state_dim,
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        include_backward = include_backward,         # For DETAILED_BALANCE
        include_flow_estimator = include_flow_estimator  # For FLOW_MATCHING
    )

    return model
end
```

**Option B: Domain-Specific Convenience Function**
```julia
# Example: Grid World convenience function
model = GFlowNet.create_grid_world_gflownet(
    grid_size = 8,
    reward_positions = Dict((3,3) => 10.0, (8,8) => 15.0),
    hidden_dim = 64,
    learning_rate = 0.01
)

# Create similar for your domain
function GFlowNet.create_your_domain_gflownet(;
    domain_param1 = default1,
    domain_param2 = default2,
    hidden_dim::Int = 64,
    learning_rate::Float64 = 0.01
)
    # Set up initial state and actions based on domain parameters
    initial_state = setup_initial_state(domain_param1, domain_param2)
    actions = setup_actions(domain_param1, domain_param2)

    # Use generic create_gflownet internally
    return GFlowNet.create_gflownet(
        initial_state, actions;
        state_dim = calculate_state_dim(domain_param1),
        hidden_dim = hidden_dim,
        learning_rate = learning_rate
    )
end
```

### Step 4: Configure Training

**Task 4.1: Create Training Configuration**
```julia
using GFlowNet

# Choose appropriate training objective
config = GFlowNet.TrainingConfig(
    # Objective selection
    objective = GFlowNet.TRAJECTORY_BALANCE,  # Default, works for most cases
    # objective = GFlowNet.DETAILED_BALANCE,  # Better credit assignment, requires backward policy
    # objective = GFlowNet.SUB_TRAJECTORY_BALANCE,  # For long trajectories

    # Partition function method
    partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,  # ⭐ RECOMMENDED
    # partition_function_method = GFlowNet.SIMPLE_ESTIMATION,  # Faster but less accurate

    # Training hyperparameters
    n_iterations = 1000,
    batch_size = 32,
    learning_rate = 0.01,

    # Logging
    verbose = true,
    log_every = 100
)
```

**Task 4.2: Run Training**
```julia
# Create model
model = create_your_domain_model()

# Train
history = GFlowNet.train_gflownet(model, config; verbose=true)

# Analyze results
println("Final loss: $(history.losses[end])")
println("Training completed in $(length(history.losses)) iterations")
```

### Step 5: Testing and Validation

**Task 5.1: Create Domain Tests**
```julia
using Test
using GFlowNet

@testset "YourDomain Interface Tests" begin
    # Setup
    state = YourDomainState(...)
    action = ACTION_1

    @testset "state_to_features" begin
        features = state_to_features(state)
        @test isa(features, Vector{Float32})
        @test length(features) > 0
        @test all(isfinite, features)
    end

    @testset "is_applicable" begin
        @test isa(is_applicable(action, state), Bool)
        # Terminal states reject actions
        terminal_state = YourDomainState(..., true)
        @test !is_applicable(ACTION_1, terminal_state)
    end

    @testset "apply_action (Zygote compatible)" begin
        new_state = apply_action(action, state)
        @test isa(new_state, YourDomainState)

        # Test Zygote compatibility
        using Zygote
        @test_nowarn gradient(x -> sum(state_to_features(apply_action(action, state))), 1.0)
    end

    @testset "reward" begin
        # Non-terminal has zero reward
        @test reward(state) == 0.0f0 || state.is_terminal

        # Terminal has positive reward
        terminal_state = YourDomainState(..., true)
        @test reward(terminal_state) > 0.0f0
        @test isa(reward(terminal_state), Float32)
    end

    @testset "is_terminal_state" begin
        @test is_terminal_state(state) == state.is_terminal
    end
end

@testset "YourDomain Training Test" begin
    model = create_your_domain_model()
    config = TrainingConfig(n_iterations=10, batch_size=8)

    # Should complete without errors
    @test_nowarn history = train_gflownet(model, config; verbose=false)

    history = train_gflownet(model, config; verbose=false)
    @test length(history.losses) > 0
    @test all(isfinite, history.losses)
end
```

**Task 5.2: Mathematical Property Validation**
```julia
@testset "Mathematical Properties" begin
    model = create_your_domain_model()

    # Sample trajectories
    trajectories = [sample_trajectory(model) for _ in 1:100]

    # All trajectories should reach terminal states
    @test all(t -> is_terminal_state(t.states[end]), trajectories)

    # All terminal rewards should be positive
    terminal_rewards = [reward(t.states[end]) for t in trajectories]
    @test all(r -> r > 0, terminal_rewards)

    # Trajectory probabilities should be computable
    @test all(t -> length(t.actions) == length(t.states) - 1, trajectories)
end
```

## Common Implementation Patterns

### Pattern 1: Discrete State Spaces
```julia
# Example: Grid World
struct GridState <: AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

function state_to_features(state::GridState)
    return Float32[state.x, state.y, Float32(state.is_terminal)]
end
```

### Pattern 2: Continuous Features
```julia
# Example: Molecular design with continuous properties
struct MoleculeState <: AbstractState
    atoms::Vector{Int}
    positions::Vector{Float64}
    energy::Float64
    is_terminal::Bool
end

function state_to_features(state::MoleculeState)
    return Float32[
        length(state.atoms),
        mean(state.positions),
        std(state.positions),
        state.energy,
        Float32(state.is_terminal)
    ]
end
```

### Pattern 3: Composite Actions
```julia
# Actions with parameters
struct AddNode <: YourAction
    node_type::Int
end

# Generate all possible actions dynamically
function get_applicable_actions(state::YourState)
    actions = []
    for node_type in 1:NUM_NODE_TYPES
        action = AddNode(node_type)
        if is_applicable(action, state)
            push!(actions, action)
        end
    end
    return actions
end
```

## Checklist for Domain Implementation

Use TodoWrite to track these required steps:

```markdown
Domain Implementation Checklist:
- [ ] Step 1: Define state type (extends AbstractState, has is_terminal)
- [ ] Step 1: Define action types (extends AbstractAction, create constants)
- [ ] Step 2.1: Implement state_to_features (returns Vector{Float32})
- [ ] Step 2.2: Implement is_applicable (returns Bool)
- [ ] Step 2.3: Implement apply_action (Zygote-compatible, no mutations!)
- [ ] Step 2.4: Implement is_terminal_state (returns Bool)
- [ ] Step 2.5: Implement reward (positive for terminals, zero otherwise)
- [ ] Step 3: Create model using create_gflownet() high-level API
- [ ] Step 4: Configure training with TrainingConfig
- [ ] Step 4: Run training with train_gflownet()
- [ ] Step 5: Write interface tests
- [ ] Step 5: Write mathematical property tests
- [ ] Step 5: Validate Zygote compatibility
- [ ] Documentation: Add example to examples/ directory
- [ ] Documentation: Update README with domain description
```

## Training Objective Selection Guide

### TRAJECTORY_BALANCE (Default)
**When to use**:
- General-purpose applications
- Dense reward signals
- Shorter trajectories (< 20 steps)
- First implementation of a new domain

**Requirements**: None special

### DETAILED_BALANCE
**When to use**:
- Sparse rewards
- Long trajectories
- Need better credit assignment
- Complex state-action dependencies

**Requirements**: `include_backward=true` in model creation

### SUB_TRAJECTORY_BALANCE
**When to use**:
- Very long trajectories (> 50 steps)
- Sparse critical decisions
- Need more learning signals

**Requirements**: Configure `sub_trajectory_config` in TrainingConfig

### FLOW_MATCHING
**When to use**:
- Direct flow estimation preferred
- Large state spaces
- Avoid recursive flow computation

**Requirements**: `include_flow_estimator=true` in model creation

## Common Pitfalls and Solutions

### Pitfall 1: Mutations in apply_action
**Symptom**: "Mutating arrays is not supported" error
**Solution**: Use pure functional style with conditional expressions

### Pitfall 2: Negative or Zero Rewards
**Symptom**: NaN in loss, training divergence
**Solution**: Ensure `reward(terminal_state) > 0` always

### Pitfall 3: Type Instability
**Symptom**: Slow training, type errors
**Solution**: Use concrete types, Vector{Float32} not Vector{Any}

### Pitfall 4: Manual Neural Networks
**Symptom**: Hard to maintain, parameter issues
**Solution**: Always use `create_gflownet()` high-level API

### Pitfall 5: Missing Interface Functions
**Symptom**: MethodError during training
**Solution**: Implement all 5 required interface functions

## Complete Minimal Example

```julia
# File: examples/my_domain/my_domain.jl
using GFlowNet

# 1. Define types
struct MyState <: AbstractState
    value::Int
    is_terminal::Bool
end

abstract type MyAction <: AbstractAction end
struct Increment <: MyAction end
struct Terminate <: MyAction end

const INCREMENT = Increment()
const TERMINATE_ACTION = Terminate()
const ALL_MY_ACTIONS = [INCREMENT, TERMINATE_ACTION]

# 2. Implement interface
GFlowNet.state_to_features(s::MyState) = Float32[s.value, Float32(s.is_terminal)]
GFlowNet.is_applicable(a::MyAction, s::MyState) = !s.is_terminal
GFlowNet.apply_action(a::Increment, s::MyState) = MyState(s.value + 1, false)
GFlowNet.apply_action(a::Terminate, s::MyState) = MyState(s.value, true)
GFlowNet.is_terminal_state(s::MyState) = s.is_terminal
GFlowNet.reward(s::MyState) = s.is_terminal ? Float32(s.value) : 0.0f0

# 3. Create and train
model = GFlowNet.create_gflownet(
    MyState(0, false),
    ALL_MY_ACTIONS;
    state_dim = 2,
    hidden_dim = 32
)

config = GFlowNet.TrainingConfig(
    objective = GFlowNet.TRAJECTORY_BALANCE,
    partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
    n_iterations = 1000,
    batch_size = 32
)

history = GFlowNet.train_gflownet(model, config; verbose=true)
println("Training complete! Final loss: $(history.losses[end])")
```

This comprehensive workflow ensures your domain implementation follows all GFlowNet requirements and best practices.
