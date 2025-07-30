# Comprehensive GFlowNet Development Rules & Best Practices

*Generic development guidelines for robust, scalable GFlowNet domain implementations*

## 🚨 CRITICAL: Automatic Differentiation (Zygote) Compatibility

### The #1 Rule: NO IN-PLACE MUTATIONS in Differentiable Functions

**Any function that might be called during gradient computation MUST avoid mutations:**

```julia
# ❌ WRONG - Causes "Mutating arrays is not supported" error
function apply_action(action::MyAction, state::MyState)
    if isa(action, MoveAction)
        state.position += action.delta  # ← MUTATION! Breaks Zygote!
    end
    return state
end

# ✅ CORRECT - Pure functional approach
function apply_action(action::MyAction, state::MyState)
    # Use conditional expressions instead of mutations
    new_position = isa(action, MoveAction) ? 
        state.position + action.delta : state.position
    
    return MyState(new_position, state.other_field)
end
```

### Validation Must Be Non-Differentiable

**CRITICAL PATTERN**: All validation functions must be wrapped with `Zygote.@ignore`:

```julia
# ✅ CORRECT - Non-differentiable validation
function safe_model_call(model, features, parameters, states)
    # All validation wrapped to prevent inclusion in computational graph
    Zygote.@ignore validate_neural_network_input(features, "features")
    Zygote.@ignore validate_model_parameters(parameters, "parameters")
    
    # Convert to Float32 for type stability
    features = convert(Array{Float32}, features)
    
    # Actual computation (differentiable)
    outputs, new_states = model(features, parameters, states)
    
    # Output validation (non-differentiable)
    Zygote.@ignore validate_neural_network_output(outputs, "model output")
    
    return outputs, new_states
end
```

### AD-Friendly vs AD-Hostile Patterns

**✅ AD-FRIENDLY:**
```julia
# Pure transformations
new_state = transform(old_state)

# Conditional assignments  
x = condition ? value_a : value_b

# Construct new objects
return SomeStruct(computed_values...)

# Functional array operations
new_array = map(f, old_array)
new_array = [f(x) for x in old_array]

# Non-mutating broadcasting
result = old_array .+ increment
```

**❌ AD-HOSTILE:**
```julia
# In-place mutations
x += 1; y -= delta; z *= factor

# Array mutations
push!(array, item); append!(array, items); array[i] = value

# Field mutations
object.field = new_value

# Dictionary mutations during AD
dict[key] = value  # Can break in AD context
```

## 🏗️ Generic Domain Development Process

### Step 1: Define Your Domain's Core Types

Every GFlowNet domain needs exactly these components:

```julia
# 1. State Type - Represents any configuration in your domain
struct YourDomainState <: GFlowNet.AbstractState
    # Domain-specific fields that fully describe a configuration
    field1::Type1
    field2::Type2
    is_terminal::Bool  # REQUIRED: marks if this is a final state
end

# 2. Action Type - Represents transitions in your domain
abstract type YourDomainAction <: GFlowNet.AbstractAction end

# Concrete action types for your domain
struct ActionType1 <: YourDomainAction
    parameters::SomeType
end

struct ActionType2 <: YourDomainAction
    # Could be singleton (no fields) or parameterized
end
```

### Step 2: Implement Required GFlowNet Interface

**These 5 functions are MANDATORY for any domain:**

```julia
# REQUIRED: Convert state to neural network input
function GFlowNet.state_to_features(state::YourDomainState)::Vector{Float32}
    # Extract numerical features that capture the state
    # MUST return Vector{Float32} for type stability
    return Float32[extract_feature_1(state), extract_feature_2(state), ...]
end

# REQUIRED: Check if state is terminal (final configuration)
function GFlowNet.is_terminal_state(state::YourDomainState)::Bool
    return state.is_terminal
end

# REQUIRED: Compute reward for terminal states (MUST be positive)
function GFlowNet.reward(state::YourDomainState)::Float64
    # Only terminal states have rewards
    is_terminal_state(state) || return 0.0
    
    # Compute domain-specific reward
    reward_value = compute_domain_reward(state)
    
    # CRITICAL: Ensure positive rewards for GFlowNet mathematics
    return max(reward_value, 1e-8)
end

# REQUIRED: Check if action can be applied from state
function GFlowNet.is_applicable(action::YourDomainAction, state::YourDomainState)::Bool
    # Domain-specific logic to determine valid transitions
    # NO MUTATIONS - pure function only
    return check_domain_constraints(action, state)
end

# REQUIRED: Apply action to state (MUST be mutation-free)
function GFlowNet.apply_action(action::YourDomainAction, state::YourDomainState)::YourDomainState
    # Create NEW state - never mutate input
    # Use conditional expressions for Zygote compatibility
    
    new_field1 = compute_new_field1(action, state)
    new_field2 = compute_new_field2(action, state)
    new_terminal = should_be_terminal(action, state)
    
    return YourDomainState(new_field1, new_field2, new_terminal)
end
```

### Step 3: Implement Equality and Hashing (CRITICAL)

```julia
# CRITICAL: Required for Set operations in state space exploration
Base.:(==)(a::YourDomainState, b::YourDomainState) = (
    a.field1 == b.field1 && 
    a.field2 == b.field2 && 
    a.is_terminal == b.is_terminal
)

Base.hash(state::YourDomainState, h::UInt) = hash((state.field1, state.field2, state.is_terminal), h)

# Also for actions if they have parameters
Base.:(==)(a::ActionType1, b::ActionType1) = a.parameters == b.parameters
Base.:(==)(a::ActionType2, b::ActionType2) = true  # Singleton actions
```

### Step 4: Create High-Level Domain Interface

```julia
# Create domain-specific model creation function
function create_your_domain_gflownet(;
    domain_param1=default1,
    domain_param2=default2,
    hidden_dim::Int=64,
    learning_rate::Float64=0.01,
    rng::AbstractRNG=Random.default_rng()
)
    # Define initial state for your domain
    initial_state = create_initial_state(domain_param1, domain_param2)
    
    # Define complete action space for your domain
    actions = create_action_space(domain_param1, domain_param2)
    
    # Calculate state feature dimension
    state_dim = length(state_to_features(initial_state))
    
    # Use generic GFlowNet creation
    return GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = state_dim,
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        rng = rng
    )
end
```

## 🎯 Current Architecture: On-Demand Computation

### Core Principle: No Explicit DAG Construction

The current implementation uses **on-demand computation** instead of pre-building DAGs:

```julia
# ✅ CURRENT APPROACH: On-demand computation
# No explicit DAG - everything computed when needed

# Get applicable actions for any state
applicable_actions = get_applicable_actions(state, all_actions)

# Compute next state for any transition
next_state = compute_next_state(action, state)

# Validate transitions when needed
is_valid = is_valid_transition(action, state)

# Analyze state space (for debugging only)
space_analysis = analyze_state_space(initial_state, actions)
```

**Benefits of On-Demand Approach:**
1. ✅ **No Cache Misses** - No pre-computed structures to maintain
2. ✅ **Memory Efficient** - No large DAG stored in memory  
3. ✅ **Robust** - Eliminates complex DAG construction errors
4. ✅ **Simple** - Easy to understand and debug
5. ✅ **Generic** - Works for any domain without modification

### Mathematical Equivalence

The DAG still exists mathematically - it's just computed on-demand:
- **States S**: All states reachable from `initial_state`
- **Edges E**: `{(s,a,s') : is_applicable(a,s) ∧ apply_action(a,s) = s'}`
- **Properties**: All GFlowNet mathematical guarantees preserved

## ⚡ Generic High-Level Interface Usage

### Model Creation Pattern (Any Domain)

```julia
# Generic pattern for any domain
model = create_your_domain_gflownet(
    # Domain-specific parameters
    domain_param1=value1,
    domain_param2=value2,
    
    # Standard GFlowNet parameters
    hidden_dim=64,
    learning_rate=0.01
)
```

### Training Pattern (Domain-Agnostic)

```julia
# Standard training configuration for any domain
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,              # Standard objective
    partition_function_method=SIMPLE_ESTIMATION,
    n_iterations=100,                          # Adjust based on complexity
    batch_size=32,                            # Adjust based on memory
    learning_rate=0.01,                       # Standard starting point
    validation_frequency=10
)

# Train any domain model
history = train_gflownet(model, config; verbose=true)
```

### Evaluation Pattern (Domain-Agnostic)

```julia
# Sample trajectories for any domain
trajectories = [sample_trajectory(model) for _ in 1:100]

# Basic analysis (works for any domain)
valid_trajectories = filter(traj -> !isempty(traj.states), trajectories)
rewards = [reward(traj.states[end]) for traj in valid_trajectories]

println("Domain Results:")
println("  Valid trajectories: $(length(valid_trajectories))/$(length(trajectories))")
println("  Mean reward: $(round(mean(rewards), digits=2))")
println("  Max reward: $(maximum(rewards))")

# Domain-specific analysis function (implement per domain)
analyze_domain_results(trajectories, domain_specific_params...)
```

## 🔧 Generic Interface Requirements

### Type Safety Requirements (Any Domain)

```julia
# ✅ Consistent Float32 usage in features
function state_to_features(state::YourDomainState)::Vector{Float32}
    # All features must be Float32 for type stability
    feature1 = Float32(normalize_value(state.field1))
    feature2 = Float32(encode_categorical(state.field2))
    terminal_flag = state.is_terminal ? Float32(1.0) : Float32(0.0)
    return Float32[feature1, feature2, terminal_flag]
end

# ✅ Proper action applicability (no mutations)
function is_applicable(action::YourDomainAction, state::YourDomainState)::Bool
    # Pure function - no side effects
    return check_constraints(action, state) && 
           check_preconditions(action, state) &&
           !state.is_terminal
end

# ✅ Pure state transitions (Zygote-safe)
function apply_action(action::YourDomainAction, state::YourDomainState)::YourDomainState
    # Use conditional expressions, never mutations
    new_value = determine_new_value(action, state)
    new_terminal = determine_if_terminal(action, state)
    
    return YourDomainState(new_value, new_terminal)
end
```

### Reward Function Guidelines (Any Domain)

```julia
# Generic reward function pattern
function reward(state::YourDomainState)::Float64
    # Only terminal states have non-zero rewards
    is_terminal_state(state) || return 0.0
    
    # Compute domain-specific objective
    raw_reward = compute_domain_objective(state)
    
    # Apply transformations for GFlowNet compatibility
    if raw_reward <= 0
        # Handle negative objectives (e.g., minimizing cost)
        transformed_reward = exp(-raw_reward)  # or other transformation
    else
        transformed_reward = raw_reward
    end
    
    # Ensure numerical stability
    return max(transformed_reward, 1e-8)
end
```

## 🧪 Generic Testing Framework

### Domain Interface Testing (Any Domain)

```julia
@testset "Domain Interface Tests: $(typeof(YourDomainState))" begin
    # Create test instances
    test_state = create_test_state()
    test_actions = create_test_actions()
    
    @testset "Required Interface Methods" begin
        # Test state_to_features
        features = state_to_features(test_state)
        @test features isa Vector{Float32}
        @test !isempty(features)
        @test all(isfinite, features)
        
        # Test terminal checking
        @test is_terminal_state(test_state) isa Bool
        
        # Test reward computation
        reward_val = reward(test_state)
        @test reward_val isa Float64
        @test reward_val >= 0.0  # GFlowNet requirement
        
        # Test equality and hashing
        state_copy = deepcopy(test_state)
        @test test_state == state_copy
        @test hash(test_state) == hash(state_copy)
    end
    
    @testset "Action Interface" begin
        for action in test_actions
            # Test applicability
            applicable = is_applicable(action, test_state)
            @test applicable isa Bool
            
            if applicable
                # Test action application
                new_state = apply_action(action, test_state)
                @test new_state isa typeof(test_state)
                @test new_state !== test_state  # Must create new object
            end
        end
    end
    
    @testset "Zygote Compatibility" begin
        # Test that interface methods work with automatic differentiation
        @test_nowarn begin
            # Simulate gradient computation
            dummy_params = Float32[1.0, 2.0, 3.0]
            Zygote.gradient(p -> sum(state_to_features(test_state) .* p), dummy_params)
        end
    end
end
```

### Generic Integration Testing

```julia
@testset "Domain Integration Tests" begin
    @testset "Model Creation" begin
        model = create_your_domain_gflownet(
            # Use minimal test parameters
            hidden_dim=8,
            learning_rate=0.1
        )
        
        @test model isa GFlowNet.GFlowNetModel
        @test length(model.all_actions) > 0
        @test model.initial_state isa YourDomainState
    end
    
    @testset "Training Integration" begin
        model = create_your_domain_gflownet(hidden_dim=8)
        config = TrainingConfig(n_iterations=5, batch_size=4)
        
        # Should complete without errors
        @test_nowarn train_gflownet(model, config; verbose=false)
    end
    
    @testset "Sampling Integration" begin
        model = create_your_domain_gflownet(hidden_dim=8)
        
        # Should be able to sample trajectories
        trajectory = sample_trajectory(model)
        @test trajectory isa GFlowNet.Trajectory
        @test !isempty(trajectory.states)
        @test is_terminal_state(trajectory.states[end])
    end
end
```

## 📋 Generic Development Checklist

### Domain Implementation Checklist

**Core Types:**
- [ ] Define `YourDomainState <: AbstractState` with `is_terminal::Bool`
- [ ] Define action types inheriting from `AbstractAction`
- [ ] Implement `Base.==` and `Base.hash` for states and actions

**Required Interface (The Big 5):**
- [ ] `state_to_features(::YourDomainState)::Vector{Float32}`
- [ ] `is_terminal_state(::YourDomainState)::Bool`
- [ ] `reward(::YourDomainState)::Float64` (positive for terminals)
- [ ] `is_applicable(::YourDomainAction, ::YourDomainState)::Bool`
- [ ] `apply_action(::YourDomainAction, ::YourDomainState)::YourDomainState`

**High-Level Interface:**
- [ ] `create_your_domain_gflownet()` function
- [ ] Domain-specific result analysis function
- [ ] Complete working example script

**Quality Assurance:**
- [ ] All interface methods are type-stable (`@inferred` passes)
- [ ] No in-place mutations anywhere in domain code
- [ ] Positive rewards for all terminal states
- [ ] Comprehensive test suite
- [ ] Performance benchmarks

### Generic Training Checklist

**Model Setup:**
- [ ] Domain parameters properly configured
- [ ] Appropriate hidden dimension for domain complexity
- [ ] Reasonable learning rate (0.01 is good default)

**Training Configuration:**
- [ ] Appropriate objective (TRAJECTORY_BALANCE for most domains)
- [ ] Suitable batch size (16-32 for most domains)
- [ ] Sufficient iterations for convergence
- [ ] Regular validation monitoring

**Results Validation:**
- [ ] Training loss decreases over time
- [ ] No NaN or Inf values in training
- [ ] Sampled trajectories are valid
- [ ] Reward distribution makes domain sense

## 🎯 Generic Success Patterns

### Model Creation Pattern

```julia
# Generic high-level creation
function create_any_domain_gflownet(domain_config; ml_config...)
    # 1. Create domain-specific initial state and actions
    initial_state = create_domain_initial_state(domain_config)
    actions = create_domain_actions(domain_config)
    
    # 2. Use generic GFlowNet infrastructure
    state_dim = length(state_to_features(initial_state))
    
    return create_gflownet(
        initial_state, actions;
        state_dim=state_dim,
        ml_config...
    )
end
```

### Training Pattern

```julia
# Generic training workflow
function train_domain_model(domain_config, training_config)
    # 1. Create model
    model = create_your_domain_gflownet(domain_config...)
    
    # 2. Train
    history = train_gflownet(model, training_config; verbose=true)
    
    # 3. Evaluate
    trajectories = [sample_trajectory(model) for _ in 1:100]
    
    # 4. Analyze (domain-specific)
    return analyze_domain_results(trajectories, domain_config)
end
```

## 🔍 Generic Debugging Guide

### Common Issues Across All Domains

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Mutation errors | "Mutating arrays not supported" | Remove `+=`, `push!`, use pure functions |
| Type instability | Slow performance, allocations | Use concrete types, `Float32` consistently |
| Negative rewards | Training instability | Ensure `reward()` always returns positive values |
| Invalid trajectories | Sampling fails | Check `is_applicable()` and `apply_action()` logic |
| Memory issues | Out of memory during training | Reduce batch size, check for memory leaks |

### Generic Debugging Process

1. **Isolate the Component**: Test state interface, action interface, training separately
2. **Check Type Stability**: Use `@code_warntype` on interface methods
3. **Validate Mathematics**: Ensure rewards are positive, transitions are valid
4. **Test with Minimal Examples**: Use smallest possible domain configuration
5. **Profile Performance**: Identify bottlenecks with `@time` and `@profile`

## 🚀 Performance Guidelines (Any Domain)

### Memory Optimization

```julia
# ✅ Efficient state representation
struct EfficientState <: AbstractState
    # Use compact types where possible
    compact_field::UInt32      # Instead of Int64 if range allows
    flags::UInt8               # Pack boolean flags
    is_terminal::Bool
end

# ✅ Minimize allocations in hot paths
function efficient_apply_action(action, state)
    # Pre-compute values to avoid repeated calculations
    new_value = compute_once(action, state)
    return EfficientState(new_value, state.flags, determine_terminal(action))
end
```

### Computational Efficiency

```julia
# ✅ Cache expensive computations if domain-appropriate
const DOMAIN_CACHE = LRU{StateType, ComputationResult}(maxsize=1000)

function cached_expensive_computation(state)
    get!(DOMAIN_CACHE, state) do
        expensive_domain_computation(state)
    end
end

# ✅ Vectorize when possible
function batch_process_states(states::Vector{<:AbstractState})
    # Process multiple states at once
    return map(state_to_features, states)
end
```

## 📚 Success Indicators

**Your domain implementation is ready when:**
- ✅ All 5 required interface methods implemented and tested
- ✅ Training completes without Zygote errors
- ✅ Sampled trajectories are valid and diverse
- ✅ Rewards reflect domain objectives correctly
- ✅ Performance is acceptable for domain size
- ✅ Code is clean, documented, and maintainable

**Red flags indicating problems:**
- ❌ Zygote mutation errors during training
- ❌ Type instability warnings in interface methods
- ❌ Negative or zero rewards for terminal states
- ❌ Invalid or stuck trajectories during sampling
- ❌ Memory usage growing without bounds
- ❌ Training loss not decreasing

---

**🎯 These generic rules apply to developing ANY GFlowNet domain - from simple toy problems to complex real-world applications.**

*Follow this process for grid worlds, supply chain optimization, molecular design, causal discovery, or any other domain.*

*Last Updated: 2024-12-19*
*Version: 3.1.0 (Generic Development Guidelines)*