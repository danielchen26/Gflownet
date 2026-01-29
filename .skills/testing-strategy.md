---
name: testing-strategy
description: Comprehensive testing strategy for GFlowNet features - property-based testing, mathematical validation, performance benchmarking
---

<EXTREMELY-IMPORTANT>
Every new feature MUST have tests. Use this skill when:
- ✅ Implementing new training objectives
- ✅ Creating new domain applications
- ✅ Adding core functionality
- ✅ Fixing bugs (regression tests)

This skill creates a TodoWrite checklist to ensure comprehensive test coverage.
</EXTREMELY-IMPORTANT>

## When to Use This Skill

Invoke this skill when:
- 🧪 Writing tests for new features
- 🧪 Creating tests for new domains
- 🧪 Adding property-based mathematical tests
- 🧪 Setting up performance benchmarks
- 🧪 Creating regression tests for bug fixes

## Testing Strategy Workflow

### Phase 1: Test Organization

GFlowNet tests follow a hierarchical structure:

```
test/
├── runtests.jl                 # Main test runner
├── core/                       # Core functionality tests
│   ├── test_graphs.jl
│   ├── test_policies.jl
│   ├── test_flows.jl
│   └── test_sampling.jl
├── training/                   # Training infrastructure tests
│   ├── test_configuration.jl
│   ├── test_training.jl
│   └── test_objectives.jl
├── objectives/                 # Specific objective tests
│   ├── trajectory_balance/
│   ├── detailed_balance/
│   ├── flow_matching/
│   └── sub_trajectory_balance/
├── applications/               # Domain-specific tests
│   ├── test_grid_world.jl
│   └── test_molecules.jl
└── utils/                      # Utility function tests
    └── test_validation.jl
```

**Task 1.1: Create Test File Structure**

Place tests in the appropriate category:
- Core algorithm tests → `test/core/`
- Training tests → `test/training/`
- Objective tests → `test/objectives/<objective_name>/`
- Domain tests → `test/applications/`

### Phase 2: Mathematical Property Tests

**The most important tests verify mathematical correctness.**

**Task 2.1: Flow Conservation Tests**

For flow-based objectives (DETAILED_BALANCE, FLOW_MATCHING):

```julia
using Test
using GFlowNet

@testset "Flow Conservation Properties" begin
    # Setup
    model = create_test_model(include_backward=true)
    initial_state = create_initial_state()

    # Sample trajectories
    trajectories = [sample_trajectory(model) for _ in 1:100]

    @testset "Forward-Backward Consistency" begin
        for traj in trajectories
            for i in 1:(length(traj.states)-1)
                s = traj.states[i]
                s_next = traj.states[i+1]
                action = traj.actions[i]

                # Compute forward and backward probabilities
                p_forward = forward_action_probability(model, s, action)
                p_backward = backward_state_probability(model, s_next, s)

                # Flow conservation (with tolerance)
                @test p_forward * flow(s) ≈ p_backward * flow(s_next) atol=1e-5
            end
        end
    end
end
```

**Task 2.2: Partition Function Tests**

For learnable partition functions:

```julia
@testset "Partition Function Properties" begin
    model = create_test_model(
        partition_function_method = LEARNABLE_ESTIMATION
    )

    @testset "Z is Positive" begin
        Z = partition_function(model)
        @test Z > 0
        @test isa(Z, Float32)
    end

    @testset "Z Updates During Training" begin
        Z_initial = partition_function(model)

        # Train for a few iterations
        config = TrainingConfig(n_iterations=10, batch_size=8)
        train_gflownet(model, config; verbose=false)

        Z_final = partition_function(model)
        @test Z_final != Z_initial  # Should have updated
    end
end
```

**Task 2.3: Reward Validation Tests**

Ensure rewards meet GFlowNet requirements:

```julia
@testset "Reward Properties" begin
    @testset "Terminal Rewards Positive" begin
        terminal_states = generate_terminal_states(10)
        for state in terminal_states
            r = reward(state)
            @test r > 0  # CRITICAL: Must be positive
            @test isa(r, Float32)
        end
    end

    @testset "Non-Terminal Rewards Zero" begin
        non_terminal_states = generate_non_terminal_states(10)
        for state in non_terminal_states
            @test reward(state) == 0.0f0
        end
    end
end
```

**Task 2.4: Probability Normalization Tests**

```julia
@testset "Probability Normalization" begin
    model = create_test_model()
    state = create_test_state()
    actions = get_applicable_actions(state)

    # Get action probabilities
    probs = [forward_action_probability(model, state, action)
             for action in actions]

    @testset "Probabilities Sum to 1" begin
        @test sum(probs) ≈ 1.0 atol=1e-6
    end

    @testset "All Probabilities Non-Negative" begin
        @test all(p -> p >= 0, probs)
    end

    @testset "All Probabilities Finite" begin
        @test all(isfinite, probs)
    end
end
```

### Phase 3: Zygote Compatibility Tests

**CRITICAL**: Test that gradient computation works.

**Task 3.1: Gradient Computation Tests**

```julia
using Zygote

@testset "Zygote Compatibility" begin
    model = create_test_model()

    @testset "apply_action is Differentiable" begin
        state = create_test_state()
        action = create_test_action()

        # Should not throw mutation errors
        @test_nowarn gradient(x -> begin
            new_state = apply_action(action, state)
            sum(state_to_features(new_state))
        end, 1.0)
    end

    @testset "Loss Gradient Computation" begin
        trajectories = [sample_trajectory(model) for _ in 1:10]

        # Compute gradients
        @test_nowarn grads = gradient(model.parameters) do params
            compute_trajectory_loss(model, trajectories)
        end

        # Gradients should be finite
        grads = gradient(model.parameters) do params
            compute_trajectory_loss(model, trajectories)
        end
        @test all(isfinite, values(grads[1]))
    end

    @testset "No Mutations in Differentiable Code" begin
        # This should NOT error
        state = create_test_state()
        action = create_test_action()

        @test_throws nothing gradient(1.0) do x
            s = apply_action(action, state)
            sum(state_to_features(s)) * x
        end
    end
end
```

**Task 3.2: Backward Policy Gradient Tests**

For objectives requiring backward policy:

```julia
@testset "Backward Policy Gradients" begin
    model = create_test_model(include_backward=true)
    trajectories = [sample_trajectory(model) for _ in 1:10]

    @testset "Backward Policy is Differentiable" begin
        @test_nowarn grads = gradient(model.parameters) do params
            total = 0.0f0
            for traj in trajectories
                for i in 2:length(traj.states)
                    s = traj.states[i-1]
                    s_next = traj.states[i]
                    p_back = backward_state_probability(model, s_next, s, params)
                    total += p_back
                end
            end
            total
        end
    end

    @testset "Backward Policy Normalization Validates" begin
        state = create_test_state()
        is_valid, total_prob, parents = validate_backward_policy_normalization(
            model, state
        )
        @test is_valid
        @test total_prob ≈ 1.0 atol=1e-5
    end
end
```

### Phase 4: Training Objective Tests

**Task 4.1: TRAJECTORY_BALANCE Tests**

```julia
@testset "TRAJECTORY_BALANCE Objective" begin
    model = create_test_model(include_backward=false)

    @testset "Loss Computation" begin
        trajectories = [sample_trajectory(model) for _ in 1:20]

        # Loss should be computable
        loss = compute_trajectory_loss(model, trajectories)
        @test isa(loss, Real)
        @test isfinite(loss)
        @test loss >= 0  # MSE-based loss
    end

    @testset "Training Reduces Loss" begin
        config = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            n_iterations = 50,
            batch_size = 16
        )

        history = train_gflownet(model, config; verbose=false)

        # Loss should generally decrease
        initial_loss = mean(history.losses[1:10])
        final_loss = mean(history.losses[end-9:end])
        @test final_loss < initial_loss
    end
end
```

**Task 4.2: DETAILED_BALANCE Tests**

```julia
@testset "DETAILED_BALANCE Objective" begin
    model = create_test_model(include_backward=true)

    @testset "Requires Backward Policy" begin
        # Should error without backward policy
        model_no_back = create_test_model(include_backward=false)
        config = TrainingConfig(objective = DETAILED_BALANCE)

        @test_throws ErrorException train_gflownet(model_no_back, config)
    end

    @testset "Edge-Level Balance" begin
        trajectories = [sample_trajectory(model) for _ in 1:20]
        loss = compute_detailed_balance_loss(model, trajectories)

        @test isa(loss, Real)
        @test isfinite(loss)
    end

    @testset "Training with DB" begin
        config = TrainingConfig(
            objective = DETAILED_BALANCE,
            n_iterations = 50,
            batch_size = 16
        )

        @test_nowarn train_gflownet(model, config; verbose=false)
    end
end
```

**Task 4.3: SUB_TRAJECTORY_BALANCE Tests**

```julia
@testset "SUB_TRAJECTORY_BALANCE Objective" begin
    model = create_test_model()

    @testset "Sub-Trajectory Sampling" begin
        trajectory = sample_trajectory(model)
        sub_len = 3

        # Should be able to extract sub-trajectories
        sub_trajs = extract_sub_trajectories(trajectory, sub_len)
        @test length(sub_trajs) >= 0
        @test all(st -> length(st.states) == sub_len + 1, sub_trajs)
    end

    @testset "O(T²) Learning Signals" begin
        trajectory = sample_trajectory(model)
        T = length(trajectory.states)

        sub_trajs = extract_sub_trajectories(trajectory, 3)
        # Should get approximately O(T²) sub-trajectories
        @test length(sub_trajs) > T  # More signals than trajectory length
    end

    @testset "Training with STB" begin
        config = TrainingConfig(
            objective = SUB_TRAJECTORY_BALANCE,
            n_iterations = 50,
            batch_size = 8
        )

        @test_nowarn train_gflownet(model, config; verbose=false)
    end
end
```

**Task 4.4: FLOW_MATCHING Tests**

```julia
@testset "FLOW_MATCHING Objective" begin
    model = create_test_model(include_flow_estimator=true)

    @testset "Requires Flow Estimator" begin
        model_no_flow = create_test_model(include_flow_estimator=false)
        config = TrainingConfig(objective = FLOW_MATCHING)

        @test_throws ErrorException train_gflownet(model_no_flow, config)
    end

    @testset "Direct Flow Estimation" begin
        state = create_test_state()

        # Should compute flow directly
        flow_est = estimate_flow(model, state)
        @test isa(flow_est, Float32)
        @test isfinite(flow_est)
    end

    @testset "Loss: (Z(s) - F(s))²" begin
        trajectories = [sample_trajectory(model) for _ in 1:20]
        loss = compute_flow_matching_loss(model, trajectories)

        @test isa(loss, Real)
        @test isfinite(loss)
        @test loss >= 0
    end
end
```

**Task 4.5: DIRECT_FLOW_OBJECTIVE Tests**

```julia
@testset "DIRECT_FLOW_OBJECTIVE" begin
    model = create_test_model(include_flow_estimator=true)

    @testset "Neural Flow Estimation" begin
        config = TrainingConfig(
            objective = DIRECT_FLOW_OBJECTIVE,
            n_iterations = 50
        )

        @test_nowarn train_gflownet(model, config; verbose=false)
    end

    @testset "Flow Network Output" begin
        state = create_test_state()
        flow_value = estimate_flow(model, state)

        @test flow_value > 0  # Flows should be positive
        @test isfinite(flow_value)
    end
end
```

### Phase 5: Domain Interface Tests

**Task 5.1: Interface Compliance Tests**

For any new domain:

```julia
@testset "YourDomain Interface Compliance" begin
    state = YourDomainState(...)
    action = YourDomainAction(...)

    @testset "state_to_features" begin
        features = state_to_features(state)
        @test isa(features, Vector{Float32})
        @test length(features) > 0
        @test all(isfinite, features)

        # Consistent dimension
        state2 = YourDomainState(...)
        @test length(state_to_features(state2)) == length(features)
    end

    @testset "is_applicable" begin
        result = is_applicable(action, state)
        @test isa(result, Bool)

        # Terminal states shouldn't accept actions
        terminal = YourDomainState(..., is_terminal=true)
        @test !is_applicable(action, terminal)
    end

    @testset "apply_action returns correct type" begin
        new_state = apply_action(action, state)
        @test isa(new_state, YourDomainState)
        @test new_state !== state  # Should be new instance
    end

    @testset "is_terminal_state" begin
        @test isa(is_terminal_state(state), Bool)
        terminal = YourDomainState(..., is_terminal=true)
        @test is_terminal_state(terminal) == true
    end

    @testset "reward" begin
        # Non-terminal → 0
        if !state.is_terminal
            @test reward(state) == 0.0f0
        end

        # Terminal → positive
        terminal = YourDomainState(..., is_terminal=true)
        @test reward(terminal) > 0
        @test isa(reward(terminal), Float32)
    end
end
```

### Phase 6: Integration Tests

**Task 6.1: End-to-End Training Tests**

```julia
@testset "End-to-End Training" begin
    @testset "Complete Training Loop" begin
        model = create_test_model()
        config = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            n_iterations = 100,
            batch_size = 16,
            verbose = false
        )

        history = train_gflownet(model, config)

        # Training completed
        @test length(history.losses) == 100
        @test all(isfinite, history.losses)

        # Model can sample
        trajectory = sample_trajectory(model)
        @test length(trajectory.states) > 0
        @test is_terminal_state(trajectory.states[end])
    end

    @testset "Different Objectives Work" begin
        objectives = [
            TRAJECTORY_BALANCE,
            DETAILED_BALANCE,
            SUB_TRAJECTORY_BALANCE,
            FLOW_MATCHING,
            DIRECT_FLOW_OBJECTIVE
        ]

        for objective in objectives
            model = create_appropriate_model(objective)
            config = TrainingConfig(
                objective = objective,
                n_iterations = 10,
                batch_size = 8
            )

            @test_nowarn train_gflownet(model, config; verbose=false)
        end
    end
end
```

### Phase 7: Performance Benchmarks

**Task 7.1: Training Speed Benchmarks**

```julia
using BenchmarkTools

@testset "Performance Benchmarks" begin
    model = create_test_model()

    @testset "Sampling Speed" begin
        # Should complete in reasonable time
        t = @belapsed sample_trajectory($model)
        @test t < 1.0  # Less than 1 second per trajectory

        println("Sampling time: $(round(t*1000, digits=2))ms")
    end

    @testset "Training Iteration Speed" begin
        config = TrainingConfig(n_iterations=1, batch_size=32)

        t = @belapsed train_gflownet($model, $config; verbose=false)
        @test t < 10.0  # Less than 10 seconds per iteration

        println("Training iteration time: $(round(t, digits=2))s")
    end

    @testset "Gradient Computation Speed" begin
        trajectories = [sample_trajectory(model) for _ in 1:32]

        t = @belapsed gradient($model.parameters) do params
            compute_trajectory_loss($model, $trajectories)
        end

        @test t < 5.0  # Less than 5 seconds

        println("Gradient computation time: $(round(t, digits=2))s")
    end
end
```

**Task 7.2: Memory Allocation Benchmarks**

```julia
@testset "Memory Efficiency" begin
    model = create_test_model()

    @testset "Sampling Allocations" begin
        allocs = @allocated sample_trajectory(model)
        # Should be reasonable (adjust threshold as needed)
        @test allocs < 1_000_000  # < 1MB

        println("Sampling allocations: $(allocs ÷ 1024)KB")
    end

    @testset "Training Allocations" begin
        config = TrainingConfig(n_iterations=1, batch_size=8)
        allocs = @allocated train_gflownet(model, config; verbose=false)

        # Should not have excessive allocations
        @test allocs < 100_000_000  # < 100MB

        println("Training allocations: $(allocs ÷ (1024^2))MB")
    end
end
```

### Phase 8: Regression Tests

**Task 8.1: Bug Fix Regression Tests**

When fixing a bug, add a regression test:

```julia
@testset "Regression: Issue #123 - Zygote mutation in apply_action" begin
    # This used to cause "Mutating arrays is not supported"
    state = GridState(1, 1, false)
    action = MoveUp()

    # Should now work with gradients
    @test_nowarn gradient(1.0) do x
        new_state = apply_action(action, state)
        sum(state_to_features(new_state)) * x
    end
end

@testset "Regression: Flow cache staleness" begin
    # Flow cache wasn't being cleared on parameter update
    model = create_test_model()

    # Compute flow
    state = create_test_state()
    flow1 = compute_flow(model, state)

    # Update parameters
    model.parameters = new_parameters()
    clear_flow_cache!()

    # Flow should be recomputed
    flow2 = compute_flow(model, state)
    @test flow1 != flow2  # Should have changed
end
```

## Complete Testing Checklist

Use TodoWrite to create this comprehensive testing checklist:

```markdown
Testing Checklist for [Feature/Domain Name]:

Mathematical Properties:
- [ ] Flow conservation tests (if applicable)
- [ ] Partition function properties
- [ ] Reward validation (positive for terminals)
- [ ] Probability normalization tests

Zygote Compatibility:
- [ ] apply_action gradient test
- [ ] Loss gradient computation test
- [ ] No mutation errors in differentiable code
- [ ] Backward policy gradients (if applicable)

Training Objectives:
- [ ] TRAJECTORY_BALANCE tests
- [ ] DETAILED_BALANCE tests (if using backward policy)
- [ ] SUB_TRAJECTORY_BALANCE tests (if applicable)
- [ ] FLOW_MATCHING tests (if using flow estimator)
- [ ] Loss decreases during training

Interface Compliance:
- [ ] state_to_features returns Vector{Float32}
- [ ] is_applicable returns Bool
- [ ] apply_action returns correct type
- [ ] is_terminal_state returns Bool
- [ ] reward returns positive Float32 for terminals

Integration:
- [ ] End-to-end training completes
- [ ] Can sample after training
- [ ] Different objectives work
- [ ] No errors during full workflow

Performance:
- [ ] Sampling speed benchmark
- [ ] Training speed benchmark
- [ ] Memory allocation reasonable
- [ ] No obvious performance regressions

Regression Tests:
- [ ] Add test for any bug fixes
- [ ] Previous bugs don't reoccur
```

## Test File Template

```julia
# test/your_feature/test_your_feature.jl

using Test
using GFlowNet
using Zygote

@testset "YourFeature Tests" begin
    @testset "Mathematical Properties" begin
        # Property tests here
    end

    @testset "Zygote Compatibility" begin
        # Gradient tests here
    end

    @testset "Interface Compliance" begin
        # Interface tests here
    end

    @testset "Integration Tests" begin
        # End-to-end tests here
    end

    @testset "Performance Benchmarks" begin
        # Benchmark tests here (optional, can be separate)
    end
end
```

## Running Tests

```bash
# Run all tests
julia --project=. test/runtests.jl

# Run specific test file
julia --project=. test/objectives/trajectory_balance/test_tb.jl

# Run with coverage
julia --project=. --code-coverage test/runtests.jl

# Run with verbose output
julia --project=. test/runtests.jl --verbose
```

This comprehensive testing strategy ensures all GFlowNet features are thoroughly validated for correctness, compatibility, and performance.
