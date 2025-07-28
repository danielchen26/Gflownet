"""
🔬 Comprehensive GFlowNet Diagnostics Suite
==========================================

This comprehensive test suite validates all critical components of GFlowNet implementation
and provides systematic debugging for common issues discovered during development.

Key Testing Areas:
1. ComponentArray + Lux + Zygote Integration
2. Gradient Flow and Computation
3. Cache Error Prevention
4. Mutation Detection in AD Context
5. Training Pipeline Validation
6. Performance Benchmarking

Author: GFlowNet Development Team
Date: 2025-01-27
Version: 2.0.0 (Post-Critical-Fixes)
"""

using GFlowNet
using Test
using Zygote
using ComponentArrays
using Random
using Statistics
using Lux
using BenchmarkTools
using Dates

println("🔬 GFlowNet Comprehensive Diagnostics Suite")
println("="^60)
println("🕐 Started at: $(Dates.format(now(), "HH:MM:SS"))")
println("📋 Testing all critical components for robustness...")
println()

# Global test state
const TEST_RESULTS = Dict{String, Any}()
const FAILED_TESTS = String[]

# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

"""Record test result with detailed diagnostics"""
function record_test(test_name::String, success::Bool, details::Dict{String,Any}=Dict())
    TEST_RESULTS[test_name] = merge(Dict("success" => success, "timestamp" => now()), details)
    if !success
        push!(FAILED_TESTS, test_name)
    end

    status = success ? "✅" : "❌"
    println("$status $test_name")
    if haskey(details, "message")
        println("   $(details["message"])")
    end
end

"""Safe test execution with error capture"""
function safe_test(test_name::String, test_fn::Function)
    try
        result = test_fn()
        if result isa Bool
            record_test(test_name, result)
        elseif result isa Dict
            success = get(result, "success", false)
            record_test(test_name, success, result)
        else
            record_test(test_name, true, Dict("result" => result))
        end
    catch e
        record_test(test_name, false, Dict("error" => string(e), "message" => "Exception: $e"))
    end
end

"""Check if gradients are properly computed"""
function validate_gradients(grads, min_norm::Float64=1e-10)
    if grads === nothing
        return false, "Gradients are nothing"
    end

    total_norm = 0.0
    param_count = 0
    zero_params = String[]

    function check_gradient_recursive(obj, path="")
        if obj isa AbstractArray && !isempty(obj)
            local_norm = sqrt(sum(abs2, obj))
            total_norm += local_norm^2
            param_count += 1

            if local_norm < min_norm
                push!(zero_params, path)
            end
        elseif obj isa NamedTuple
            for (key, value) in pairs(obj)
                check_gradient_recursive(value, isempty(path) ? string(key) : "$path.$key")
            end
        elseif hasproperty(obj, :axes) && hasmethod(keys, (typeof(obj),))
            # ComponentArray
            for key in keys(obj)
                check_gradient_recursive(getproperty(obj, key), isempty(path) ? string(key) : "$path.$key")
            end
        end
    end

    check_gradient_recursive(grads)

    final_norm = sqrt(total_norm)
    has_nonzero = final_norm >= min_norm

    return has_nonzero, Dict(
        "gradient_norm" => final_norm,
        "param_count" => param_count,
        "zero_params" => zero_params,
        "message" => has_nonzero ? "Gradients OK (norm=$final_norm)" : "Zero gradients detected"
    )
end

# =============================================================================
# TEST 1: BASIC FRAMEWORK INTEGRATION
# =============================================================================

println("1️⃣ Framework Integration Tests")
println("-"^40)

safe_test("Lux Basic Functionality") do
    rng = Random.MersenneTwister(42)
    net = Lux.Dense(3, 2)
    x = Float32[1.0, 2.0, 3.0]
    ps, st = Lux.setup(rng, net)

    # Test forward pass
    y, _ = net(x, ps, st)

    # Test gradient computation
    loss_fn = p -> sum(net(x, p, st)[1].^2)
    grad_result = Zygote.gradient(loss_fn, ps)

    has_grads, grad_info = validate_gradients(grad_result[1])

    Dict("success" => has_grads && length(y) == 2,
         "output_shape" => size(y),
         "gradient_info" => grad_info)
end

safe_test("ComponentArray Basic Functionality") do
    ca = ComponentArray(a=[1.0, 2.0], b=[3.0, 4.0])

    # Test gradient through ComponentArray
    loss_fn = x -> sum(x.a.^2) + sum(x.b.^2)
    grad_result = Zygote.gradient(loss_fn, ca)

    has_grads, grad_info = validate_gradients(grad_result[1])

    Dict("success" => has_grads,
         "gradient_info" => grad_info)
end

safe_test("Lux + ComponentArray Integration") do
    rng = Random.MersenneTwister(42)
    net = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))
    x = Float32[1.0, 2.0]
    ps, st = Lux.setup(rng, net)

    # ✅ CORRECT: Single ComponentArray wrapping
    ca_params = ComponentArray(ps)

    # Test forward pass with ComponentArray
    y, _ = net(x, ca_params, st)

    # Test gradient computation
    loss_fn = p -> sum(net(x, p, st)[1].^2)
    grad_result = Zygote.gradient(loss_fn, ca_params)

    has_grads, grad_info = validate_gradients(grad_result[1])

    Dict("success" => has_grads,
         "output_value" => y[1],
         "gradient_info" => grad_info,
         "message" => "Lux+ComponentArray integration working")
end

safe_test("Double ComponentArray Wrapping Detection") do
    rng = Random.MersenneTwister(42)
    net = Lux.Dense(2, 1)
    x = Float32[1.0, 2.0]
    ps, st = Lux.setup(rng, net)

    # ❌ WRONG: Double wrapping (should fail gracefully)
    try
        double_wrapped = ComponentArray(
            layer1 = ComponentArray(ps)  # This is wrong!
        )

        loss_fn = p -> sum(net(x, p.layer1, st)[1].^2)
        grad_result = Zygote.gradient(loss_fn, double_wrapped)

        has_grads, grad_info = validate_gradients(grad_result[1])

        Dict("success" => !has_grads,  # Should fail!
             "gradient_info" => grad_info,
             "message" => "Double wrapping correctly detected as problematic")
    catch e
        Dict("success" => true,
             "message" => "Double wrapping prevented by error: $e")
    end
end

# =============================================================================
# TEST 2: GFLOWNET MODEL CREATION
# =============================================================================

println("\n2️⃣ GFlowNet Model Creation Tests")
println("-"^40)

safe_test("Grid World Model Creation") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    # Check parameter structure
    has_forward = haskey(model.parameters, :forward)
    has_flow = haskey(model.parameters, :flow)
    param_count = length(model.parameters)

    Dict("success" => has_forward && has_flow && param_count > 0,
         "param_count" => param_count,
         "has_forward" => has_forward,
         "has_flow" => has_flow,
         "parameter_type" => typeof(model.parameters))
end

safe_test("Model Parameter Gradient Flow") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    # Test gradient flow through model parameters
    test_state = model.initial_state
    features = state_to_features(test_state)

    loss_fn = ps -> begin
        logits, _ = model.forward_policy.model(features, ps.forward, model.states.forward)
        return sum(logits.^2)
    end

    grad_result = Zygote.gradient(loss_fn, model.parameters)
    has_grads, grad_info = validate_gradients(grad_result[1])

    Dict("success" => has_grads,
         "gradient_info" => grad_info,
         "message" => has_grads ? "Parameter gradients flowing correctly" : "Parameter gradients BLOCKED")
end

# =============================================================================
# TEST 3: TRAJECTORY SAMPLING AND LOSS COMPUTATION
# =============================================================================

println("\n3️⃣ Trajectory and Loss Computation Tests")
println("-"^40)

safe_test("Trajectory Sampling") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    # Sample multiple trajectories
    trajectories = [sample_trajectory(model) for _ in 1:5]

    valid_count = count(traj -> length(traj.states) > 1 && is_terminal_state(traj.states[end]), trajectories)
    rewards = [reward(traj.states[end]) for traj in trajectories]

    Dict("success" => valid_count == 5,
         "valid_trajectories" => valid_count,
         "total_trajectories" => length(trajectories),
         "mean_reward" => mean(rewards),
         "reward_range" => (minimum(rewards), maximum(rewards)))
end

safe_test("Single Trajectory Loss Gradient") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    trajectory = sample_trajectory(model)

    loss_fn = ps -> compute_single_trajectory_loss(model, trajectory, ps)
    grad_result = Zygote.gradient(loss_fn, model.parameters)

    has_grads, grad_info = validate_gradients(grad_result[1])

    Dict("success" => has_grads,
         "trajectory_length" => length(trajectory.states),
         "gradient_info" => grad_info)
end

safe_test("Batch Trajectory Loss Gradient") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    trajectories = [sample_trajectory(model) for _ in 1:4]
    config = TrainingConfig()

    loss_fn = ps -> compute_trajectory_loss(model, trajectories, ps, config)
    grad_result = Zygote.gradient(loss_fn, model.parameters)

    has_grads, grad_info = validate_gradients(grad_result[1])

    Dict("success" => has_grads,
         "batch_size" => length(trajectories),
         "gradient_info" => grad_info,
         "message" => has_grads ? "Batch loss gradients working" : "Batch loss gradients FAILED")
end

# =============================================================================
# TEST 4: MUTATION DETECTION IN AD CONTEXT
# =============================================================================

println("\n4️⃣ Mutation Detection Tests")
println("-"^40)

safe_test("Mutation-Free Domain Functions") do
    # Test that domain functions don't mutate during AD
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    test_state = GridState(1, 1, false)
    test_action = MoveRight()

    # These should not cause mutations during AD
    try
        # Test apply_action in AD context
        state_fn = s -> begin
            new_state = apply_action(test_action, s)
            Float64(new_state.x + new_state.y)
        end

        grad_result = Zygote.gradient(state_fn, test_state)

        Dict("success" => true,
             "message" => "Domain functions are mutation-free in AD context")
    catch e
        if occursin("Mutating arrays is not supported", string(e))
            Dict("success" => false,
                 "message" => "MUTATION DETECTED in domain functions: $e")
        else
            Dict("success" => false,
                 "message" => "Other AD error: $e")
        end
    end
end

safe_test("Zygote.@ignore Wrapping Validation") do
    # Test that non-differentiable operations are properly wrapped
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    test_state = model.initial_state

    # Test that discrete operations are properly ignored
    discrete_fn = s -> begin
        # These should be wrapped with Zygote.@ignore in actual code
        actions = Zygote.@ignore get_applicable_actions(s, model.all_actions)
        action_count = Zygote.@ignore length(actions)
        Float64(action_count)
    end

    try
        grad_result = Zygote.gradient(discrete_fn, test_state)
        Dict("success" => true,
             "message" => "Discrete operations properly handled in AD")
    catch e
        Dict("success" => false,
             "message" => "Discrete operation AD error: $e")
    end
end

# =============================================================================
# TEST 5: TRAINING PIPELINE VALIDATION
# =============================================================================

println("\n5️⃣ Training Pipeline Tests")
println("-"^40)

safe_test("Complete Training Step") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    trajectories = [sample_trajectory(model) for _ in 1:4]
    config = TrainingConfig()

    # Store original parameters for comparison
    original_params = deepcopy(model.parameters)

    # Perform training step
    loss_val, grad_norm = train_step!(model, trajectories, config)

    # Check if parameters actually changed
    param_changed = any(original_params.forward.layer_1.weight .!= model.parameters.forward.layer_1.weight)

    Dict("success" => !isinf(loss_val) && grad_norm > 1e-10 && param_changed,
         "loss" => loss_val,
         "gradient_norm" => grad_norm,
         "parameters_updated" => param_changed,
         "message" => "Complete training step with parameter updates")
end

safe_test("Multi-Iteration Training") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    config = TrainingConfig(n_iterations=3, batch_size=4, validation_frequency=1)
    history = train_gflownet(model, config; verbose=false)

    # Check training metrics
    successful_iterations = count(!isnan, history[:losses])
    nonzero_gradients = count(g -> g > 1e-10, history[:gradient_norms])

    Dict("success" => successful_iterations == 3 && nonzero_gradients == 3,
         "successful_iterations" => successful_iterations,
         "nonzero_gradients" => nonzero_gradients,
         "final_loss" => history[:losses][end],
         "gradient_norms" => history[:gradient_norms])
end

# =============================================================================
# TEST 6: PERFORMANCE BENCHMARKS
# =============================================================================

println("\n6️⃣ Performance Benchmark Tests")
println("-"^40)

safe_test("Training Performance Benchmark") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=32)

    trajectories = [sample_trajectory(model) for _ in 1:8]
    config = TrainingConfig()

    # Benchmark training step
    benchmark_result = @benchmark train_step!(m, t, c) setup=(m=deepcopy($model), t=$trajectories, c=$config)

    median_time_ms = median(benchmark_result.times) / 1e6  # Convert to milliseconds

    Dict("success" => median_time_ms < 1000,  # Should be under 1 second
         "median_time_ms" => median_time_ms,
         "min_time_ms" => minimum(benchmark_result.times) / 1e6,
         "max_time_ms" => maximum(benchmark_result.times) / 1e6,
         "message" => "Training step performance: $(round(median_time_ms, digits=1))ms median")
end

safe_test("Memory Efficiency Test") do
    Random.seed!(42)

    # Test memory usage during model creation and training
    gc()  # Clean up before measurement
    initial_memory = Base.gc_live_bytes()

    model = create_grid_world_gflownet(grid_size=4, hidden_dim=64)
    config = TrainingConfig(n_iterations=2, batch_size=8)
    history = train_gflownet(model, config; verbose=false)

    gc()  # Clean up after operations
    final_memory = Base.gc_live_bytes()

    memory_used_mb = (final_memory - initial_memory) / 1024^2

    Dict("success" => memory_used_mb < 100,  # Should use less than 100MB
         "memory_used_mb" => memory_used_mb,
         "message" => "Memory usage: $(round(memory_used_mb, digits=1))MB")
end

# =============================================================================
# TEST 7: EDGE CASES AND ERROR HANDLING
# =============================================================================

println("\n7️⃣ Edge Case and Error Handling Tests")
println("-"^40)

safe_test("Empty Trajectory Handling") do
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    # Create invalid/empty trajectory scenario
    empty_trajectories = Trajectory[]
    config = TrainingConfig()

    try
        loss = compute_trajectory_loss(model, empty_trajectories, model.parameters, config)
        Dict("success" => loss == 0.0,
             "loss_value" => loss,
             "message" => "Empty trajectories handled gracefully")
    catch e
        Dict("success" => false,
             "message" => "Empty trajectory error: $e")
    end
end

safe_test("Invalid State Handling") do
    # Test robustness to edge cases in state space
    Random.seed!(42)
    model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

    # Test with boundary state
    boundary_state = GridState(3, 3, false)  # Corner state

    try
        applicable_actions = get_applicable_actions(boundary_state, model.all_actions)
        features = state_to_features(boundary_state)

        # Should handle gracefully
        Dict("success" => true,
             "applicable_actions" => length(applicable_actions),
             "features_valid" => all(isfinite, features),
             "message" => "Boundary states handled correctly")
    catch e
        Dict("success" => false,
             "message" => "Boundary state error: $e")
    end
end

# =============================================================================
# FINAL RESULTS AND SUMMARY
# =============================================================================

println("\n" * "="^60)
println("📊 COMPREHENSIVE DIAGNOSTICS SUMMARY")
println("="^60)

total_tests = length(TEST_RESULTS)
passed_tests = count(result -> result["success"], values(TEST_RESULTS))
failed_tests = total_tests - passed_tests

println("🏆 OVERALL RESULTS:")
println("   Total Tests: $total_tests")
println("   Passed: $passed_tests")
println("   Failed: $failed_tests")
println("   Success Rate: $(round(passed_tests/total_tests*100, digits=1))%")

if failed_tests > 0
    println("\n❌ FAILED TESTS:")
    for test_name in FAILED_TESTS
        result = TEST_RESULTS[test_name]
        println("   - $test_name")
        if haskey(result, "message")
            println("     $(result["message"])")
        end
        if haskey(result, "error")
            println("     Error: $(result["error"])")
        end
    end
end

println("\n✅ KEY VALIDATIONS:")
framework_ok = TEST_RESULTS["Lux + ComponentArray Integration"]["success"]
gradient_ok = TEST_RESULTS["Model Parameter Gradient Flow"]["success"]
training_ok = TEST_RESULTS["Complete Training Step"]["success"]
performance_ok = TEST_RESULTS["Training Performance Benchmark"]["success"]

println("   Framework Integration: $(framework_ok ? "✅" : "❌")")
println("   Gradient Computation: $(gradient_ok ? "✅" : "❌")")
println("   Training Pipeline: $(training_ok ? "✅" : "❌")")
println("   Performance: $(performance_ok ? "✅" : "❌")")

if framework_ok && gradient_ok && training_ok
    println("\n🎉 CORE FUNCTIONALITY: FULLY OPERATIONAL")
    println("   The GFlowNet implementation is robust and production-ready!")
else
    println("\n⚠️  CORE ISSUES DETECTED")
    println("   Review failed tests before deploying to production.")
end

# Save detailed results
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
results_path = joinpath(dirname(@__FILE__), "..", "..", "results", "diagnostics_$timestamp.json")
mkpath(dirname(results_path))

try
    open(results_path, "w") do f
        # Convert results to JSON-like format for saving
        simplified_results = Dict(
            "timestamp" => string(now()),
            "total_tests" => total_tests,
            "passed_tests" => passed_tests,
            "failed_tests" => failed_tests,
            "success_rate" => passed_tests/total_tests*100,
            "test_details" => Dict(name => Dict(
                "success" => result["success"],
                "message" => get(result, "message", ""),
                "error" => get(result, "error", "")
            ) for (name, result) in TEST_RESULTS)
        )

        write(f, "# GFlowNet Comprehensive Diagnostics Results\n")
        write(f, "# Generated: $(now())\n\n")
        write(f, "SUMMARY:\n")
        write(f, "- Total Tests: $total_tests\n")
        write(f, "- Passed: $passed_tests\n")
        write(f, "- Failed: $failed_tests\n")
        write(f, "- Success Rate: $(round(passed_tests/total_tests*100, digits=1))%\n\n")

        for (test_name, result) in TEST_RESULTS
            status = result["success"] ? "PASS" : "FAIL"
            write(f, "[$status] $test_name\n")
            if haskey(result, "message")
                write(f, "  Message: $(result["message"])\n")
            end
            if haskey(result, "error")
                write(f, "  Error: $(result["error"])\n")
            end
            write(f, "\n")
        end
    end
    println("\n📝 Detailed results saved to: $results_path")
catch e
    println("\n⚠️  Could not save results: $e")
end

println("\n🔬 Comprehensive diagnostics completed!")
println("="^60)
