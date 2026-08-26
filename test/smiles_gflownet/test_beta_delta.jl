# Test β-scheduling (Direction A) and |δ|-priority replay (Direction B)

using Test
using GFlowNet

include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))

println("=" ^ 60)
println("TESTING: β-scheduling and |δ|-priority replay")
println("=" ^ 60)

n_passed = 0
n_failed = 0
n_total = 8

# Test 1: FinetuningConfig with new fields
print("Test 1: FinetuningConfig β-scheduling fields... ")
try
    config = FinetuningConfig(;
        n_iterations=100,
        beta_schedule=:linear_ramp,
        beta_start=0.0,
        beta_end=8.0,
        delta_priority_replay=true,
        reward_weighted=true,
        training_mode=:tb,
    )
    @assert config.beta_schedule == :linear_ramp
    @assert config.beta_start == 0.0
    @assert config.beta_end == 8.0
    @assert config.delta_priority_replay == true
    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# Test 2: _compute_current_beta linear ramp
print("Test 2: β linear ramp schedule... ")
try
    config = FinetuningConfig(; n_iterations=100, beta_schedule=:linear_ramp,
        beta_start=0.0, beta_end=8.0)

    beta_0 = GFlowNet._compute_current_beta(config, 0, 100)
    @assert abs(beta_0 - 0.0) < 0.01 "Expected 0.0, got $beta_0"

    beta_50 = GFlowNet._compute_current_beta(config, 50, 100)
    @assert abs(beta_50 - 4.0) < 0.01 "Expected 4.0, got $beta_50"

    beta_100 = GFlowNet._compute_current_beta(config, 100, 100)
    @assert abs(beta_100 - 8.0) < 0.01 "Expected 8.0, got $beta_100"

    println("PASSED (0.0 -> 4.0 -> 8.0)")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# Test 3: β schedule :none uses fixed reward_exponent
print("Test 3: β schedule :none uses fixed value... ")
try
    config = FinetuningConfig(; reward_exponent=4.0, beta_schedule=:none)
    beta = GFlowNet._compute_current_beta(config, 50, 100)
    @assert abs(beta - 4.0) < 0.01 "Expected 4.0, got $beta"
    println("PASSED (fixed at 4.0)")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# Test 4: update_deltas! tracking
print("Test 4: update_deltas! tracks |TB error|... ")
try
    buf = SMILESReplayBuffer(100)
    add_to_replay!(buf, "CCO", [1,2,3,4,5], 0.8)
    add_to_replay!(buf, "CCC", [1,2,3,6,5], 0.6)
    add_to_replay!(buf, "CCN", [1,2,3,7,5], 0.9)

    update_deltas!(buf, ["CCO", "CCC", "CCN"], [2.5, 0.1, 1.8])
    @assert abs(buf.tb_deltas["CCO"] - 2.5) < 0.01
    @assert abs(buf.tb_deltas["CCC"] - 0.1) < 0.01
    @assert abs(buf.tb_deltas["CCN"] - 1.8) < 0.01
    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# Test 5: sample_replay_with_delta returns correct count
print("Test 5: sample_replay_with_delta returns n samples... ")
try
    buf = SMILESReplayBuffer(100)
    add_to_replay!(buf, "CCO", [1,2,3,4,5], 0.8)
    add_to_replay!(buf, "CCC", [1,2,3,6,5], 0.6)
    add_to_replay!(buf, "CCN", [1,2,3,7,5], 0.9)
    update_deltas!(buf, ["CCO", "CCC", "CCN"], [2.5, 0.1, 1.8])

    samples = sample_replay_with_delta(buf, 50; rank_weighted=true, delta_priority=true)
    @assert length(samples) == 50 "Expected 50 samples, got $(length(samples))"
    println("PASSED (50 samples)")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# Test 6: δ-priority favors high-|δ| molecules
print("Test 6: δ-priority favors high-|delta| molecules... ")
try
    buf = SMILESReplayBuffer(100)
    add_to_replay!(buf, "CCO", [1,2,3,4,5], 0.5)   # Low reward, HIGH delta
    add_to_replay!(buf, "CCC", [1,2,3,6,5], 0.9)   # High reward, LOW delta
    update_deltas!(buf, ["CCO", "CCC"], [5.0, 0.01])

    # Sample many times with delta priority
    samples = sample_replay_with_delta(buf, 200; rank_weighted=true, delta_priority=true)
    cco_count = count(s -> s.smiles == "CCO", samples)
    ccc_count = count(s -> s.smiles == "CCC", samples)

    # CCO has much higher |δ| (5.0 vs 0.01), should dominate
    @assert cco_count > ccc_count "Expected CCO ($cco_count) > CCC ($ccc_count)"
    println("PASSED (CCO: $cco_count, CCC: $ccc_count)")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# Test 7: Standard rank-by-reward still works with sample_replay_with_delta
print("Test 7: rank-by-reward mode unaffected... ")
try
    buf = SMILESReplayBuffer(100)
    add_to_replay!(buf, "CCO", [1,2,3,4,5], 0.3)   # Low reward
    add_to_replay!(buf, "CCC", [1,2,3,6,5], 0.9)   # High reward

    samples = sample_replay_with_delta(buf, 200; rank_weighted=true, delta_priority=false)
    cco_count = count(s -> s.smiles == "CCO", samples)
    ccc_count = count(s -> s.smiles == "CCC", samples)

    # CCC has higher reward, should dominate in reward-rank mode
    @assert ccc_count > cco_count "Expected CCC ($ccc_count) > CCO ($cco_count)"
    println("PASSED (CCC: $ccc_count, CCO: $cco_count)")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# Test 8: New exports exist
print("Test 8: New exports accessible... ")
try
    @assert isdefined(GFlowNet, :update_deltas!)
    @assert isdefined(GFlowNet, :sample_replay_with_delta)
    @assert isdefined(GFlowNet, :_compute_current_beta)
    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

println("\n" * "=" ^ 60)
println("Results: $n_passed passed, $n_failed failed out of $n_total")
println("=" ^ 60)

if n_failed > 0
    error("$n_failed tests failed")
end
