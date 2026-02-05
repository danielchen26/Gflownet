# Extreme Mode Collapse Test - ALL IMPLEMENTED FEATURES
# Tests whether combining ALL exploration improvements can solve the 70:1 case
#
# Run with: julia --project=. examples/core_features/extreme_mode_collapse_full_features.jl
#
# Implemented Methods (from literature):
# 1. ε-Uniform Exploration (Malkin et al. 2022)
# 2. Entropy Regularization (AISTATS 2024)
# 3. Experience Replay Buffer with Prioritized Sampling (JMLR 2023)
# 4. Adaptive Z Learning Rate (peptide paper - 10x faster)

using Test
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

"""
    test_full_feature_mode_collapse()

Test mode discovery using ALL implemented exploration features.
"""
function test_full_feature_mode_collapse()
    println("=" ^ 70)
    println("EXTREME MODE COLLAPSE - ALL FEATURES TEST")
    println("=" ^ 70)
    println()
    println("Testing whether combining ALL exploration improvements solves 70:1 case:")
    println("  • ε-Uniform Exploration (Malkin et al. 2022)")
    println("  • Entropy Regularization (AISTATS 2024)")
    println("  • Experience Replay Buffer (JMLR 2023)")
    println("  • Faster Z Learning (peptide paper)")
    println()

    # 5×5 grid with extreme path asymmetry
    grid_size = 5
    reward_positions = Dict(
        (5, 5) => 10.0,  # 70 paths
        (1, 5) => 10.0   # 1 path
    )

    println("Configuration:")
    println("  Grid: 5×5, peaks at (5,5) and (1,5)")
    println("  Path ratio: 70:1 (structural asymmetry)")
    println()

    # Pure policy evaluation config
    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    # =========================================================================
    # TEST 1: Baseline - No features
    # =========================================================================
    println("-" ^ 70)
    println("TEST 1: BASELINE (no exploration features)")
    println("-" ^ 70)

    model_baseline = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_baseline = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 1000,
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.0,
        epsilon_decay = false,
        entropy_weight = 0.0,
        use_replay_buffer = false,
        verbose = false
    )

    print("Training baseline (1000 iter)... ")
    GFlowNet.train_gflownet(model_baseline, config_baseline; verbose=false)
    println("done!")

    samples_baseline = [GFlowNet.sample_trajectory(model_baseline; config=eval_config) for _ in 1:1000]
    p1_base, p2_base = count_peaks(samples_baseline)
    println("  Results: Peak(5,5)=$p1_base, Peak(1,5)=$p2_base")
    println()

    # =========================================================================
    # TEST 2: Epsilon + Entropy only (no replay)
    # =========================================================================
    println("-" ^ 70)
    println("TEST 2: ε-Uniform + Entropy (no replay buffer)")
    println("-" ^ 70)

    model_eps_ent = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_eps_ent = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 1000,
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.3,           # High ε
        epsilon_decay = true,
        entropy_weight = 0.05,   # High entropy
        use_replay_buffer = false,
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training ε+entropy (1000 iter)... ")
    GFlowNet.train_gflownet(model_eps_ent, config_eps_ent; verbose=false)
    println("done!")

    samples_eps_ent = [GFlowNet.sample_trajectory(model_eps_ent; config=eval_config) for _ in 1:1000]
    p1_ee, p2_ee = count_peaks(samples_eps_ent)
    println("  Results: Peak(5,5)=$p1_ee, Peak(1,5)=$p2_ee")
    println()

    # =========================================================================
    # TEST 3: ALL FEATURES - Epsilon + Entropy + Replay Buffer
    # =========================================================================
    println("-" ^ 70)
    println("TEST 3: ALL FEATURES (ε + entropy + replay buffer)")
    println("-" ^ 70)

    model_all = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_all = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 1000,
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.3,           # High ε-uniform
        epsilon_decay = true,    # Anneal to 0
        entropy_weight = 0.05,   # High entropy
        use_replay_buffer = true,         # ENABLE REPLAY
        replay_buffer_size = 5000,        # Large buffer
        replay_ratio = 0.5,               # 50% replay samples
        replay_priority_alpha = 0.6,      # Prioritized sampling
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training ALL FEATURES (1000 iter)... ")
    GFlowNet.train_gflownet(model_all, config_all; verbose=false)
    println("done!")

    samples_all = [GFlowNet.sample_trajectory(model_all; config=eval_config) for _ in 1:1000]
    p1_all, p2_all = count_peaks(samples_all)
    println("  Results: Peak(5,5)=$p1_all, Peak(1,5)=$p2_all")
    println()

    # =========================================================================
    # TEST 4: Extended training with ALL features (2000 iterations)
    # =========================================================================
    println("-" ^ 70)
    println("TEST 4: ALL FEATURES + Extended Training (2000 iter)")
    println("-" ^ 70)

    model_extended = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_extended = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 2000,      # More iterations
        batch_size = 64,          # Larger batch
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.4,            # Very high ε
        epsilon_decay = true,
        entropy_weight = 0.1,     # Very high entropy
        use_replay_buffer = true,
        replay_buffer_size = 10000,
        replay_ratio = 0.5,
        replay_priority_alpha = 0.6,
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training ALL FEATURES extended (2000 iter)... ")
    GFlowNet.train_gflownet(model_extended, config_extended; verbose=false)
    println("done!")

    samples_ext = [GFlowNet.sample_trajectory(model_extended; config=eval_config) for _ in 1:1000]
    p1_ext, p2_ext = count_peaks(samples_ext)
    println("  Results: Peak(5,5)=$p1_ext, Peak(1,5)=$p2_ext")
    println()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    println("=" ^ 70)
    println("SUMMARY - Extreme 70:1 Mode Collapse Test")
    println("=" ^ 70)
    println()
    println("| Configuration                    | Peak(5,5) | Peak(1,5) | Modes |")
    println("|----------------------------------|-----------|-----------|-------|")
    m1 = (p1_base > 10 ? 1 : 0) + (p2_base > 10 ? 1 : 0)
    m2 = (p1_ee > 10 ? 1 : 0) + (p2_ee > 10 ? 1 : 0)
    m3 = (p1_all > 10 ? 1 : 0) + (p2_all > 10 ? 1 : 0)
    m4 = (p1_ext > 10 ? 1 : 0) + (p2_ext > 10 ? 1 : 0)
    println("| Baseline (no features)           | $(lpad(p1_base, 4))      | $(lpad(p2_base, 4))      | $m1/2   |")
    println("| ε-Uniform + Entropy              | $(lpad(p1_ee, 4))      | $(lpad(p2_ee, 4))      | $m2/2   |")
    println("| ALL Features (+ replay)          | $(lpad(p1_all, 4))      | $(lpad(p2_all, 4))      | $m3/2   |")
    println("| ALL Features Extended (2000)     | $(lpad(p1_ext, 4))      | $(lpad(p2_ext, 4))      | $m4/2   |")
    println()

    # Check if replay buffer helps
    best_p2 = max(p2_base, p2_ee, p2_all, p2_ext)
    if best_p2 > 10
        println("✅ SUCCESS: Minority mode (1,5) discovered with $best_p2 samples!")
        if p2_all > p2_ee || p2_ext > p2_ee
            println("   → Replay buffer HELPED improve minority mode discovery")
        end
    else
        println("⚠️  Mode collapse persists even with ALL features")
        println("   The 70:1 structural asymmetry is too extreme.")
        println()
        println("   This confirms that for EXTREME path imbalances, additional")
        println("   techniques beyond standard exploration may be needed:")
        println("   • Reward shaping (boost minority mode reward)")
        println("   • Curiosity-driven exploration")
        println("   • Hindsight experience replay")
        println("   • Backward policy entropy (TLM 2024)")
    end
    println()
    println("=" ^ 70)

    return (p1_base, p2_base), (p1_all, p2_all), (p1_ext, p2_ext)
end

function count_peaks(samples)
    p1, p2 = 0, 0
    for traj in samples
        terminal = traj.states[end]
        pos = (terminal.x, terminal.y)
        if pos == (5, 5)
            p1 += 1
        elseif pos == (1, 5)
            p2 += 1
        end
    end
    return p1, p2
end

# Run test
if abspath(PROGRAM_FILE) == @__FILE__
    test_full_feature_mode_collapse()
end
