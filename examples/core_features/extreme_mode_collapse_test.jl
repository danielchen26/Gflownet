# Extreme Mode Collapse Test - 70:1 Path Asymmetry
# Tests the hardest case: 5×5 grid where structural asymmetry creates 70:1 path ratio
#
# Run with: julia --project=. examples/core_features/extreme_mode_collapse_test.jl
#
# Path Analysis:
#   - Grid only allows MoveRight and MoveUp from (1,1)
#   - Peak at (5,5): paths = binomial(8,4) = 70 (4 right + 4 up in any order)
#   - Peak at (1,5): paths = binomial(4,0) = 1 (only 4 ups, no other option)
#   - Ratio: 70:1 structural asymmetry
#
# Expected sampling ratio (if R1=R2=10): 70:1 based on paths
# With rewards R(5,5)=10, R(1,5)=10: p ∝ R × paths = 70:1

using Test
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

"""
    test_extreme_mode_collapse()

Test mode discovery in the extreme 70:1 path asymmetry case.
"""
function test_extreme_mode_collapse()
    println("=" ^ 70)
    println("EXTREME MODE COLLAPSE TEST - 70:1 Path Asymmetry")
    println("=" ^ 70)
    println()

    # 5×5 grid with two reward peaks
    grid_size = 5
    reward_positions = Dict(
        (5, 5) => 10.0,  # 70 paths (far corner)
        (1, 5) => 10.0   # 1 path (same column as start)
    )

    println("Configuration:")
    println("  Grid: 5×5 (start at (1,1), only MoveRight/MoveUp)")
    println("  Peak 1: (5,5) R=10, paths=binomial(8,4)=70")
    println("  Peak 2: (1,5) R=10, paths=binomial(4,0)=1")
    println("  Path ratio: 70:1 (structural, not reward-based)")
    println("  Expected sampling ratio: 70:1 (equal rewards)")
    println()

    # =========================================================================
    # TEST 1: WITHOUT exploration
    # =========================================================================
    println("-" ^ 70)
    println("TEST 1: Training WITHOUT exploration (ε=0, entropy=0)")
    println("-" ^ 70)

    model_no_exp = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,  # Larger network for harder problem
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_no_exp = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 1000,  # More iterations for extreme case
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.0,          # NO exploration
        epsilon_decay = false,
        entropy_weight = 0.0,   # NO entropy
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training 1000 iterations... ")
    history_no_exp = GFlowNet.train_gflownet(model_no_exp, config_no_exp; verbose=false)
    println("done!")

    # Sample with pure policy
    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    print("Sampling 1000 trajectories... ")
    samples_no_exp = [GFlowNet.sample_trajectory(model_no_exp; config=eval_config) for _ in 1:1000]
    println("done!")

    # Count modes
    peak1_no = 0
    peak2_no = 0
    other_no = 0
    for traj in samples_no_exp
        terminal = traj.states[end]
        pos = (terminal.x, terminal.y)
        if pos == (5, 5)
            peak1_no += 1
        elseif pos == (1, 5)
            peak2_no += 1
        else
            other_no += 1
        end
    end

    modes_no = (peak1_no > 10 ? 1 : 0) + (peak2_no > 10 ? 1 : 0)

    println()
    println("Results WITHOUT exploration:")
    println("  Peak (5,5) samples: $peak1_no / 1000 ($(round(peak1_no/10, digits=1))%)")
    println("  Peak (1,5) samples: $peak2_no / 1000 ($(round(peak2_no/10, digits=1))%)")
    println("  Other terminals:    $other_no / 1000")
    println("  Modes discovered: $modes_no / 2")
    if peak2_no <= 10
        println("  → MODE COLLAPSE: Peak (1,5) essentially not sampled!")
    end
    println()

    # =========================================================================
    # TEST 2: WITH HIGH exploration
    # =========================================================================
    println("-" ^ 70)
    println("TEST 2: Training WITH HIGH exploration (ε=0.2, entropy=0.05)")
    println("-" ^ 70)

    model_with_exp = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_with_exp = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 1000,
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.2,           # HIGH exploration
        epsilon_decay = true,    # Anneal to 0
        entropy_weight = 0.05,   # Higher entropy for extreme case
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training 1000 iterations with exploration... ")
    history_with_exp = GFlowNet.train_gflownet(model_with_exp, config_with_exp; verbose=false)
    println("done!")

    print("Sampling 1000 trajectories (pure policy)... ")
    samples_with_exp = [GFlowNet.sample_trajectory(model_with_exp; config=eval_config) for _ in 1:1000]
    println("done!")

    # Count modes
    peak1_with = 0
    peak2_with = 0
    other_with = 0
    for traj in samples_with_exp
        terminal = traj.states[end]
        pos = (terminal.x, terminal.y)
        if pos == (5, 5)
            peak1_with += 1
        elseif pos == (1, 5)
            peak2_with += 1
        else
            other_with += 1
        end
    end

    modes_with = (peak1_with > 10 ? 1 : 0) + (peak2_with > 10 ? 1 : 0)

    println()
    println("Results WITH exploration:")
    println("  Peak (5,5) samples: $peak1_with / 1000 ($(round(peak1_with/10, digits=1))%)")
    println("  Peak (1,5) samples: $peak2_with / 1000 ($(round(peak2_with/10, digits=1))%)")
    println("  Other terminals:    $other_with / 1000")
    println("  Modes discovered: $modes_with / 2")

    # Analyze ratio
    if peak1_with > 10 && peak2_with > 10
        actual_ratio = peak1_with / peak2_with
        expected_ratio = 70.0  # Due to 70:1 path asymmetry
        println("  Actual ratio: $(round(actual_ratio, digits=1)):1")
        println("  Expected ratio: 70:1 (from path counts)")
    end
    println()

    # =========================================================================
    # TEST 3: VERY HIGH exploration (last resort)
    # =========================================================================
    println("-" ^ 70)
    println("TEST 3: Training with EXTREME exploration (ε=0.4, entropy=0.1)")
    println("-" ^ 70)

    model_extreme = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_extreme = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 2000,      # More iterations
        batch_size = 64,          # Larger batch
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.4,            # VERY HIGH exploration
        epsilon_decay = true,     # Anneal to 0
        entropy_weight = 0.1,     # Very high entropy
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training 2000 iterations with extreme exploration... ")
    history_extreme = GFlowNet.train_gflownet(model_extreme, config_extreme; verbose=false)
    println("done!")

    print("Sampling 1000 trajectories (pure policy)... ")
    samples_extreme = [GFlowNet.sample_trajectory(model_extreme; config=eval_config) for _ in 1:1000]
    println("done!")

    # Count modes
    peak1_ext = 0
    peak2_ext = 0
    other_ext = 0
    for traj in samples_extreme
        terminal = traj.states[end]
        pos = (terminal.x, terminal.y)
        if pos == (5, 5)
            peak1_ext += 1
        elseif pos == (1, 5)
            peak2_ext += 1
        else
            other_ext += 1
        end
    end

    modes_ext = (peak1_ext > 10 ? 1 : 0) + (peak2_ext > 10 ? 1 : 0)

    println()
    println("Results with EXTREME exploration:")
    println("  Peak (5,5) samples: $peak1_ext / 1000 ($(round(peak1_ext/10, digits=1))%)")
    println("  Peak (1,5) samples: $peak2_ext / 1000 ($(round(peak2_ext/10, digits=1))%)")
    println("  Other terminals:    $other_ext / 1000")
    println("  Modes discovered: $modes_ext / 2")

    if peak1_ext > 10 && peak2_ext > 10
        actual_ratio = peak1_ext / peak2_ext
        println("  Actual ratio: $(round(actual_ratio, digits=1)):1")
    end
    println()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    println("=" ^ 70)
    println("EXTREME MODE COLLAPSE TEST - SUMMARY")
    println("=" ^ 70)
    println()
    println("Configuration: 5×5 grid, peaks at (5,5) and (1,5), 70:1 path ratio")
    println()
    println("| Test                    | Peak(5,5) | Peak(1,5) | Modes |")
    println("|-------------------------|-----------|-----------|-------|")
    println("| No exploration (ε=0)    | $(lpad(peak1_no, 4))      | $(lpad(peak2_no, 4))      | $modes_no/2   |")
    println("| High (ε=0.2, H=0.05)    | $(lpad(peak1_with, 4))      | $(lpad(peak2_with, 4))      | $modes_with/2   |")
    println("| Extreme (ε=0.4, H=0.1)  | $(lpad(peak1_ext, 4))      | $(lpad(peak2_ext, 4))      | $modes_ext/2   |")
    println()

    # Verdict
    if modes_with > modes_no || modes_ext > modes_no
        println("✅ EXPLORATION HELPS with mode discovery even in extreme 70:1 case!")
        if peak2_with > 10 || peak2_ext > 10
            println("   The minority mode (1,5) was successfully discovered.")
        end
    else
        println("⚠️  STRUCTURAL ASYMMETRY TOO EXTREME")
        println("   The 70:1 path ratio creates fundamental bias that exploration alone")
        println("   cannot overcome. This may require:")
        println("   - Reward shaping (higher reward for minority mode)")
        println("   - Backward policy entropy")
        println("   - Modified action space")
        println("   - Curiosity-driven exploration")
    end
    println()
    println("=" ^ 70)

    return (modes_no, modes_with, modes_ext)
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_extreme_mode_collapse()
end
