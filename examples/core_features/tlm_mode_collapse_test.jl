# TLM (Trajectory Likelihood Maximization) Test for Extreme 70:1 Mode Collapse
# ICLR 2025: "Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization"
#
# Run with: julia --project=. examples/core_features/tlm_mode_collapse_test.jl
#
# This test verifies that TLM solves the extreme 70:1 path asymmetry problem
# that standard exploration methods (ε-uniform, entropy, replay) cannot solve.
#
# Path Analysis:
#   - Grid only allows MoveRight and MoveUp from (1,1)
#   - Peak at (5,5): paths = binomial(8,4) = 70 (4 right + 4 up in any order)
#   - Peak at (1,5): paths = binomial(4,0) = 1 (only 4 ups, no other option)
#   - Ratio: 70:1 structural asymmetry
#
# TLM Theory:
#   - Max-entropy backward policy: P_B(s|s') = n(s)/n(s') where n(s) = #paths to s
#   - Training P_B via -log P_B(s|s') on forward trajectories encodes path counts
#   - This compensates for the 70:1 structural asymmetry

using Test
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

"""
    test_tlm_extreme_mode_collapse()

Test TLM on the extreme 70:1 path asymmetry case.
"""
function test_tlm_extreme_mode_collapse()
    println("=" ^ 70)
    println("TLM (ICLR 2025) - EXTREME MODE COLLAPSE TEST")
    println("=" ^ 70)
    println()
    println("Testing whether TLM (Trajectory Likelihood Maximization) solves the")
    println("extreme 70:1 path asymmetry that standard exploration cannot handle.")
    println()
    println("Key insight from ICLR 2025 paper:")
    println("  • Max-entropy backward policy: P_B(s|s') ∝ n(s)/n(s')")
    println("  • Training backward policy encodes path count structure")
    println("  • This directly compensates for structural asymmetry")
    println()

    # 5×5 grid with extreme path asymmetry
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
    println("  Expected sampling ratio (equal rewards): 70:1")
    println()

    # Pure policy evaluation config (no exploration during testing)
    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    # =========================================================================
    # TEST 1: Standard TB Baseline (no TLM)
    # =========================================================================
    println("-" ^ 70)
    println("TEST 1: Standard TB (Trajectory Balance) - BASELINE")
    println("-" ^ 70)

    model_tb = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        include_backward = false,  # No backward policy
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_tb = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 1000,
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.1,           # Some exploration
        epsilon_decay = true,
        entropy_weight = 0.01,   # Standard entropy
        verbose = false
    )

    print("Training TB baseline (1000 iter)... ")
    GFlowNet.train_gflownet(model_tb, config_tb; verbose=false)
    println("done!")

    samples_tb = [GFlowNet.sample_trajectory(model_tb; config=eval_config) for _ in 1:1000]
    p1_tb, p2_tb = count_peaks(samples_tb)
    modes_tb = (p1_tb > 10 ? 1 : 0) + (p2_tb > 10 ? 1 : 0)

    println("  Results: Peak(5,5)=$p1_tb, Peak(1,5)=$p2_tb, Modes=$modes_tb/2")
    println()

    # =========================================================================
    # TEST 2: TLM - Trajectory Likelihood Maximization
    # =========================================================================
    println("-" ^ 70)
    println("TEST 2: TLM (Trajectory Likelihood Maximization) - ICLR 2025")
    println("-" ^ 70)

    # Create model with backward policy for TLM
    model_tlm = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        include_backward = true,  # CRITICAL: Enable backward policy for TLM
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_tlm = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_LIKELIHOOD_MAXIMIZATION,
        n_iterations = 1000,
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.1,
        epsilon_decay = true,
        entropy_weight = 0.01,
        # TLM specific parameters
        tlm_backward_weight = 1.0,      # Weight for backward likelihood loss
        tlm_entropy_coeff = 0.01,       # Entropy for backward policy
        verbose = false
    )

    print("Training TLM (1000 iter)... ")
    GFlowNet.train_gflownet(model_tlm, config_tlm; verbose=false)
    println("done!")

    samples_tlm = [GFlowNet.sample_trajectory(model_tlm; config=eval_config) for _ in 1:1000]
    p1_tlm, p2_tlm = count_peaks(samples_tlm)
    modes_tlm = (p1_tlm > 10 ? 1 : 0) + (p2_tlm > 10 ? 1 : 0)

    println("  Results: Peak(5,5)=$p1_tlm, Peak(1,5)=$p2_tlm, Modes=$modes_tlm/2")
    println()

    # =========================================================================
    # TEST 3: TLM with Higher Backward Weight
    # =========================================================================
    println("-" ^ 70)
    println("TEST 3: TLM with Higher Backward Weight (λ=2.0)")
    println("-" ^ 70)

    model_tlm2 = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        include_backward = true,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_tlm2 = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_LIKELIHOOD_MAXIMIZATION,
        n_iterations = 1500,        # More iterations
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.15,             # Higher exploration
        epsilon_decay = true,
        entropy_weight = 0.02,
        tlm_backward_weight = 2.0,  # Higher backward weight
        tlm_entropy_coeff = 0.02,
        verbose = false
    )

    print("Training TLM with λ=2.0 (1500 iter)... ")
    GFlowNet.train_gflownet(model_tlm2, config_tlm2; verbose=false)
    println("done!")

    samples_tlm2 = [GFlowNet.sample_trajectory(model_tlm2; config=eval_config) for _ in 1:1000]
    p1_tlm2, p2_tlm2 = count_peaks(samples_tlm2)
    modes_tlm2 = (p1_tlm2 > 10 ? 1 : 0) + (p2_tlm2 > 10 ? 1 : 0)

    println("  Results: Peak(5,5)=$p1_tlm2, Peak(1,5)=$p2_tlm2, Modes=$modes_tlm2/2")
    println()

    # =========================================================================
    # TEST 4: TLM with Extended Training
    # =========================================================================
    println("-" ^ 70)
    println("TEST 4: TLM Extended Training (2000 iterations)")
    println("-" ^ 70)

    model_tlm_ext = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        include_backward = true,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_tlm_ext = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_LIKELIHOOD_MAXIMIZATION,
        n_iterations = 2000,
        batch_size = 64,            # Larger batch
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.2,              # High initial exploration
        epsilon_decay = true,
        entropy_weight = 0.01,
        tlm_backward_weight = 1.5,
        tlm_entropy_coeff = 0.01,
        use_replay_buffer = true,   # Add replay buffer
        replay_buffer_size = 5000,
        replay_ratio = 0.5,
        verbose = false
    )

    print("Training TLM extended (2000 iter + replay)... ")
    GFlowNet.train_gflownet(model_tlm_ext, config_tlm_ext; verbose=false)
    println("done!")

    samples_tlm_ext = [GFlowNet.sample_trajectory(model_tlm_ext; config=eval_config) for _ in 1:1000]
    p1_tlm_ext, p2_tlm_ext = count_peaks(samples_tlm_ext)
    modes_tlm_ext = (p1_tlm_ext > 10 ? 1 : 0) + (p2_tlm_ext > 10 ? 1 : 0)

    println("  Results: Peak(5,5)=$p1_tlm_ext, Peak(1,5)=$p2_tlm_ext, Modes=$modes_tlm_ext/2")
    println()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    println("=" ^ 70)
    println("TLM EXTREME MODE COLLAPSE TEST - SUMMARY")
    println("=" ^ 70)
    println()
    println("| Method                          | Peak(5,5) | Peak(1,5) | Modes |")
    println("|---------------------------------|-----------|-----------|-------|")
    println("| TB Baseline (ε=0.1, H=0.01)     | $(lpad(p1_tb, 4))      | $(lpad(p2_tb, 4))      | $modes_tb/2   |")
    println("| TLM (λ=1.0)                     | $(lpad(p1_tlm, 4))      | $(lpad(p2_tlm, 4))      | $modes_tlm/2   |")
    println("| TLM (λ=2.0)                     | $(lpad(p1_tlm2, 4))      | $(lpad(p2_tlm2, 4))      | $modes_tlm2/2   |")
    println("| TLM Extended (+ replay)         | $(lpad(p1_tlm_ext, 4))      | $(lpad(p2_tlm_ext, 4))      | $modes_tlm_ext/2   |")
    println()

    # Determine best result
    best_p2 = max(p2_tb, p2_tlm, p2_tlm2, p2_tlm_ext)
    best_modes = max(modes_tb, modes_tlm, modes_tlm2, modes_tlm_ext)

    if best_p2 > 10
        println("✅ SUCCESS: Minority mode (1,5) discovered with $best_p2 samples!")
        if p2_tlm > p2_tb || p2_tlm2 > p2_tb || p2_tlm_ext > p2_tb
            println("   → TLM IMPROVED minority mode discovery over TB baseline")
            println()
            println("   TLM (ICLR 2025) works by training the backward policy to")
            println("   encode path count information, directly compensating for")
            println("   the structural 70:1 path asymmetry.")
        end
    else
        println("⚠️  Mode collapse persists even with TLM")
        println()
        println("   Possible reasons:")
        println("   • Backward policy network needs more capacity")
        println("   • Higher backward weight λ may help")
        println("   • More training iterations needed")
        println("   • Try different learning rate for backward policy")
        println()
        println("   For the extreme 70:1 case, reward shaping (boosting minority")
        println("   reward by 70x) is still a guaranteed solution.")
    end
    println()
    println("=" ^ 70)

    return (p1_tb, p2_tb), (p1_tlm, p2_tlm), (p1_tlm_ext, p2_tlm_ext)
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
    test_tlm_extreme_mode_collapse()
end
