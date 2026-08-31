# Exploration Features on the 70-paths-against-1 Grid - ALL IMPLEMENTED FEATURES
# Tests what each exploration feature changes when one reward peak is reachable by 70
# trajectories and an equally rewarded one by a single trajectory.
#
# Run with: julia --project=. examples/core_features/extreme_mode_collapse_full_features.jl
#
# Implemented Methods (from literature):
# 1. ε-Uniform Exploration (Malkin et al. 2022)
# 2. Entropy Regularization (AISTATS 2024)
# 3. Experience Replay Buffer with Prioritized Sampling (JMLR 2023)
# 4. Adaptive Z Learning Rate (peptide paper - 10x faster)
#
# WHAT THIS SCRIPT MEASURES
# -------------------------
# It used to ask whether these four features together could "solve the 70:1 case".
# There is no 70:1 case to solve. Trajectory Balance with its backward term properly
# accounted converges to p(x) = R(x)/Z, an expression with no path-count factor in
# it, so two peaks of equal reward take equal mass however many trajectories reach
# them. The starved minority peak this file was built around came from
# `compute_single_trajectory_loss` dropping its `sum log P_B` term when no backward
# policy was supplied — sound only if every state has one parent, and every interior
# cell of this lattice has two. The loss was therefore minimised by the
# path-count-biased law n(x)R(x) / sum_y n(y)R(y), which does give (5,5) 65.9% of the
# mass against 0.9% for (1,5).
#
# So the question the four arms answer now is what the features cost and buy on a
# problem whose fixed point they cannot move. Measured: all four reach the same
# path-count-invariant law, and the two replay arms need roughly four times the
# iterations to get there, because half of every replay batch is stale. At 120-150
# iterations the replay arms sit at a closing loss near 7.7 with total variation
# 0.31-0.36 against the enumerated law, while the two on-policy arms are already at
# a closing loss near 0.07 and TV 0.14-0.18.
#
# Enumerated ground truth for exactly this configuration
# (test/theory/enumerate.jl, grid 5, R(5,5) = R(1,5) = 10.0):
#   exact_Z(5) = 59.0 — the two peaks plus the background rewards of the other 22
#                       cells that are allowed to terminate; (1,1) is not one.
#   analytic_optimum_terminal_law_corrected(5):
#     (5,5) = (1,5) = 10/59 = 16.9492%  ->  67.80 of 400 samples each, with
#                                          binomial sd sqrt(400 p (1-p)) = 7.50
#   analytic_optimum_terminal_law(5), the optimum of the OLD loss, for contrast:
#     (5,5) = 65.913% -> 263.7/400,    (1,5) = 0.942% -> 3.8/400

using Test
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

include(joinpath(@__DIR__, "..", "convergence_assertions.jl"))

"""
    test_full_feature_mode_collapse()

Check what each exploration feature changes on the 70-paths-against-1 grid, whose
correct terminal law is path-count invariant.
"""
function test_full_feature_mode_collapse()
    println("=" ^ 70)
    println("EXPLORATION FEATURES ON THE 70-PATHS-AGAINST-1 GRID")
    println("=" ^ 70)
    println()
    println("Testing what each exploration feature costs and buys here:")
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
    println("  Grid: 5×5, peaks at (5,5) and (1,5), R = 10.0 each")
    println("  Path ratio: 70:1 (structural; does not enter the target law)")
    println("  Enumerated target: each peak 10/59 = 16.95% -> 67.8 of 400 samples")
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
        # Cut from 1000 / batch 32, then raised 120 -> 400: at 120-150 iterations
        # this arm is still off the enumerated law (TV to
        # analytic_optimum_terminal_law_corrected = 0.142 / 0.179 / 0.150 on seeds
        # 1-3), at 400 it is 0.099 / 0.097 / 0.112 with the closing loss at
        # 0.008 / 0.022 / 0.020.
        n_iterations = 400,
        batch_size = 16,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.0,
        epsilon_decay = false,
        entropy_weight = 0.0,
        use_replay_buffer = false,
        verbose = false
    )

    print("Training baseline ($(config_baseline.n_iterations) iter)... ")
    history_baseline = GFlowNet.train_gflownet(model_baseline, config_baseline; verbose=false)
    println("done!")

    assert_finite_iterations(history_baseline, config_baseline.n_iterations, "baseline")
    # No replay buffer here, so the loss IS a progress statistic. Measured ratio
    # 0.0016 / 0.0041 / 0.0028 on seeds 1-3 at this budget; bar 0.5.
    assert_loss_decreased(history_baseline, "baseline"; window=10, max_ratio=0.5)

    samples_baseline = [GFlowNet.sample_trajectory(model_baseline; config=eval_config) for _ in 1:400]
    p1_base, p2_base = count_peaks(samples_baseline)
    println("  Results: Peak(5,5)=$p1_base, Peak(1,5)=$p2_base")
    # BOTH peaks are asserted, because both are supposed to be there. The enumerated
    # correct law gives each of them 10/59 = 16.9492% -> 67.80 of 400 samples, with
    # binomial sd 7.50. Bar 30 of 400 (7.5%):
    #   - 67.80 - 30 = 37.8, i.e. 5.0 binomial sd of head room for finite-training
    #     error and for the fact that this script does not seed its runs. The worst
    #     count over 3 seeds x 4 arms at these budgets was 51.
    #   - it still fails on path-count collapse, where the 1-path peak takes
    #     analytic_optimum_terminal_law(5)[(1,5)] = 0.942% -> 3.8/400, and
    #   - on a reward-blind sampler, which spreads over the 24 terminable cells for
    #     1/24 = 4.17% -> 16.7/400 (sd 4.0), leaving the bar 3.3 sd above it.
    # Measured here: peaks 57/82, 53/74, 51/84 on seeds 1-3.
    assert_modes_discovered([p1_base, p2_base], "baseline both peaks";
                            min_per_mode=30, n_samples=400)
    # Path-count invariance itself: equal rewards must give 1:1. Target 1.0 with
    # |ratio - 1| <= 3.0, i.e. up to 4:1. Three binomial sd on both counts bounds the
    # ratio by 90.3/45.3 = 1.99 and 4.0 is twice that; the old path-count-biased law
    # sat at 263.7/3.8 = 69.4:1, which fails the bar by 17x. Worst ratio observed
    # over 3 seeds x 4 arms at these budgets: 1.65.
    assert_relative_error_below(max(p1_base, p2_base) / max(min(p1_base, p2_base), 1),
                                1.0, "baseline peak ratio"; max_rel_error=3.0)
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
        # Cut from 1000 / batch 32, then 120 -> 400; same measured basis as TEST 1.
        # At 400 / batch 16: loss 5.981 -> -0.032 and TV to the enumerated law
        # 0.082 / 0.118 / 0.079 on seeds 1-3.
        n_iterations = 400,
        batch_size = 16,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.3,           # High ε
        epsilon_decay = true,
        entropy_weight = 0.05,   # High entropy
        use_replay_buffer = false,
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training ε+entropy ($(config_eps_ent.n_iterations) iter)... ")
    history_eps_ent = GFlowNet.train_gflownet(model_eps_ent, config_eps_ent; verbose=false)
    println("done!")

    assert_finite_iterations(history_eps_ent, config_eps_ent.n_iterations, "eps+entropy")
    # Still no replay buffer, so the loss remains a progress statistic.
    assert_loss_decreased(history_eps_ent, "eps+entropy"; window=10, max_ratio=0.5)

    samples_eps_ent = [GFlowNet.sample_trajectory(model_eps_ent; config=eval_config) for _ in 1:400]
    p1_ee, p2_ee = count_peaks(samples_eps_ent)
    println("  Results: Peak(5,5)=$p1_ee, Peak(1,5)=$p2_ee")
    # Same two bars as TEST 1; the derivation is stated there. Measured at this
    # budget: peaks 65/56, 95/58, 65/67 on seeds 1-3, ratios 1.16 / 1.64 / 1.03
    # against the enumerated 1.00.
    assert_modes_discovered([p1_ee, p2_ee], "eps+entropy both peaks";
                            min_per_mode=30, n_samples=400)
    assert_relative_error_below(max(p1_ee, p2_ee) / max(min(p1_ee, p2_ee), 1),
                                1.0, "eps+entropy peak ratio"; max_rel_error=3.0)
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
        # Cut from 1000 / batch 32, then raised 120 -> 1500. Half of every batch is
        # replayed here, so this arm converges roughly four times slower: measured at
        # 120 iterations the closing loss is 7.67 / 7.76 / 8.01 with TV to the
        # enumerated law 0.309 / 0.330 / 0.332, at 800 it is 0.57-0.66 and 0.10-0.14,
        # and at 1500 it is -0.029 / -0.026 / -0.019 with TV 0.058 / 0.094 / 0.093.
        n_iterations = 1500,
        batch_size = 16,
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

    print("Training ALL FEATURES ($(config_all.n_iterations) iter)... ")
    history_all = GFlowNet.train_gflownet(model_all, config_all; verbose=false)
    println("done!")

    assert_finite_iterations(history_all, config_all.n_iterations, "all features")
    # At the 120 iterations this arm used to run for, the loss mean RISES
    # (23.314 -> 26.550, ratio 1.139) because half of every batch is replayed and the
    # buffer keeps feeding back trajectories from an older policy. At 1500 iterations
    # that transient is over and the loss is a progress statistic again: measured
    # 10.52 -> -0.029 on seed 1, 8.57 -> -0.026 on seed 2, 11.81 -> -0.019 on seed 3.
    assert_loss_decreased(history_all, "all features"; window=10, max_ratio=0.5)

    samples_all = [GFlowNet.sample_trajectory(model_all; config=eval_config) for _ in 1:400]
    p1_all, p2_all = count_peaks(samples_all)
    println("  Results: Peak(5,5)=$p1_all, Peak(1,5)=$p2_all")
    # Same two bars as TEST 1. Measured at this budget: peaks 68/65, 71/63, 64/66 on
    # seeds 1-3, ratios 1.05 / 1.13 / 1.03 against the enumerated 1.00.
    assert_modes_discovered([p1_all, p2_all], "all features both peaks";
                            min_per_mode=30, n_samples=400)
    assert_relative_error_below(max(p1_all, p2_all) / max(min(p1_all, p2_all), 1),
                                1.0, "all features peak ratio"; max_rel_error=3.0)
    println()

    # =========================================================================
    # TEST 4: ALL FEATURES at the highest exploration setting
    # =========================================================================
    println("-" ^ 70)
    println("TEST 4: ALL FEATURES at ε=0.4, entropy=0.1")
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
        # Cut from 2000 / batch 64, then 120 -> 1500 for the same reason as TEST 3.
        # Measured at 1500 / batch 16: loss 10.01 -> -0.061 on seed 1, TV to the
        # enumerated law 0.090 / 0.084 / 0.111 on seeds 1-3.
        n_iterations = 1500,
        batch_size = 16,
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

    print("Training ALL FEATURES extended ($(config_extended.n_iterations) iter)... ")
    history_extended = GFlowNet.train_gflownet(model_extended, config_extended; verbose=false)
    println("done!")

    assert_finite_iterations(history_extended, config_extended.n_iterations,
                             "all features extended")
    # Replay again, and again converged by 1500: measured 10.01 -> -0.061,
    # 7.94 -> -0.060, 12.08 -> -0.058 on seeds 1-3.
    assert_loss_decreased(history_extended, "all features extended"; window=10, max_ratio=0.5)

    samples_ext = [GFlowNet.sample_trajectory(model_extended; config=eval_config) for _ in 1:400]
    p1_ext, p2_ext = count_peaks(samples_ext)
    println("  Results: Peak(5,5)=$p1_ext, Peak(1,5)=$p2_ext")
    # Same two bars as TEST 1. Measured at this budget: peaks 73/57, 60/66, 69/60 on
    # seeds 1-3, ratios 1.28 / 1.10 / 1.15 against the enumerated 1.00.
    assert_modes_discovered([p1_ext, p2_ext], "all features extended both peaks";
                            min_per_mode=30, n_samples=400)
    assert_relative_error_below(max(p1_ext, p2_ext) / max(min(p1_ext, p2_ext), 1),
                                1.0, "all features extended peak ratio"; max_rel_error=3.0)
    println()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    println("=" ^ 70)
    println("SUMMARY - Exploration features on the 70-paths-against-1 grid")
    println("=" ^ 70)
    println()
    println("Enumerated target law: each peak 10/59 = 16.95%, i.e. 67.8 of 400")
    println()
    println("| Configuration                    | Peak(5,5) | Peak(1,5) | Modes |")
    println("|----------------------------------|-----------|-----------|-------|")
    m1 = (p1_base > 10 ? 1 : 0) + (p2_base > 10 ? 1 : 0)
    m2 = (p1_ee > 10 ? 1 : 0) + (p2_ee > 10 ? 1 : 0)
    m3 = (p1_all > 10 ? 1 : 0) + (p2_all > 10 ? 1 : 0)
    m4 = (p1_ext > 10 ? 1 : 0) + (p2_ext > 10 ? 1 : 0)
    println("| Baseline (no features, 400)      | $(lpad(p1_base, 4))      | $(lpad(p2_base, 4))      | $m1/2   |")
    println("| ε-Uniform + Entropy (400)        | $(lpad(p1_ee, 4))      | $(lpad(p2_ee, 4))      | $m2/2   |")
    println("| ALL Features (+ replay, 1500)    | $(lpad(p1_all, 4))      | $(lpad(p2_all, 4))      | $m3/2   |")
    println("| ALL Features, ε=0.4 (1500)       | $(lpad(p1_ext, 4))      | $(lpad(p2_ext, 4))      | $m4/2   |")
    println()

    # Verdict
    worst = min(p1_base, p2_base, p1_ee, p2_ee, p1_all, p2_all, p1_ext, p2_ext)
    if worst >= 30
        println("Every configuration reached both peaks, the weakest at $worst of 400")
        println("against an enumerated target of 67.8 each. The 70:1 path ratio does")
        println("not appear in R(x)/Z, so there is no mode collapse here for these")
        println("features to fix. What they change is cost: the two replay arms need")
        println("1500 iterations to reach the law the two on-policy arms reach in 400,")
        println("because half of every replay batch comes from an older policy.")
    else
        println("A peak fell to $worst of 400, under the bar of 30, against an")
        println("enumerated target of 67.8. That is a regression rather than the 70:1")
        println("asymmetry reasserting itself: with the backward term accounted for,")
        println("the terminal law is path-count invariant. Check sum log P_B in the")
        println("trajectory balance loss before reaching for another exploration knob.")
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
