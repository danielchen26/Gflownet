# Path-Count Invariance Test - two equal-reward peaks, 70 paths against 1
# Tests the hardest structural case: a 5×5 grid where one peak is reachable by 70
# distinct trajectories and the other by exactly one.
#
# Run with: julia --project=. examples/core_features/extreme_mode_collapse_test.jl
#
# Path Analysis:
#   - Grid only allows MoveRight and MoveUp from (1,1)
#   - Peak at (5,5): paths = binomial(8,4) = 70 (4 right + 4 up in any order)
#   - Peak at (1,5): paths = binomial(4,0) = 1 (only 4 ups, no other option)
#   - Ratio: 70:1 structural asymmetry
#
# WHAT THIS SCRIPT MEASURES
# -------------------------
# The 70:1 ratio is a property of the state graph, not of the target distribution.
# Trajectory Balance with its backward term properly accounted converges to
# p(x) = R(x)/Z, and that expression contains no path-count factor: two peaks of
# equal reward take equal mass however many trajectories reach them. Path-count
# INVARIANCE is the claim the three arms below test.
#
# This file used to advertise "Expected sampling ratio: 70:1 (equal rewards)". That
# was never a prediction of GFlowNet theory, it was a symptom.
# `compute_single_trajectory_loss` dropped its `sum log P_B` term whenever no
# backward policy was supplied, which is sound only when every state has exactly one
# parent. On this lattice every interior cell has two, so the loss was minimised by
# the path-count-biased law n(x)R(x) / sum_y n(y)R(y), under which (5,5) really did
# hold 65.9% of the mass against 0.9% for (1,5) — the advertised 70:1. With the
# backward term restored both peaks hold 16.9%, and the 70 vs 1 path counts stay a
# true structural fact that no longer biases anything.
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
    test_extreme_mode_collapse()

Check that the terminal law is path-count invariant in the extreme 70-paths-against-1
case, with and without exploration.
"""
function test_extreme_mode_collapse()
    println("=" ^ 70)
    println("PATH-COUNT INVARIANCE TEST - 70 paths against 1, equal rewards")
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
    println("  Path ratio: 70:1 (structural; does not enter the target law)")
    println("  Enumerated target: each peak 10/59 = 16.95% -> 67.8 of 400 samples")
    println("  Expected sampling ratio: 1:1 (equal rewards, path count cancels)")
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
        # Cut from 1000 iterations / batch 32, then raised 150 -> 400: at 150 the
        # sampler is still visibly off the enumerated law (TV to
        # analytic_optimum_terminal_law_corrected = 0.142 / 0.179 / 0.150 on seeds
        # 1-3), at 400 it is 0.099 / 0.097 / 0.112 with the closing loss at
        # 0.008 / 0.022 / 0.020.
        n_iterations = 400,
        batch_size = 16,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.0,          # NO exploration
        epsilon_decay = false,
        entropy_weight = 0.0,   # NO entropy
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training $(config_no_exp.n_iterations) iterations... ")
    history_no_exp = GFlowNet.train_gflownet(model_no_exp, config_no_exp; verbose=false)
    println("done!")

    assert_finite_iterations(history_no_exp, config_no_exp.n_iterations,
                             "TB no exploration")
    # Measured ratio 0.0016 / 0.0041 / 0.0028 on seeds 1-3 at this budget; bar 0.5.
    assert_loss_decreased(history_no_exp, "TB no exploration"; window=10, max_ratio=0.5)

    # Sample with pure policy
    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    print("Sampling 400 trajectories... ")
    samples_no_exp = [GFlowNet.sample_trajectory(model_no_exp; config=eval_config) for _ in 1:400]
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
    println("  Peak (5,5) samples: $peak1_no / 400 ($(round(peak1_no/4, digits=1))%)")
    println("  Peak (1,5) samples: $peak2_no / 400 ($(round(peak2_no/4, digits=1))%)")
    println("  Other terminals:    $other_no / 400")
    println("  Modes discovered: $modes_no / 2")
    if peak2_no <= 10
        println("  → REGRESSION: the 1-path peak is starved. The correct law gives it")
        println("     16.95% (67.8/400), so this points at a mishandled backward term,")
        println("     not at a structural limit of the problem.")
    end
    println()
    # BOTH peaks are asserted, because both are supposed to be there. The enumerated
    # correct law gives each of them 10/59 = 16.9492% -> 67.80 of 400 samples, with
    # binomial sd 7.50. Bar 30 of 400 (7.5%):
    #   - 67.80 - 30 = 37.8, i.e. 5.0 binomial sd of head room for finite-training
    #     error and for the fact that this script does not seed its runs. The worst
    #     count over 3 seeds x 3 arms at this budget was 51.
    #   - it still fails on path-count collapse, where the 1-path peak takes
    #     analytic_optimum_terminal_law(5)[(1,5)] = 0.942% -> 3.8/400, and
    #   - on a reward-blind sampler, which spreads over the 24 terminable cells for
    #     1/24 = 4.17% -> 16.7/400 (sd 4.0), leaving the bar 3.3 sd above it.
    assert_modes_discovered([peak1_no, peak2_no], "TB no exploration both peaks";
                            min_per_mode=30, n_samples=400)
    # Path-count invariance itself: equal rewards must give 1:1. Target 1.0 with
    # |ratio - 1| <= 3.0, i.e. up to 4:1. Three binomial sd on both counts bounds the
    # ratio by 90.3/45.3 = 1.99 and 4.0 is twice that; the old path-count-biased law
    # sat at 263.7/3.8 = 69.4:1, which fails the bar by 17x. Worst ratio observed
    # over 3 seeds x 3 arms at this budget: 1.65.
    assert_relative_error_below(max(peak1_no, peak2_no) / max(min(peak1_no, peak2_no), 1),
                                1.0, "TB no exploration peak ratio"; max_rel_error=3.0)


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
        # Cut from 1000 / batch 32, then 150 -> 400 as in TEST 1. Measured at 400 /
        # batch 16: loss 6.517 -> -0.032, peaks 67 and 68 of 400 (seed 1); TV to the
        # enumerated law 0.049 / 0.112 / 0.123 on seeds 1-3.
        n_iterations = 400,
        batch_size = 16,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.2,           # HIGH exploration
        epsilon_decay = true,    # Anneal to 0
        entropy_weight = 0.05,   # Higher entropy for extreme case
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training $(config_with_exp.n_iterations) iterations with exploration... ")
    history_with_exp = GFlowNet.train_gflownet(model_with_exp, config_with_exp; verbose=false)
    println("done!")

    assert_finite_iterations(history_with_exp, config_with_exp.n_iterations,
                             "TB high exploration")
    assert_loss_decreased(history_with_exp, "TB high exploration"; window=10, max_ratio=0.5)

    print("Sampling 400 trajectories (pure policy)... ")
    samples_with_exp = [GFlowNet.sample_trajectory(model_with_exp; config=eval_config) for _ in 1:400]
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
    println("  Peak (5,5) samples: $peak1_with / 400 ($(round(peak1_with/4, digits=1))%)")
    println("  Peak (1,5) samples: $peak2_with / 400 ($(round(peak2_with/4, digits=1))%)")
    println("  Other terminals:    $other_with / 400")
    println("  Modes discovered: $modes_with / 2")

    # Same two bars as TEST 1; the derivation is stated there. Measured at this
    # budget: peaks 67/68, 89/86, 88/75 on seeds 1-3, so ratios 1.01 / 1.03 / 1.17
    # against the enumerated 1.00.
    assert_modes_discovered([peak1_with, peak2_with], "TB high exploration both peaks";
                            min_per_mode=30, n_samples=400)
    assert_relative_error_below(max(peak1_with, peak2_with) / max(min(peak1_with, peak2_with), 1),
                                1.0, "TB high exploration peak ratio"; max_rel_error=3.0)

    # Analyze ratio
    if peak1_with > 0 && peak2_with > 0
        actual_ratio = peak1_with / peak2_with
        println("  Actual ratio: $(round(actual_ratio, digits=2)):1")
        println("  Expected ratio: 1:1 (equal rewards; the 70:1 path count cancels)")
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
        # Cut from 2000 / batch 64, then 150 -> 400 as in TEST 1. Measured at 400 /
        # batch 16: loss 6.289 -> -0.054, peaks 63 and 71 of 400 (seed 1); TV to the
        # enumerated law 0.087 / 0.074 / 0.121 on seeds 1-3.
        n_iterations = 400,
        batch_size = 16,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.4,            # VERY HIGH exploration
        epsilon_decay = true,     # Anneal to 0
        entropy_weight = 0.1,     # Very high entropy
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    print("Training $(config_extreme.n_iterations) iterations with extreme exploration... ")
    history_extreme = GFlowNet.train_gflownet(model_extreme, config_extreme; verbose=false)
    println("done!")

    assert_finite_iterations(history_extreme, config_extreme.n_iterations,
                             "TB extreme exploration")
    assert_loss_decreased(history_extreme, "TB extreme exploration"; window=10, max_ratio=0.5)

    print("Sampling 400 trajectories (pure policy)... ")
    samples_extreme = [GFlowNet.sample_trajectory(model_extreme; config=eval_config) for _ in 1:400]
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
    println("  Peak (5,5) samples: $peak1_ext / 400 ($(round(peak1_ext/4, digits=1))%)")
    println("  Peak (1,5) samples: $peak2_ext / 400 ($(round(peak2_ext/4, digits=1))%)")
    println("  Other terminals:    $other_ext / 400")
    println("  Modes discovered: $modes_ext / 2")

    # Same two bars as TEST 1. Measured at this budget: peaks 63/71, 61/66, 84/59 on
    # seeds 1-3, so ratios 1.13 / 1.08 / 1.42 against the enumerated 1.00.
    assert_modes_discovered([peak1_ext, peak2_ext], "TB extreme exploration both peaks";
                            min_per_mode=30, n_samples=400)
    assert_relative_error_below(max(peak1_ext, peak2_ext) / max(min(peak1_ext, peak2_ext), 1),
                                1.0, "TB extreme exploration peak ratio"; max_rel_error=3.0)

    if peak1_ext > 0 && peak2_ext > 0
        actual_ratio = peak1_ext / peak2_ext
        println("  Actual ratio: $(round(actual_ratio, digits=2)):1")
    end
    println()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    println("=" ^ 70)
    println("PATH-COUNT INVARIANCE TEST - SUMMARY")
    println("=" ^ 70)
    println()
    println("Configuration: 5×5 grid, peaks at (5,5) and (1,5), 70:1 path ratio")
    println("Enumerated target law: each peak 10/59 = 16.95%, i.e. 67.8 of 400")
    println()
    println("| Test                    | Peak(5,5) | Peak(1,5) | Modes |")
    println("|-------------------------|-----------|-----------|-------|")
    println("| No exploration (ε=0)    | $(lpad(peak1_no, 4))      | $(lpad(peak2_no, 4))      | $modes_no/2   |")
    println("| High (ε=0.2, H=0.05)    | $(lpad(peak1_with, 4))      | $(lpad(peak2_with, 4))      | $modes_with/2   |")
    println("| Extreme (ε=0.4, H=0.1)  | $(lpad(peak1_ext, 4))      | $(lpad(peak2_ext, 4))      | $modes_ext/2   |")
    println()

    # Verdict
    if modes_no == 2 && modes_with == 2 && modes_ext == 2
        println("PATH-COUNT INVARIANCE HOLDS in all three arms: both peaks were")
        println("   sampled, and the 70:1 path asymmetry did not tilt the sampler")
        println("   toward (5,5). Exploration is not what rescues the 1-path peak —")
        println("   R(x)/Z has no path-count factor in it, so there is nothing for")
        println("   exploration to rescue. What ε and entropy change here is how")
        println("   fast the loss reaches that fixed point, not where it lands.")
    else
        println("A peak was missed. Under the correct law both peaks carry 16.95% of")
        println("   the mass, so a missing peak is a regression rather than a limit")
        println("   of the problem. Check the trajectory balance loss first: dropping")
        println("   sum log P_B reinstates the n(x)R(x) law that used to make this")
        println("   demo report 70:1 and starve (1,5) at 3.8 of 400 samples.")
    end
    println()
    println("=" ^ 70)

    return (modes_no, modes_with, modes_ext)
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_extreme_mode_collapse()
end
