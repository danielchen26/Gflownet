# Reward Shaping 8x8 Grid - what is left of the 3432:1 asymmetry once P_B is correct
# Run with: julia --project=. examples/core_features/reward_shaping_8x8_tuning.jl

using Statistics
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

include(joinpath(@__DIR__, "..", "convergence_assertions.jl"))

# WHAT THIS SCRIPT USED TO DO, AND WHY THE REWARDS CHANGED
# -------------------------------------------------------
# It used to train on Dict((8,8) => 10.0, (1,8) => 34320.0) and it called that
# "reward shaping for the 3432:1 path asymmetry". (8,8) sits at the end of
# binomial(14,7) = 3432 distinct lattice paths and (1,8) at the end of exactly one,
# and 34320 = 3432 x 10 exactly. That integer identity is the whole story: the
# reward was picked to cancel a path count, not for any property of reward shaping.
# The 5x5 script in this directory was built the same way, with 700 = 70 x 10.
#
# The cancellation only means anything under the bug that used to sit in
# `compute_single_trajectory_loss`, which dropped its `sum log P_B` term when the
# model had no backward policy. That is P_B = 1 unnormalised, valid only if every
# state has one parent; in this lattice (x,y) has two. The optimum of that loss is
# n(x)R(x) / sum_y n(y)R(y), so 3432 x 10 against 1 x 34320 tied at 39.221% each.
# With the backward term restored (uniform over parents) path multiplicity cancels
# inside the objective and the optimum is p(x) = R(x)/Z for any fixed P_B.
#
# So the reward compensation is not needed and is now actively misleading: at
# R(1,8) = 34320 the correct law puts 99.631% on (1,8) and 0.029% on (8,8). This
# script therefore trains on EQUAL rewards, where the correct law splits the two
# corners evenly REGARDLESS of the 3432:1 path ratio, and the question the script
# always claimed to ask -- can a sampler cover both modes under extreme path
# asymmetry -- is finally the question it actually asks.
#
# ENUMERATED LAWS (test/theory/enumerate.jl, `analytic_optimum_terminal_law_corrected`
# for the correct column and `analytic_optimum_terminal_law` for the biased one; grid 8)
#
#   equal rewards, R(8,8) = R(1,8) = 10, Z = exact_Z(8) = 137.0
#                    n_paths      correct        biased
#       (8,8)           3432       7.2993%      64.5185%
#       (1,8)              1       7.2993%       0.0188%
#       everything else            85.4015%     35.4627%
#
#   original shaped rewards, R(8,8) = 10 and R(1,8) = 34320, Z = 34447.0
#                    n_paths      correct        biased
#       (8,8)           3432       0.0290%      39.2211%
#       (1,8)              1      99.6313%      39.2211%
#       everything else             0.3397%     21.5578%
#
# Z = 137.0 is 10 + 10 + 117, the 117 being the distance-based rewards on the other
# 61 cells that may terminate. (1,1) may not terminate so it is not in Z. The biased
# denominator is sum_x n(x)R(x) = 53194.0 for equal rewards and 87504.0 for shaped.
#
# THE OLD ASSERTION WAS SATISFIABLE WITHOUT TRAINING
# --------------------------------------------------
# This file used to assert `min_per_mode = 60` of 500 on the (8,8) count and cite
# measured values of 287/500 and 319/500 as its evidence. Those are 57.4% and 63.8%,
# and the biased law puts 64.5185% on (8,8). The old bar was not measuring what the
# network had learned. It was measuring the shape of the DAG: the missing backward
# term parked 39-64% of the mass on the 3432-path corner no matter what the policy
# did, so a run that had not trained at all still cleared 60/500 comfortably.
#
# That is not hypothetical here. Post-repair the three replay arms below do not train
# at 150 iterations -- measured end/start loss ratios 0.64 to 0.95, against 0.011 to
# 0.034 for the on-policy arm -- and pre-repair those same arms passed the (8,8) floor
# anyway. An assertion a demo can satisfy by standing still is worse than no
# assertion, because it reports path multiplicity as learning. Anyone weighing how
# much to trust the rest of the numbers in this file should weigh that first.
#
# WHY THERE IS NO TERMINAL-LAW ASSERTION BELOW: KNOWN NOT ROBUST
# --------------------------------------------------------------
# Grid 8 does not reach the enumerated law at any budget this script can afford. The
# single all-up path to (1,8) is one of 3432 under a uniform walk, and finding it is
# now purely an exploration problem rather than a distributional one. Measured on the
# on-policy arm, equal rewards, 2000 evaluation samples, seeds [42,1,2], against an
# enumerated 7.2993% for each corner:
#
#     iters      (8,8) share              (1,8) share             loss ratio
#      150    7.45%  6.85%  4.75%     0.75%  2.55%  1.70%     0.011 0.034 0.013
#      300   12.35%  7.15%  7.50%     1.90%  2.90%  2.40%     0.008 0.016 0.002
#      600    4.50%  3.60%  3.85%     4.00%  6.40%  2.15%     0.009 0.006 0.002
#     1200    8.75%  6.95%  6.65%     2.50%  6.25%  4.25%     0.002 0.001 0.005
#
# The loss converges four iterations-doublings before the terminal law does. (1,8)
# climbs with budget but is still at 2.50-6.25% after 1200 iterations, and it is
# non-monotone across seeds -- 600 gives 4.00/6.40/2.15 and 1200 gives 2.50/6.25/4.25 --
# so a monotonicity check would fail on seed noise and teach the next reader that the
# demo is flaky rather than that the budget is short. There is no honest threshold in
# that table, so this file asserts none. Following the precedent in
# test/theory/enumerate.jl's caller `test_samples_proportional_to_reward.jl`, this is
# recorded as KNOWN NOT ROBUST: the numbers still run and still print, they just do
# not gate, because asserting a pass the run does not earn would be a lie and
# deleting the finding would hide it.
#
# What it would take: the 5x5 sibling reaches the enumerated law to under 1% relative
# in 800 iterations on a grid with 70:1 asymmetry, so the shortfall here is the 3432:1
# exploration cost, not the objective. A run long enough to settle (1,8) at grid 8 is
# beyond an example script -- 1200 iterations already measured 4-23 minutes -- and
# would want a targeted exploration mechanism rather than more epsilon.
#
# For reference, not asserted: at the ORIGINAL shaped rewards the same on-policy arm
# at 150 iterations measured (8,8) 4.75%/0.65%/7.00% and (1,8) 3.25%/81.95%/2.35%
# over seeds [42,1,2], against an enumerated 0.0290% and 99.6313%. It reproduces
# neither law. In particular (8,8) at 4.75% is 95/2000, so the old 60-of-500 floor
# (12% of samples) is now unreachable there as well as meaningless.
#
# BUDGET
# ------
# Cut from 2000-3000 iterations at batch 64 / hidden 128 to 150 at batch 16 / hidden
# 64. This was by far the most expensive script in the set: measured per-iteration
# cost at grid 8 was 6.3-8.6 s/iteration at batch 64 / hidden 128 under load, so the
# old 3000-iteration runs were hours. At batch 16 / hidden 64 a 150-iteration run
# measures 19-90 s.
#
# WHY 2000 EVALUATION SAMPLES AND NOT 500
# ---------------------------------------
# Each corner's enumerated share is 7.2993%. The binomial s.d. of its count is
# sqrt(n p (1-p)): at n = 500 that is 36.5 +- 5.82 counts, a 15.9% relative standard
# error, and at n = 2000 it is 146.0 +- 11.63, a 7.97% relative standard error.
# Sampling 2000 trajectories from a trained model costs about a second. The eval size
# was raised to resolve the enumerated share, not to move any bar -- there are no
# terminal-law bars here to move.
const N_EVAL = 2000

# Enumerated correct-law share of each corner under the equal rewards used below,
# from test/theory/enumerate.jl. A target, not a measurement.
const SHARE_CORNER = 0.0729927   # 10/137
const SD_CORNER = sqrt(N_EVAL * SHARE_CORNER * (1 - SHARE_CORNER))  # 11.63 counts

function count_peaks_8x8(samples)
    p1, p2, other = 0, 0, 0
    for traj in samples
        terminal = traj.states[end]
        pos = (terminal.x, terminal.y)
        if pos == (8, 8)
            p1 += 1
        elseif pos == (1, 8)
            p2 += 1
        else
            other += 1
        end
    end
    return p1, p2, other
end

function run_8x8_experiment(;label, reward_positions, n_iterations, epsilon, entropy_weight,
                             use_replay=false, epsilon_decay=true, seed=42)
    println("-" ^ 70)
    println("  $label")
    println("-" ^ 70)

    Random.seed!(seed)

    model = GFlowNet.create_grid_world_gflownet(
        grid_size = 8,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        n_iterations = n_iterations,
        batch_size = 16,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = epsilon,
        epsilon_decay = epsilon_decay,
        entropy_weight = entropy_weight,
        z_learning_rate_multiplier = 10.0,
        use_replay_buffer = use_replay,
        replay_buffer_size = 5000,
        replay_ratio = 0.5,
        verbose = false
    )

    print("  Training ($n_iterations iter, ε=$epsilon, H=$entropy_weight, replay=$use_replay)... ")
    history = GFlowNet.train_gflownet(model, config; verbose=false)

    # The anti-silent-failure gate, on every arm, with no threshold: every requested
    # iteration must have produced a finite loss. This is the assertion that catches
    # what the old (8,8) floor could not, namely an arm that does not train.
    assert_finite_iterations(history, n_iterations, label)
    # `filter(!isnan, ...)` then `isempty(...) ? NaN : ...` was the silent-pass shape:
    # a run where every iteration threw printed "loss: NaN" and continued. The
    # assertion above makes that a hard failure; the filter is kept only for printing.
    valid_losses = filter(!isnan, history.losses)
    final_loss = isempty(valid_losses) ? NaN : valid_losses[end]
    println("done! (loss: $(round(final_loss, digits=4)))")

    # Loss progress is asserted only for the on-policy arm, and it certifies loss
    # progress ONLY -- the terminal law is not asserted anywhere in this file, see the
    # KNOWN NOT ROBUST note at the top. The replay arms mix 50% stale trajectories
    # into every batch and measurably do not train at this budget: end/start ratios
    # 0.909/0.914/0.921, 0.951/0.674/0.692 and 0.726/0.713/0.637 over seeds [42,1,2],
    # and raising them to 600 iterations only reaches 0.18-0.30 across all three arms
    # on seeds [42,1], against 0.002-0.009 for the on-policy arm at the same budget.
    # Measured on-policy ratios at this budget:
    # 0.011/0.034/0.013. Bar 0.5, cleared by 15x at the worst seed.
    use_replay || assert_loss_decreased(history, label; window=10, max_ratio=0.5)

    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    samples = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:N_EVAL]
    p1, p2, other = count_peaks_8x8(samples)

    expected = SHARE_CORNER * N_EVAL
    ratio = p2 > 0 ? round(p1 / p2, digits=1) : Inf
    modes = (p1 > 10 ? 1 : 0) + (p2 > 10 ? 1 : 0)

    println("  Peak(8,8)=$p1, Peak(1,8)=$p2, Other=$other, Ratio=$ratio:1, Modes=$modes/2")
    println("  enumerated correct law: $(round(expected, digits=1)) at EACH corner " *
            "($(round(SHARE_CORNER * 100, digits=4))%), " *
            "$(round((1 - 2 * SHARE_CORNER) * N_EVAL, digits=1)) elsewhere")
    println("  deviation: (8,8) $(round((p1 - expected) / SD_CORNER, digits=1)) s.d., " *
            "(1,8) $(round((p2 - expected) / SD_CORNER, digits=1)) s.d. " *
            "(s.d. = $(round(SD_CORNER, digits=2)) counts)")
    println()
    return (p1=p1, p2=p2, other=other, ratio=ratio, modes=modes, loss=final_loss)
end

function main()
    println("=" ^ 70)
    println("8×8 EXPLORATION TUNING UNDER 3432:1 PATH ASYMMETRY")
    println("Equal rewards: the correct law splits the corners evenly regardless")
    println("=" ^ 70)
    println()

    # EQUAL rewards. The old Dict((8,8) => 10.0, (1,8) => 34320.0) is not used: 34320
    # is 3432 x 10, i.e. compensation for a path count that no longer biases anything.
    # See the enumerated tables at the top of this file for both settings.
    equal_rewards = Dict((8,8) => 10.0, (1,8) => 10.0)

    println("Enumerated laws (test/theory/enumerate.jl), grid 8:")
    println("  equal rewards R=10/R=10,   Z =   137.0 : correct 7.2993% / 7.2993%," *
            "  biased 64.5185% / 0.0188%")
    println("  old shaped  R=10/R=34320, Z = 34447.0 : correct 0.0290% / 99.6313%," *
            " biased 39.2211% / 39.2211%")
    println("  34320 = 3432 x 10 exactly, which is why the biased column ties.")
    println("  No terminal-law assertion is made below; see KNOWN NOT ROBUST in this file.")
    println()

    # Test 1: Higher epsilon, no decay
    r1 = run_8x8_experiment(
        label = "Test 1: Equal R + high ε=0.3, no decay, 150 iter",
        reward_positions = equal_rewards,
        n_iterations = 150,
        epsilon = 0.3,
        entropy_weight = 0.02,
        epsilon_decay = false,
        seed = 42
    )

    # Test 2: Replay buffer + moderate epsilon
    r2 = run_8x8_experiment(
        label = "Test 2: Equal R + ε=0.2 + replay buffer, 150 iter",
        reward_positions = equal_rewards,
        n_iterations = 150,
        epsilon = 0.2,
        entropy_weight = 0.02,
        use_replay = true,
        epsilon_decay = true,
        seed = 42
    )

    # Test 3: Very high epsilon + high entropy + replay
    r3 = run_8x8_experiment(
        label = "Test 3: Equal R + ε=0.3 + H=0.05 + replay, 150 iter",
        reward_positions = equal_rewards,
        n_iterations = 150,
        epsilon = 0.3,
        entropy_weight = 0.05,
        use_replay = true,
        epsilon_decay = false,
        seed = 42
    )

    # Test 4: Extreme epsilon to force discovery
    r4 = run_8x8_experiment(
        label = "Test 4: Equal R + ε=0.5 + H=0.05 + replay, 150 iter",
        reward_positions = equal_rewards,
        n_iterations = 150,
        epsilon = 0.5,
        entropy_weight = 0.05,
        use_replay = true,
        epsilon_decay = false,
        seed = 42
    )

    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()
    expected = SHARE_CORNER * N_EVAL
    println("Enumerated correct law: $(round(expected, digits=1)) of $N_EVAL at each " *
            "corner. Deviations in binomial s.d. ($(round(SD_CORNER, digits=2)) counts).")
    println()
    println("| Test | Replay | Peak(8,8) | vs law | Peak(1,8) | vs law | Loss   |")
    println("|------|--------|-----------|--------|-----------|--------|--------|")
    for (i, (r, rep)) in enumerate(zip([r1, r2, r3, r4], [false, true, true, true]))
        d1 = round((r.p1 - expected) / SD_CORNER, digits=1)
        d2 = round((r.p2 - expected) / SD_CORNER, digits=1)
        println("| $i    | $(lpad(rep ? "yes" : "no", 6)) | $(lpad(r.p1, 9)) | " *
                "$(lpad(d1, 6)) | $(lpad(r.p2, 9)) | $(lpad(d2, 6)) | " *
                "$(lpad(round(r.loss, digits=3), 6)) |")
    end
    println()
    println("No arm reaches the enumerated law, and this is reported rather than")
    println("asserted; see KNOWN NOT ROBUST at the top of this file. The arms are NOT")
    println("ranked by their (1,8) count here: that count is a single seed of a")
    println("quantity measured at 0.75-2.55% across seeds on the best arm, so an")
    println("argmax over four of them ranks seed noise. What does separate the arms is")
    println("whether they train at all: the on-policy arm's loss falls by ~100x at this")
    println("budget, the three replay arms by under 2x.")
    println("=" ^ 70)

    return (r1, r2, r3, r4)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
