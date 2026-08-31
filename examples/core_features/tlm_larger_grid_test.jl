# TLM / TB / reward-shaping comparison on larger grids
# Runs 4x4, 5x5 and 6x6 to verify scalability
#
# Run with: julia --project=. examples/core_features/tlm_larger_grid_test.jl
#
# WHAT THIS SCRIPT USED TO CLAIM, AND WHY THAT CHANGED
# ----------------------------------------------------
# This script was built around the belief that a trajectory-balance sampler
# converges to P(x) proportional to R(x) * n(x), where n(x) counts the distinct
# (1,1) -> x paths; hence that the 1-path corner (1,g) is structurally starved next
# to the binomial(2(g-1), g-1)-path corner (g,g), and needs its reward boosted by
# the path ratio to compete. That law was an artefact of a bug. The TB loss in
# src/training/losses.jl dropped its `sum log P_B` term whenever the model had no
# backward policy -- i.e. it assumed P_B = 1 unnormalised, which holds only if every
# state has one parent, and the grid lattice gives interior states two. With the
# term restored (uniform over parents) the sampler converges to P(x) = R(x)/Z,
# Z = sum_x R(x) over the states that may terminate, and n(x) does not appear.
#
# Two consequences run through this whole file:
#   * With EQUAL rewards the two corners are now EQUALLY likely, whatever the path
#     ratio. There is no structural starvation left, so the TB and TLM arms are no
#     longer a starved/rescued pair -- they are two routes to the same law, and the
#     interesting observation is that they AGREE. Enumerated shares for R=10 at both
#     corners: 24.39% each on 4x4, 16.95% on 5x5, 12.35% on 6x6.
#   * Boosting the minority reward BY the path ratio is now an OVER-correction of
#     exactly that ratio. On 4x4 it does not balance the corners; it hands (1,4)
#     86.58% against 4.33% for (4,4). The reward-shaping arm is kept for that reason
#     and is reported as an over-correction, not as a fix.
#
# All shares quoted below come from `analytic_optimum_terminal_law_corrected` in
# test/theory/enumerate.jl, which is exact enumeration of the DAG, not a measurement.
#
# THRESHOLD RULE USED AT EVERY assert_modes_discovered BELOW
# ----------------------------------------------------------
# m = p * N_EVAL is the count the enumerated correct law predicts for a mode of
# share p, and s = sqrt(N_EVAL * p * (1-p)) is the binomial standard deviation of
# that count. A bar is admissible only if it sits at least 5s below m -- so a
# sampler that IS at the correct law cannot fail it through sampling noise -- and at
# most m/2, so it still fails on a sampler that has lost the mode:
#   bar = floor(min(m - 5s, m/2))
# If m - 5s <= 0 the mode is too rare at this N to carry a lower bound at all, and
# the assertion must move to the mode that carries the mass. Every bar below quotes
# its own m and s, and the grid size that binds the shared value.

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet
using Statistics
using Random

include(joinpath(@__DIR__, "..", "convergence_assertions.jl"))

Random.seed!(42)

# Budget cut from 500 iterations / batch 32 to 60 / batch 16, and evaluation from 500
# to 300 samples. This script trains NINE models (3 grid sizes x 3 methods), so the
# old budget was 4500 iterations and could not finish inside 300s.
#
# Measured at 60 iterations / batch 16, AFTER the P_B repair:
#   grid 4, TB : loss 5.555 -> 0.214 (ratio 0.0385), corners 78 and 55 of 300
#   grid 4, TLM: loss 7.751 -> 0.533 (ratio 0.0687), corners 101 and 42 of 300
#   grid 4, RS : loss 8.238 -> 0.103 (ratio 0.0124), corners 0 and 274 of 300
#                (enumerated 12.99 and 259.74 -- the shaped corner matches, and
#                 (4,4) at 4.33% is too rare at N=300 to bound, see its assertion)
# i.e. the loss is converged everywhere at this budget. The pre-repair figures that
# used to sit here (grid 4 TB majority peak 218/300, grid 6 TB 204/300) were the
# path-count-biased law and are not comparable: the enumerated correct share of the
# majority corner is 24.39% on 4x4 and 12.35% on 6x6, so 218/300 and 204/300 were
# never reachable targets under a correct TB loss.
const N_EVAL = 300

function test_grid(grid_size::Int; n_iterations::Int=60)
    println("\n" * "=" ^ 60)
    println("Grid Size: $(grid_size)×$(grid_size)")
    println("=" ^ 60)

    # Path counts still describe the DAG, but they no longer describe the sampling
    # law: post-repair P(x) = R(x)/Z with no n(x) factor. The ratio is printed
    # because the reward-shaping arm below is scaled by it, and because it is the
    # exact factor by which that scaling now OVER-shoots.
    # Peak at (grid_size, grid_size): binomial(2*(grid_size-1), grid_size-1) paths
    # Peak at (1, grid_size): binomial(grid_size-1, 0) = 1 path
    n = grid_size - 1
    majority_paths = binomial(2*n, n)
    minority_paths = 1
    path_ratio = majority_paths / minority_paths

    println("Path analysis (structural only -- P(x) = R(x)/Z does not depend on it):")
    println("  Peak ($grid_size,$grid_size): $majority_paths paths")
    println("  Peak (1,$grid_size): $minority_paths path")
    println("  Path ratio: $(round(path_ratio, digits=1)):1")

    reward_positions = Dict(
        (grid_size, grid_size) => 10.0,
        (1, grid_size) => 10.0
    )

    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 50
    )

    # Test 1: Standard TB
    println("\n--- Standard TB ---")
    model_tb = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        include_backward = false,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_tb = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = n_iterations,
        batch_size = 16,
        learning_rate = 0.005,
        epsilon = 0.1,
        epsilon_decay = true,
        entropy_weight = 0.01,
        verbose = false
    )

    print("Training TB ($n_iterations iter)... ")
    history_tb = GFlowNet.train_gflownet(model_tb, config_tb; verbose=false)
    println("done!")

    samples_tb = [GFlowNet.sample_trajectory(model_tb; config=eval_config) for _ in 1:N_EVAL]
    p1_tb, p2_tb = count_peaks(samples_tb, grid_size)
    assert_finite_iterations(history_tb, n_iterations, "TB $(grid_size)x$(grid_size)")
    assert_loss_decreased(history_tb, "TB $(grid_size)x$(grid_size)"; window=6, max_ratio=0.5)
    # Equal rewards, so the enumerated correct law gives BOTH corners the same share
    # and neither is "the majority" any more; the name is kept only for continuity
    # with the printed table. Shares / m / s at N_EVAL=300 for (g,g):
    #   g=4  24.39%  m=73.17  s=7.44  ->  min(m-5s, m/2) = 35.98
    #   g=5  16.95%  m=50.85  s=6.50  ->                    18.36
    #   g=6  12.35%  m=37.04  s=5.70  ->                     8.55   <- binds
    # One bar serves all three grid sizes, so it must be the smallest: 8.
    # The old bar of 30 was derived from 218/300 and 204/300, which were the
    # path-count-biased law; under the correct law grid 6 expects only 37.04, so a
    # bar of 30 there sat 1.2s below the mean and would have failed roughly one run
    # in nine on noise alone. 8 sits 5.1s below.
    assert_modes_discovered([p1_tb], "TB $(grid_size)x$(grid_size) majority peak";
                            min_per_mode=8, n_samples=N_EVAL)
    println("  Results: Peak($grid_size,$grid_size)=$p1_tb, Peak(1,$grid_size)=$p2_tb")

    # Test 2: TLM
    println("\n--- TLM (ICLR 2025) ---")
    model_tlm = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        include_backward = true,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_tlm = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_LIKELIHOOD_MAXIMIZATION,
        n_iterations = n_iterations,
        batch_size = 16,
        learning_rate = 0.005,
        epsilon = 0.1,
        epsilon_decay = true,
        entropy_weight = 0.01,
        tlm_backward_weight = 1.0,
        tlm_entropy_coeff = 0.01,
        verbose = false
    )

    print("Training TLM ($n_iterations iter)... ")
    history_tlm = GFlowNet.train_gflownet(model_tlm, config_tlm; verbose=false)
    println("done!")

    samples_tlm = [GFlowNet.sample_trajectory(model_tlm; config=eval_config) for _ in 1:N_EVAL]
    p1_tlm, p2_tlm = count_peaks(samples_tlm, grid_size)
    assert_finite_iterations(history_tlm, n_iterations, "TLM $(grid_size)x$(grid_size)")
    assert_loss_decreased(history_tlm, "TLM $(grid_size)x$(grid_size)"; window=6, max_ratio=0.5)
    # SAME bar as the TB arm above, and that is the point. TLM learns P_B instead of
    # fixing it uniform, and trajectory balance is valid for ANY P_B, so both arms
    # must land on the same law P(x) = R(x)/Z -- an invariance the theory demands,
    # which is stronger evidence than the old dissociation this comment used to
    # celebrate ("TLM moves mass OFF the majority peak"). There is no path-count bias
    # left for TLM to compensate, so the pre-repair figures quoted here (TLM majority
    # 25/300 and 12/300 against 204-218/300 for TB) described the bug, not TLM.
    # Enumerated shares are identical to the TB arm's: grid 6 binds at
    # min(m-5s, m/2) = 8.55 for m=37.04, s=5.70.
    assert_modes_discovered([p1_tlm], "TLM $(grid_size)x$(grid_size) majority peak";
                            min_per_mode=8, n_samples=N_EVAL)
    println("  Results: Peak($grid_size,$grid_size)=$p1_tlm, Peak(1,$grid_size)=$p2_tlm")

    # Test 3: reward shaping by the path ratio -- NOW AN OVER-CORRECTION
    #
    # This arm was built to cancel the path-count bias: boost the 1-path corner by
    # n((g,g)) so that R * n balances. With the bias gone the sampling law is R(x)/Z
    # and there is nothing to cancel, so multiplying R((1,g)) by the path ratio
    # over-shoots by exactly that ratio. The arm is kept because that over-shoot is
    # now the thing worth showing: the enumerated correct law hands (1,g) 86.58%
    # (g=4), 93.46% (g=5), 97.26% (g=6) and leaves (g,g) with 4.33%, 1.34%, 0.39%.
    # For equal-reward mode balance under a correct TB loss, no shaping is needed at
    # all -- see the TB arm above, where both corners already share the mass.
    println("\n--- Reward Shaping by path ratio (over-corrects post-repair) ---")
    shaped_reward = Dict(
        (grid_size, grid_size) => 10.0,
        (1, grid_size) => 10.0 * path_ratio  # path-ratio boost, now an over-shoot
    )

    model_rs = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = shaped_reward,
        hidden_dim = 64,
        learning_rate = 0.005,
        include_backward = false,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config_rs = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = n_iterations,
        batch_size = 16,
        learning_rate = 0.005,
        epsilon = 0.1,
        epsilon_decay = true,
        entropy_weight = 0.01,
        verbose = false
    )

    print("Training with reward shaping ($n_iterations iter)... ")
    history_rs = GFlowNet.train_gflownet(model_rs, config_rs; verbose=false)
    println("done!")

    samples_rs = [GFlowNet.sample_trajectory(model_rs; config=eval_config) for _ in 1:N_EVAL]
    p1_rs, p2_rs = count_peaks(samples_rs, grid_size)
    assert_finite_iterations(history_rs, n_iterations, "reward shaping $(grid_size)x$(grid_size)")
    assert_loss_decreased(history_rs, "reward shaping $(grid_size)x$(grid_size)"; window=6, max_ratio=0.5)
    # The assertion has MOVED from (g,g) to (1,g), because (g,g) can no longer carry
    # a lower bound here at all: its enumerated shares give m = 12.99 (g=4), 4.01
    # (g=5), 1.16 (g=6) at N_EVAL=300, and m - 5s is NEGATIVE in all three cases
    # (-4.64, -5.93, -4.21), so any positive bar on (g,g) could be tripped by
    # sampling noise from a sampler that is exactly at the correct law. The old bar
    # of 30 was worse than fragile, it was unreachable: 30 against an expected 12.99
    # on 4x4 is 4.8s ABOVE the mean, and the observed count was 0.
    # (1,g) is the corner the shaping now favours, so it is the one that is bounded:
    #   g=4  86.58%  m=259.74  s=5.90  ->  min(m-5s, m/2) = 129.87  <- binds
    #   g=5  93.46%  m=280.37  s=4.28  ->                    140.19
    #   g=6  97.26%  m=291.78  s=2.83  ->                    145.89
    assert_modes_discovered([p2_rs], "reward shaping $(grid_size)x$(grid_size) shaped peak";
                            min_per_mode=129, n_samples=N_EVAL)
    println("  Results: Peak($grid_size,$grid_size)=$p1_rs, Peak(1,$grid_size)=$p2_rs")

    # Summary
    modes_tb = (p1_tb > 5 ? 1 : 0) + (p2_tb > 5 ? 1 : 0)
    modes_tlm = (p1_tlm > 5 ? 1 : 0) + (p2_tlm > 5 ? 1 : 0)
    modes_rs = (p1_rs > 5 ? 1 : 0) + (p2_rs > 5 ? 1 : 0)

    # Expected counts under the enumerated correct law, for reading the table:
    #   TB and TLM (R=10 at both corners): both columns 73.17 at g=4, 50.85 at g=5,
    #     37.04 at g=6, out of 300 -- equal, because P(x) = R(x)/Z ignores n(x).
    #   Reward shaping (R((1,g)) = 10 * path_ratio): 12.99 / 259.74 at g=4,
    #     4.01 / 280.37 at g=5, 1.16 / 291.78 at g=6.
    println("\n--- Summary for $(grid_size)×$(grid_size) (path ratio $(round(path_ratio, digits=0)):1, which the sampling law no longer follows) ---")
    println("| Method          | ($grid_size,$grid_size) | (1,$grid_size) | Modes |")
    println("|-----------------|----------|----------|-------|")
    println("| TB              | $(lpad(p1_tb, 4))     | $(lpad(p2_tb, 4))     | $modes_tb/2   |")
    println("| TLM             | $(lpad(p1_tlm, 4))     | $(lpad(p2_tlm, 4))     | $modes_tlm/2   |")
    println("| Reward Shaping  | $(lpad(p1_rs, 4))     | $(lpad(p2_rs, 4))     | $modes_rs/2   |")

    return (tb=(p1_tb, p2_tb), tlm=(p1_tlm, p2_tlm), rs=(p1_rs, p2_rs), path_ratio=path_ratio)
end

function count_peaks(samples, grid_size)
    p1, p2 = 0, 0
    for traj in samples
        if !isempty(traj.states)
            terminal = traj.states[end]
            pos = (terminal.x, terminal.y)
            if pos == (grid_size, grid_size)
                p1 += 1
            elseif pos == (1, grid_size)
                p2 += 1
            end
        end
    end
    return p1, p2
end

function main()
    println("=" ^ 60)
    println("TLM LARGER GRID TEST")
    println("Testing TLM vs TB vs Reward Shaping on different grid sizes")
    println("=" ^ 60)

    results = Dict()

    # Test on different grid sizes
    for grid_size in [4, 5, 6]
        results[grid_size] = test_grid(grid_size; n_iterations=60)
    end

    # Final summary
    println("\n" * "=" ^ 60)
    println("FINAL SUMMARY - All Grid Sizes")
    println("=" ^ 60)
    println()
    println("| Grid | Path Ratio | TB (1,g) | TLM (1,g) | RS (1,g) | Correct-law (1,g) |")
    println("|------|------------|----------|-----------|----------|-------------------|")
    # Correct-law column: R((1,g))/Z * 300, enumerated. Equal rewards for TB and TLM
    # give 73.17 / 50.85 / 37.04; the reward-shaped run targets 259.74 / 280.37 /
    # 291.78 because its boost is now an over-correction rather than a compensation.
    for (grid_size, expect_eq) in zip([4, 5, 6], [73.17, 50.85, 37.04])
        r = results[grid_size]
        println("| $(grid_size)×$(grid_size) | $(lpad(round(Int, r.path_ratio), 5)):1    | $(lpad(r.tb[2], 6))   | $(lpad(r.tlm[2], 7))   | $(lpad(r.rs[2], 6))   | $(lpad(expect_eq, 15))   |")
    end
    println()
    println("Key observations:")
    println("  • Path ratio grows exponentially with grid size, but the sampling law")
    println("    does not depend on it: P(x) = R(x)/Z, so equal-reward corners are")
    println("    equally likely on every grid size (TB and TLM (1,g) columns).")
    println("  • TB and TLM agree, as trajectory balance requires: the objective is")
    println("    valid for any fixed or learned P_B, so both reach the same law.")
    println("  • Reward shaping by the path ratio OVER-corrects by exactly that ratio.")
    println("    It is no longer the fix for mode imbalance; equal rewards already are.")
    println()
end

main()
