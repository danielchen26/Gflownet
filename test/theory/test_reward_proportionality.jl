# test/theory/test_reward_proportionality.jl
#
# THE ORACLE. This is the acceptance gate for the core mathematics.
#
# Two kinds of check live here, for a reason.
#
# The analytic testsets do not train. They construct the exact analytic optimum
# of the loss AS CODED and ask whether that optimum satisfies the defining
# GFlowNet theorem. That is a strictly stronger statement than "a training run
# did not converge", and it runs in seconds.
#
# The last testset DOES train, because the analytic testsets alone were not
# enough. Every assertion in them reads only the helpers in enumerate.jl, so this
# whole file passed green while src/training/losses.jl was still dropping its
# `sum log P_B` term: the file asserted "the bug that was fixed" about a repair
# that had never been applied to the loss the trainer actually calls. The
# training testset closes that gap. It drives `GFlowNet.train_step!` on the exact
# configuration that was broken -- 3x3 grid, `include_backward=false`,
# TRAJECTORY_BALANCE -- and checks the learned Z and the sampled terminal law
# against both the correct and the biased analytic laws. It costs about four
# minutes per seed, which is why it is last and why the analytic checks stay.

using Test
using GFlowNet
using Random

include(joinpath(@__DIR__, "enumerate.jl"))

@testset "GFlowNet defining theorem: p(x) proportional to R(x)" begin
    n = 3
    set_grid!(n, Dict((3, 3) => 10.0))

    Z_true = exact_Z(n)
    R = reward_table(n)
    u = path_counts(n)

    # Ground-truth sanity. If any of these fails, the HELPER is wrong and must be
    # fixed before drawing any conclusion about the implementation.
    @test Z_true ≈ 19.0
    @test u[(3, 3)] == 6

    # The optimum of the objective AS NOW CODED: TB including the sum log P_B
    # term, with a fixed uniform backward policy. Trajectory Balance is valid for
    # ANY fixed P_B, so the terminal law at the optimum must be R(x)/Z regardless
    # of which P_B is used -- that invariance is the real content of the theorem.
    law = analytic_optimum_terminal_law_corrected(n)
    @test sum(values(law)) ≈ 1.0 atol = 1e-10

    ratios = Dict(k => law[k] / (R[k] / Z_true) for k in keys(law))
    worst = maximum(abs(r - 1) for r in values(ratios))
    @info "reward-proportionality (corrected TB)" Z_true worst_deviation = worst

    # THE DEFINING THEOREM.
    # Before the P_B repair this was 0.7564 (ratios 0.2436 .. 1.4615, exactly
    # 0.2436*n(x)). After, it is 0.0.
    @test worst < 0.05

    # Regression guard on the specific bug: the uncorrected objective's optimum
    # was the path-count-biased law, and it must NOT be what we converge to now.
    biased = analytic_optimum_terminal_law(n)
    worst_biased = maximum(abs(biased[k] / (R[k] / Z_true) - 1) for k in keys(biased))
    @info "uncorrected objective, for contrast" worst_deviation = worst_biased
    @test worst_biased > 0.5          # documents the bug that was fixed
    @test worst < worst_biased        # and that the fix strictly improves it
end

@testset "mechanism: the old bias was exactly the path count n(x)" begin
    # Kept as documentation of the defect that was fixed. It characterises the
    # UNCORRECTED objective's optimum, so it must continue to hold -- it is a
    # statement about a formula, not about current behaviour.
    n = 3
    set_grid!(n, Dict((3, 3) => 10.0))
    R = reward_table(n)
    u = path_counts(n)
    law = analytic_optimum_terminal_law(n)

    Z_coded = sum(u[k] * R[k] for k in keys(R))
    @test Z_coded ≈ 78.0

    for k in keys(law)
        @test law[k] ≈ u[k] * R[k] / Z_coded atol = 1e-12
    end
    @info "path-count bias" Z_coded Z_true = exact_Z(n) ratio = Z_coded / exact_Z(n)
end

# =============================================================================
# The coded path. Everything above this line reads enumerate.jl only.
# =============================================================================

"""
    sampled_terminal_law(model; n_samples) -> Dict{Tuple{Int,Int},Float64}

Empirical distribution over terminal grid positions under the model's own
forward policy. Endpoints that are not terminal states are dropped rather than
counted, so the result is conditional on having terminated.
"""
function sampled_terminal_law(model; n_samples::Int)
    counts = Dict{Tuple{Int,Int},Int}()
    for _ in 1:n_samples
        s = sample_trajectory(model).states[end]
        GFlowNet.is_terminal_state(s) || continue
        counts[(s.x, s.y)] = get(counts, (s.x, s.y), 0) + 1
    end
    total = sum(values(counts); init = 0)
    total == 0 && return Dict{Tuple{Int,Int},Float64}()
    return Dict(k => v / total for (k, v) in counts)
end

"""Half the L1 distance over the UNION of both supports."""
tv_distance(p, q) =
    sum(abs(get(p, k, 0.0) - get(q, k, 0.0)) for k in union(keys(p), keys(q))) / 2

# Budget and tolerances are MEASURED on this exact configuration. 1000
# iterations, batch 32, learning_rate 0.005, z_learning_rate_multiplier 10.0,
# 4000 evaluation samples, three seeds:
#
#   seed        learned Z   |Z-19|/19   TV to correct law   TV to biased law
#   20260828     19.0004     0.00002        0.0207              0.2520
#   7            19.0004     0.00002        0.0167              0.2560
#   11           19.0080     0.00042        0.0199              0.2402
#
# and the same run with the `sum log P_B` term deleted from
# compute_single_trajectory_loss (the pre-repair loss, executed from a scratch
# copy of the package so this repository was untouched):
#
#   seed        learned Z   |Z-19|/19   TV to correct law   TV to biased law
#   20260828     77.9995     3.1052         0.2432              0.0106
#   7            78.3380     3.1231         0.2522              0.0144
#   11           77.6663     3.0877         0.2284              0.0189
#
# So each of the four assertions below separates the two losses by more than an
# order of magnitude, and all four fail on the pre-repair loss.
const CODED_ITERS = 1000
const CODED_BATCH = 32
const CODED_LR = 0.005
const CODED_Z_MULT = 10.0
const CODED_SAMPLES = 4000

# THREE seeds. The sibling file test_samples_proportional_to_reward.jl records a
# case where one seed in three converged and a single-seed test called the
# objective fixed; 20260828 alone would be that same mistake.
const CODED_SEEDS = (20260828, 7, 11)

@testset "coded TB loss learns Z = sum_x R(x), not sum_x n(x) R(x)" begin
    n = 3
    set_grid!(n, Dict((3, 3) => 10.0))

    Z_true = exact_Z(n)                                     # 19.0
    Z_biased = sum(path_counts(n)[k] * reward_table(n)[k]   # 78.0
                   for k in keys(reward_table(n)))
    @test Z_true ≈ 19.0
    @test Z_biased ≈ 78.0

    law_correct = analytic_optimum_terminal_law_corrected(n)
    law_biased = analytic_optimum_terminal_law(n)

    for seed in CODED_SEEDS
        @testset "seed $seed" begin
            Random.seed!(seed)
            set_grid!(n, Dict((3, 3) => 10.0))

            model = create_grid_world_gflownet(
                grid_size = n,
                reward_positions = Dict((3, 3) => 10.0),
                hidden_dim = 64,
                include_backward = false,
                partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
            )

            config = TrainingConfig(
                objective = TRAJECTORY_BALANCE,
                n_iterations = 1,
                batch_size = CODED_BATCH,
                learning_rate = CODED_LR,
                z_learning_rate_multiplier = CODED_Z_MULT,
            )

            for _ in 1:CODED_ITERS
                GFlowNet.train_step!(
                    model,
                    [sample_trajectory(model) for _ in 1:CODED_BATCH],
                    config,
                )
            end

            Z_learned = exp(model.parameters.log_Z)
            err_true = abs(Z_learned - Z_true) / Z_true
            err_biased = abs(Z_learned - Z_biased) / Z_biased

            law = sampled_terminal_law(model; n_samples = CODED_SAMPLES)
            tv_correct = tv_distance(law, law_correct)
            tv_biased = tv_distance(law, law_biased)

            @info "coded TB, forward-only P_B" seed Z_learned err_true tv_correct tv_biased

            # BOTH directions are required. Closeness to 19.0 alone would admit a
            # model that learned nothing if the tolerance were loose; distance
            # from 78.0 alone admits any wrong answer that is merely not 78.
            # Measured: 0.00002, 0.00002, 0.00042 here against 3.09..3.12 for the
            # pre-repair loss, so 0.03 sits two orders of magnitude from either.
            @test err_true < 0.03
            # Measured: 0.756..0.757 here against 6e-6..0.004 pre-repair.
            @test err_biased > 0.5

            # The terminal law, which is the property enumerate.jl names as
            # "precisely how the fix is verified": the sampler must sit on the
            # correct law, not the path-count-biased one.
            # Measured: 0.0167..0.0207 against 0.2402..0.2560, a 12x gap; the
            # pre-repair loss inverts both comparisons exactly.
            @test tv_correct < 0.05
            @test tv_correct < tv_biased
        end
    end
end
