# test/theory/test_reward_proportionality.jl
#
# THE ORACLE. This is the acceptance gate for the core mathematics.
#
# It does not train. Training is slow and noisy; instead it constructs the exact
# analytic optimum of the loss AS CODED and asks whether that optimum satisfies
# the defining GFlowNet theorem. That is a strictly stronger statement than "a
# training run did not converge", and it runs in seconds.

using Test
using GFlowNet

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
