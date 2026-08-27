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
    law = analytic_optimum_terminal_law(n)

    # Ground-truth sanity: these are independently reproduced figures. If any of
    # these three fails, the HELPER is wrong and must be fixed before drawing any
    # conclusion about the implementation.
    @test Z_true ≈ 19.0
    @test u[(3, 3)] == 6
    @test sum(values(law)) ≈ 1.0 atol = 1e-10

    ratios = Dict(k => law[k] / (R[k] / Z_true) for k in keys(law))
    worst = maximum(abs(r - 1) for r in values(ratios))
    @info "reward-proportionality" Z_true worst_deviation = worst ratios = ratios

    # ---------------------------------------------------------------------------
    # The defining theorem. At the optimum of a CORRECT GFlowNet objective the
    # terminal law must be R(x)/Z for every x.
    #
    # Measured before the repair: ratios span 0.2436 (n=1) to 1.4615 (n=6),
    # i.e. worst deviation about 0.46.
    # ---------------------------------------------------------------------------
    @test worst < 0.05
end

@testset "mechanism: the bias is exactly the path count n(x)" begin
    # This testset documents WHY the theorem above fails, so the failure is
    # actionable rather than mysterious.
    #
    # IMPORTANT: this testset is EXPECTED TO PASS while the bug is present and to
    # FAIL once the missing `sum log P_B` term is restored. Its transition from
    # pass to fail is the proof that the fix worked -- at which point delete it.
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
