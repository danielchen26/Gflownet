# test/theory/test_objective_health.jl
#
# Structural health checks on the objectives. These are properties of the code at
# ANY parameter setting, so they need no training and are decisive.
#
# Three things are asserted:
#   1. every objective produces a non-zero gradient for the components it claims
#      to use -- a component with norm 0 can never learn;
#   2. every objective actually responds to the reward -- an objective whose loss
#      is invariant to a 100x reward change cannot be a GFlowNet objective;
#   3. P_B is a probability distribution over the parent set.

using Test
using GFlowNet
using Zygote
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "enumerate.jl"))

# Component-wise gradient norm; log_Z is a scalar, so `vec` would throw on it.
function component_norm(g, c::Symbol)
    haskey(g, c) || return nothing
    v = getproperty(g, c)
    return v isa Number ? abs(v) : norm(vec(collect(v)))
end

function build_model(n::Int)
    Random.seed!(20260826)
    return create_grid_world_gflownet(
        grid_size = n,
        reward_positions = Dict((3, 3) => 10.0),
        hidden_dim = 8,
        include_backward = true,
        include_flow_estimator = true,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
    )
end

# Components each objective is supposed to be learning. A zero gradient here is a
# structural defect, not a tuning issue.
const REQUIRED_COMPONENTS = Dict(
    GFlowNet.TRAJECTORY_BALANCE => (:forward, :log_Z),
    GFlowNet.DETAILED_BALANCE => (:forward, :backward, :flow),
    GFlowNet.FLOW_MATCHING => (:forward, :flow),
    GFlowNet.SUB_TRAJECTORY_BALANCE => (:forward, :flow),
)

@testset "no objective has a dead gradient component" begin
    n = 3
    set_grid!(n, Dict((3, 3) => 10.0))
    model = build_model(n)
    trajs = [sample_trajectory(model) for _ in 1:8]

    for (obj, required) in REQUIRED_COMPONENTS
        cfg = TrainingConfig(objective = obj, n_iterations = 1, batch_size = 8)
        g = Zygote.gradient(p -> GFlowNet.compute_trajectory_loss(model, trajs, p, cfg),
                            model.parameters)[1]
        @test g !== nothing
        g === nothing && continue
        norms = Dict(c => component_norm(g, c) for c in required)
        @info "gradient norms" objective = obj norms = norms
        for c in required
            nrm = norms[c]
            @test nrm !== nothing
            # 1e-2 rather than 1e-6: FLOW_MATCHING leaks 0.004535 through the
            # entropy term alone, which is not a learning signal. Measured
            # reference: TB forward = 14.42.
            @test nrm !== nothing && nrm > 1e-2
        end
    end
end

@testset "every objective responds to the reward" begin
    n = 3
    set_grid!(n, Dict((3, 3) => 10.0))
    model = build_model(n)
    trajs = [sample_trajectory(model) for _ in 1:8]

    for obj in keys(REQUIRED_COMPONENTS)
        cfg = TrainingConfig(objective = obj, n_iterations = 1, batch_size = 8)
        set_grid!(n, Dict((3, 3) => 10.0))
        l1 = GFlowNet.compute_trajectory_loss(model, trajs, model.parameters, cfg)
        set_grid!(n, Dict((3, 3) => 1000.0))
        l2 = GFlowNet.compute_trajectory_loss(model, trajs, model.parameters, cfg)
        set_grid!(n, Dict((3, 3) => 10.0))
        @info "reward sensitivity" objective = obj loss_R10 = l1 loss_R1000 = l2 delta = abs(l1 - l2)
        # Measured before the repair: SUB_TRAJECTORY_BALANCE gives exactly 0.0,
        # both losses being 9.990423551228066.
        @test abs(l1 - l2) > 1e-6
    end
end

@testset "P_B is a distribution over parents" begin
    n = 3
    set_grid!(n, Dict((3, 3) => 10.0))
    model = build_model(n)
    bw = model.backward_policy
    @test bw !== nothing

    worst_multi = 0.0
    worst_single = 0.0
    for x in 1:n, y in 1:n
        child = GS(x, y, false)
        ps = parents_of(child)
        isempty(ps) && continue
        # Argument order is (policy, child, parent, ...): the DB loss calls it as
        # (policy, target, source, ...) with source earlier in the trajectory, so
        # the third argument is the parent. Reversing them silently returns 0.0.
        total = sum(GFlowNet.compute_backward_probability(
                        bw, child, p, model.parameters.backward,
                        model.states.backward, model.all_actions) for p in ps)
        dev = abs(total - 1.0)
        if length(ps) >= 2
            worst_multi = max(worst_multi, dev)
        else
            worst_single = max(worst_single, dev)
        end
    end
    @info "P_B normalisation" worst_multi_parent = worst_multi worst_single_parent = worst_single
    # Measured before the repair: multi-parent sums 1.1967-1.2922 (dev up to
    # 0.292); single-parent P_B 0.51-0.68 (dev up to 0.488).
    @test worst_multi < 1e-6
    @test worst_single < 1e-6
end
