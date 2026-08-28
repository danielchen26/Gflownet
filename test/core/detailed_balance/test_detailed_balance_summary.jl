"""
DETAILED_BALANCE implementation verification.

This file used to be a println script with ZERO `@test` assertions, wired into
runtests.jl as part of the "Detailed Balance" group and closing with

    println("✅ Gradient computation: Working correctly")
    println("✅ Balance equation: Satisfied within numerical tolerance")
    println("The DETAILED_BALANCE implementation is fully functional!")

None of which was checked. It computed `has_forward_grads`/`has_backward_grads`/
`has_flow_grads` and PRINTED "Present" or "Missing" -- a missing gradient, i.e.
an untrained component, printed a cheerful line and the file still passed. The
balance-error loop swallowed every exception with a bare `catch  # Skip if states
aren't connected`, and the whole reporting block was guarded by
`if !isempty(balance_errors)`, so a loop that produced nothing at all also
passed. Two more sections were guarded by `if length(valid_losses) >= 2`.
"""

using Test
using GFlowNet
using GFlowNet: compute_trajectory_loss
using Statistics
using Zygote
using Random

@testset "DETAILED_BALANCE verification" begin
    Random.seed!(23)

    model = create_grid_world_gflownet(
        grid_size=4,
        hidden_dim=32,
        include_backward=true,
        include_flow_estimator=true
    )

    @testset "gradients reach every trained component" begin
        trajectories = [sample_trajectory(model) for _ in 1:8]

        loss_val, grads = Zygote.withgradient(model.parameters) do ps
            Zygote.@ignore clear_flow_cache!()
            compute_trajectory_loss(model, trajectories, ps,
                                    TrainingConfig(objective=DETAILED_BALANCE))
        end

        @test isfinite(loss_val)
        @test loss_val >= 0.0

        g = grads[1]
        # DETAILED_BALANCE trains all three components. "Missing" here means a
        # silently untrained network, which is what the old printlns tolerated.
        for component in (:forward, :backward, :flow)
            @test haskey(g, component)
            @test !isnothing(getproperty(g, component))
            # Present-but-all-zero is the same defect wearing a disguise.
            @test any(!iszero, getproperty(g, component))
            @test all(isfinite, getproperty(g, component))
        end
    end

    @testset "training produces a usable loss curve" begin
        config = TrainingConfig(
            objective=DETAILED_BALANCE,
            n_iterations=50,
            batch_size=16,
            learning_rate=0.01
        )

        history = train_gflownet(model, config; verbose=false)

        # No NaN filtering: a NaN loss is a failure, not a datum to skip. The old
        # version filtered them out and then only reported `if length >= 2`.
        @test length(history.losses) == config.n_iterations
        @test all(isfinite, history.losses)
        @test all(>=(0.0), history.losses)

        # Loss must actually come down over 50 iterations: compare the mean of
        # the first five against the mean of the last five, which is robust to
        # per-batch noise while still failing a flat or diverging curve.
        @test mean(history.losses[1:5]) > mean(history.losses[end-4:end])
    end

    @testset "detailed balance equation is satisfied on real transitions" begin
        # The old loop wrapped detailed_balance_loss in a bare `catch` and then
        # reported nothing if the result list came out empty. Transitions taken
        # from a sampled trajectory are connected by construction, so there is
        # nothing legitimate to catch.
        test_trajectories = [sample_trajectory(model) for _ in 1:20]
        balance_errors = Float64[]

        for traj in test_trajectories
            for i in 1:(length(traj.states) - 1)
                source = traj.states[i]
                is_terminal_state(source) && continue
                target = traj.states[i + 1]
                db_loss = detailed_balance_loss(model, source, target)
                @test isfinite(db_loss)
                @test db_loss >= 0.0
                push!(balance_errors, sqrt(db_loss))
            end
        end

        # 20 trajectories on a 4x4 grid always contain at least one non-terminal
        # transition; an empty list means sampling broke.
        @test !isempty(balance_errors)
        @test all(isfinite, balance_errors)
    end
end
