"""
Grid world with forward-only and full backward-policy models.

This file used to be a println script with ZERO `@test` assertions, wired into
runtests.jl as half the "Grid World Application" group and closing with
`println("✅ Both versions work correctly!")`. It could only fail by throwing,
and section 3 was additionally wrapped in `if length(traj.states) >= 2`, so its
comparison could vanish without a trace.
"""

using Test
using GFlowNet
using Random
using Statistics: mean

@testset "Grid World Policy Versions" begin
    Random.seed!(11)

    config = TrainingConfig(
        objective = TRAJECTORY_BALANCE,
        n_iterations = 10,
        batch_size = 8,
        learning_rate = 0.01
    )

    model_forward_only = create_grid_world_gflownet(
        grid_size = 4,
        hidden_dim = 32,
        learning_rate = 0.01,
        include_backward = false
    )
    model_with_backward = create_grid_world_gflownet(
        grid_size = 4,
        hidden_dim = 32,
        learning_rate = 0.01,
        include_backward = true
    )

    """Every recorded action must actually produce the next recorded state."""
    function trajectory_is_consistent(traj)
        for i in 1:length(traj.actions)
            is_applicable(traj.actions[i], traj.states[i]) || return false
            apply_action(traj.actions[i], traj.states[i]) == traj.states[i+1] || return false
        end
        return true
    end

    @testset "forward-only model trains and samples" begin
        @test isnothing(model_forward_only.backward_policy)
        @test !haskey(model_forward_only.parameters, :backward)

        history = train_gflownet(model_forward_only, config; verbose=false)
        @test length(history.losses) == 10
        @test all(isfinite, history.losses)
        @test all(>=(0.0), history.losses)

        trajectories = [sample_trajectory(model_forward_only) for _ in 1:5]
        for t in trajectories
            @test length(t.states) >= 2
            @test length(t.actions) == length(t.states) - 1
            @test t.states[1] == model_forward_only.initial_state
            @test is_terminal_state(t.states[end])
            @test trajectory_is_consistent(t)
        end
    end

    @testset "backward-policy model trains and samples" begin
        @test !isnothing(model_with_backward.backward_policy)
        @test haskey(model_with_backward.parameters, :backward)

        history = train_gflownet(model_with_backward, config; verbose=false)
        @test length(history.losses) == 10
        @test all(isfinite, history.losses)
        @test all(>=(0.0), history.losses)

        trajectories = [sample_trajectory(model_with_backward) for _ in 1:5]
        for t in trajectories
            @test length(t.states) >= 2
            @test length(t.actions) == length(t.states) - 1
            @test is_terminal_state(t.states[end])
            @test trajectory_is_consistent(t)
        end
    end

    @testset "the two versions differ in P_B, not just in prose" begin
        # Explicit trajectory through (2,2), which has two parents, so P_B there
        # is a real choice rather than a forced 1. The old version sampled a
        # trajectory and only inspected it `if length(traj.states) >= 2`.
        states = GridState[
            GridState(1, 1, false),
            GridState(2, 1, false),
            GridState(2, 2, false),
            GridState(2, 2, true),
        ]
        actions = GridAction[MoveRight(), MoveUp(), Terminate()]
        traj = GFlowNet.Trajectory(states, actions)

        p_forward = forward_transition_probability(model_with_backward,
                                                  states[1], states[2])
        @test 0.0 < p_forward <= 1.0

        p_backward = backward_transition_probability(model_with_backward,
                                                    states[3], states[2])
        @test 0.0 < p_backward < 1.0

        # Give both models IDENTICAL forward parameters, so the only remaining
        # difference in the trajectory-balance loss is the sum of log P_B terms.
        # Without this the two losses would differ merely because the two models
        # were initialized from different random draws, and the assertion below
        # would prove nothing.
        model_with_backward.parameters.forward .= model_forward_only.parameters.forward

        loss_forward_only = trajectory_balance_loss(model_forward_only, traj)
        loss_with_backward = trajectory_balance_loss(model_with_backward, traj)
        @test isfinite(loss_forward_only)
        @test isfinite(loss_with_backward)

        # Forward-only TB uses P_B = 1, i.e. log P_B = 0. The backward model adds
        # a learned, strictly-less-than-1 P_B at (2,2), so the losses MUST differ.
        # This is the assertion the closing println only claimed.
        @test loss_forward_only != loss_with_backward
    end
end
