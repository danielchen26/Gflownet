"""
Tests for the backward policy implementation.

This file used to be a println script: it created a model, printed values, and
ended with `println("✅ All tests passed!")`. It contained ZERO `@test`
assertions, and runtests.jl wires it in as the whole "Policies" group -- so the
group could only fail by throwing, and every claim in it ("P_B computed
successfully", "trajectory balance now uses the full formula with P_B terms")
was unverified prose. Three of its five sections were additionally wrapped in
`if !isempty(applicable_actions)`.
"""

using Test
using GFlowNet
using Random

@testset "Backward Policy" begin
    Random.seed!(7)

    model = create_grid_world_gflownet(
        grid_size = 3,
        hidden_dim = 32,
        learning_rate = 0.01,
        include_backward = true
    )

    @testset "model carries a backward policy" begin
        @test !isnothing(model.backward_policy)
        @test haskey(model.parameters, :backward)
        @test haskey(model.states, :backward)
    end

    @testset "P_F is a distribution over children" begin
        s0 = model.initial_state
        applicable_actions = get_applicable_actions(s0, model.all_actions)
        # At (1,1) of a 3x3 grid the applicable actions are MoveRight and MoveUp;
        # Terminate is not. The old `if !isempty(...)` guard could only ever have
        # hidden a regression here.
        @test length(applicable_actions) == 2

        total = 0.0
        for a in applicable_actions
            next_state = apply_action(a, s0)
            p = forward_transition_probability(model, s0, next_state)
            @test 0.0 < p <= 1.0
            total += p
        end
        @test total ≈ 1.0 atol=1e-5
    end

    @testset "P_B is a distribution over parents" begin
        # (2,2) is reachable from (1,2) and (2,1), so its backward distribution
        # is a genuine choice rather than the trivial P_B = 1.
        child = GridState(2, 2, false)
        parents = GFlowNet.get_previous_states(model, child)
        @test Set(parents) == Set([GridState(1, 2, false), GridState(2, 1, false)])

        total = 0.0
        for p in parents
            prob = backward_transition_probability(model, child, p)
            @test 0.0 < prob < 1.0
            total += prob
        end
        @test total ≈ 1.0 atol=1e-5

        # A state that cannot reach `child` gets exactly zero, not a share.
        @test backward_transition_probability(model, child, GridState(3, 3, false)) == 0.0
    end

    @testset "trajectory balance consumes P_B" begin
        # Explicit trajectory through (2,2) -- a two-parent state, so P_B there is
        # not pinned to 1 and the backward parameters can actually move the loss.
        states = GridState[
            GridState(1, 1, false),
            GridState(2, 1, false),
            GridState(2, 2, false),
            GridState(2, 2, true),
        ]
        actions = GridAction[MoveRight(), MoveUp(), Terminate()]
        trajectory = GFlowNet.Trajectory(states, actions)

        loss = trajectory_balance_loss(model, trajectory)
        @test isfinite(loss)
        @test loss >= 0.0

        # The file's headline claim, now checked: perturbing ONLY the backward
        # parameters changes the trajectory-balance loss. If P_B were dropped
        # from the loss, this would be exactly equal.
        perturbed = deepcopy(model)
        perturbed.parameters.backward .*= 1.5
        @test trajectory_balance_loss(perturbed, trajectory) != loss
    end

    @testset "sampled trajectories are well formed" begin
        trajectory = sample_trajectory(model)
        @test length(trajectory.states) >= 2
        @test length(trajectory.actions) == length(trajectory.states) - 1
        @test trajectory.states[1] == model.initial_state
        @test is_terminal_state(trajectory.states[end])
        @test isfinite(trajectory_balance_loss(model, trajectory))
    end
end
