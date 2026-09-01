"""
Replay path parity with the non-replay path.

WHY THIS FILE EXISTS

`train_step!` and `train_step_weighted!` each carried their own copy of the parameter-update
tail, and the copies diverged. The replay copy scaled the log_Z gradient and handed it to
Adam, and skipped `clip_gradients!` entirely. Both of those had already been repaired in
`train_step!`, so setting `use_replay_buffer = true` silently reverted two fixes at once --
without an error, without a warning, and without any observable except the training dynamics.

Scaling a gradient before Adam is a no-op because Adam divides by the gradient's own second
moment, so `z_learning_rate_multiplier` had no route to the step size on that path.

MEASURED before the repair, one step on a 4x4 grid with reward 10 at (4,4), seed 7:

    path         dlog_Z at multiplier 1    at multiplier 10    ratio
    non-replay          0.060095               0.600954         10.0
    replay              0.010000               0.010000          1.0

After factoring the shared tail into `_apply_gradient_step!`, both paths give 0.060095 and
0.600954 -- identical to the digit, ratio exactly 10.0.

WHAT THIS TEST PINS

Not "the multiplier works" -- that would pass on one path alone. It pins PARITY: the two paths
must respond to the knob the same way. A future edit to either one that does not go through the
shared helper breaks this, which is the only thing that would have caught the original
divergence.
"""

using Test
using GFlowNet
using Random

@testset "Replay and non-replay paths agree" begin

    # One step, so the measurement is the step itself rather than accumulated dynamics.
    function logz_movement(multiplier::Float64, use_replay::Bool)
        Random.seed!(7)
        model = create_grid_world_gflownet(
            grid_size = 4,
            reward_positions = Dict((4, 4) => 10.0),
            hidden_dim = 32,
            learning_rate = 0.01,
            partition_function_method = LEARNABLE_ESTIMATION,
        )
        z0 = model.parameters.log_Z
        config = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            partition_function_method = LEARNABLE_ESTIMATION,
            n_iterations = 1,
            batch_size = 8,
            learning_rate = 0.01,
            z_learning_rate_multiplier = multiplier,
            use_replay_buffer = use_replay,
            replay_buffer_size = 64,
        )
        trajectories = [sample_trajectory(model) for _ in 1:8]
        if use_replay
            GFlowNet.train_step_weighted!(model, trajectories, ones(8), config)
        else
            GFlowNet.train_step!(model, trajectories, config)
        end
        return abs(model.parameters.log_Z - z0)
    end

    @testset "z_learning_rate_multiplier scales log_Z on BOTH paths" begin
        for use_replay in (false, true)
            at1 = logz_movement(1.0, use_replay)
            at10 = logz_movement(10.0, use_replay)

            # Measured 0.060095 and 0.600954 on both paths. Before the repair the replay path
            # gave 0.010000 for BOTH multipliers, i.e. a ratio of 1.0 -- which this rejects.
            @test at1 > 1e-6
            @test isapprox(at10 / at1, 10.0; rtol = 1e-3)
        end
    end

    @testset "the two paths agree to the digit" begin
        # The parity assertion proper. Same seed, same configuration, same trajectories: the
        # only difference is which function applies the step, and after factoring the shared
        # tail there is no difference left to find.
        for multiplier in (1.0, 10.0)
            plain = logz_movement(multiplier, false)
            replay = logz_movement(multiplier, true)
            @test isapprox(plain, replay; rtol = 1e-9)
        end
    end

    @testset "gradient clipping is reached on both paths" begin
        # `clip_gradients!` was absent from the replay tail. Its measured effect under Adam is
        # small -- clip_norm 1e6 vs 1e-6 over 40 steps moved parameters 1.2488 vs 1.2518, a
        # 0.25% difference, because Adam divides out a uniform rescale -- so asserting a large
        # behavioural change would be asserting something false. What IS assertable is that an
        # extreme clip norm produces the SAME small perturbation on both paths, which can only
        # hold if both paths call it.
        function movement(clip::Float64, use_replay::Bool)
            Random.seed!(11)
            model = create_grid_world_gflownet(
                grid_size = 4, reward_positions = Dict((4, 4) => 10.0), hidden_dim = 32,
                learning_rate = 0.01, partition_function_method = LEARNABLE_ESTIMATION)
            p0 = copy(model.parameters)
            config = TrainingConfig(
                objective = TRAJECTORY_BALANCE,
                partition_function_method = LEARNABLE_ESTIMATION,
                n_iterations = 1, batch_size = 8, learning_rate = 0.01,
                gradient_clip_norm = clip,
                use_replay_buffer = use_replay, replay_buffer_size = 64)
            trajectories = [sample_trajectory(model) for _ in 1:8]
            if use_replay
                GFlowNet.train_step_weighted!(model, trajectories, ones(8), config)
            else
                GFlowNet.train_step!(model, trajectories, config)
            end
            return maximum(abs.(model.parameters .- p0))
        end

        for clip in (1e-6, 1e6)
            @test isapprox(movement(clip, false), movement(clip, true); rtol = 1e-9)
        end
    end
end
