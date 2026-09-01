"""
Objective x flag matrix: no configuration may return a history containing NaN.

WHY THIS FILE EXISTS

`train_gflownet` catches every per-iteration exception and pushes NaN (training.jl, the
`catch e` in the iteration loop), then prints its completion banner regardless. So an
objective with no implementation, or one whose loss dereferences a component the model does
not carry, does not fail -- it returns a full-length history of NaN and looks like a training
run that went badly.

An adversarial audit measured 12 of 32 cells in this matrix doing exactly that:

    DIRECT_FLOW_OBJECTIVE   4 of 4 flag combinations, 0 of 5 finite losses
    COMBINED_OBJECTIVES     4 of 4, 0 of 5
    MULTI_OBJECTIVE_TB      4 of 4, 0 of 5   (on a plain model)

`validate_training_config` refused none of them. DIRECT_FLOW_OBJECTIVE and
COMBINED_OBJECTIVES had entries in the requirement table whose `why` text said, in prose,
that they were disabled or unimplemented -- while declaring zero required components, so the
validator found nothing missing and let them through. A table that can only express MISSING
COMPONENTS cannot express NOT TRAINABLE; that is why the table now carries a `refuse` field.

THE ASSERTION

For every (objective, include_backward, include_flow_estimator) cell: either training
produces `count(isfinite, losses) == n_iterations`, or the call THROWS. A history with any
NaN in it is a failure. That single rule is what makes this class of defect impossible to
reintroduce quietly, and it needs no threshold -- which is the point, because a threshold
would have to be calibrated and could then be wrong.
"""

using Test
using GFlowNet
using Random

@testset "Objective x flag matrix has no silent NaN" begin
    objectives = (
        TRAJECTORY_BALANCE,
        DETAILED_BALANCE,
        FLOW_MATCHING,
        SUB_TRAJECTORY_BALANCE,
        TRAJECTORY_LIKELIHOOD_MAXIMIZATION,
        DIRECT_FLOW_OBJECTIVE,
        COMBINED_OBJECTIVES,
        MULTI_OBJECTIVE_TB,
    )

    # Deliberately cheap: 5 iterations, batch 4, hidden 8, grid 3. This test is about which
    # cells RUN, not about convergence, so the budget only has to be enough to reach the loss
    # branch. The whole matrix is 32 cells.
    n_iter = 5

    for objective in objectives,
        include_backward in (false, true),
        include_flow_estimator in (false, true)

        Random.seed!(20260828)
        label = "$objective ib=$include_backward fe=$include_flow_estimator"

        model = create_grid_world_gflownet(
            grid_size = 3,
            reward_positions = Dict((3, 3) => 10.0),
            hidden_dim = 8,
            include_backward = include_backward,
            include_flow_estimator = include_flow_estimator,
            partition_function_method = LEARNABLE_ESTIMATION,
        )

        outcome = try
            history = train_gflownet(
                model,
                TrainingConfig(objective = objective,
                               partition_function_method = LEARNABLE_ESTIMATION,
                               n_iterations = n_iter, batch_size = 4);
                verbose = false)
            (:ran, count(isfinite, history.losses), length(history.losses))
        catch e
            # A throw is an acceptable outcome -- it tells the caller something. Only
            # ArgumentError is acceptable, though: any other exception type escaping this far
            # means an unhandled defect rather than a refusal.
            e isa ArgumentError || rethrow()
            (:refused, 0, 0)
        end

        if outcome[1] === :ran
            # Every iteration finite. NOT `any(!isnan)`, NOT `!all(isnan)`, and NOT a length
            # check -- twenty NaN satisfy the length check, which is how this went unnoticed.
            @test outcome[2] == outcome[3]
            @test outcome[2] == n_iter
        else
            @test outcome[1] === :refused
        end
    end

    # The three cells the audit found, pinned individually so a regression names itself rather
    # than appearing as one of thirty-two anonymous failures.
    plain = create_grid_world_gflownet(grid_size = 3, hidden_dim = 8,
                                      partition_function_method = LEARNABLE_ESTIMATION)
    for objective in (DIRECT_FLOW_OBJECTIVE, COMBINED_OBJECTIVES, MULTI_OBJECTIVE_TB)
        @test_throws ArgumentError train_gflownet(
            plain,
            TrainingConfig(objective = objective,
                           partition_function_method = LEARNABLE_ESTIMATION,
                           n_iterations = n_iter, batch_size = 4);
            verbose = false)
    end

    # And the two partition-function methods that do nothing are refused at CONSTRUCTION, so
    # they cannot be smuggled past a loss function that never sees validate_training_config.
    # The refusal used to live only in validate_training_config, reachable from one of six
    # entry points; the audit measured the other five accepting them and pinning Z = 1.
    for method in (SAMPLING_ESTIMATION, ADAPTIVE_ESTIMATION)
        @test_throws ArgumentError TrainingConfig(objective = TRAJECTORY_BALANCE,
                                                 partition_function_method = method,
                                                 n_iterations = 5, batch_size = 4)
    end
end
