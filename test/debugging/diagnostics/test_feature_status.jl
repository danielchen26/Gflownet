# Feature-status regression test.
#
# HISTORY, and the reason this file is now the inverse of what it was: it used to
# be titled "Working vs Broken Features Documentation" and asserted that
# DETAILED_BALANCE training, FLOW_MATCHING training, flow(),
# compute_recursive_flow(), partition_function(), validate_flow_conservation()
# and the DAG helpers get_next_states()/get_previous_states()/get_root_state()
# were all BROKEN. Two of those claims were `@test_broken`, the rest were
# `@test_throws UndefVarError`, and it closed with a 70-line println telling
# readers to "stick to TRAJECTORY_BALANCE".
#
# Every one of those claims is false as of 2026-08-28 on Julia 1.11.6. Observed:
# DETAILED_BALANCE and FLOW_MATCHING both train to completion with finite
# losses; flow/partition_function/compute_recursive_flow return 19.0 for the 3x3
# grid (the enumerated Z that test/theory/test_reward_proportionality.jl pins);
# validate_flow_conservation returns true; all three DAG helpers are defined in
# src/core/graphs.jl and work when handed the model itself. `@test_broken
# train_gflownet(...)` even reported "Expression evaluated to non-Boolean"
# because train_gflownet returns a TrainingHistory -- the marker was structurally
# incapable of ever passing OR failing usefully.
#
# So the file now pins the capabilities the repair delivered. A regression to the
# old broken state fails here instead of being documented as normal.

using Test
using GFlowNet

@testset "Feature Status" begin

    @testset "objectives that used to be documented as broken" begin
        # A model with backward policy and flow estimator: the configuration
        # DETAILED_BALANCE and FLOW_MATCHING actually need.
        for objective in (TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING)
            model = create_grid_world_gflownet(
                grid_size=3,
                reward_positions=Dict((3, 3) => 10.0),
                hidden_dim=16,
                include_backward=true,
                include_flow_estimator=true
            )
            config = TrainingConfig(
                objective=objective,
                n_iterations=5,
                batch_size=4,
                learning_rate=0.01
            )

            history = train_gflownet(model, config; verbose=false)

            @test history isa GFlowNet.TrainingHistory
            @test length(history.losses) == 5
            @test all(isfinite, history.losses)
            @test all(>=(0.0), history.losses)
            # A silently-dead objective produces a constant loss and zero
            # gradients; both would slip past an isfinite check alone.
            @test all(isfinite, history.gradient_norms)
            @test any(>(0.0), history.gradient_norms)
        end
    end

    @testset "core interface functions" begin
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=16
        )
        state = GridState(2, 2, false)

        @test length(state_to_features(state)) == 3
        @test !is_terminal_state(state)
        @test is_applicable(MoveRight(), state)

        new_state = apply_action(MoveRight(), state)
        @test new_state.x == 3

        # On-demand action enumeration
        actions = get_applicable_actions(state, model.all_actions)
        @test length(actions) > 0
        @test all(a -> is_applicable(a, state), actions)
    end

    @testset "flow computation" begin
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=16,
            include_backward=true,
            include_flow_estimator=true
        )
        s0 = model.initial_state

        # These enumerate the DAG, so they do not depend on the (untrained)
        # weights: F(s_0) == Z, and Z == 19 exactly for this grid.
        @test flow(model, s0) ≈ 19.0 atol=1e-6
        @test partition_function(model) ≈ flow(model, s0)
        @test compute_recursive_flow(model, s0) ≈ flow(model, s0)
        @test validate_flow_conservation(model, s0)

        # Downstream flow must be strictly smaller than the total.
        @test 0.0 < flow(model, GridState(2, 2, false)) < flow(model, s0)
    end

    @testset "DAG helpers" begin
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=16
        )
        state = GridState(2, 2, false)

        # The `dag` argument is duck-typed: anything with `all_actions` and
        # `initial_state`, i.e. the model (src/core/graphs.jl:217-224).
        nexts = GFlowNet.get_next_states(model, state)
        @test GridState(3, 2, false) in nexts
        @test GridState(2, 3, false) in nexts
        @test GridState(2, 2, true) in nexts

        prevs = GFlowNet.get_previous_states(model, state)
        @test GridState(1, 2, false) in prevs
        @test GridState(2, 1, false) in prevs
        @test !(state in prevs)  # no self-loops

        @test GFlowNet.get_root_state(model) == model.initial_state

        # Contract: a non-model DAG argument is rejected, not silently accepted.
        @test_throws ArgumentError GFlowNet.get_previous_states(model.all_actions, state)
    end

    @testset "model fields" begin
        model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)

        # backward_policy and flow_estimator ARE fields; they hold `nothing`
        # unless requested. The old test asserted the fields did not exist.
        @test hasproperty(model, :backward_policy)
        @test hasproperty(model, :flow_estimator)
        @test isnothing(model.backward_policy)
        @test isnothing(model.flow_estimator)

        equipped = create_grid_world_gflownet(
            grid_size=3,
            hidden_dim=16,
            include_backward=true,
            include_flow_estimator=true
        )
        @test !isnothing(equipped.backward_policy)
        @test !isnothing(equipped.flow_estimator)
        @test keys(equipped.parameters) == (:forward, :backward, :flow)

        # Still true: there is no explicit DAG object and no stored state_dim.
        @test !hasproperty(model, :dag)
        @test !hasproperty(model, :state_dim)
    end
end
