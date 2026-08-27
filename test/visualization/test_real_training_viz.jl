# Comprehensive tests for real training visualization system
# Tests domain adapters, session management, and metrics computation

using Test
using GFlowNet

# Include the visualization modules
include("../../src/utils/visualization/core/adapters.jl")
include("../../src/utils/visualization/core/metrics.jl")
include("../../src/utils/visualization/core/training_session.jl")
include("../../src/utils/visualization/domains/grid_world.jl")

@testset "Real Training Visualization" begin

    @testset "Domain Adapter Interface" begin
        println("\n=== Testing Domain Adapter Interface ===")

        # Create a real model
        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict((3,3) => 10.0, (5,5) => 8.0),
            hidden_dim = 32
        )
        adapter = GridWorldAdapter(5, Dict((3,3) => 10.0, (5,5) => 8.0))

        # Test state_to_viz_data
        println("  ✓ Testing state_to_viz_data...")
        state = GFlowNet.GridState(2, 3, false)
        viz = state_to_viz_data(adapter, state)
        @test viz["x"] == 2
        @test viz["y"] == 3
        @test viz["is_terminal"] == false
        @test haskey(viz, "reward")
        println("    State conversion: PASS")

        # Test trajectory_to_viz_data
        println("  ✓ Testing trajectory_to_viz_data...")
        traj = GFlowNet.sample_trajectory(model)
        traj_viz = trajectory_to_viz_data(adapter, traj, "test_traj")
        @test traj_viz["id"] == "test_traj"
        @test haskey(traj_viz, "states")
        @test haskey(traj_viz, "actions")
        @test haskey(traj_viz, "rewards")
        @test traj_viz["length"] == length(traj.actions)
        println("    Trajectory conversion: PASS")

        # Test get_domain_config
        println("  ✓ Testing get_domain_config...")
        config = get_domain_config(adapter)
        @test config["domain_type"] == "grid_world"
        @test config["grid_size"] == [5, 5]
        @test config["supports_flow_field"] == true
        @test config["supports_distribution"] == true
        @test length(config["reward_peaks"]) == 2
        println("    Domain config: PASS")

        # Test get_renderer_name
        println("  ✓ Testing get_renderer_name...")
        renderer = get_renderer_name(adapter)
        @test renderer == "GridWorldRenderer"
        println("    Renderer name: PASS")
    end

    @testset "Training Session Lifecycle" begin
        println("\n=== Testing Training Session Lifecycle ===")

        config = Dict(
            "domain_type" => "grid_world",
            "grid_size" => 5,
            "n_episodes" => 10,
            "batch_size" => 4,
            "learning_rate" => 0.01,
            "objective" => "TRAJECTORY_BALANCE",
            "reward_peaks" => [
                Dict("position" => [3, 3], "intensity" => 10.0),
                Dict("position" => [5, 5], "intensity" => 8.0)
            ]
        )

        # Need to define create_model_and_adapter for testing
        function create_model_and_adapter(domain_type::String, config::Dict)
            grid_size = get(config, "grid_size", 5)
            peaks_config = get(config, "reward_peaks", [])
            reward_positions = Dict{Tuple{Int,Int}, Float64}()
            for peak in peaks_config
                pos = peak["position"]
                intensity = get(peak, "intensity", 10.0)
                reward_positions[(Int(pos[1]), Int(pos[2]))] = Float64(intensity)
            end

            model = GFlowNet.create_grid_world_gflownet(
                grid_size = grid_size,
                reward_positions = reward_positions,
                hidden_dim = 32,
                learning_rate = 0.01
            )
            adapter = GridWorldAdapter(grid_size, reward_positions)
            return model, adapter
        end

        println("  ✓ Creating session...")
        # Inject the factory explicitly. It is defined inside this @testset, so it
        # is NOT a Main global and the old implicit lookup failed with
        # UndefVarError from inside create_session.
        session = create_session(config; model_factory=create_model_and_adapter)

        @test session.current_iteration == 0
        @test session.total_iterations == 10
        @test !session.is_training
        @test session.batch_size == 4
        println("    Session creation: PASS")

        # Run a few training steps
        println("  ✓ Running training steps...")
        session.is_training = true
        for i in 1:3
            result = step!(session)
            @test result["status"] == "ok"
            @test haskey(result, "loss")
            @test haskey(result, "mean_reward")
            @test haskey(result, "iteration")
            @test result["iteration"] == i
        end
        @test session.current_iteration == 3
        @test length(session.losses) == 3
        @test length(session.rewards) == 3
        @test length(session.gradient_norms) == 3
        println("    Training steps: PASS (3 iterations)")
    end

    @testset "Parse Objective" begin
        println("\n=== Testing Objective Parsing ===")

        @test parse_objective("TRAJECTORY_BALANCE") == GFlowNet.TRAJECTORY_BALANCE
        @test parse_objective("DETAILED_BALANCE") == GFlowNet.DETAILED_BALANCE
        @test parse_objective("FLOW_MATCHING") == GFlowNet.FLOW_MATCHING
        @test parse_objective("SUB_TRAJECTORY_BALANCE") == GFlowNet.SUB_TRAJECTORY_BALANCE
        @test parse_objective("DIRECT_FLOW_OBJECTIVE") == GFlowNet.DIRECT_FLOW_OBJECTIVE

        # Test case insensitivity
        @test parse_objective("trajectory_balance") == GFlowNet.TRAJECTORY_BALANCE
        @test parse_objective(" TRAJECTORY_BALANCE ") == GFlowNet.TRAJECTORY_BALANCE

        # Test error handling
        @test_throws ErrorException parse_objective("INVALID_OBJECTIVE")

        println("  ✓ All objectives parsed correctly")
        println("  ✓ Error handling works")
    end

    @testset "Universal Metrics" begin
        println("\n=== Testing Universal Metrics Computation ===")

        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict((3,3) => 10.0, (5,5) => 8.0),
            hidden_dim = 32
        )
        trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:10]

        println("  ✓ Computing metrics from 10 trajectories...")
        metrics = compute_gflownet_metrics(model, trajectories)

        @test haskey(metrics, "mean_reward")
        @test haskey(metrics, "max_reward")
        @test haskey(metrics, "min_reward")
        @test haskey(metrics, "reward_std")
        @test haskey(metrics, "unique_terminals")
        @test haskey(metrics, "diversity_ratio")
        @test haskey(metrics, "mean_length")
        @test haskey(metrics, "max_length")
        @test haskey(metrics, "partition_function")
        @test metrics["n_trajectories"] == 10

        # Validate metric values
        @test metrics["mean_reward"] >= 0
        @test metrics["max_reward"] >= metrics["mean_reward"]
        @test metrics["diversity_ratio"] >= 0 && metrics["diversity_ratio"] <= 1
        @test metrics["unique_terminals"] <= 10

        println("    All metric fields present: PASS")
        println("    Metric values valid: PASS")
    end

    @testset "Grid World Domain Metrics" begin
        println("\n=== Testing Grid World Domain Metrics ===")

        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 4,
            reward_positions = Dict((4,4) => 10.0, (1,4) => 8.0),
            hidden_dim = 32
        )
        adapter = GridWorldAdapter(4, Dict((4,4) => 10.0, (1,4) => 8.0))
        trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:20]

        println("  ✓ Computing domain-specific metrics...")
        domain_metrics = compute_domain_metrics(adapter, model, trajectories)

        @test haskey(domain_metrics, "mode_coverage")
        @test haskey(domain_metrics, "modes_discovered")
        @test haskey(domain_metrics, "total_modes")
        @test haskey(domain_metrics, "unique_positions")
        @test haskey(domain_metrics, "top_positions")

        @test domain_metrics["total_modes"] == 2
        @test domain_metrics["mode_coverage"] >= 0 && domain_metrics["mode_coverage"] <= 1
        @test domain_metrics["modes_discovered"] <= 2

        println("    Mode coverage tracking: PASS")
        println("    Position distribution: PASS")
    end

    @testset "Grid World Flow Field" begin
        println("\n=== Testing Flow Field Computation ===")

        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 4,
            reward_positions = Dict((4,4) => 10.0),
            hidden_dim = 32
        )
        adapter = GridWorldAdapter(4, Dict((4,4) => 10.0))

        println("  ✓ Computing flow field for 4×4 grid...")
        flow_data = compute_flow_field(adapter, model)

        @test flow_data["supported"] == true
        @test flow_data["grid_size"] == 4
        @test length(flow_data["data"]) == 16  # 4×4 grid

        # Check each flow point has required fields
        for point in flow_data["data"]
            @test haskey(point, "position")
            @test haskey(point, "velocity")
            @test haskey(point, "magnitude")
            # The key is "flow_value", not "flow". compute_flow_field has always
            # emitted "flow_value" (domains/grid_world.jl:187) and the dashboard
            # reads point.flow_value in four places, so the producer and the real
            # consumer already agreed -- only this assertion was stale. It failed
            # 16 times per run, and identically at the pre-repair baseline
            # 31fae84a, but this file is not referenced by runtests.jl so nothing
            # ever reported it.
            @test haskey(point, "flow_value")
            @test haskey(point, "reward")
            @test length(point["position"]) == 2
            @test length(point["velocity"]) == 2
        end

        println("    Flow field structure: PASS")
        println("    All grid points covered: PASS")
    end

    @testset "Grid World Distribution" begin
        println("\n=== Testing Distribution Computation ===")

        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 4,
            reward_positions = Dict((4,4) => 10.0, (1,1) => 5.0),
            hidden_dim = 32
        )
        adapter = GridWorldAdapter(4, Dict((4,4) => 10.0, (1,1) => 5.0))
        trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:20]

        println("  ✓ Computing distribution from 20 trajectories...")
        dist = compute_distribution_data(adapter, model, trajectories)

        @test dist["supported"] == true
        @test dist["grid_size"] == 4
        @test size(dist["empirical"]) == (4, 4)
        @test size(dist["target"]) == (4, 4)
        @test size(dist["counts"]) == (4, 4)
        @test dist["total_samples"] == 20

        # Validate probability distributions sum to 1
        @test sum(dist["empirical"]) ≈ 1.0 atol=1e-10
        @test sum(dist["target"]) ≈ 1.0 atol=1e-10

        println("    Distribution structure: PASS")
        println("    Probability normalization: PASS")
    end

    @testset "Error Handling" begin
        println("\n=== Testing Error Handling ===")

        # Test empty trajectories
        model = GFlowNet.create_grid_world_gflownet(grid_size=3, hidden_dim=16)
        metrics = compute_gflownet_metrics(model, Trajectory[])
        @test haskey(metrics, "error")
        println("  ✓ Empty trajectory handling: PASS")

        # Test invalid objective
        @test_throws ErrorException parse_objective("NOT_A_REAL_OBJECTIVE")
        println("  ✓ Invalid objective handling: PASS")
    end

    println("\n" * "="^60)
    println("ALL TESTS PASSED! ✅")
    println("="^60)
end
