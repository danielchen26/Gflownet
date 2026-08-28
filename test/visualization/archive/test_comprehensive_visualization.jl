# DEAD TEST — targets a visualization API that no longer exists. Do not trust it,
# do not wire it into runtests.jl, and do not "fix" it without deciding first
# whether the subsystem it tests is coming back.
#
# Measured 2026-08-28:
#   julia --project=. -e 'using Test, GFlowNet; include(...)'
#   -> Test Summary: GFlowNet Visualization System Tests | Error 1, Total 1, 1.4s
#      "Some tests did not pass: 0 passed, 0 failed, 1 errored, 0 broken."
#
# It errors on the very first statement, line 19 below:
# `create_grid_world_gflownet(; target_states, target_rewards, include_backward)`
# destructured into `(gflownet, grid_size)`. The current signature
# (src/applications/grid_world.jl:238-243) takes `reward_positions::Dict` and has
# no `target_states`, `target_rewards`, or `include_backward` kwarg, and returns
# one model rather than a tuple.
#
# Every symbol the rest of the file exercises is also gone: a repo-wide grep for
# GFlowNetVisualizer, set_gflownet_theme!, TrainingMonitor, get_color_palette and
# export_figure finds no definition anywhere under src/. This was the Makie-based
# visualization layer, which the Oxygen/JS dashboard under
# src/utils/visualization/ replaced.
#
# Original header: Comprehensive Test for GFlowNet Visualization System
# Tests all visualization features to ensure they work correctly

using Test
using GFlowNet
using Random
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

@testset "GFlowNet Visualization System Tests" begin
    
    println("\n🧪 Starting Comprehensive Visualization Tests")
    println("=" ^ 50)
    
    # Create test model
    println("\n📦 Setting up test environment...")
    gflownet, grid_size = create_grid_world_gflownet(
        grid_size = 5,
        target_states = [(4, 4)],
        target_rewards = [10.0],
        include_backward = true
    )
    
    # Generate test data
    println("📊 Generating test trajectories...")
    test_trajectories = [sample_trajectory(gflownet) for _ in 1:20]
    test_rewards = [reward(traj.states[end]) for traj in test_trajectories]
    
    @testset "Theme System Tests" begin
        println("\n🎨 Testing Theme System...")
        
        @test_nowarn set_gflownet_theme!(dark_mode=false)
        @test_nowarn set_gflownet_theme!(dark_mode=true)
        
        # Test different theme styles
        for style in [:light, :dark, :publication, :presentation]
            @test_nowarn theme = get_theme(style)
            println("  ✓ Theme style '$style' loaded successfully")
        end
        
        # Test color palettes
        @test length(get_color_palette(5)) == 5
        @test length(get_color_palette(10, palette=:sequential)) == 10
        
        # Test custom theme creation
        @test_nowarn custom_theme = create_custom_theme(:light, fontsize=20)
        
        println("  ✅ All theme tests passed!")
    end
    
    @testset "Core Infrastructure Tests" begin
        println("\n🏗️ Testing Core Infrastructure...")
        
        # Test visualizer creation with different backends.
        # The catch below used to swallow every exception and only println a
        # warning, so if all three backends failed to construct, this testset ran
        # ZERO assertions and still reported green. `working_backends` makes that
        # impossible: an unavailable backend now WARNS with the concrete reason,
        # and the run still fails if none of them worked.
        working_backends = Symbol[]
        for backend in [:wgl, :gl, :cairo]
            try
                viz = GFlowNetVisualizer(
                    backend = backend,
                    theme = :light,
                    update_frequency = 10,
                    export_path = "test_viz/",
                    auto_export = false
                )
                println("  ✓ Created visualizer with $backend backend")

                # Test observable updates
                @test_nowarn update!(viz, Dict(
                    :loss => 1.0,
                    :reward => 5.0,
                    :gradient_norm => 0.5,
                    :iteration => 1
                ))

                @test length(viz.loss_history[]) == 1
                @test viz.loss_history[][1] == 1.0
                @test viz.current_iteration[] == 1
                push!(working_backends, backend)

            catch e
                @warn "Backend $backend unavailable — its assertions did NOT run" backend exception=(e, catch_backtrace())
            end
        end
        @test !isempty(working_backends)
        
        # Test export functionality
        viz = GFlowNetVisualizer(backend = :cairo)  # Cairo for headless testing
        update!(viz, Dict(:loss => 1.0, :iteration => 1))
        
        # Test data export
        mkpath("test_viz/")
        @test_nowarn export_data(viz, :json)
        
        println("  ✅ Core infrastructure tests passed!")
    end
    
    @testset "Training Monitor Tests" begin
        println("\n📈 Testing Training Monitor...")
        
        monitor = TrainingMonitor(
            backend = :cairo,  # Use Cairo for headless testing
            loss_window = 10,
            smooth_factor = 0.9,
            export_path = "test_viz/"
        )
        
        # Simulate training updates
        println("  📊 Simulating training progress...")
        for i in 1:50
            loss = 10.0 / (1.0 + 0.1 * i) + 0.1 * randn()
            reward = min(10.0, i * 0.2 + randn())
            grad_norm = abs(0.5 + 0.2 * randn())
            
            update!(monitor.visualizer, Dict(
                :loss => loss,
                :reward => reward,
                :gradient_norm => grad_norm,
                :iteration => i
            ))
        end
        
        # Check derived observables
        @test length(monitor.smoothed_losses[]) == 50
        @test monitor.convergence_rate[] >= 0  # Should show improvement
        
        # Test visualization creation
        @test_nowarn fig1 = plot_training_losses(monitor, show_confidence=true)
        println("  ✓ Training loss plot created")
        
        @test_nowarn fig2 = plot_gradient_health(monitor.visualizer.gradient_norms)
        println("  ✓ Gradient health plot created")
        
        # Test multi-objective comparison
        objective_data = Dict(
            :TB => monitor.visualizer.loss_history,
            :DB => Observable(monitor.visualizer.loss_history[] .* 0.9),
            :FM => Observable(monitor.visualizer.loss_history[] .* 1.1)
        )
        
        @test_nowarn fig3 = plot_multi_objective_comparison(monitor, objective_data)
        println("  ✓ Multi-objective comparison created")
        
        # Test dashboard creation
        @test_nowarn dashboard = create_training_dashboard(monitor)
        println("  ✓ Training dashboard created")
        
        println("  ✅ Training monitor tests passed!")
    end
    
    @testset "Trajectory Visualization Tests" begin
        println("\n🚀 Testing Trajectory Visualizations...")
        
        # Use Cairo backend for headless testing
        CairoMakie.activate!()
        
        # Test 2D trajectory plot
        @test_nowarn fig_2d = plot_trajectory_2d(
            test_trajectories[1],
            color_by = :gradient,
            show_actions = true,
            show_rewards = true
        )
        println("  ✓ 2D trajectory plot created")
        
        # Test 3D trajectory plot (without rotation for testing)
        @test_nowarn fig_3d = plot_trajectory_3d(
            test_trajectories[1],
            color_by = :time,
            show_projection = true,
            rotation_speed = 0.0  # No rotation for testing
        )
        println("  ✓ 3D trajectory plot created")
        
        # Test trajectory bundle
        @test_nowarn bundle_fig = plot_trajectory_bundle(
            test_trajectories,
            max_trajectories = 10,
            alpha = 0.5,
            highlight_best = true,
            style = :lines
        )
        println("  ✓ Trajectory bundle plot created")
        
        # Test different bundle styles
        for style in [:lines, :streamline, :heatmap]
            @test_nowarn plot_trajectory_bundle(
                test_trajectories[1:5],
                style = style
            )
            println("  ✓ Bundle style '$style' works")
        end
        
        # Test trajectory animation (without saving)
        @test_nowarn anim_fig = animate_trajectory(
            test_trajectories[1],
            fps = 10,
            trail_length = 3,
            save_path = ""  # Don't save
        )
        println("  ✓ Trajectory animation created")
        
        # Test trajectory comparison
        trajectories_set1 = test_trajectories[1:10]
        trajectories_set2 = test_trajectories[11:20]
        
        @test_nowarn comp_fig = plot_trajectory_comparison(
            trajectories_set1,
            trajectories_set2,
            labels = ["First Half", "Second Half"],
            metric = :reward
        )
        println("  ✓ Trajectory comparison created")
        
        println("  ✅ Trajectory visualization tests passed!")
    end
    
    @testset "Export and Save Tests" begin
        println("\n💾 Testing Export Functionality...")
        
        # Create a simple figure
        fig = Figure()
        ax = Axis(fig[1, 1], title = "Test Plot")
        lines!(ax, 1:10, randn(10))
        
        # Test different export formats
        test_dir = "test_viz/exports/"
        mkpath(test_dir)
        
        # PNG export
        @test_nowarn export_figure(fig, joinpath(test_dir, "test.png"), dpi=150)
        @test isfile(joinpath(test_dir, "test.png"))
        println("  ✓ PNG export successful")
        
        # PDF export
        @test_nowarn export_figure(fig, joinpath(test_dir, "test.pdf"))
        @test isfile(joinpath(test_dir, "test.pdf"))
        println("  ✓ PDF export successful")
        
        # SVG export
        @test_nowarn export_figure(fig, joinpath(test_dir, "test.svg"))
        @test isfile(joinpath(test_dir, "test.svg"))
        println("  ✓ SVG export successful")
        
        println("  ✅ Export tests passed!")
    end
    
    @testset "Integration Tests" begin
        println("\n🔗 Testing Integration Features...")
        
        # Test callback creation
        monitor = TrainingMonitor(backend = :cairo)
        callback = create_loss_logger(monitor)
        
        # Simulate training loop integration
        for i in 1:10
            loss = 1.0 / (1.0 + 0.1 * i)
            metrics = Dict(
                :gradient_norm => 0.5,
                :reward => 5.0 * i / 10
            )
            @test callback(i, loss, metrics) == true
        end
        
        @test length(monitor.visualizer.loss_history[]) == 10
        println("  ✓ Training callback integration works")
        
        # Test observable model creation
        obs_model = create_observable_model(gflownet)
        @test haskey(obs_model, :parameters)
        @test haskey(obs_model, :training_state)
        println("  ✓ Observable model creation works")
        
        println("  ✅ Integration tests passed!")
    end
    
    @testset "Error Handling Tests" begin
        println("\n⚠️  Testing Error Handling...")
        
        # Test invalid theme
        @test_throws ErrorException get_theme(:invalid_theme)
        
        # Test invalid export format
        fig = Figure()
        @test_throws ErrorException export_figure(fig, "test.invalid")
        
        # Test empty trajectory handling
        empty_traj = Trajectory([], [], [])
        @test_nowarn plot_trajectory_2d(empty_traj)
        
        println("  ✅ Error handling tests passed!")
    end
    
    # Clean up test files
    rm("test_viz", recursive=true, force=true)
    
    println("\n" * "=" * 50)
    println("✅ All Visualization Tests Passed Successfully!")
    println("=" * 50)
end

# Run performance benchmarks
@testset "Performance Benchmarks" begin
    println("\n⏱️  Running Performance Benchmarks...")
    
    using BenchmarkTools
    
    # Create test data
    gflownet, _ = create_grid_world_gflownet(grid_size = 5)
    trajectories = [sample_trajectory(gflownet) for _ in 1:100]
    
    # Benchmark visualization creation
    println("\n📊 Benchmarking visualization creation times:")
    
    # Theme switching
    t1 = @elapsed set_gflownet_theme!(dark_mode=false)
    println("  Theme switching: $(round(t1*1000, digits=2))ms")
    
    # Monitor creation
    t2 = @elapsed TrainingMonitor(backend = :cairo)
    println("  Monitor creation: $(round(t2*1000, digits=2))ms")
    
    # Plot creation
    CairoMakie.activate!()
    t3 = @elapsed plot_trajectory_2d(trajectories[1])
    println("  2D trajectory plot: $(round(t3*1000, digits=2))ms")
    
    t4 = @elapsed plot_trajectory_bundle(trajectories[1:20])
    println("  Bundle plot (20 trajectories): $(round(t4*1000, digits=2))ms")
    
    # Update performance
    monitor = TrainingMonitor(backend = :cairo)
    t5 = @elapsed for i in 1:100
        update!(monitor.visualizer, Dict(:loss => randn(), :iteration => i))
    end
    println("  100 monitor updates: $(round(t5*1000, digits=2))ms")
    
    println("\n✅ Performance benchmarks completed!")
end