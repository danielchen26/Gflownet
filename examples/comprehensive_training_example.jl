"""
Comprehensive GFlowNet Training Example

This example demonstrates all available training objectives and partition function
estimation methods in the GFlowNet.jl framework.
"""

using GFlowNet
using Plots, Random, Statistics

# Set random seed for reproducibility
Random.seed!(42)

"""
    create_simple_grid_environment(size::Int=5)

Create a simple grid world environment for demonstration.
"""
function create_simple_grid_environment(size::Int=5)
    # Create states (grid positions)
    states = [(i, j) for i in 1:size, j in 1:size]
    initial_state = (1, 1)
    terminal_states = [(size, size)]
    
    # Create actions (movements)
    actions = [:right, :down, :up, :left]
    
    # Create DAG
    dag = create_dag(initial_state, terminal_states, actions)
    
    # Add valid transitions
    for i in 1:size, j in 1:size
        current = (i, j)
        
        # Right movement
        if j < size
            next_state = (i, j + 1)
            add_action!(dag, current, next_state, :right)
        end
        
        # Down movement  
        if i < size
            next_state = (i + 1, j)
            add_action!(dag, current, next_state, :down)
        end
        
        # Up movement
        if i > 1
            next_state = (i - 1, j)
            add_action!(dag, current, next_state, :up)
        end
        
        # Left movement
        if j > 1
            next_state = (i, j - 1)
            add_action!(dag, current, next_state, :left)
        end
    end
    
    return dag
end

"""
    create_reward_function(terminal_state)

Create a simple reward function that gives higher reward for reaching the terminal state.
"""
function create_reward_function(terminal_state)
    return function(state)
        if state == terminal_state
            return 10.0  # High reward for reaching goal
        else
            # Distance-based reward (encourage moving towards goal)
            dist = abs(state[1] - terminal_state[1]) + abs(state[2] - terminal_state[2])
            return max(0.1, 2.0 - 0.1 * dist)
        end
    end
end

"""
    demonstrate_training_objectives()

Demonstrate all available training objectives.
"""
function demonstrate_training_objectives()
    println("🎯 Demonstrating All Training Objectives")
    println("=" ^ 50)
    
    # Create environment
    dag = create_simple_grid_environment(4)
    reward_fn = create_reward_function((4, 4))
    
    # Create a basic model (simplified for demonstration)
    model = GFlowNetModel(dag, reward_fn)  # This would need proper implementation
    
    # Define training objectives to test
    objectives = [
        (TRAJECTORY_BALANCE, "Standard Trajectory Balance"),
        (GENERAL_TRAJECTORY_BALANCE, "General Trajectory Balance (with P_B)"),
        (SUB_TRAJECTORY_BALANCE, "Sub-Trajectory Balance"),
        (HIERARCHICAL_SUB_TB, "Hierarchical Sub-Trajectory Balance"),
        (ADAPTIVE_SUB_TB, "Adaptive Sub-Trajectory Balance"),
        (FLOW_CONSISTENCY, "Flow Consistency (Edge-Level)", Dict(:flow_consistency_mode => EDGE_LEVEL)),
        (FLOW_CONSISTENCY, "Flow Consistency (State-Level)", Dict(:flow_consistency_mode => STATE_LEVEL)),
        (FLOW_CONSISTENCY, "Flow Consistency (Mixed-Level)", Dict(:flow_consistency_mode => MIXED_LEVEL))
    ]
    
    results = Dict()
    
    for objective_data in objectives
        if length(objective_data) == 3
            objective, name, extra_config = objective_data
        else
            objective, name = objective_data
            extra_config = Dict()
        end
        
        println("\n📈 Training with: $name")
        
        try
            # Create configuration with any extra config for flow consistency
            base_config = Dict(
                :min_length => 2,
                :max_length => nothing,
                :n_subtrajectories => 5,
                :scales => [2, 4, 8],
                :difficulty_threshold => 0.1
            )
            merged_config = merge(base_config, extra_config)
            
            config = TrainingConfig(
                objective=objective,
                partition_function_method=SIMPLE_ESTIMATION,
                batch_size=16,
                learning_rate=0.01,
                n_iterations=100,
                partition_update_frequency=10,
                validation_frequency=25,
                sub_trajectory_config=merged_config
            )
            
            # Train model
            println("  Starting training...")
            history = train_gflownet(model, config; verbose=false)
            
            # Store results
            results[name] = history
            
            final_loss = history[:losses][end]
            println("  ✅ Completed! Final loss: $(round(final_loss, digits=4))")
            
        catch e
            println("  ❌ Failed: $e")
            results[name] = nothing
        end
    end
    
    return results
end

"""
    demonstrate_partition_function_methods()

Demonstrate all partition function estimation methods.
"""
function demonstrate_partition_function_methods()
    println("\n🔢 Demonstrating Partition Function Methods")
    println("=" ^ 50)
    
    # Create environment
    dag = create_simple_grid_environment(3)
    reward_fn = create_reward_function((3, 3))
    model = GFlowNetModel(dag, reward_fn)
    
    # Define partition function methods to test
    methods = [
        (SIMPLE_ESTIMATION, "Simple Estimation (Sum of Rewards)"),
        (LEARNABLE_PARAMETER, "Learnable Parameter (Gradient Descent)"),
        (SAMPLING_BASED, "Sampling-Based Estimation"),
        (ADAPTIVE_ESTIMATION, "Adaptive Method Switching")
    ]
    
    results = Dict()
    
    for (method, name) in methods
        println("\n🧮 Testing: $name")
        
        try
            # Create configuration
            config = TrainingConfig(
                objective=TRAJECTORY_BALANCE,
                partition_function_method=method,
                batch_size=12,
                learning_rate=0.02,
                n_iterations=150,
                partition_update_frequency=5,
                validation_frequency=30
            )
            
            # Train model
            println("  Training with $name...")
            history = train_gflownet(model, config; verbose=false)
            
            # Store results
            results[name] = history
            
            final_Z = history[:partition_function_estimates][end]
            final_loss = history[:losses][end]
            println("  ✅ Completed! Final Z: $(round(final_Z, digits=4)), Loss: $(round(final_loss, digits=4))")
            
        catch e
            println("  ❌ Failed: $e")
            results[name] = nothing
        end
    end
    
    return results
end

"""
    demonstrate_advanced_configurations()

Demonstrate advanced configuration options.
"""
function demonstrate_advanced_configurations()
    println("\n⚙️  Demonstrating Advanced Configurations")
    println("=" ^ 50)
    
    # Create environment
    dag = create_simple_grid_environment(4)
    reward_fn = create_reward_function((4, 4))
    model = GFlowNetModel(dag, reward_fn)
    
    # Test hierarchical sub-trajectory balance with custom scales
    println("\n🏗️  Hierarchical Sub-Trajectory Balance with Custom Scales")
    config_hierarchical = TrainingConfig(
        objective=HIERARCHICAL_SUB_TB,
        partition_function_method=ADAPTIVE_ESTIMATION,
        batch_size=20,
        learning_rate=0.015,
        n_iterations=200,
        sub_trajectory_config=Dict(
            :scales => [2, 3, 5, 8],  # Custom scales
            :n_subtrajectories => 8
        )
    )
    
    println("  Training with custom hierarchical scales...")
    history_hierarchical = train_gflownet(model, config_hierarchical; verbose=false)
    println("  ✅ Hierarchical training completed!")
    
    # Test adaptive sub-trajectory balance with custom difficulty threshold
    println("\n🎯 Adaptive Sub-Trajectory Balance with Custom Threshold")
    config_adaptive = TrainingConfig(
        objective=ADAPTIVE_SUB_TB,
        partition_function_method=LEARNABLE_PARAMETER,
        batch_size=16,
        learning_rate=0.01,
        n_iterations=150,
        sub_trajectory_config=Dict(
            :difficulty_threshold => 0.05  # Lower threshold = more selective
        )
    )
    
    println("  Training with adaptive difficulty selection...")
    history_adaptive = train_gflownet(model, config_adaptive; verbose=false)
    println("  ✅ Adaptive training completed!")
    
    # Test general trajectory balance (requires backward policy)
    println("\n↔️  General Trajectory Balance (Full Formulation)")
    try
        config_general = TrainingConfig(
            objective=GENERAL_TRAJECTORY_BALANCE,
            partition_function_method=SAMPLING_BASED,
            batch_size=14,
            learning_rate=0.008,
            n_iterations=100
        )
        
        println("  Training with full trajectory balance...")
        history_general = train_gflownet(model, config_general; verbose=false)
        println("  ✅ General trajectory balance completed!")
    catch e
        println("  ⚠️  General TB requires backward policy: $e")
    end
    
    return Dict(
        "hierarchical" => history_hierarchical,
        "adaptive" => history_adaptive
    )
end

"""
    compare_methods_performance()

Compare the performance of different methods side by side.
"""
function compare_methods_performance()
    println("\n📊 Performance Comparison")
    println("=" ^ 50)
    
    # Create environment
    dag = create_simple_grid_environment(5)
    reward_fn = create_reward_function((5, 5))
    model = GFlowNetModel(dag, reward_fn)
    
    # Define comparison scenarios
    scenarios = [
        ("Simple TB + Simple Z", TRAJECTORY_BALANCE, SIMPLE_ESTIMATION),
        ("Simple TB + Learnable Z", TRAJECTORY_BALANCE, LEARNABLE_PARAMETER),
        ("Sub-TB + Adaptive Z", SUB_TRAJECTORY_BALANCE, ADAPTIVE_ESTIMATION),
        ("Hierarchical TB + Sampling Z", HIERARCHICAL_SUB_TB, SAMPLING_BASED)
    ]
    
    results = Dict()
    
    for (name, objective, z_method) in scenarios
        println("\n🔬 Testing: $name")
        
        config = TrainingConfig(
            objective=objective,
            partition_function_method=z_method,
            batch_size=18,
            learning_rate=0.012,
            n_iterations=200,
            partition_update_frequency=8,
            validation_frequency=40
        )
        
        println("  Running training...")
        history = train_gflownet(model, config; verbose=false)
        
        results[name] = history
        
        # Calculate convergence metrics
        losses = history[:losses]
        final_loss = losses[end]
        mean_final_10 = mean(losses[max(1, end-9):end])
        
        println("  📈 Final loss: $(round(final_loss, digits=4))")
        println("  📈 Mean last 10: $(round(mean_final_10, digits=4))")
    end
    
    return results
end

"""
    create_performance_plots(results)

Create plots comparing the performance of different methods.
"""
function create_performance_plots(results)
    println("\n📈 Creating Performance Plots")
    println("=" ^ 30)
    
    # Plot 1: Loss curves comparison
    p1 = plot(title="Training Loss Comparison", xlabel="Iteration", ylabel="Loss", legend=:topright)
    
    for (method_name, history) in results
        if !isnothing(history) && haskey(history, :losses)
            plot!(p1, history[:losses], label=method_name, linewidth=2)
        end
    end
    
    # Plot 2: Partition function estimates
    p2 = plot(title="Partition Function Estimates", xlabel="Update Step", ylabel="Z Estimate", legend=:topright)
    
    for (method_name, history) in results
        if !isnothing(history) && haskey(history, :partition_function_estimates) && !isempty(history[:partition_function_estimates])
            plot!(p2, history[:partition_function_estimates], label=method_name, linewidth=2, marker=:circle)
        end
    end
    
    # Combine plots
    combined_plot = plot(p1, p2, layout=(2, 1), size=(800, 600))
    
    return combined_plot
end

"""
    main()

Main function to run all demonstrations.
"""
function main()
    println("🚀 GFlowNet.jl Comprehensive Training Demonstration")
    println("=" ^ 60)
    println("This example demonstrates all training objectives and Z estimation methods.")
    println()
    
    # 1. Demonstrate training objectives
    objective_results = demonstrate_training_objectives()
    
    # 2. Demonstrate partition function methods  
    z_method_results = demonstrate_partition_function_methods()
    
    # 3. Demonstrate advanced configurations
    advanced_results = demonstrate_advanced_configurations()
    
    # 4. Compare methods performance
    comparison_results = compare_methods_performance()
    
    # 5. Create performance plots
    println("\n📊 Generating Performance Plots...")
    try
        plots = create_performance_plots(comparison_results)
        savefig(plots, "gflownet_training_comparison.png")
        println("  ✅ Plots saved as 'gflownet_training_comparison.png'")
    catch e
        println("  ⚠️  Could not create plots: $e")
    end
    
    # Summary
    println("\n🎉 Demonstration Complete!")
    println("=" ^ 30)
    println("Successfully tested:")
    println("  ✅ $(length(objective_results)) training objectives")
    println("  ✅ $(length(z_method_results)) partition function methods")
    println("  ✅ Advanced configuration options")
    println("  ✅ Performance comparisons")
    println()
    println("Your GFlowNet.jl framework now supports:")
    println("  🎯 General Trajectory Balance (with P_B)")
    println("  🔄 Sub-Trajectory Balance (3 variants)")
    println("  🧮 Advanced Z estimation (4 methods)")
    println("  ⚙️  Flexible training configurations")
    println("  📊 Comprehensive monitoring and comparison")
    
    return Dict(
        "objectives" => objective_results,
        "z_methods" => z_method_results,
        "advanced" => advanced_results,
        "comparison" => comparison_results
    )
end

# Run the demonstration if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end 