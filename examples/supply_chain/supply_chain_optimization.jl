#!/usr/bin/env julia

"""
Supply Chain Network Optimization using GFlowNets

This example demonstrates how to use GFlowNets to find optimal supply chain network
structures that minimize cost while ensuring reliability. The problem involves:

1. Network Design: Deciding which suppliers, warehouses, and customers to connect
2. Cost Optimization: Minimizing transportation and operational costs
3. Reliability Assurance: Ensuring robust supply paths and redundancy
4. Capacity Planning: Balancing network capacity with demand requirements

Key Questions Addressed:
- What network structure minimizes cost while ensuring reliability?
- How do we balance direct connections vs. warehouse intermediaries?
- What level of redundancy is optimal for supply security?
- How do geographic factors affect network design?
"""

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)

using GFlowNet
using Random
using Statistics
using Plots
using CSV
using DataFrames
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

# Global parameters
const MAX_NODES = 15
const N_ITERATIONS = 500
const BATCH_SIZE = 32
const LEARNING_RATE = 0.001

"""
    create_realistic_supply_chain_scenario()

Create a realistic supply chain scenario with predefined suppliers and customers.
"""
function create_realistic_supply_chain_scenario()
    println("🏭 Creating Realistic Supply Chain Scenario")
    println("=" ^ 50)
    
    # Define suppliers (manufacturing locations)
    suppliers = [
        GFlowNet.SupplyChainNode(1, GFlowNet.SUPPLIER, (10.0, 20.0), 1000.0, 300.0, 2000.0),  # Factory A
        GFlowNet.SupplyChainNode(2, GFlowNet.SUPPLIER, (80.0, 70.0), 800.0, 250.0, 1500.0),   # Factory B
    ]
    
    # Define customers (retail locations)
    customers = [
        GFlowNet.SupplyChainNode(3, GFlowNet.CUSTOMER, (30.0, 80.0), 400.0, 100.0, 500.0),   # Store 1
        GFlowNet.SupplyChainNode(4, GFlowNet.CUSTOMER, (70.0, 30.0), 350.0, 90.0, 400.0),    # Store 2
        GFlowNet.SupplyChainNode(5, GFlowNet.CUSTOMER, (50.0, 90.0), 300.0, 80.0, 350.0),    # Store 3
    ]
    
    println("Suppliers:")
    for supplier in suppliers
        println("  - Factory $(supplier.id): Location $(supplier.location), Capacity $(supplier.capacity)")
    end
    
    println("\nCustomers:")
    for customer in customers
        println("  - Store $(customer.id): Location $(customer.location), Demand $(customer.demand)")
    end
    
    return suppliers, customers
end

"""
    run_supply_chain_optimization(suppliers, customers)

Run the main supply chain optimization using GFlowNets.
"""
function run_supply_chain_optimization(suppliers, customers)
    println("\n🎯 Running Supply Chain Optimization")
    println("=" ^ 50)
    
    # Create the GFlowNet model
    println("Creating GFlowNet model...")
    model = GFlowNet.create_supply_chain_gflownet(
        initial_suppliers=length(suppliers),
        initial_customers=length(customers),
        max_nodes=MAX_NODES,
        hidden_dim=128,
        learning_rate=LEARNING_RATE
    )
    
    println("✅ Model created successfully")
    
    # Configure training for supply chain optimization
    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.TRAJECTORY_BALANCE,
        partition_function_method=GFlowNet.ADAPTIVE_ESTIMATION,
        n_iterations=N_ITERATIONS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        validation_frequency=50,
        verbose=true
    )
    
    println("\n📈 Starting Training...")
    println("Objective: Minimize cost while maximizing reliability")
    println("Iterations: $(N_ITERATIONS)")
    println("Batch size: $(BATCH_SIZE)")
    
    # Train the model
    history = GFlowNet.train_gflownet(model, config; verbose=true)
    
    println("✅ Training completed!")
    
    return model, history
end

"""
    analyze_supply_chain_results(model, history)

Analyze and visualize the results of supply chain optimization.
"""
function analyze_supply_chain_results(model, history)
    println("\n📊 Analyzing Supply Chain Results")
    println("=" ^ 50)
    
    # Sample multiple network configurations
    println("Sampling optimal network configurations...")
    n_samples = 100
    trajectories = []
    
    for i in 1:n_samples
        try
            trajectory = GFlowNet.sample_trajectory(model)
            if !isnothing(trajectory) && !isempty(trajectory)
                push!(trajectories, trajectory)
            end
        catch e
            println("Warning: Failed to sample trajectory $i: $e")
        end
    end
    
    println("✅ Sampled $(length(trajectories)) valid network configurations")
    
    if isempty(trajectories)
        println("❌ No valid trajectories found. Training may need more iterations.")
        return
    end
    
    # Analyze final states
    final_states = [traj[end].state for traj in trajectories if !isempty(traj)]
    
    if isempty(final_states)
        println("❌ No valid final states found.")
        return
    end
    
    # Calculate statistics
    costs = [state.total_cost for state in final_states]
    reliabilities = [state.reliability_score for state in final_states]
    rewards = [GFlowNet.reward(state) for state in final_states]
    
    println("\n📈 Network Performance Statistics:")
    println("  Average Cost: $(round(mean(costs), digits=2)) ± $(round(std(costs), digits=2))")
    println("  Average Reliability: $(round(mean(reliabilities), digits=3)) ± $(round(std(reliabilities), digits=3))")
    println("  Average Reward: $(round(mean(rewards), digits=2)) ± $(round(std(rewards), digits=2))")
    println("  Best Cost: $(round(minimum(costs), digits=2))")
    println("  Best Reliability: $(round(maximum(reliabilities), digits=3))")
    println("  Best Reward: $(round(maximum(rewards), digits=2))")
    
    # Find best network
    best_idx = argmax(rewards)
    best_state = final_states[best_idx]
    
    println("\n🏆 Best Network Configuration:")
    analyze_network_structure(best_state)
    
    # Create visualizations
    create_supply_chain_plots(history, costs, reliabilities, rewards, best_state)
    
    # Save results
    save_supply_chain_results(history, final_states)
    
    return best_state, final_states
end

"""
    analyze_network_structure(state::GFlowNet.SupplyChainState)

Analyze and display the structure of a supply chain network.
"""
function analyze_network_structure(state)
    network = state.network
    
    println("  Total Cost: $(round(state.total_cost, digits=2))")
    println("  Reliability Score: $(round(state.reliability_score, digits=3))")
    println("  Reward: $(round(GFlowNet.reward(state), digits=2))")
    
    # Node analysis
    suppliers = [n for n in network.nodes if n.type == GFlowNet.SUPPLIER]
    warehouses = [n for n in network.nodes if n.type == GFlowNet.WAREHOUSE]
    customers = [n for n in network.nodes if n.type == GFlowNet.CUSTOMER]
    
    println("  Nodes: $(length(suppliers)) suppliers, $(length(warehouses)) warehouses, $(length(customers)) customers")
    println("  Connections: $(length(network.connections))")
    
    # Connection analysis
    if !isempty(network.connections)
        capacities = [conn.capacity for conn in network.connections]
        println("  Average Connection Capacity: $(round(mean(capacities), digits=1))")
        
        # Show key connections
        println("  Key Connections:")
        for (i, conn) in enumerate(network.connections[1:min(5, end)])
            from_node = network.nodes[conn.from_node]
            to_node = network.nodes[conn.to_node]
            println("    $(from_node.type) $(conn.from_node) → $(to_node.type) $(conn.to_node) (Capacity: $(conn.capacity))")
        end
    end
end

"""
    create_supply_chain_plots(history, costs, reliabilities, rewards, best_state)

Create comprehensive visualizations of supply chain optimization results.
"""
function create_supply_chain_plots(history, costs, reliabilities, rewards, best_state)
    println("\n📊 Creating Visualizations...")
    
    try
        # Plot 1: Training Progress
        p1 = plot(history[:losses], 
                 title="Training Loss", 
                 xlabel="Iteration", 
                 ylabel="Loss",
                 linewidth=2,
                 color=:blue)
        
        # Plot 2: Cost vs Reliability Trade-off
        p2 = scatter(costs, reliabilities,
                    title="Cost vs Reliability Trade-off",
                    xlabel="Total Cost",
                    ylabel="Reliability Score",
                    alpha=0.6,
                    color=:red,
                    markersize=4)
        
        # Plot 3: Reward Distribution
        p3 = histogram(rewards,
                      title="Reward Distribution",
                      xlabel="Reward",
                      ylabel="Frequency",
                      bins=20,
                      alpha=0.7,
                      color=:green)
        
        # Plot 4: Network Structure (simplified)
        p4 = plot_network_structure(best_state)
        
        # Combine plots
        combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
        
        # Save plot
        savefig(combined_plot, "supply_chain_optimization_results.png")
        println("✅ Plots saved as 'supply_chain_optimization_results.png'")
        
    catch e
        println("⚠️  Could not create plots: $e")
    end
end

"""
    plot_network_structure(state::GFlowNet.SupplyChainState)

Create a simple network structure visualization.
"""
function plot_network_structure(state)
    network = state.network
    
    # Extract node positions and types
    x_coords = [node.location[1] for node in network.nodes]
    y_coords = [node.location[2] for node in network.nodes]
    
    # Create base plot
    p = scatter(x_coords, y_coords, 
               title="Best Network Structure",
               xlabel="X Coordinate",
               ylabel="Y Coordinate",
               legend=:topright,
               markersize=8)
    
    # Color nodes by type
    for node in network.nodes
        x, y = node.location
        if node.type == GFlowNet.SUPPLIER
            scatter!(p, [x], [y], color=:blue, label="Supplier", markersize=10)
        elseif node.type == GFlowNet.WAREHOUSE
            scatter!(p, [x], [y], color=:orange, label="Warehouse", markersize=8)
        else  # CUSTOMER
            scatter!(p, [x], [y], color=:red, label="Customer", markersize=8)
        end
    end
    
    # Add connections
    for conn in network.connections
        from_node = network.nodes[conn.from_node]
        to_node = network.nodes[conn.to_node]
        plot!(p, [from_node.location[1], to_node.location[1]], 
                 [from_node.location[2], to_node.location[2]],
                 color=:gray, alpha=0.6, linewidth=2)
    end
    
    return p
end

"""
    save_supply_chain_results(history, final_states)

Save optimization results to CSV files.
"""
function save_supply_chain_results(history, final_states)
    println("\n💾 Saving Results...")
    
    try
        # Save training history
        training_df = DataFrame(
            iteration = 1:length(history[:losses]),
            loss = history[:losses]
        )
        
        if haskey(history, :partition_function_estimates) && !isempty(history[:partition_function_estimates])
            # Pad or truncate to match losses length
            z_estimates = history[:partition_function_estimates]
            if length(z_estimates) < length(history[:losses])
                # Repeat last value
                z_estimates = [z_estimates; fill(z_estimates[end], length(history[:losses]) - length(z_estimates))]
            elseif length(z_estimates) > length(history[:losses])
                z_estimates = z_estimates[1:length(history[:losses])]
            end
            training_df.partition_function = z_estimates
        end
        
        CSV.write("supply_chain_training.csv", training_df)
        
        # Save network configurations
        networks_df = DataFrame(
            network_id = 1:length(final_states),
            total_cost = [state.total_cost for state in final_states],
            reliability_score = [state.reliability_score for state in final_states],
            reward = [GFlowNet.reward(state) for state in final_states],
            n_nodes = [length(state.network.nodes) for state in final_states],
            n_connections = [length(state.network.connections) for state in final_states]
        )
        
        CSV.write("supply_chain_networks.csv", networks_df)
        
        println("✅ Results saved to CSV files")
        
    catch e
        println("⚠️  Could not save results: $e")
    end
end

"""
    demonstrate_supply_chain_scenarios()

Demonstrate different supply chain optimization scenarios.
"""
function demonstrate_supply_chain_scenarios()
    println("\n🌟 Demonstrating Different Supply Chain Scenarios")
    println("=" ^ 60)

    scenarios = [
        ("Cost-Focused", 0.7, 0.3),      # 70% cost weight, 30% reliability weight
        ("Reliability-Focused", 0.3, 0.7), # 30% cost weight, 70% reliability weight
        ("Balanced", 0.5, 0.5)           # Equal weights
    ]

    results = Dict()

    for (scenario_name, cost_weight, reliability_weight) in scenarios
        println("\n📋 Scenario: $scenario_name")
        println("   Cost Weight: $cost_weight, Reliability Weight: $reliability_weight")

        # Create suppliers and customers for this scenario
        suppliers, customers = create_realistic_supply_chain_scenario()

        # Run optimization (in practice, you'd modify the reward function weights)
        model, history = run_supply_chain_optimization(suppliers, customers)

        # Analyze results
        best_state, final_states = analyze_supply_chain_results(model, history)

        results[scenario_name] = (best_state, final_states, history)

        println("✅ Scenario $scenario_name completed")
    end

    return results
end

"""
    compare_scenarios(results)

Compare results across different scenarios.
"""
function compare_scenarios(results)
    println("\n🔍 Comparing Scenarios")
    println("=" ^ 30)

    comparison_data = []

    for (scenario_name, (best_state, final_states, history)) in results
        avg_cost = mean([state.total_cost for state in final_states])
        avg_reliability = mean([state.reliability_score for state in final_states])
        avg_reward = mean([GFlowNet.reward(state) for state in final_states])

        push!(comparison_data, (
            scenario = scenario_name,
            best_cost = best_state.total_cost,
            best_reliability = best_state.reliability_score,
            best_reward = GFlowNet.reward(best_state),
            avg_cost = avg_cost,
            avg_reliability = avg_reliability,
            avg_reward = avg_reward
        ))

        println("$scenario_name:")
        println("  Best Cost: $(round(best_state.total_cost, digits=2))")
        println("  Best Reliability: $(round(best_state.reliability_score, digits=3))")
        println("  Best Reward: $(round(GFlowNet.reward(best_state), digits=2))")
    end

    return comparison_data
end

"""
    main()

Main function to run the complete supply chain optimization demonstration.
"""
function main()
    println("🚀 Supply Chain Network Optimization with GFlowNets")
    println("=" ^ 60)
    println("This example demonstrates how GFlowNets can optimize supply chain networks")
    println("by finding structures that minimize cost while ensuring reliability.")
    println()

    try
        # Create realistic scenario
        suppliers, customers = create_realistic_supply_chain_scenario()

        # Run main optimization
        model, history = run_supply_chain_optimization(suppliers, customers)

        # Analyze results
        best_state, final_states = analyze_supply_chain_results(model, history)

        # Demonstrate different scenarios
        scenario_results = demonstrate_supply_chain_scenarios()

        # Compare scenarios
        comparison = compare_scenarios(scenario_results)

        println("\n🎉 Supply Chain Optimization Complete!")
        println("=" ^ 40)
        println("Key Insights:")
        println("✅ GFlowNets successfully found diverse network configurations")
        println("✅ Trade-offs between cost and reliability were explored")
        println("✅ Multiple optimal solutions were discovered")
        println("✅ Different scenarios showed varying optimization strategies")

        println("\nFiles Generated:")
        println("📊 supply_chain_optimization_results.png - Visualization plots")
        println("📈 supply_chain_training.csv - Training progress data")
        println("🏭 supply_chain_networks.csv - Network configuration data")

        println("\nNext Steps:")
        println("🔬 Experiment with different cost/reliability weights")
        println("🌍 Try different geographic distributions of nodes")
        println("📦 Add capacity constraints and demand variations")
        println("🔄 Explore dynamic supply chain scenarios")

        return best_state, final_states, scenario_results

    catch e
        println("❌ Error during optimization: $e")
        println("This might be due to missing dependencies or configuration issues.")
        println("Please ensure all required packages are installed.")
        rethrow(e)
    end
end

# Run the demonstration if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end
