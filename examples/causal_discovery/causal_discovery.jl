#!/usr/bin/env julia

# Example script for causal discovery using GFlowNets
# This demonstrates how GFlowNets can be used to find directed acyclic graphs
# that best explain observed data

# IMPORTANT: This script must be run from the example directory
# Run with: julia causal_discovery.jl

using Pkg
Pkg.activate(@__DIR__)  # Activate the project in the current directory (the example directory)

using GFlowNet
using GFlowNet.GFlowNetUtils
using Plots
using Random
using LinearAlgebra
using Statistics
using StatsBase
using Distributions  # For probability distributions
using Lux, Optimisers, NNlib  # Moved from inside the main function

# Generate synthetic causal data
function generate_synthetic_causal_data(n_samples::Int, n_variables::Int; 
                                       sparsity::Float64=0.3, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    
    # Generate a random DAG
    adjacency = zeros(Int, n_variables, n_variables)
    
    # Lower triangular ensures acyclicity
    for i in 2:n_variables
        for j in 1:(i-1)
            if rand(rng) < sparsity
                adjacency[i, j] = 1
            end
        end
    end
    
    # Generate weights
    weights = adjacency .* (rand(rng, n_variables, n_variables) .* 2 .- 1)
    
    # Generate data
    data = zeros(n_samples, n_variables)
    
    for sample in 1:n_samples
        # Generate in topological order
        for i in 1:n_variables
            # Compute parents contribution
            parent_contribution = 0.0
            for j in 1:n_variables
                parent_contribution += data[sample, j] * weights[i, j]
            end
            
            # Add noise
            data[sample, i] = parent_contribution + 0.1 * randn(rng)
        end
    end
    
    # Compute covariance and correlation matrices
    cov_matrix = cov(data)
    cor_matrix = cor(data)
    
    return data, adjacency, weights, cov_matrix, cor_matrix
end

# Add a custom has_cycle function that converts BitVectors to Vector{Bool}
function has_cycle(adjacency_matrix::Matrix{Bool})
    n = size(adjacency_matrix, 1)
    # Using Vector{Bool} instead of falses() to ensure type compatibility
    visited = Vector{Bool}(falses(n))
    rec_stack = Vector{Bool}(falses(n))
    
    function is_cyclic_util(adjacency_matrix, v, visited, rec_stack)
        visited[v] = true
        rec_stack[v] = true
        
        # Visit all neighbors
        for i in 1:size(adjacency_matrix, 1)
            if adjacency_matrix[v, i]
                if !visited[i]
                    if is_cyclic_util(adjacency_matrix, i, visited, rec_stack)
                        return true
                    end
                elseif rec_stack[i]
                    return true
                end
            end
        end
        
        rec_stack[v] = false
        return false
    end
    
    for i in 1:n
        if !visited[i]
            if is_cyclic_util(adjacency_matrix, i, visited, rec_stack)
                return true
            end
        end
    end
    
    return false
end

# Create a DAGState from our causal_discovery.jl implementation
function create_dag_from_adjacency(adjacency::Matrix{Int}, node_names::Vector{String})
    # Convert Int matrix to Bool matrix, ensuring it's Matrix{Bool} and not BitMatrix
    bool_adjacency = Matrix{Bool}(adjacency .!= 0)
    return GFlowNet.DAGState(bool_adjacency, node_names, false)
end

# Skip visualizations that require specific modules
function visualize_causal_graph(dag_state)
    # Simplified version that just prints adjacency matrix
    println("Causal Graph Adjacency Matrix:")
    println(dag_state.adjacency_matrix)
    return plot(title="Causal Graph", legend=false)
end

function visualize_reward_distribution(model, n_samples)
    # Simplified version that returns a placeholder plot
    return plot(title="Reward Distribution", legend=false)
end

function visualize_training_progress(losses)
    # Simplified version that returns a placeholder plot
    return plot(title="Training Progress", legend=false)
end

# Main function to run the example
function main()
    println("Setting up Causal Discovery GFlowNet example...")
    
    # Generate synthetic causal data
    n_variables = 5
    n_samples = 1000
    
    variable_names = ["X$i" for i in 1:n_variables]
    data, true_adjacency, true_weights, cov_matrix, cor_matrix = 
        generate_synthetic_causal_data(n_samples, n_variables)
    
    println("Generated synthetic causal data with:\n",
            "- True adjacency matrix:\n", true_adjacency)
    
    # Create initial state with empty DAG
    initial_state = GFlowNet.create_initial_dag_state(n_variables, variable_names)
    
    # Create all possible actions for this DAG
    actions = GFlowNet.create_dag_actions(n_variables)
    
    # Create terminal states - in practice, these would be discovered during search
    # For this example, we'll create a few terminal states including the true DAG
    true_dag = create_dag_from_adjacency(true_adjacency, variable_names)
    
    # Create some variations by randomly adding/removing edges from the true DAG
    rng = Random.MersenneTwister(42)
    terminal_states = [GFlowNet.DAGState(Matrix{Bool}(true_adjacency .!= 0), variable_names, true)]
    
    for i in 1:5
        # Copy true adjacency and make random modifications
        adj_copy = copy(true_adjacency)
        
        # Randomly modify some edges
        for _ in 1:3
            i, j = rand(rng, 1:n_variables), rand(rng, 1:n_variables)
            # Convert to Matrix{Bool} before calling has_cycle
            temp_matrix = Matrix{Bool}((adj_copy .| [i == x && j == y for x in 1:n_variables, y in 1:n_variables]) .!= 0)
            if i != j && adj_copy[i, j] == 0 && !has_cycle(temp_matrix)
                adj_copy[i, j] = 1
            end
        end
        
        # Add this as a terminal state
        push!(terminal_states, GFlowNet.DAGState(Matrix{Bool}(adj_copy .!= 0), variable_names, true))
    end
    
    # Terminal sink state
    terminal_sink = GFlowNet.DAGState(Matrix{Bool}(zeros(Int, n_variables, n_variables) .!= 0), variable_names, true)
    
    # Create DAG
    dag = GFlowNet.create_dag(initial_state, terminal_states, terminal_sink, actions)
    
    # Create neural network models for policies
    rng = Random.default_rng()
    
    # Get feature dimension from a state
    input_dim = length(GFlowNet.state_to_features(initial_state))
    
    # Output dimension is the number of possible states
    output_dim = length(dag.states)
    
    # Create forward policy
    forward_policy, forward_ps, forward_st = GFlowNet.create_forward_policy(
        input_dim, 128, output_dim, rng
    )
    
    # Create flow estimator
    flow_estimator, flow_ps, flow_st = GFlowNet.create_flow_estimator(
        input_dim, 128, rng
    )
    
    # Create optimizer
    opt = Optimisers.Adam(0.001)
    
    forward_opt_state = Optimisers.setup(opt, forward_ps)
    flow_opt_state = Optimisers.setup(opt, flow_ps)
    
    # Define optimizer structure for GFlowNet
    optimizer = (forward = forward_opt_state, flow = flow_opt_state)
    
    # Create GFlowNet model with trajectory balance objective
    model = GFlowNet.GFlowNetModel(
        dag,
        forward_policy,
        nothing,  # No backward policy
        flow_estimator,
        nothing,  # Will be estimated during training
        [GFlowNet.TrajectoryBalanceObjective(1.0)],
        optimizer,
        (forward = forward_ps, backward = nothing, flow = flow_ps),  # Parameters
        (forward = forward_st, backward = nothing, flow = flow_st)   # States
    )
    
    # Train using modern interface
    println("Training GFlowNet with modern training interface...")
    
    # Create training configuration optimized for causal discovery
    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.GENERAL_TRAJECTORY_BALANCE,  # Use general TB for non-deterministic paths
        partition_function_method=GFlowNet.SAMPLING_BASED,  # Complex graph spaces need sampling
        batch_size=24,
        learning_rate=0.001,
        n_iterations=1500,
        partition_update_frequency=25,
        validation_frequency=100,
        early_stopping_patience=150
    )
    
    println("Training configuration for causal discovery:")
    println("  Objective: $(config.objective) (handles non-deterministic graph construction)")
    println("  Partition function method: $(config.partition_function_method) (optimal for complex spaces)")
    println("  Batch size: $(config.batch_size)")
    println("  Iterations: $(config.n_iterations)")
    
    training_history = GFlowNet.train_gflownet(model, config; verbose=true)
    
    println("Training completed!")
    println("  Final loss: $(round(training_history[:losses][end], digits=6))")
    println("  Final Z estimate: $(round(training_history[:partition_function_estimates][end], digits=6))")
    println("  Total training iterations: $(length(training_history[:losses]))")
    
    # Visualize results
    println("Visualizing results...")
    
    # Create output directory if it doesn't exist
    output_dir = "."
    
    # Plot loss curve from training history
    loss_plot = plot(
        1:length(training_history[:losses]),
        training_history[:losses],
        title="Causal Discovery Training Loss",
        xlabel="Iteration",
        ylabel="Loss",
        lw=2,
        legend=false
    )
    savefig(loss_plot, joinpath(output_dir, "causal_discovery_loss.png"))
    
    # Sample DAGs and visualize them
    n_samples = 20
    sampled_trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:n_samples]
    sampled_dags = [trajectory.states[end] for trajectory in sampled_trajectories]
    
    # Calculate rewards for each sampled DAG
    rewards = [GFlowNet.reward(dag, data) for dag in sampled_dags]
    
    # Find the highest reward DAG
    best_idx = argmax(rewards)
    best_dag = sampled_dags[best_idx]
    
    println("Best discovered DAG has reward $(rewards[best_idx]):")
    println(best_dag.adjacency_matrix)
    
    # Visualize the best discovered DAG
    dag_plot = visualize_causal_graph(best_dag)
    savefig(dag_plot, joinpath(output_dir, "causal_discovery_best_dag.png"))
    
    # Visualize the true DAG
    true_dag_plot = visualize_causal_graph(true_dag)
    savefig(true_dag_plot, joinpath(output_dir, "causal_discovery_true_dag.png"))
    
    # Plot reward distribution
    reward_plot = visualize_reward_distribution(model, 100)
    savefig(reward_plot, joinpath(output_dir, "causal_discovery_rewards.png"))
    
    # Calculate structural hamming distance between true DAG and discovered DAGs
    function structural_hamming_distance(adj1, adj2)
        return sum(abs.(adj1 - adj2))
    end
    
    distances = [structural_hamming_distance(dag.adjacency_matrix, true_adjacency) for dag in sampled_dags]
    mean_distance = mean(distances)
    min_distance = minimum(distances)
    
    println("Structural Hamming Distance statistics:")
    println("- Mean SHD: $mean_distance")
    println("- Min SHD: $min_distance")
    
    # Plot distribution of SHDs
    shd_plot = histogram(distances, 
                         bins=0:maximum(distances), 
                         title="Structural Hamming Distance", 
                         xlabel="SHD", 
                         ylabel="Frequency", 
                         legend=false)
    savefig(shd_plot, joinpath(output_dir, "causal_discovery_shd.png"))
    
    println("Example completed. Results saved to causal_discovery_*.png")
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 