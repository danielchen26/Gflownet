#!/usr/bin/env julia

# Example script for causal discovery using GFlowNets
# This demonstrates how GFlowNets can be used to find directed acyclic graphs
# that best explain observed data

# IMPORTANT: This script must be run from the project root directory
# Run with: julia examples/causal_discovery.jl

using Pkg
Pkg.activate(".")  # Activate the project in the current directory (should be the project root)

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

# Create a DAGState from our causal_discovery.jl implementation
function create_dag_from_adjacency(adjacency::Matrix{Int}, node_names::Vector{String})
    return GFlowNet.DAGState(adjacency, node_names, false)
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
    initial_state = GFlowNet.create_initial_dag_state(variable_names)
    
    # Create all possible actions for this DAG
    actions = GFlowNet.create_dag_actions(variable_names)
    
    # Create terminal states - in practice, these would be discovered during search
    # For this example, we'll create a few terminal states including the true DAG
    true_dag = create_dag_from_adjacency(true_adjacency, variable_names)
    
    # Create some variations by randomly adding/removing edges from the true DAG
    rng = Random.MersenneTwister(42)
    terminal_states = [GFlowNet.DAGState(true_adjacency, variable_names, true)]
    
    for i in 1:5
        # Copy true adjacency and make random modifications
        adj_copy = copy(true_adjacency)
        
        # Randomly modify some edges
        for _ in 1:3
            i, j = rand(rng, 1:n_variables), rand(rng, 1:n_variables)
            if i != j && adj_copy[i, j] == 0 && !GFlowNet.has_cycle(adj_copy .| [i == x && j == y for x in 1:n_variables, y in 1:n_variables])
                adj_copy[i, j] = 1
            end
        end
        
        # Add this as a terminal state
        push!(terminal_states, GFlowNet.DAGState(adj_copy, variable_names, true))
    end
    
    # Terminal sink state
    terminal_sink = GFlowNet.DAGState(zeros(Int, n_variables, n_variables), variable_names, true)
    
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
    
    # Create logger
    logger = GFlowNetLogger("causal_discovery_training.csv", log_frequency=10, verbose=true)
    
    # Train the model
    println("Training GFlowNet...")
    n_iterations = 500
    batch_size = 16
    
    for iter in 1:n_iterations
        # Sample trajectories
        trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:batch_size]
        
        # Compute loss and gradients
        total_loss, total_grad = GFlowNet.compute_loss_and_grad(model, trajectories)
        
        # Apply optimizer updates
        GFlowNet.apply_optimizer!(model, total_grad)
        
        # Log metrics
        if iter % 10 == 0
            # Get terminal states from trajectories
            terminal_states = [trajectory.states[end] for trajectory in trajectories]
            
            # Compute rewards
            rewards = [GFlowNet.reward(state, data) for state in terminal_states]
            
            # Log
            log_iteration!(
                logger, 
                total_loss,
                reward_mean=mean(rewards),
                reward_std=std(rewards)
            )
        end
        
        # Re-estimate partition function periodically
        if iter % 50 == 0
            model.partition_function = GFlowNet.estimate_partition_function(model)
        end
    end
    
    # Visualize results
    println("Visualizing results...")
    
    # Plot loss curve
    losses = get_metric(logger, "loss")
    loss_plot = visualize_training_progress(losses)
    savefig(loss_plot, "causal_discovery_loss.png")
    
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
    savefig(dag_plot, "causal_discovery_best_dag.png")
    
    # Visualize the true DAG
    true_dag_plot = visualize_causal_graph(true_dag)
    savefig(true_dag_plot, "causal_discovery_true_dag.png")
    
    # Plot reward distribution
    reward_plot = visualize_reward_distribution(model, 100)
    savefig(reward_plot, "causal_discovery_rewards.png")
    
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
    savefig(shd_plot, "causal_discovery_shd.png")
    
    println("Example completed. Results saved to causal_discovery_*.png")
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 