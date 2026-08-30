#!/usr/bin/env julia

# Example script for causal discovery using GFlowNets
# This demonstrates how GFlowNets can be used to find directed acyclic graphs
# that best explain observed data

# IMPORTANT: This script must be run from the example directory
# Run with: julia causal_discovery.jl

using Pkg
Pkg.activate(@__DIR__)  # Activate the project in the current directory (the example directory)

# `examples/**/Manifest.toml` is intentionally untracked (a manifest resolved on
# one Julia version cannot be instantiated on another), so a clean checkout has
# no manifest here and `using GFlowNet` would otherwise die with an opaque
# "required but does not seem to be installed". Project.toml carries a
# `[sources]` entry pointing GFlowNet at the repository root, so resolving and
# installing on first run needs no registry entry and no `Pkg.develop` step.
try
    Pkg.instantiate()
catch err
    @error """Could not instantiate the example environment at $(@__DIR__).
             Run `julia --project=. examples/setup_examples.jl` from the \
             repository root, then re-run this script.""" exception = err
    rethrow()
end

using GFlowNet
using GFlowNet.GFlowNetUtils
using Plots
using Random
using LinearAlgebra
using Statistics
using StatsBase
using Distributions  # For probability distributions

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
    
    # The true DAG is only needed for the evaluation further down.
    true_dag = create_dag_from_adjacency(true_adjacency, variable_names)

    # Build the model with the current on-demand API. There is no `create_dag`
    # helper and no `GFlowNetModel(dag, ...)` constructor any more, and
    # `TrajectoryBalanceObjective` is gone -- the objective now lives in
    # `TrainingConfig`. `create_gflownet` takes the initial state plus the full
    # action set and builds the policy network, the `ComponentArray` parameters,
    # the optimiser and the layer states itself; reachable states are enumerated
    # on demand during sampling, so no bootstrap list of terminal states is
    # required.
    rng = Random.default_rng()
    input_dim = length(GFlowNet.state_to_features(initial_state))

    model = GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = input_dim,
        hidden_dim = 128,
        learning_rate = 0.001,
        include_flow_estimator = true,
        # LEARNABLE, not SAMPLING: SAMPLING_ESTIMATION is not implemented and
        # silently pinned Z = 1 here, so trajectory balance was unsatisfiable on a
        # reward set that does not sum to 1.
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        rng = rng
    )
    
    # Train using modern interface
    println("Training GFlowNet with modern training interface...")
    
    # Demo budget: a DAG trajectory can add and remove edges up to
    # `SamplingConfig`'s 100-step cap, so keep the iteration count small enough
    # that the example finishes in well under a minute of training.
    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.TRAJECTORY_BALANCE,
        # Must match the model. The old comment here claimed "complex graph spaces
        # need sampling", asserting a benefit from an unimplemented no-op.
        partition_function_method=GFlowNet.LEARNABLE_ESTIMATION,
        batch_size=4,
        learning_rate=0.001,
        n_iterations=10,
        validation_frequency=5,
        early_stopping_patience=10
    )
    
    println("Training configuration for causal discovery:")
    println("  Objective: $(config.objective)")
    println("  Partition function method: $(config.partition_function_method) (optimal for complex spaces)")
    println("  Batch size: $(config.batch_size)")
    println("  Iterations: $(config.n_iterations)")
    
    training_history = GFlowNet.train_gflownet(model, config; verbose=true)
    
    println("Training completed!")
    println("  Final loss: $(round(training_history[:losses][end], digits=6))")
    println("  Mean gradient norm: $(round(sum(training_history[:gradient_norms]) / length(training_history[:gradient_norms]), digits=6))")
    println("  Total training iterations: $(length(training_history[:losses]))")
    
    # Visualize results
    println("Visualizing results...")
    
    # Create output directory if it doesn't exist
    # Write next to the script, not into whatever directory the user launched
    # from -- this example is normally run as `julia examples/causal_discovery/…`
    # from the repository root.
    output_dir = @__DIR__
    
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
                         bins=0:max(1, maximum(distances)),  # 0:0 is not a valid bin edge set
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