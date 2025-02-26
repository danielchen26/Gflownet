#!/usr/bin/env julia

# Example script for active learning using GFlowNets
# This demonstrates how GFlowNets can be used for experimental design
# and selecting informative experiments

# IMPORTANT: This script must be run from the project root directory
# Run with: julia examples/active_learning.jl

using Pkg
Pkg.activate(".")  # Activate the project in the current directory (should be the project root)

using GFlowNet
using GFlowNet.GFlowNetUtils
using Plots
using Random
using LinearAlgebra
using Statistics
using Lux, Optimisers, NNlib
using MultivariateStats

# Main function to run the example
function main()
    println("Setting up Active Learning GFlowNet example...")
    
    # Simulate experiment data
    n_experiments = 50  # Total number of possible experiments
    feature_dim = 10    # Feature dimension for each experiment
    
    # Generate synthetic experiment data
    experiment_features, experiment_values = 
        GFlowNet.simulate_experiment_data(n_experiments, feature_dim)
    
    println("Generated synthetic data for $n_experiments experiments with $feature_dim features")
    
    # Set maximum experiments to select
    max_experiments = 5
    
    # Create initial state with no experiments selected
    initial_state = GFlowNet.create_initial_experiment_state(max_experiments)
    
    # Terminal states - in practice, these would be discovered during search
    # Here, we'll generate some representative terminal states by selecting random experiments
    rng = Random.MersenneTwister(42)
    terminal_states = []
    
    # Add some representative terminal states
    for _ in 1:10
        # Select a random number of experiments between 1 and max_experiments
        n_selected = rand(rng, 1:max_experiments)
        selected_experiments = sample(rng, 1:n_experiments, n_selected, replace=false)
        
        # Create a terminal state with these experiments
        term_state = GFlowNet.ExperimentState(selected_experiments, max_experiments, true)
        push!(terminal_states, term_state)
    end
    
    # Terminal sink state
    terminal_sink = GFlowNet.ExperimentState(Int[], max_experiments, true)
    
    # Create all possible actions for this task
    actions = GFlowNet.create_experiment_actions(n_experiments)
    
    # Create DAG
    dag = GFlowNet.create_dag(initial_state, terminal_states, terminal_sink, actions)
    
    # Create neural network models for policies
    rng = Random.default_rng()
    
    # Get feature dimension from a state (using experiment_features)
    input_dim = length(GFlowNet.state_to_features(initial_state, experiment_features))
    
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
    
    # Define the reward function that captures the state_to_features and experiment data
    function experiment_reward(state)
        return GFlowNet.reward(state, experiment_features, experiment_values)
    end
    
    # Modify the GFlowNet.reward function to use our custom function
    # (In a proper implementation, we would use method specialization)
    GFlowNet.reward(state::GFlowNet.ExperimentState) = experiment_reward(state)
    
    # Create logger
    logger = GFlowNetLogger("active_learning_training.csv", log_frequency=10, verbose=true)
    
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
            rewards = [GFlowNet.reward(state) for state in terminal_states]
            
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
    savefig(loss_plot, "active_learning_loss.png")
    
    # Sample experiment selections and visualize them
    n_samples = 10
    sampled_trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:n_samples]
    sampled_states = [trajectory.states[end] for trajectory in sampled_trajectories]
    
    # Calculate rewards for each sampled state
    rewards = [GFlowNet.reward(state) for state in sampled_states]
    
    # Find the best experiment selection
    best_idx = argmax(rewards)
    best_state = sampled_states[best_idx]
    
    println("Best experiment selection has reward $(rewards[best_idx]):")
    println("Selected experiments: $(best_state.experiments)")
    
    # Visualize the best experiment selection
    exp_plot = visualize_experiment_selection(best_state, experiment_features, experiment_values)
    savefig(exp_plot, "active_learning_best_selection.png")
    
    # Plot experiment values
    values_plot = scatter(1:n_experiments, experiment_values, 
                         marker_z=experiment_values,
                         title="Experiment Values",
                         xlabel="Experiment Index",
                         ylabel="Value",
                         legend=false,
                         color=:viridis)
    
    # Highlight selected experiments
    scatter!(values_plot, best_state.experiments, experiment_values[best_state.experiments],
            markersize=10,
            markershape=:star5,
            markercolor=:red,
            label="Selected")
    
    savefig(values_plot, "active_learning_values.png")
    
    # Plot reward distribution
    reward_plot = visualize_reward_distribution(model, 100)
    savefig(reward_plot, "active_learning_rewards.png")
    
    # Analyze the diversity of selected experiments
    function experiment_diversity(state)
        if isempty(state.experiments) || length(state.experiments) == 1
            return 0.0
        end
        
        selected_features = experiment_features[state.experiments, :]
        
        # Calculate pairwise distances
        diversity = 0.0
        n_pairs = 0
        for i in 1:length(state.experiments)
            for j in i+1:length(state.experiments)
                diversity += norm(selected_features[i, :] - selected_features[j, :])
                n_pairs += 1
            end
        end
        
        return diversity / n_pairs
    end
    
    # Calculate diversity for each sampled state
    diversities = [experiment_diversity(state) for state in sampled_states]
    mean_diversity = mean(diversities)
    
    println("Analysis of selected experiments:")
    println("- Mean reward: $(mean(rewards))")
    println("- Best reward: $(maximum(rewards))")
    println("- Mean diversity: $mean_diversity")
    
    # PCA plot of experiment features and selection
    # Perform PCA
    X = experiment_features'
    M = fit(PCA, X; maxoutdim=2)
    Y = transform(M, X)
    
    # Plot PCA results
    pca_plot = scatter(Y[1,:], Y[2,:],
                      title="PCA of Experiment Features",
                      xlabel="PC1",
                      ylabel="PC2",
                      label=nothing,
                      markersize=8,
                      markerstrokewidth=1,
                      markerstrokecolor=:black,
                      markercolor=:lightblue)
    
    # Highlight selected experiments
    scatter!(pca_plot, Y[1,best_state.experiments], Y[2,best_state.experiments],
            markersize=10,
            markershape=:star5,
            markercolor=:red,
            label="Selected")
    
    savefig(pca_plot, "active_learning_pca.png")
    
    println("Example completed. Results saved to active_learning_*.png")
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 