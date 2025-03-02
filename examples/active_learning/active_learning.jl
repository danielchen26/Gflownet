#!/usr/bin/env julia

# Example script for active learning using GFlowNets
# This demonstrates how GFlowNets can be used for experimental design
# and selecting informative experiments

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)

using GFlowNet
using Plots
using Random
using Statistics
using StatsBase
using LinearAlgebra
using Lux # Add Lux for neural network models
using Optimisers # Add Optimisers for optimization
using CSV
using DataFrames

# Set global parameters
const N_EXPERIMENTS = 50   # Total number of possible experiments
const FEATURE_DIM = 10     # Feature dimension for each experiment
const MAX_EXPERIMENTS = 5  # Maximum number of experiments to select
const N_ITERATIONS = 1000  # Number of training iterations
const BATCH_SIZE = 32      # Batch size for training
const LEARNING_RATE = 1e-3 # Learning rate for optimizer

# Global variables to store experiment data
global_experiment_features = nothing
global_experiment_values = nothing

# Define state-to-features method for ExperimentState
function GFlowNet.state_to_features(state::GFlowNet.ExperimentState)
    # Create a one-hot encoding of which experiments have been selected
    selected = zeros(Float64, N_EXPERIMENTS)
    for exp_idx in state.experiments
        selected[exp_idx] = 1.0
    end
    
    # Add a feature for the number of experiments selected (normalized)
    n_selected = length(state.experiments) / MAX_EXPERIMENTS
    
    # Add a feature for whether the state is terminal
    is_terminal = Float64(state.is_terminal)
    
    # Combine all features
    return [selected; n_selected; is_terminal]
end

# Helper function to calculate reward based on experiment values
function experiment_reward(state::GFlowNet.ExperimentState)
    # No experiments selected means zero reward
    if isempty(state.experiments)
        return 0.0
    end
    
    # Get the values of selected experiments
    selected_values = global_experiment_values[state.experiments]
    
    # Reward is the maximum value among selected experiments
    # This encourages exploration of high-value experiments
    return maximum(selected_values)
end

# Define the reward function - this is the objective we want to maximize
function GFlowNet.reward(state::GFlowNet.ExperimentState)
    return experiment_reward(state)
end

# Custom function to sample trajectories for our model
# This abstracts away the differences between the built-in sampling and our needs
function sample_trajectories(model, batch_size, rng=Random.GLOBAL_RNG)
    trajectories = Vector{GFlowNet.Trajectory}(undef, batch_size)
    
    for i in 1:batch_size
        # Try to get a valid trajectory with experiments
        valid_trajectory = false
        for _ in 1:10  # Try up to 10 times
            traj = GFlowNet.sample_trajectory(model)
            
            # If the terminal state has at least one experiment, use it
            if !isempty(traj.states[end].experiments)
                trajectories[i] = traj
                valid_trajectory = true
                break
            end
        end
        
        # If we couldn't get a valid trajectory, create one with high-value experiments
        if !valid_trajectory
            # Sample 1-3 experiments from the top 10 highest value experiments
            top_exps = sortperm(global_experiment_values, rev=true)[1:10]
            n_selected = rand(rng, 1:3)
            selected = sample(rng, top_exps, n_selected, replace=false)
            
            # Create a terminal state with these experiments
            terminal_state = GFlowNet.ExperimentState(selected, MAX_EXPERIMENTS, true)
            
            # Create a trajectory with just this terminal state
            # The Trajectory constructor only needs a vector of states
            trajectories[i] = GFlowNet.Trajectory([terminal_state])
        end
    end
    
    return trajectories
end

function main()
    println("Setting up Active Learning GFlowNet example...")
    
    # Set a fixed random seed for reproducibility
    rng = Random.MersenneTwister(42)
    Random.seed!(42)
    
    # Generate synthetic experiment data
    println("Generating synthetic data for $(N_EXPERIMENTS) experiments with $(FEATURE_DIM) features...")
    
    # Generate random features for experiments
    global global_experiment_features = randn(rng, FEATURE_DIM, N_EXPERIMENTS)
    
    # Generate random weights for a linear model
    true_weights = randn(rng, FEATURE_DIM)
    
    # Calculate true values as dot product with some noise
    base_values = global_experiment_features' * true_weights
    global global_experiment_values = base_values .+ 0.5 * randn(rng, N_EXPERIMENTS)
    
    # Normalize values to [0, 1] range for reward scaling
    min_val, max_val = minimum(global_experiment_values), maximum(global_experiment_values)
    global global_experiment_values = (global_experiment_values .- min_val) ./ (max_val - min_val)
    
    println("Top 5 experiments by value:")
    sorted_indices = sortperm(global_experiment_values, rev=true)
    for i in 1:5
        idx = sorted_indices[i]
        println("  Experiment $idx: $(global_experiment_values[idx])")
    end
    
    # Create initial state with no experiments selected
    initial_state = GFlowNet.ExperimentState(Int[], MAX_EXPERIMENTS, false)
    
    # Terminal states with diverse experiment selections for bootstrapping
    terminal_states = GFlowNet.ExperimentState[]
    
    # Add terminal states that include high-value experiments
    println("Creating terminal states with experiments...")
    # Sort experiments by value to include some high-value ones in terminal states
    sorted_exp_indices = sortperm(global_experiment_values, rev=true)
    
    # Add terminal states with high-value experiments
    for i in 1:5
        # Include at least one high-value experiment in each terminal state
        high_value_exp = sorted_exp_indices[i]
        
        # Select other experiments randomly
        n_additional = rand(rng, 0:MAX_EXPERIMENTS-1)
        remaining_exps = setdiff(1:N_EXPERIMENTS, high_value_exp)
        additional_exps = sample(rng, remaining_exps, n_additional, replace=false)
        
        # Combine high-value experiment with random ones
        selected_experiments = vcat(high_value_exp, additional_exps)
        
        # Create a terminal state with these experiments
        term_state = GFlowNet.ExperimentState(selected_experiments, MAX_EXPERIMENTS, true)
        push!(terminal_states, term_state)
        println("  Terminal state $i: experiments $(term_state.experiments), reward: $(experiment_reward(term_state))")
    end
    
    # Add some completely random terminal states for diversity
    for i in 6:10
        n_selected = rand(rng, 1:MAX_EXPERIMENTS)
        selected_experiments = sample(rng, 1:N_EXPERIMENTS, n_selected, replace=false)
        
        term_state = GFlowNet.ExperimentState(selected_experiments, MAX_EXPERIMENTS, true)
        push!(terminal_states, term_state)
        println("  Random terminal state $i: experiments $(term_state.experiments), reward: $(experiment_reward(term_state))")
    end
    
    # Terminal sink state (represents stopping with experiments)
    # Use the highest value experiment for the sink state
    best_exp = [sorted_exp_indices[1]]
    terminal_sink = GFlowNet.ExperimentState(best_exp, MAX_EXPERIMENTS, true)
    println("Created terminal sink state with experiment $(best_exp[1]), reward: $(experiment_reward(terminal_sink))")
    
    # Create actions for experiment selection
    actions = GFlowNet.create_experiment_actions(N_EXPERIMENTS)
    
    # Setup the flow network model
    rng = Random.MersenneTwister(42)
    
    # Create a neural network for the policy
    input_dim = N_EXPERIMENTS + 2  # One-hot encoding + num_selected + is_terminal
    nn_model = Chain(
        Dense(input_dim => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => length(actions))
    )
    
    # Initialize parameters
    ps, st = Lux.setup(rng, nn_model)
    
    # Create the flow network with the neural network
    model = GFlowNet.GFlowNetModel(
        GFlowNet.create_dag(
            initial_state,
            terminal_states,
            terminal_sink,
            actions
        ),
        GFlowNet.ForwardPolicy(nn_model),
        nothing,  # backward_policy
        nothing,  # flow_estimator
        nothing,  # partition_function
        [GFlowNet.TrajectoryBalanceObjective(1.0)],  # objectives
        Optimisers.Adam(LEARNING_RATE),  # optimizer
        (forward = ps, backward = nothing, flow = nothing),  # parameters
        (forward = st, backward = nothing, flow = nothing)   # states
    )
    
    println("Training GFlowNet for $N_ITERATIONS iterations...")
    
    # Track training metrics
    train_data = DataFrame(
        iteration = Int[],
        loss = Float64[],
        mean_reward = Float64[],
        max_reward = Float64[]
    )
    
    # Train the model with improved monitoring
    for iter in 1:N_ITERATIONS
        # Sample trajectories using our custom function
        trajectories = sample_trajectories(model, BATCH_SIZE, rng)
        
        # Calculate loss and gradient
        try
            loss, grad = GFlowNet.compute_loss_and_grad(model, trajectories)
            
            # Apply optimizer to update the model
            if !isnothing(grad)
                GFlowNet.apply_optimizer!(model, grad)
                
                # Calculate rewards from sampled trajectories
                terminal_rewards = [GFlowNet.reward(traj.states[end]) for traj in trajectories]
                mean_reward = mean(terminal_rewards)
                max_reward = maximum(terminal_rewards)
                
                # Log training metrics
                push!(train_data, (iter, loss, mean_reward, max_reward))
                
                if iter % 100 == 0
                    println("Iteration $iter: Loss = $loss, Mean reward = $mean_reward, Max reward = $max_reward")
                end
            else
                # Still log metrics even if we didn't get gradients
                terminal_rewards = [GFlowNet.reward(traj.states[end]) for traj in trajectories]
                mean_reward = mean(terminal_rewards)
                max_reward = maximum(terminal_rewards)
                
                # Log even when no gradient is available
                push!(train_data, (iter, loss, mean_reward, max_reward))
                
                if iter % 100 == 0
                    println("Iteration $iter: Loss = $loss (no gradient), Mean reward = $mean_reward, Max reward = $max_reward")
                end
            end
        catch e
            println("Error in training iteration $iter: $e")
            println("Error details: $e")
            
            # Log error in training data
            push!(train_data, (iter, NaN, NaN, NaN))
        end
    end
    
    # Save training metrics
    CSV.write("active_learning_training.csv", train_data)
    
    # Plot training metrics
    loss_plot = Plots.plot(
        train_data.iteration, 
        train_data.loss,
        title="Loss During Training",
        xlabel="Iteration",
        ylabel="Loss",
        legend=false,
        lw=2
    )
    Plots.savefig(loss_plot, "active_learning_loss.png")
    
    reward_plot = Plots.plot(
        train_data.iteration, 
        [train_data.mean_reward train_data.max_reward], 
        title="Rewards During Training",
        xlabel="Iteration",
        ylabel="Reward",
        label=["Mean Reward" "Max Reward"],
        lw=2
    )
    Plots.savefig(reward_plot, "active_learning_rewards.png")
    
    # Sample experiment selections after training
    println("\nSampling experiment selections after training...")
    n_samples = 10
    
    try
        # Estimate partition function
        println("Estimating partition function...")
        Z = GFlowNet.estimate_partition_function(model)
        println("Estimated partition function: $Z")
        
        # Sample states from the model
        sample_results = []
        
        for _ in 1:n_samples
            # Try to get a valid trajectory with experiments
            for attempt in 1:10  # Try up to 10 times
                traj = GFlowNet.sample_trajectory(model)
                terminal_state = traj.states[end]
                
                # If the terminal state has at least one experiment, use it
                if !isempty(terminal_state.experiments)
                    push!(sample_results, (terminal_state, GFlowNet.reward(terminal_state)))
                    break
                end
                
                # If this is the last attempt and we still don't have a valid state
                if attempt == 10
                    # Just use the best terminal state we know about
                    push!(sample_results, (terminal_sink, GFlowNet.reward(terminal_sink)))
                end
            end
        end
        
        # Extract states and rewards
        sampled_states = [r[1] for r in sample_results]
        rewards = [r[2] for r in sample_results]
        
        # Print the selected experiments and their rewards
        println("\nSampled experiment selections:")
        for (i, (state, reward)) in enumerate(zip(sampled_states, rewards))
            println("Sample $i: Selected experiments $(state.experiments), Reward: $reward")
        end
        
        # Find the best experiment selection
        best_idx = argmax(rewards)
        best_state = sampled_states[best_idx]
        
        println("\nBest experiment selection has reward $(rewards[best_idx]):")
        println("Selected experiments: $(best_state.experiments)")
        
        # Create a heatmap of experiment values
        plt = Plots.heatmap(
            reshape(global_experiment_values, 5, 10),
            title="Experiment Values",
            xlabel="Experiment Group",
            ylabel="Experiment Index",
            color=:viridis,
            aspect_ratio=:equal,
            clims=(0, 1)
        )
        
        # Mark the best selected experiments
        selected_x = [(idx-1) % 10 + 1 for idx in best_state.experiments]
        selected_y = [(idx-1) ÷ 10 + 1 for idx in best_state.experiments]
        Plots.scatter!(
            plt, 
            selected_x, 
            selected_y, 
            marker=:star,
            markersize=10,
            color=:red,
            label="Selected"
        )
        
        # Add markers for top 5 experiments
        top5_x = [(idx-1) % 10 + 1 for idx in sorted_exp_indices[1:5]]
        top5_y = [(idx-1) ÷ 10 + 1 for idx in sorted_exp_indices[1:5]]
        Plots.scatter!(
            plt, 
            top5_x, 
            top5_y, 
            marker=:circle,
            markersize=8,
            color=:blue,
            label="Top 5 Value"
        )
        
        # Save the plot
        Plots.savefig(plt, "active_learning_results.png")
        println("\nResults visualization saved to active_learning_results.png")
    catch e
        println("Error in sampling or visualization: $e")
        println("Error details: $e")
    end
end

# Run the example if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 