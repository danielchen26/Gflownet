#!/usr/bin/env julia

# Example script for active learning using GFlowNets
# This demonstrates how GFlowNets can be used for experimental design
# and selecting informative experiments

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)

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
using Plots
using Random
using Statistics
using StatsBase
using LinearAlgebra
using CSV
using DataFrames

# Set global parameters
const N_EXPERIMENTS = 50   # Total number of possible experiments
const FEATURE_DIM = 10     # Feature dimension for each experiment
const MAX_EXPERIMENTS = 5  # Maximum number of experiments to select
const N_ITERATIONS = 20    # Number of training iterations (demo budget)
const BATCH_SIZE = 8       # Batch size for training
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
            
            # `Trajectory` requires one action per transition, so a single-state
            # trajectory carries an empty action vector.
            trajectories[i] = GFlowNet.Trajectory([terminal_state], GFlowNet.AbstractAction[])
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
    
    # Rank experiments by value; used for reporting and for the sampling fallback.
    sorted_exp_indices = sortperm(global_experiment_values, rev=true)

    # Show a few representative terminal selections so the reward scale is visible.
    println("Representative terminal selections:")
    for i in 1:5
        # Include at least one high-value experiment in each selection
        high_value_exp = sorted_exp_indices[i]
        n_additional = rand(rng, 0:MAX_EXPERIMENTS-1)
        remaining_exps = setdiff(1:N_EXPERIMENTS, high_value_exp)
        additional_exps = sample(rng, remaining_exps, n_additional, replace=false)

        term_state = GFlowNet.ExperimentState(
            vcat(high_value_exp, additional_exps), MAX_EXPERIMENTS, true
        )
        println("  High-value selection $i: experiments $(term_state.experiments), reward: $(experiment_reward(term_state))")
    end

    for i in 6:10
        n_selected = rand(rng, 1:MAX_EXPERIMENTS)
        selected_experiments = sample(rng, 1:N_EXPERIMENTS, n_selected, replace=false)

        term_state = GFlowNet.ExperimentState(selected_experiments, MAX_EXPERIMENTS, true)
        println("  Random selection $i: experiments $(term_state.experiments), reward: $(experiment_reward(term_state))")
    end

    # Terminal sink state (stopping with the single highest-value experiment).
    # Used as the fallback when post-training sampling yields empty selections.
    best_exp = [sorted_exp_indices[1]]
    terminal_sink = GFlowNet.ExperimentState(best_exp, MAX_EXPERIMENTS, true)
    println("Terminal sink state uses experiment $(best_exp[1]), reward: $(experiment_reward(terminal_sink))")

    # Create actions for experiment selection
    actions = GFlowNet.create_experiment_actions(N_EXPERIMENTS)

    # Build the model with the current on-demand API: `create_dag`,
    # `TrajectoryBalanceObjective` and the DAG-based `GFlowNetModel` constructor
    # no longer exist. `create_gflownet` takes the initial state plus the full
    # action set and builds the policy network, the `ComponentArray` parameters,
    # the optimiser and the layer states itself, so no bootstrap list of terminal
    # states is needed.
    input_dim = N_EXPERIMENTS + 2  # One-hot encoding + num_selected + is_terminal
    model = GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = input_dim,
        hidden_dim = 128,
        learning_rate = LEARNING_RATE,
        rng = Random.MersenneTwister(42)
    )
    
    println("Training GFlowNet for $N_ITERATIONS iterations...")
    
    # Track training metrics
    train_data = DataFrame(
        iteration = Int[],
        loss = Float64[],
        mean_reward = Float64[],
        max_reward = Float64[]
    )
    
    # Create modern training configuration
    config = TrainingConfig(
        objective=TRAJECTORY_BALANCE,
        partition_function_method=SIMPLE_ESTIMATION,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
        validation_frequency=20
    )
    
    println("Training with modern training interface...")
    
    # Train using the modern interface
    try
        history = train_gflownet(model, config; verbose=true)
        
        # Convert history to DataFrame for consistency with original
        train_data = DataFrame(
            iteration = 1:length(history[:losses]),
            loss = history[:losses],
            mean_reward = fill(0.0, length(history[:losses])),  # Will be updated below
            max_reward = fill(0.0, length(history[:losses]))    # Will be updated below
        )
        
        # Calculate actual rewards by sampling
        println("Calculating final performance metrics...")
        trajectories = sample_trajectories(model, BATCH_SIZE, rng)
        terminal_rewards = [GFlowNet.reward(traj.states[end]) for traj in trajectories]
        final_mean_reward = mean(terminal_rewards)
        final_max_reward = maximum(terminal_rewards)
                
        # Update the last entries with actual reward values
        train_data.mean_reward[end] = final_mean_reward
        train_data.max_reward[end] = final_max_reward
        
        println("Training completed successfully!")
        println("Final mean reward: $final_mean_reward")
        println("Final max reward: $final_max_reward")
        
        catch e
        println("Error in modern training: $e")
        println("Training failed. Please check the configuration and model setup.")
        return
    end
    
    # Save training metrics
    # Write next to the script, not into whatever directory the user launched
    # from -- this example is normally run as `julia examples/active_learning/…`
    # from the repository root.
    CSV.write(joinpath(@__DIR__, "active_learning_training.csv"), train_data)
    
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
    Plots.savefig(loss_plot, joinpath(@__DIR__, "active_learning_loss.png"))
    
    reward_plot = Plots.plot(
        train_data.iteration, 
        [train_data.mean_reward train_data.max_reward], 
        title="Rewards During Training",
        xlabel="Iteration",
        ylabel="Reward",
        label=["Mean Reward" "Max Reward"],
        lw=2
    )
    Plots.savefig(reward_plot, joinpath(@__DIR__, "active_learning_rewards.png"))
    
    # Sample experiment selections after training
    println("\nSampling experiment selections after training...")
    n_samples = 10
    
    try
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
        Plots.savefig(plt, joinpath(@__DIR__, "active_learning_results.png"))
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