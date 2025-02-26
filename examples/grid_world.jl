#!/usr/bin/env julia

# Example script for grid world navigation using GFlowNets
# This demonstrates the basic concepts of GFlowNets in a simple environment

# IMPORTANT: This script must be run from the project root directory
# Run with: julia examples/grid_world.jl

using Pkg
Pkg.activate(".")  # Activate the project in the current directory (should be the project root)

using GFlowNet
using GFlowNet.GFlowNetUtils
using Plots
using Lux, Random, Optimisers, NNlib
using Statistics  # Add Statistics for mean and std functions

# Define the grid world state and action types
struct GridState <: GFlowNet.AbstractState
    x::Int  # x-coordinate
    y::Int  # y-coordinate
    is_terminal::Bool
end

abstract type GridAction <: GFlowNet.AbstractAction end

struct MoveRightAction <: GridAction end
struct MoveLeftAction <: GridAction end
struct MoveUpAction <: GridAction end
struct MoveDownAction <: GridAction end
struct TerminateAction <: GridAction end

# Define grid world parameters
const GRID_SIZE = 5
const REWARD_POSITIONS = [(5, 5) => 10.0, (3, 4) => 5.0, (2, 2) => 2.0]

# Implementation of required interface functions
function GFlowNet.is_applicable(action::MoveRightAction, state::GridState)
    !state.is_terminal && state.x < GRID_SIZE
end

function GFlowNet.is_applicable(action::MoveLeftAction, state::GridState)
    !state.is_terminal && state.x > 1
end

function GFlowNet.is_applicable(action::MoveUpAction, state::GridState)
    !state.is_terminal && state.y < GRID_SIZE
end

function GFlowNet.is_applicable(action::MoveDownAction, state::GridState)
    !state.is_terminal && state.y > 1
end

function GFlowNet.is_applicable(action::TerminateAction, state::GridState)
    !state.is_terminal
end

function GFlowNet.apply_action(action::MoveRightAction, state::GridState)
    GridState(state.x + 1, state.y, false)
end

function GFlowNet.apply_action(action::MoveLeftAction, state::GridState)
    GridState(state.x - 1, state.y, false)
end

function GFlowNet.apply_action(action::MoveUpAction, state::GridState)
    GridState(state.x, state.y + 1, false)
end

function GFlowNet.apply_action(action::MoveDownAction, state::GridState)
    GridState(state.x, state.y - 1, false)
end

function GFlowNet.apply_action(action::TerminateAction, state::GridState)
    GridState(state.x, state.y, true)
end

function GFlowNet.state_to_features(state::GridState)
    # Calculate the position index
    pos_idx = (state.x - 1) * GRID_SIZE + state.y
    grid_size_sq = GRID_SIZE * GRID_SIZE
    
    # Use vcat and zeros to create features (completely non-mutating)
    # Create a one-hot vector for position
    features = vcat(
        # Position feature (one hot)
        Float32.([(i == pos_idx) for i in 1:grid_size_sq]),
        # Terminal state feature
        Float32[state.is_terminal]
    )
    
    return features
end

function GFlowNet.reward(state::GridState)
    if !state.is_terminal
        return 0.0
    end
    
    # Return the reward for the current position if it's in REWARD_POSITIONS
    for (pos, reward_value) in REWARD_POSITIONS
        if (state.x, state.y) == pos
            return reward_value
        end
    end
    
    # Default reward for terminal states not in REWARD_POSITIONS
    return 0.0
end

# Create grid world actions
function create_grid_actions()
    actions = GridAction[
        MoveRightAction(),
        MoveLeftAction(),
        MoveUpAction(),
        MoveDownAction(),
        TerminateAction()
    ]
    return actions
end

# Helper function to visualize the grid world
function visualize_grid(trajectories=nothing; show_rewards=true)
    p = plot(
        title="Grid World",
        xlim=(0.5, GRID_SIZE + 0.5),
        ylim=(0.5, GRID_SIZE + 0.5),
        xticks=1:GRID_SIZE,
        yticks=1:GRID_SIZE,
        aspect_ratio=:equal,
        legend=true,
        grid=true
    )
    
    # Draw grid lines
    for i in 1:GRID_SIZE
        plot!(p, [0.5, GRID_SIZE + 0.5], [i + 0.5, i + 0.5], color=:gray, alpha=0.5, label=nothing)
        plot!(p, [i + 0.5, i + 0.5], [0.5, GRID_SIZE + 0.5], color=:gray, alpha=0.5, label=nothing)
    end
    
    # Show rewards
    if show_rewards
        for ((x, y), r) in REWARD_POSITIONS
            annotate!(p, x, y, text("R=$r", 8, :black))
            scatter!(p, [x], [y], color=:gold, markersize=20, alpha=0.5, label=nothing)
        end
    end
    
    # Plot trajectories if provided
    if !isnothing(trajectories)
        for (i, trajectory) in enumerate(trajectories)
            xs = [state.x for state in trajectory.states]
            ys = [state.y for state in trajectory.states]
            
            # Only show first 5 trajectories in legend
            label = i <= 5 ? "Path $i" : nothing
            
            plot!(p, xs, ys, color=i, linewidth=2, label=label, marker=:circle, markersize=4)
        end
    end
    
    return p
end

# Main function to run the example
function main()
    println("Setting up Grid World GFlowNet example...")
    
    # Create states and actions
    initial_state = GridState(1, 1, false)
    
    # Terminal states are all grid positions when terminated
    terminal_states = [GridState(x, y, true) for x in 1:GRID_SIZE for y in 1:GRID_SIZE]
    
    # Terminal sink state (special state to collect all terminal states)
    terminal_sink = GridState(0, 0, true)
    
    # Create actions
    actions = create_grid_actions()
    
    # Create DAG
    dag = GFlowNet.create_dag(initial_state, terminal_states, terminal_sink, actions)
    
    # Create neural network models for policies
    rng = Random.default_rng()
    
    # Feature dimension is grid_size^2 + 1 (for terminal flag)
    input_dim = GRID_SIZE * GRID_SIZE + 1
    
    # Output dimension is the number of possible states
    output_dim = length(dag.states)
    
    # Create forward policy
    forward_policy, forward_ps, forward_st = GFlowNet.create_forward_policy(
        input_dim, 64, output_dim, rng
    )
    
    # Create flow estimator
    flow_estimator, flow_ps, flow_st = GFlowNet.create_flow_estimator(
        input_dim, 64, rng
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
    logger = GFlowNetLogger("grid_world_training.csv", log_frequency=10, verbose=true)
    
    # Train the model
    println("Training GFlowNet...")
    n_iterations = 1000
    batch_size = 32
    
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
    savefig(loss_plot, "grid_world_loss.png")
    
    # Sample and visualize trajectories
    n_samples = 10
    sampled_trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:n_samples]
    grid_plot = visualize_grid(sampled_trajectories)
    savefig(grid_plot, "grid_world_paths.png")
    
    # Plot reward distribution
    reward_plot = visualize_reward_distribution(model, 100)
    savefig(reward_plot, "grid_world_rewards.png")
    
    println("Example completed. Results saved to grid_world_*.png")
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 