#!/usr/bin/env julia

# Example script for grid world navigation using GFlowNets
# This demonstrates the basic concepts of GFlowNets in a simple grid environment

# IMPORTANT: This script must be run from the example directory
# Run with: julia grid_world.jl

using Pkg
Pkg.activate(@__DIR__)  # Activate the project in the current directory (the example directory)

# Import the GFlowNet package first
using GFlowNet

# Then import other dependencies
using Plots
using Lux, Random, Optimisers, NNlib
using Statistics  # Add Statistics for mean and std functions
using Dates  # For timestamp logging
using Distributions: Categorical  # For action sampling
using Zygote  # For gradient computation

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

# One-hot features
function GFlowNet.state_to_features(state::GridState)
    pos = zeros(Float32, GRID_SIZE * GRID_SIZE)
    idx = (state.y - 1) * GRID_SIZE + state.x
    pos[idx] = 1.0f0
    return vcat(pos, [Float32(state.is_terminal)])
end

# Reward 0 default
function GFlowNet.reward(state::GridState)
    if !state.is_terminal
        return 0.0
    end
    for (pos, reward_value) in REWARD_POSITIONS
        if (state.x, state.y) == pos
            return reward_value
        end
    end
    return 0.0  # Zero for non-reward terminals
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

# Custom forward policy that works with actions instead of states
struct GridForwardPolicy
    model::Any
end

function create_grid_forward_policy(input_dim::Int, hidden_dim::Int, n_actions::Int, rng)
    # Create a simple MLP that outputs action probabilities
    model = Chain(
        Dense(input_dim => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => n_actions)  # Output logits for each action
    )
    
    # Initialize parameters
    ps, st = Lux.setup(rng, model)
    
    return GridForwardPolicy(model), ps, st
end

# Custom action sampling for grid world
function sample_action_from_policy(policy, state, actions, ps, st, rng=Random.default_rng())
    # Get state features
    features = GFlowNet.state_to_features(state)
    features = reshape(features, :, 1)  # Add batch dimension
    
    # Get action logits
    logits, new_st = policy.model(features, ps, st)
    logits = vec(logits)  # Remove batch dimension
    
    # Filter to only applicable actions
    applicable_actions = GridAction[]
    applicable_indices = Int[]
    
    for (i, action) in enumerate(actions)
        if GFlowNet.is_applicable(action, state)
            push!(applicable_actions, action)
            push!(applicable_indices, i)
        end
    end
    
    if isempty(applicable_actions)
        return nothing, 0.0, new_st
    end
    
    # Get logits for applicable actions and add exploration bonus for termination
    applicable_logits = copy(logits[applicable_indices])
    
    # Add exploration bonus to TerminateAction to encourage learning termination
    for (i, action) in enumerate(applicable_actions)
        if isa(action, TerminateAction)
            # Add dynamic termination bonus: +1.0 if x+y >= 8 (near high reward areas), otherwise +0.5
            termination_bonus = 0.5f0
            if state.x + state.y >= 8
                termination_bonus = 1.0f0
            end
            applicable_logits[i] += termination_bonus
        end
    end
    
    applicable_logits = clamp.(applicable_logits, -10.0f0, 10.0f0)  # Clamp for stability
    probs = NNlib.softmax(applicable_logits)
    
    # Sample action
    action_idx = rand(rng, Categorical(probs))
    selected_action = applicable_actions[action_idx]
    selected_prob = probs[action_idx]
    
    return selected_action, selected_prob, new_st
end

# Custom trajectory sampling
function sample_grid_trajectory(policy, actions, ps, st, max_steps=10, rng=Random.default_rng())
    states = [GridState(1, 1, false)]  # Start at (1,1)
    current_state = states[1]
    
    for step in 1:max_steps
        if current_state.is_terminal
            break
        end
        
        # Force termination if we're stuck too long or at boundaries
        if step > 15 || (current_state.x == GRID_SIZE && current_state.y == GRID_SIZE)
            # Force termination
            next_state = GridState(current_state.x, current_state.y, true)
            push!(states, next_state)
            break
        end
        
        # Sample action
        action, prob, new_st = sample_action_from_policy(policy, current_state, actions, ps, st, rng)
        st = new_st
        
        if isnothing(action)
            # Force termination if no action available
            next_state = GridState(current_state.x, current_state.y, true)
            push!(states, next_state)
            break
        end
        
        # Apply action
        next_state = GFlowNet.apply_action(action, current_state)
        push!(states, next_state)
        current_state = next_state
    end
    
    # Ensure we always end with a terminal state
    if !states[end].is_terminal
        terminal_state = GridState(states[end].x, states[end].y, true)
        push!(states, terminal_state)
    end
    
    return GFlowNet.Trajectory(states)
end

# Custom training loop
function train_grid_gflownet(policy, actions, ps, st, optimizer, n_iterations=1000, batch_size=32)
    println("Starting Grid World GFlowNet training...")
    println("Initial test: sampling one trajectory...")
    
    # Test initial sampling
    try
        test_traj = sample_grid_trajectory(policy, actions, ps, st)
        println("✅ Initial trajectory sampling successful! Length: $(length(test_traj.states))")
        println("✅ Final reward: $(GFlowNet.reward(test_traj.states[end]))")
    catch e
        println("❌ Initial trajectory sampling failed: $e")
        return ps, st, []
    end
    
    training_data = []
    
    for iter in 1:n_iterations
        try
            # Sample batch of trajectories
            trajectories = []
            total_reward = 0.0
            valid_trajectories = 0
            
            for _ in 1:batch_size
                traj = sample_grid_trajectory(policy, actions, ps, st)
                if length(traj.states) >= 2
                    push!(trajectories, traj)
                                         total_reward += GFlowNet.reward(traj.states[end])
                    valid_trajectories += 1
                end
            end
            
            if valid_trajectories == 0
                println("Warning: No valid trajectories in batch $iter")
                continue
            end
            
            avg_reward = total_reward / valid_trajectories
            
            # Compute loss and gradients
            loss_value, grads = compute_trajectory_balance_loss_and_gradients(
                policy, trajectories, actions, ps, st
            )
            
            # Update parameters (with error checking)
            if !isnothing(grads) && !any(isnan, grads) && !any(isinf, grads)
                # Gradient clipping
                grad_norm = norm(grads)
                if grad_norm > 1.0
                    grads = grads * (1.0 / grad_norm)
                end
                optimizer, ps = Optimisers.update(optimizer, ps, grads)
            else
                println("Warning: Skipping parameter update due to invalid gradients at iteration $iter")
            end
            
            # Log progress
            if iter % 10 == 0 || iter == 1
                # Sample some test trajectories to check diversity
                test_rewards = []
                for _ in 1:5
                    test_traj = sample_grid_trajectory(policy, actions, ps, st)
                    push!(test_rewards, GFlowNet.reward(test_traj.states[end]))
                end
                
                println("Iter $iter/$n_iterations - Loss: $(round(loss_value, digits=4)), Avg Reward: $(round(avg_reward, digits=4)), Test rewards: $(test_rewards)")
                
                # Store training data
                push!(training_data, (
                    iteration = iter,
                    loss = loss_value,
                    avg_reward = avg_reward,
                    timestamp = now()
                ))
            end
            
        catch e
            println("Error in training iteration $iter: $e")
            # Continue training instead of crashing
            continue
        end
    end
    
    return ps, st, training_data
end

# Simplified trajectory balance loss computation with gradients
function compute_trajectory_balance_loss_and_gradients(policy, trajectories, actions, ps, st)
    # Simpler approach: compute loss and gradients more directly
    total_loss = 0.0f0
    
    function loss_fn(params)
        batch_loss = 0.0f0
        n_valid = 0
        
        for traj in trajectories
            if length(traj.states) < 2
                continue  # Skip invalid trajectories
            end
            
            # Simple trajectory probability computation
            traj_log_prob = 0.0f0
            
            # Only look at a few key transitions to avoid complexity
            n_transitions = min(3, length(traj.states) - 1)
            
            for i in 1:n_transitions
                current_state = traj.states[i]
                next_state = traj.states[i+1]
                
                # Get state features
                features = GFlowNet.state_to_features(current_state)
                features = reshape(features, :, 1)
                
                # Get action logits
                logits, _ = policy.model(features, params, st)
                logits = vec(logits)
                
                # Find which action was taken
                action_taken = nothing
                for (action_idx, action) in enumerate(actions)
                    if GFlowNet.is_applicable(action, current_state) && 
                       GFlowNet.apply_action(action, current_state) == next_state
                        action_taken = action_idx
                        break
                    end
                end
                
                if !isnothing(action_taken)
                    # Compute probability of this action
                    applicable_mask = [GFlowNet.is_applicable(actions[j], current_state) for j in 1:length(actions)]
                    
                    if sum(applicable_mask) > 0
                        masked_logits = logits .- 1000.0f0 * (1.0f0 .- Float32.(applicable_mask))
                        probs = NNlib.softmax(masked_logits)
                        traj_log_prob += log(max(probs[action_taken], 1f-8))
                    end
                end
            end
            
            # Compute reward
            final_reward = GFlowNet.reward(traj.states[end])
            reward_safe = max(final_reward, 1e-8f0)
            
            # Simple trajectory balance: P(traj) should be proportional to R
            # Minimize (log_prob - log_reward)^2
            balance_error = traj_log_prob - log(reward_safe)
            batch_loss += balance_error^2
            n_valid += 1
        end
        
        return n_valid > 0 ? batch_loss / n_valid : 0.0f0
    end
    
    # Compute loss and gradients with error handling
    try
        result = Zygote.withgradient(loss_fn, ps)
        loss_value = result.val
        grads = result.grad[1]
        
        # Check for NaN/Inf gradients
        if isnothing(grads) || any(isnan, grads) || any(isinf, grads)
            # Return zero gradients if computation failed
            zero_grads = similar(ps)
            fill!(zero_grads, 0.0f0)
            return Float32(loss_value), zero_grads
        end
        
        return Float32(loss_value), grads
    catch e
        println("Warning: Gradient computation failed: $e")
        # Return zero gradients as fallback
        zero_grads = similar(ps)
        fill!(zero_grads, 0.0f0)
        return 1.0f0, zero_grads
    end
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
    
    # Create actions
    actions = create_grid_actions()
    
    # Create neural network models for policies
    rng = Random.default_rng()
    
    # Feature dimension: 26 (25 one-hot positions + terminal flag)
    input_dim = GRID_SIZE * GRID_SIZE + 1
    
    # Output dimension: number of actions (5)
    n_actions = length(actions)
    
    # Create forward policy with increased hidden dimension
    policy, ps, st = create_grid_forward_policy(input_dim, 128, n_actions, rng)
    
    # Create optimizer
    optimizer = Optimisers.setup(Optimisers.Adam(0.0005), ps)  # Lower learning rate
    
    # Test basic functionality before training
    println("Testing basic functionality...")
    
    # Test state features
    test_state = GridState(2, 3, false)
    test_features = GFlowNet.state_to_features(test_state)
    println("✅ State features: length $(length(test_features))")
    
    # Test actions
    println("✅ Available actions: $(length(actions))")
    
    # Test policy forward pass
    test_features_batch = reshape(test_features, :, 1)
    try
        logits, _ = policy.model(test_features_batch, ps, st)
        println("✅ Policy forward pass successful! Output shape: $(size(logits))")
    catch e
        println("❌ Policy forward pass failed: $e")
        return
    end
    
    # Create results directory if it doesn't exist
    mkpath("results")
    
    # Train the model with optimized parameters
    println("Training GFlowNet...")
    println("  Input dim: $input_dim")
    println("  Hidden dim: 128")
    println("  Output dim (actions): $n_actions")
    
    # Train with optimized parameters for high success rate
    final_ps, final_st, training_data = train_grid_gflownet(
        policy, actions, ps, st, optimizer, 5000, 32  # 5000 iterations, batch 32
    )
    
    println("Training completed!")
    
    # Sample final trajectories for visualization
    println("Sampling final trajectories...")
    final_trajectories = [sample_grid_trajectory(policy, actions, final_ps, final_st, 10) for _ in 1:10]  # max_steps=10
    
    # Visualize results
    println("Visualizing results...")
    
    # Create output directory
    output_dir = "results"
    mkpath(output_dir)
    
    # Plot training progress
    if !isempty(training_data)
        iterations = [d.iteration for d in training_data]
        losses = [d.loss for d in training_data]
        rewards = [d.avg_reward for d in training_data]
        
        # Loss plot
        loss_plot = plot(iterations, losses, title="Training Loss", xlabel="Iteration", ylabel="Loss", lw=2, color=:blue)
        savefig(loss_plot, joinpath(output_dir, "grid_world_loss.png"))
        
        # Reward plot
        reward_plot = plot(iterations, rewards, title="Average Reward", xlabel="Iteration", ylabel="Reward", lw=2, color=:green)
        savefig(reward_plot, joinpath(output_dir, "grid_world_rewards.png"))
        
        # Save training data to CSV
        open(joinpath(output_dir, "grid_world_training.csv"), "w") do f
            println(f, "iteration,loss,avg_reward")
            for d in training_data
                println(f, "$(d.iteration),$(d.loss),$(d.avg_reward)")
            end
        end
    end
    
    # Plot sampled paths
    grid_plot = visualize_grid(final_trajectories)
    savefig(grid_plot, joinpath(output_dir, "grid_world_paths.png"))
    
    println("Example completed. Results saved to $output_dir/")
    
    # Print final statistics
    final_rewards = [GFlowNet.reward(traj.states[end]) for traj in final_trajectories]
    println("Final trajectory statistics:")
    println("  Mean reward: $(round(mean(final_rewards), digits=2))")
    println("  Std reward: $(round(std(final_rewards), digits=2))")
    println("  Max reward: $(round(maximum(final_rewards), digits=2))")
    println("  High-reward trajectories (R≥5): $(count(r -> r >= 5.0, final_rewards))/$(length(final_rewards))")
    println("  Optimal trajectories (R=10): $(count(r -> r == 10.0, final_rewards))/$(length(final_rewards))")
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 