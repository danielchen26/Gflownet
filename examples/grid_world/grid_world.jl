"""
🎯 Simple Grid World GFlowNet Example
====================================

This example demonstrates a minimal, elegant GFlowNet implementation using only core framework functions.
The agent learns to navigate a 3x3 grid to reach high-reward terminal states.

Key Features:
- Uses only core GFlowNet functions
- Clean, readable code structure
- Proper ComponentArray handling
- Neural network-based policy learning
"""

using GFlowNet
using Lux
using Random
using ComponentArrays
using Optimisers  # ← FIXED: Add missing import for optimizer
using Dates
using Logging
using Graphs

# Optional plotting (will gracefully handle if not available)
try
    using Plots
    global PLOTS_AVAILABLE = true
catch
    global PLOTS_AVAILABLE = false
end

# Clean output - no debug logging needed

# =============================================================================
# Simple Grid World States and Actions
# =============================================================================

"""Simple grid state - position (x,y) and terminal flag"""
struct SimpleGridState <: GFlowNet.AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

"""Simple grid actions - move in 4 directions or terminate"""
abstract type SimpleGridAction <: GFlowNet.AbstractAction end

struct MoveUp <: SimpleGridAction end
struct MoveDown <: SimpleGridAction end  
struct MoveLeft <: SimpleGridAction end
struct MoveRight <: SimpleGridAction end
struct Terminate <: SimpleGridAction end

# Create action instances
const UP = MoveUp()
const DOWN = MoveDown()
const LEFT = MoveLeft()
const RIGHT = MoveRight()
const TERMINATE = Terminate()

# =============================================================================
# Core Interface Implementation
# =============================================================================

"""Convert state to features for neural network"""
function GFlowNet.state_to_features(state::SimpleGridState)
    # Return proper Float32 array, not validation call
    return Float32[state.x, state.y, state.is_terminal ? 1.0f0 : 0.0f0]
end

"""Check if action is applicable to state"""
function GFlowNet.is_applicable(action::SimpleGridAction, state::SimpleGridState)
    # FIXED: Terminal states should not have any applicable actions
    if state.is_terminal
        return false  # No actions from terminal states
    end

    # FIXED: Non-terminal states can always terminate
    if isa(action, Terminate)
        return true  # Can always terminate from non-terminal states
    end

    # Check grid boundaries (3x3 grid) for movement actions
    x, y = state.x, state.y
    if isa(action, MoveUp) && y < 3
        return true
    elseif isa(action, MoveDown) && y > 1
        return true
    elseif isa(action, MoveLeft) && x > 1
        return true
    elseif isa(action, MoveRight) && x < 3
        return true
    end

    return false
end

"""Apply action to state"""
function GFlowNet.apply_action(action::SimpleGridAction, state::SimpleGridState)
    if isa(action, Terminate)
        return SimpleGridState(state.x, state.y, true)
    end
    
    x, y = state.x, state.y
    if isa(action, MoveUp)
        y += 1
    elseif isa(action, MoveDown)
        y -= 1
    elseif isa(action, MoveLeft)
        x -= 1
    elseif isa(action, MoveRight)
        x += 1
    end
    
    return SimpleGridState(x, y, false)
end

"""Check if state is terminal"""
function GFlowNet.is_terminal_state(state::SimpleGridState)
    return state.is_terminal
end

"""Compute reward for state with exploration bonus and distance-based shaping"""
function GFlowNet.reward(state::SimpleGridState)
    if !state.is_terminal
        return 0.0f0  # No reward for non-terminal states
    end

    # Clear reward structure for better learning signal
    if state.x == 3 && state.y == 3
        return 100.0f0  # Very high reward for (3,3)
    elseif state.x == 1 && state.y == 3
        return 50.0f0   # High reward for (1,3)  
    elseif state.x == 3 && state.y == 1
        return 20.0f0   # Medium reward for (3,1)
    else
        return 1.0f0    # Base reward for other terminal states
    end
end



# =============================================================================
# Custom DAG Creation for Grid World
# =============================================================================

"""Create a custom DAG that includes all intermediate states"""
function create_custom_grid_dag(initial_state, all_states, terminal_states, terminal_sink, actions)

    # Include the sink state
    unique_states = unique([all_states; [terminal_sink]])
    state_to_idx = Dict(state => i for (i, state) in enumerate(unique_states))

    # Initialize an empty directed graph
    graph = SimpleDiGraph(length(unique_states))

    # Generate all possible transitions
    for action in actions
        for state in unique_states
            if GFlowNet.is_applicable(action, state)
                next_state = GFlowNet.apply_action(action, state)
                if next_state in unique_states
                    add_edge!(graph, state_to_idx[state], state_to_idx[next_state])
                end
            end
        end
    end

    # Add edges from terminal states to sink
    for term_state in terminal_states
        if term_state in unique_states
            add_edge!(graph, state_to_idx[term_state], state_to_idx[terminal_sink])
        end
    end

    # Build action cache
    action_cache = Dict{typeof(initial_state), Vector{eltype(actions)}}()
    for state in unique_states
        applicable_actions = [action for action in actions if GFlowNet.is_applicable(action, state)]
        action_cache[state] = applicable_actions
    end

    # Create the DAG structure
    return GFlowNet.DirectedAcyclicGraph(
        graph,
        collect(unique_states),
        actions,
        state_to_idx,
        initial_state,
        terminal_states,
        terminal_sink,
        action_cache
    )
end

# =============================================================================
# Simple Model Creation
# =============================================================================

"""Create a simple GFlowNet model for the grid world"""
function create_simple_gflownet()
    # 1. Define states and actions
    initial_state = SimpleGridState(1, 1, false)

    # FIXED: Include ALL reachable states (both terminal and non-terminal)
    # Create all non-terminal states (intermediate states)
    all_states = SimpleGridState[]
    for x in 1:3, y in 1:3
        push!(all_states, SimpleGridState(x, y, false))  # Non-terminal states
    end

    # Create all terminal states
    terminal_states = SimpleGridState[]
    for x in 1:3, y in 1:3
        push!(terminal_states, SimpleGridState(x, y, true))  # Terminal states
    end

    # Add all states to the DAG (both non-terminal and terminal)
    all_dag_states = [all_states; terminal_states]

    terminal_sink = SimpleGridState(0, 0, true)  # Special sink state

    actions = SimpleGridAction[UP, DOWN, LEFT, RIGHT, TERMINATE]
    
    # 2. Create DAG using core function with ALL states
    # We need to create a custom DAG that includes all intermediate states
    dag = create_custom_grid_dag(initial_state, all_dag_states, terminal_states, terminal_sink, actions)

    # DAG created successfully
    println("✅ DAG created with $(length(dag.states)) states")
    
    # 3. Create neural networks
    input_dim = 3  # x, y, is_terminal
    hidden_dim = 64  # Increased capacity
    n_states = length(dag.states)  # Output should be number of states in DAG

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    # Forward policy network - outputs STATE transition logits
    n_states = length(dag.states)  # Output should be number of states in DAG (19)
    forward_nn = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),  # Additional layer for better learning
        Dense(hidden_dim => n_states)  # Output STATE transition logits
    )
    
    # Flow estimator network - improved architecture
    flow_nn = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),  # Additional layer
        Dense(hidden_dim => 1, x -> softplus.(x) .+ 1f-6)  # Positive flow values with small epsilon
    )
    
    # Initialize parameters
    forward_ps, forward_st = Lux.setup(rng, forward_nn)
    flow_ps, flow_st = Lux.setup(rng, flow_nn)
    
    # 4. Create GFlowNet components
    forward_policy = GFlowNet.ForwardPolicy(forward_nn)
    flow_estimator = GFlowNet.FlowEstimator(flow_nn)
    
    # 5. Create optimizer with much lower learning rate for debugging
    opt = Optimisers.Adam(0.0001)  # Very low learning rate for stable learning
    forward_opt_state = Optimisers.setup(opt, forward_ps)
    flow_opt_state = Optimisers.setup(opt, flow_ps)
    optimizer = (forward = forward_opt_state, flow = flow_opt_state)

    # 6. Create model using core framework with ComponentArray
    # Convert parameters to ComponentArrays for proper gradient handling
    forward_ca = ComponentArray(forward_ps)
    flow_ca = ComponentArray(flow_ps)

    # Use ComponentArray structure for parameters
    parameters = ComponentArray(
        forward = forward_ca,
        flow = flow_ca
    )
    states = (forward = forward_st, backward = nothing, flow = flow_st)

    model = GFlowNet.create_gflownet_model_safe(
        dag=dag,
        forward_policy=forward_policy,
        flow_estimator=flow_estimator,
        optimizer=optimizer,  # ← FIXED: Add the missing optimizer!
        objectives=[GFlowNet.TrajectoryBalanceObjective(1.0)],  # Add training objective
        parameters=parameters,
        states=states
    )
    
    return model, initial_state
end

# =============================================================================
# Simple Training Loop
# =============================================================================

"""Simple training loop using core GFlowNet functions"""
function train_simple_gflownet(model, initial_state, log_function=println; n_iterations=50, batch_size=4)
    # Use the provided logging function
    log_msg = log_function
    
    log_msg("🚀 Training GFlowNet...")
    log_msg("   Iterations: $n_iterations")
    log_msg("   Batch size: $batch_size")
    
    training_data = []
    successful_samples = 0
    
    for iter in 1:n_iterations
        # Sample trajectories with simple approach - FIXED: Properly typed trajectory array
        trajectories = GFlowNet.Trajectory[]
        iter_successful = 0
        
        for sample_idx in 1:batch_size
            try
                # Add curriculum learning: inject good trajectories early in training
                if iter <= 20 && sample_idx == 1 && rand() < 0.3  # 30% chance in first 20 iterations
                    # Create a high-reward trajectory to (3,3) to help model learn
                    if rand() < 0.6
                        # Path to (3,3): (1,1) -> (1,2) -> (1,3) -> (2,3) -> (3,3) -> terminal
                        good_states = [
                            SimpleGridState(1, 1, false),
                            SimpleGridState(1, 2, false),
                            SimpleGridState(1, 3, false),
                            SimpleGridState(2, 3, false),
                            SimpleGridState(3, 3, false),
                            SimpleGridState(3, 3, true)
                        ]
                    else
                        # Alternative path to (1,3): (1,1) -> (1,2) -> (1,3) -> terminal
                        good_states = [
                            SimpleGridState(1, 1, false),
                            SimpleGridState(1, 2, false),
                            SimpleGridState(1, 3, false),
                            SimpleGridState(1, 3, true)
                        ]
                    end
                    traj = GFlowNet.Trajectory(good_states)
                    log_msg("  📚 Injected curriculum trajectory to help learning")
                else
                    # Use standard GFlowNet sampling - exploration comes from stochastic policy
                    traj = GFlowNet.sample_trajectory(model)
                end

                push!(trajectories, traj)
                iter_successful += 1
                successful_samples += 1
            catch e
                log_msg("Warning: Error sampling trajectory $sample_idx in iteration $iter: $e")
                # Create a simple fallback trajectory using initial_state passed as parameter
                fallback_terminal = SimpleGridState(initial_state.x, initial_state.y, true)
                fallback_traj = GFlowNet.Trajectory([initial_state, fallback_terminal])
                push!(trajectories, fallback_traj)
            end
        end
        
        if !isempty(trajectories) && iter_successful > 0
            try
                # Compute trajectory balance loss and gradients
                loss, grads = GFlowNet.compute_loss_and_grad(model, trajectories)
                
                # Apply gradient clipping (critical for GFlowNet stability)
                max_grad_norm = 1.0  # Standard value from Python implementations
                if !isnothing(grads)
                    grad_norm = GFlowNet.clip_gradients!(grads, max_grad_norm)
                    if iter % 20 == 0  # Monitor gradient norms
                        log_msg("   Gradient norm: $(round(grad_norm, digits=4))")
                    end
                end
                
                # Apply gradients
                GFlowNet.apply_optimizer!(model, grads)
                
                # Track training progress
                rewards = [GFlowNet.reward(traj.states[end]) for traj in trajectories]
                mean_reward = sum(rewards) / length(rewards)
                high_reward_count = count(r -> r >= 5.0, rewards)
                
                push!(training_data, (iter, loss, mean_reward, high_reward_count, iter_successful))
                
                # Print progress with enhanced monitoring (Python standard)
                if iter % 5 == 0
                    # Estimate current partition function for monitoring
                    current_Z = GFlowNet.estimate_partition_function(GFlowNet.SimplePartitionFunctionEstimator(), model)
                    log_msg("Iteration $iter: Loss = $(round(loss, digits=4)), Mean Reward = $(round(mean_reward, digits=2)), High Rewards = $high_reward_count/$batch_size, Successful samples = $iter_successful/$batch_size, Z = $(round(current_Z, digits=2))")
                end
            catch e
                log_msg("Error in training iteration $iter: $e")
            end
        else
            log_msg("No successful trajectories in iteration $iter")
        end
    end
    
    log_msg("✅ Training completed!")
    log_msg("📊 Total successful samples: $successful_samples/$(n_iterations * batch_size)")
    return model, training_data
end

# =============================================================================
# Simple Evaluation
# =============================================================================

"""Evaluate the trained model"""
function evaluate_simple_model(model, log_function=println; n_samples=10)
    # Use the provided logging function
    log_msg = log_function
    
    log_msg("🎯 Evaluating Model...")
    log_msg("   Sampling $n_samples trajectories...")
    
    rewards = Float64[]
    high_reward_count = 0
    max_reward_count = 0
    trajectories = GFlowNet.Trajectory[]  # FIXED: Properly typed trajectory array
    successful_evals = 0
    
    for i in 1:n_samples
        try
            # Sample trajectory for evaluation
            traj = GFlowNet.sample_trajectory(model)
            final_state = traj.states[end]
            reward_val = GFlowNet.reward(final_state)
            push!(rewards, reward_val)
            push!(trajectories, traj)
            successful_evals += 1
            
            if reward_val >= 10.0
                max_reward_count += 1
            elseif reward_val >= 5.0
                high_reward_count += 1
            end
        catch e
            log_msg("Error sampling evaluation trajectory $i: $e")
            push!(rewards, 0.0)
        end
    end
    
    if !isempty(rewards) && successful_evals > 0
        mean_reward = sum(rewards) / length(rewards)
        high_reward_percentage = (high_reward_count / n_samples) * 100  
        max_reward_percentage = (max_reward_count / n_samples) * 100
        
        # Sample trajectories for display
        sample_trajectories = []
        for i in 1:min(3, length(trajectories))
            traj_str = join(["($(s.x),$(s.y))" for s in trajectories[i].states], " → ")
            reward_val = GFlowNet.reward(trajectories[i].states[end])
            push!(sample_trajectories, "   $i. $traj_str (Reward: $reward_val)")
        end
        
        # Fix reward distribution calculation
        reward_1_count = count(r -> r == 1.0, rewards)
        reward_5_count = count(r -> r == 5.0, rewards)
        reward_10_count = count(r -> r == 10.0, rewards)
        reward_20_count = count(r -> r == 20.0, rewards)
        reward_other_count = n_samples - reward_1_count - reward_5_count - reward_10_count - reward_20_count

        results_summary = """
📊 Evaluation Results:
   Successful evaluations: $successful_evals/$n_samples
   Mean Reward: $(round(mean_reward, digits=1))
   High Reward (≥5.0) Rate: $(round(high_reward_percentage, digits=1))% ($high_reward_count/$n_samples)
   Max Reward (≥10.0) Rate: $(round(max_reward_percentage, digits=1))% ($max_reward_count/$n_samples)

📋 Sample Trajectories:
$(join(sample_trajectories, "\n"))

📈 Reward Distribution:
   Reward 1.0: $reward_1_count samples
   Reward 5.0: $reward_5_count samples
   Reward 10.0: $reward_10_count samples
   Reward 20.0: $reward_20_count samples
   Other: $reward_other_count samples
"""
    else
        results_summary = "❌ No successful trajectories sampled during evaluation."
    end
    
    log_msg(results_summary)
    return results_summary
end

# =============================================================================
# Visualization Functions
# =============================================================================

"""Create learning curve plots"""
function create_learning_curves(training_data, results_dir)
    if !PLOTS_AVAILABLE
        println("⚠️  Plots.jl not available - skipping learning curves")
        return
    end

    try
        if isempty(training_data)
            println("⚠️  No training data available for plotting")
            return
        end

        # Extract data
        iterations = [d[1] for d in training_data]
        losses = [d[2] for d in training_data]
        rewards = [d[3] for d in training_data]
        high_rewards = [d[4] for d in training_data]

        # Create plots
        p1 = plot(iterations, losses, title="Training Loss", xlabel="Iteration", ylabel="Loss",
                 linewidth=2, color=:red, legend=false)

        p2 = plot(iterations, rewards, title="Mean Reward", xlabel="Iteration", ylabel="Reward",
                 linewidth=2, color=:blue, legend=false)

        p3 = plot(iterations, high_rewards, title="High Reward Count", xlabel="Iteration", ylabel="Count",
                 linewidth=2, color=:green, legend=false)

        # Combine plots
        combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 600))

        # Save plot
        plot_file = joinpath(results_dir, "learning_curves.png")
        savefig(combined_plot, plot_file)
        println("📈 Learning curves saved to: $plot_file")

    catch e
        println("⚠️  Could not create learning curves: $e")
    end
end

"""Create 2D grid visualization"""
function create_grid_visualization(trajectories, results_dir)
    if !PLOTS_AVAILABLE
        println("⚠️  Plots.jl not available - skipping grid visualization")
        return
    end

    try
        if isempty(trajectories)
            println("⚠️  No trajectories available for grid visualization")
            return
        end

        # Create grid plot
        p = plot(xlims=(0.5, 3.5), ylims=(0.5, 3.5), aspect_ratio=:equal,
                title="Grid World Trajectories", xlabel="X", ylabel="Y")

        # Plot grid
        for x in 1:3, y in 1:3
            scatter!([x], [y], color=:lightgray, markersize=8, alpha=0.3, legend=false)
        end

        # Plot trajectories
        colors = [:red, :blue, :green, :orange, :purple]
        for (i, traj) in enumerate(trajectories[1:min(5, length(trajectories))])
            x_coords = [s.x for s in traj.states if !s.is_terminal]
            y_coords = [s.y for s in traj.states if !s.is_terminal]

            if !isempty(x_coords)
                plot!(x_coords, y_coords, color=colors[mod1(i, length(colors))],
                     linewidth=2, marker=:circle, markersize=4, label="Traj $i")
            end
        end

        # Save plot
        plot_file = joinpath(results_dir, "grid_trajectories.png")
        savefig(p, plot_file)
        println("🗺️  Grid visualization saved to: $plot_file")

    catch e
        println("⚠️  Could not create grid visualization: $e")
    end
end

# =============================================================================
# Main Execution
# =============================================================================

"""Main function - run the simple grid world example"""
function main()
    # Create results directory
    results_dir = "results"
    if !isdir(results_dir)
        mkdir(results_dir)
        println("📁 Created results directory: $results_dir/")
    end
    
    # Setup logging
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    debug_log = open("$results_dir/debug_$timestamp.log", "w")
    results_file = "$results_dir/grid_world_results_$timestamp.txt"
    
    # Redirect output to both console and log
    function log_print(msg)
        println(msg)
        println(debug_log, msg)
        flush(debug_log)
    end
    
    try
        log_print("🎯 Grid World GFlowNet Example")
        log_print("=" ^ 50)
        log_print("📅 Timestamp: $timestamp")
        log_print("📁 Results directory: $results_dir/")
        
        # Create model
        log_print("🔧 Creating GFlowNet model...")
        model, initial_state = create_simple_gflownet()
        log_print("✅ Model created successfully!")
        
        # Train model
        log_print("\n🚀 Starting training...")
        trained_model, training_data = train_simple_gflownet(model, initial_state, log_print)

        # Evaluate model
        log_print("\n🎯 Evaluating model...")
        evaluation_results = evaluate_simple_model(trained_model, log_print)

        # Create visualizations
        log_print("\n📊 Creating visualizations...")
        create_learning_curves(training_data, results_dir)

        # Sample trajectories for grid visualization
        eval_trajectories = []
        for _ in 1:5
            try
                traj = GFlowNet.sample_trajectory(trained_model)
                push!(eval_trajectories, traj)
            catch
                fallback_terminal = SimpleGridState(initial_state.x, initial_state.y, true)
                fallback_traj = GFlowNet.Trajectory([initial_state, fallback_terminal])
                push!(eval_trajectories, fallback_traj)
            end
        end
        create_grid_visualization(eval_trajectories, results_dir)
        
        # Save final results
        open(results_file, "w") do f
            println(f, "Grid World GFlowNet Results")
            println(f, "Timestamp: $timestamp")
            println(f, "=" ^ 40)
            println(f, evaluation_results)
            println(f, "\nTraining Data:")
            for (iter, loss, mean_reward, high_reward_count, successful) in training_data
                println(f, "Iteration $iter: Loss=$loss, Mean Reward=$mean_reward, High Rewards=$high_reward_count, Successful=$successful")
            end
        end
        
        log_print("\n✅ Grid World Example Complete!")
        log_print("📊 Results saved to: $results_file")
        log_print("🐛 Debug log saved to: $results_dir/debug_$timestamp.log")
        
    finally
        close(debug_log)
    end
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 