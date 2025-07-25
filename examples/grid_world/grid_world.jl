"""
🎯 Grid World GFlowNet Example
==============================

This example demonstrates GFlowNet training on a 5×5 grid world environment.
The agent learns to navigate to high-reward terminal states at positions (3,3), (1,5), and (5,1).

Key Features:
- 5×5 grid navigation with multiple reward positions
- Uses core GFlowNet framework functions
- Trajectory balance objective with proper training
- Comprehensive visualization and analysis
- Modern training interface demonstration
"""

using GFlowNet
using Lux
using Random
using ComponentArrays
using Optimisers  # ← FIXED: Add missing import for optimizer
using Zygote     # ← FIXED: Add missing import for gradients
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
# Grid World Configuration
# =============================================================================

const GRID_SIZE = 5
const REWARD_POSITIONS = Dict(
    (3, 3) => 10.0,  # High reward at center
    (1, 5) => 5.0,   # Medium reward at top-left
    (5, 1) => 5.0    # Medium reward at bottom-right
)

# =============================================================================
# Grid World States and Actions
# =============================================================================

"""Grid state - position (x,y) and terminal flag"""
struct GridState <: GFlowNet.AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

"""Grid actions - move in 4 directions or terminate"""
abstract type GridAction <: GFlowNet.AbstractAction end

struct MoveUp <: GridAction end
struct MoveDown <: GridAction end
struct MoveLeft <: GridAction end
struct MoveRight <: GridAction end
struct Terminate <: GridAction end

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
function GFlowNet.state_to_features(state::GridState)
    # Use both coordinate and one-hot representations for better learning
    # Normalized coordinates (0-1 range)
    x_norm = Float32((state.x - 1) / (GRID_SIZE - 1))
    y_norm = Float32((state.y - 1) / (GRID_SIZE - 1))

    # Distance to reward positions (helps with learning)
    dist_to_center = Float32(sqrt((state.x - 3)^2 + (state.y - 3)^2) / sqrt(8))  # Normalized
    dist_to_top_left = Float32(sqrt((state.x - 1)^2 + (state.y - 5)^2) / sqrt(16))
    dist_to_bottom_right = Float32(sqrt((state.x - 5)^2 + (state.y - 1)^2) / sqrt(16))

    # One-hot encoding for position (sparse but precise)
    pos_idx = (state.x - 1) * GRID_SIZE + state.y
    grid_size_sq = GRID_SIZE * GRID_SIZE
    one_hot = Float32.([(i == pos_idx) for i in 1:grid_size_sq])

    features = vcat(
        # Coordinate features (dense representation)
        [x_norm, y_norm],
        # Distance features (reward-aware)
        [dist_to_center, dist_to_top_left, dist_to_bottom_right],
        # One-hot position (precise representation)
        one_hot,
        # Terminal state feature
        [state.is_terminal ? 1.0f0 : 0.0f0]
    )

    return features
end

"""Check if action is applicable to state"""
function GFlowNet.is_applicable(action::GridAction, state::GridState)
    # Terminal states should not have any applicable actions
    if state.is_terminal
        return false
    end

    # Non-terminal states can always terminate
    if isa(action, Terminate)
        return true
    end

    # Check grid boundaries (5x5 grid) for movement actions
    x, y = state.x, state.y
    if isa(action, MoveUp) && y < GRID_SIZE
        return true
    elseif isa(action, MoveDown) && y > 1
        return true
    elseif isa(action, MoveLeft) && x > 1
        return true
    elseif isa(action, MoveRight) && x < GRID_SIZE
        return true
    end

    return false
end

"""Apply action to state"""
function GFlowNet.apply_action(action::GridAction, state::GridState)
    if isa(action, Terminate)
        return GridState(state.x, state.y, true)
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

    return GridState(x, y, false)
end

"""Check if state is terminal"""
function GFlowNet.is_terminal_state(state::GridState)
    return state.is_terminal
end

"""Compute reward for terminal states"""
function GFlowNet.reward(state::GridState)
    if !state.is_terminal
        return 0.0f0  # No reward for non-terminal states
    end

    # Check if the position has a special reward
    position = (state.x, state.y)
    if haskey(REWARD_POSITIONS, position)
        return Float32(REWARD_POSITIONS[position])
    end

    # Default reward for other terminal states
    return 0.1f0
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

"""Create a GFlowNet model for the 5x5 grid world"""
function create_grid_world_gflownet()
    # 1. Define states and actions
    initial_state = GridState(1, 1, false)

    # Create all non-terminal states (intermediate states)
    all_states = GridState[]
    for x in 1:GRID_SIZE, y in 1:GRID_SIZE
        push!(all_states, GridState(x, y, false))  # Non-terminal states
    end

    # Create all terminal states
    terminal_states = GridState[]
    for x in 1:GRID_SIZE, y in 1:GRID_SIZE
        push!(terminal_states, GridState(x, y, true))  # Terminal states
    end

    # Add all states to the DAG (both non-terminal and terminal)
    all_dag_states = [all_states; terminal_states]

    terminal_sink = GridState(0, 0, true)  # Special sink state

    actions = GridAction[UP, DOWN, LEFT, RIGHT, TERMINATE]
    
    # 2. Create DAG using core function with ALL states
    # We need to create a custom DAG that includes all intermediate states
    dag = create_custom_grid_dag(initial_state, all_dag_states, terminal_states, terminal_sink, actions)

    # DAG created successfully
    println("✅ DAG created with $(length(dag.states)) states")
    
    # 3. Create neural networks
    # Input: 2 (coords) + 3 (distances) + 25 (one-hot) + 1 (terminal) = 31 features
    input_dim = 2 + 3 + GRID_SIZE * GRID_SIZE + 1
    hidden_dim = 128  # Increased capacity for better learning
    n_states = length(dag.states)  # Output should be number of states in DAG

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    # Forward policy network - improved architecture for stability
    forward_nn = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim ÷ 2, relu),  # Bottleneck layer
        Dense(hidden_dim ÷ 2 => n_states)  # Output STATE transition logits
    )

    # Flow estimator network - improved architecture
    flow_nn = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim ÷ 2, relu),  # Bottleneck layer
        Dense(hidden_dim ÷ 2 => 1, x -> softplus.(x) .+ 1f-6)  # Positive flow values
    )
    
    # Initialize parameters
    forward_ps, forward_st = Lux.setup(rng, forward_nn)
    flow_ps, flow_st = Lux.setup(rng, flow_nn)
    
    # 4. Create GFlowNet components
    forward_policy = GFlowNet.ForwardPolicy(forward_nn)
    flow_estimator = GFlowNet.FlowEstimator(flow_nn)
    
    # 5. Create optimizer with reduced learning rate for stability
    opt = Optimisers.Adam(0.0005)  # Reduced learning rate for better stability
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

"""Training loop for grid world GFlowNet using modern interface"""
function train_grid_world_gflownet(model, initial_state, log_function=println; n_iterations=1000, batch_size=32)
    # Use the provided logging function
    log_msg = log_function

    log_msg("🚀 Training GFlowNet with modern interface...")
    log_msg("   Iterations: $n_iterations")
    log_msg("   Batch size: $batch_size")

    # Create modern training configuration
    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        partition_function_method = GFlowNet.ADAPTIVE_ESTIMATION,  # Use adaptive Z estimation
        batch_size = batch_size,
        learning_rate = 0.001,
        n_iterations = n_iterations,
        partition_update_frequency = 10,  # Update Z every 10 iterations
        validation_frequency = 50,
        early_stopping_patience = 100,
        sub_trajectory_config = Dict(
            :max_grad_norm => 1.0,
            :curriculum_learning => true,
            :curriculum_rate => 0.3  # 30% curriculum injection
        )
    )

    log_msg("✅ Training configuration created")
    log_msg("   Objective: $(config.objective)")
    log_msg("   Partition function method: $(config.partition_function_method)")
    log_msg("   Learning rate: $(config.learning_rate)")

    # Custom training loop with curriculum learning
    training_data = []
    successful_samples = 0

    for iter in 1:n_iterations
        # Sample trajectories with curriculum learning
        trajectories = GFlowNet.Trajectory[]
        iter_successful = 0

        for sample_idx in 1:batch_size
            try
                # Enhanced curriculum learning: inject good trajectories early in training
                if iter <= 100 && rand() < 0.3  # 30% chance in first 100 iterations
                    # Create a high-reward trajectory to help model learn
                    if rand() < 0.33
                        # Path to (3,3): (1,1) -> (2,1) -> (3,1) -> (3,2) -> (3,3) -> terminal
                        good_states = [
                            GridState(1, 1, false),
                            GridState(2, 1, false),
                            GridState(3, 1, false),
                            GridState(3, 2, false),
                            GridState(3, 3, false),
                            GridState(3, 3, true)
                        ]
                    elseif rand() < 0.5
                        # Path to (1,5): (1,1) -> (1,2) -> (1,3) -> (1,4) -> (1,5) -> terminal
                        good_states = [
                            GridState(1, 1, false),
                            GridState(1, 2, false),
                            GridState(1, 3, false),
                            GridState(1, 4, false),
                            GridState(1, 5, false),
                            GridState(1, 5, true)
                        ]
                    else
                        # Path to (5,1): (1,1) -> (2,1) -> (3,1) -> (4,1) -> (5,1) -> terminal
                        good_states = [
                            GridState(1, 1, false),
                            GridState(2, 1, false),
                            GridState(3, 1, false),
                            GridState(4, 1, false),
                            GridState(5, 1, false),
                            GridState(5, 1, true)
                        ]
                    end
                    traj = GFlowNet.Trajectory(good_states)
                    if iter % 100 == 0
                        log_msg("  📚 Curriculum learning active (30% injection rate)")
                    end
                else
                    # Use standard GFlowNet sampling
                    traj = GFlowNet.sample_trajectory(model)
                end

                push!(trajectories, traj)
                iter_successful += 1
                successful_samples += 1
            catch e
                log_msg("Warning: Error sampling trajectory $sample_idx in iteration $iter: $e")
                # Create a simple fallback trajectory
                fallback_terminal = GridState(initial_state.x, initial_state.y, true)
                fallback_traj = GFlowNet.Trajectory([initial_state, fallback_terminal])
                push!(trajectories, fallback_traj)
            end
        end

        if !isempty(trajectories) && iter_successful > 0
            try
                # Use legacy but working approach with improvements
                loss, grads = GFlowNet.compute_loss_and_grad(model, trajectories)

                # Apply gradient clipping (critical for stability)
                max_grad_norm = 0.5  # Reduced for better stability
                if !isnothing(grads)
                    grad_norm = GFlowNet.clip_gradients!(grads, max_grad_norm)
                    if iter % 50 == 0  # Monitor gradient norms less frequently
                        log_msg("   Gradient norm: $(round(grad_norm, digits=4))")
                    end
                end

                # Apply gradients
                GFlowNet.apply_optimizer!(model, grads)

                loss_val = loss

                # Update partition function periodically
                if iter % config.partition_update_frequency == 0
                    # Update Z using adaptive estimation
                    try
                        new_Z = GFlowNet.estimate_partition_function(GFlowNet.AdaptivePartitionFunctionEstimator(), model)
                        if iter % 50 == 0
                            log_msg("   Updated partition function Z = $(round(new_Z, digits=2))")
                        end
                    catch
                        # Fallback to simple estimation if adaptive fails
                        new_Z = GFlowNet.estimate_partition_function(GFlowNet.SimplePartitionFunctionEstimator(), model)
                    end
                end

                # Track training progress
                rewards = [GFlowNet.reward(traj.states[end]) for traj in trajectories]
                mean_reward = sum(rewards) / length(rewards)
                high_reward_count = count(r -> r >= 5.0, rewards)
                max_reward_count = count(r -> r >= 10.0, rewards)

                push!(training_data, (iter, loss_val, mean_reward, high_reward_count, iter_successful))

                # Print progress with enhanced monitoring
                if iter % 50 == 0
                    log_msg("Iteration $iter: Loss = $(round(loss_val, digits=4)), Mean Reward = $(round(mean_reward, digits=2)), High Rewards = $high_reward_count/$batch_size, Max Rewards = $max_reward_count/$batch_size, Successful = $iter_successful/$batch_size")
                end
            catch e
                log_msg("Error in training iteration $iter: $e")
                # Continue training even if one iteration fails
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
function evaluate_grid_world_model(model, log_function=println; n_samples=50)
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
        
        # Reward distribution calculation for 5x5 grid
        reward_01_count = count(r -> r ≈ 0.1, rewards)  # Default terminal reward
        reward_5_count = count(r -> r ≈ 5.0, rewards)   # (1,5) and (5,1)
        reward_10_count = count(r -> r ≈ 10.0, rewards) # (3,3)
        reward_other_count = n_samples - reward_01_count - reward_5_count - reward_10_count

        results_summary = """
📊 Evaluation Results:
   Successful evaluations: $successful_evals/$n_samples
   Mean Reward: $(round(mean_reward, digits=1))
   High Reward (≥5.0) Rate: $(round(high_reward_percentage, digits=1))% ($high_reward_count/$n_samples)
   Max Reward (≥10.0) Rate: $(round(max_reward_percentage, digits=1))% ($max_reward_count/$n_samples)

📋 Sample Trajectories:
$(join(sample_trajectories, "\n"))

📈 Reward Distribution:
   Reward 0.1: $reward_01_count samples (default terminal)
   Reward 5.0: $reward_5_count samples (positions (1,5) and (5,1))
   Reward 10.0: $reward_10_count samples (position (3,3))
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
        plot_file = joinpath(results_dir, "grid_world_loss.png")
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
        p = plot(xlims=(0.5, GRID_SIZE + 0.5), ylims=(0.5, GRID_SIZE + 0.5), aspect_ratio=:equal,
                title="5×5 Grid World Trajectories", xlabel="X", ylabel="Y")

        # Plot grid
        for x in 1:GRID_SIZE, y in 1:GRID_SIZE
            # Color code reward positions
            if (x, y) == (3, 3)
                scatter!([x], [y], color=:gold, markersize=12, alpha=0.8, legend=false)
            elseif (x, y) == (1, 5) || (x, y) == (5, 1)
                scatter!([x], [y], color=:silver, markersize=10, alpha=0.8, legend=false)
            else
                scatter!([x], [y], color=:lightgray, markersize=8, alpha=0.3, legend=false)
            end
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
        plot_file = joinpath(results_dir, "grid_world_paths.png")
        savefig(p, plot_file)
        println("🗺️  Grid visualization saved to: $plot_file")

    catch e
        println("⚠️  Could not create grid visualization: $e")
    end
end

"""Create reward distribution histogram"""
function create_reward_distribution(trajectories, results_dir)
    if !PLOTS_AVAILABLE
        println("⚠️  Plots.jl not available - skipping reward distribution")
        return
    end

    try
        if isempty(trajectories)
            println("⚠️  No trajectories available for reward distribution")
            return
        end

        # Extract rewards
        rewards = [GFlowNet.reward(traj.states[end]) for traj in trajectories]

        # Create histogram
        p = histogram(rewards, bins=10, title="Reward Distribution",
                     xlabel="Reward Value", ylabel="Frequency",
                     color=:blue, alpha=0.7, legend=false)

        # Add vertical lines for special reward values
        vline!([0.1], color=:gray, linestyle=:dash, linewidth=2, label="Default (0.1)")
        vline!([5.0], color=:orange, linestyle=:dash, linewidth=2, label="Medium (5.0)")
        vline!([10.0], color=:red, linestyle=:dash, linewidth=2, label="High (10.0)")

        # Save plot
        plot_file = joinpath(results_dir, "grid_world_rewards.png")
        savefig(p, plot_file)
        println("📊 Reward distribution saved to: $plot_file")

    catch e
        println("⚠️  Could not create reward distribution: $e")
    end
end

"""Save training data to CSV file"""
function save_training_data_csv(training_data, results_dir, _timestamp)
    try
        if isempty(training_data)
            println("⚠️  No training data available for CSV export")
            return
        end

        csv_file = joinpath(results_dir, "grid_world_training.csv")

        open(csv_file, "w") do f
            # Write header
            println(f, "iteration,loss,mean_reward,high_reward_count,successful_samples")

            # Write data
            for (iter, loss, mean_reward, high_reward_count, successful) in training_data
                println(f, "$iter,$loss,$mean_reward,$high_reward_count,$successful")
            end
        end

        println("💾 Training data saved to: $csv_file")

    catch e
        println("⚠️  Could not save training data CSV: $e")
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
        log_print("🔧 Creating 5×5 Grid World GFlowNet model...")
        model, initial_state = create_grid_world_gflownet()
        log_print("✅ Model created successfully!")

        # Train model with shorter run for testing
        log_print("\n🚀 Starting training...")
        trained_model, training_data = train_grid_world_gflownet(model, initial_state, log_print; n_iterations=100, batch_size=16)

        # Evaluate model
        log_print("\n🎯 Evaluating model...")
        evaluation_results = evaluate_grid_world_model(trained_model, log_print)

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
                fallback_terminal = GridState(initial_state.x, initial_state.y, true)
                fallback_traj = GFlowNet.Trajectory([initial_state, fallback_terminal])
                push!(eval_trajectories, fallback_traj)
            end
        end
        create_grid_visualization(eval_trajectories, results_dir)

        # Create reward distribution plot
        log_print("\n📊 Creating reward distribution...")
        create_reward_distribution(eval_trajectories, results_dir)

        # Save training data as CSV
        log_print("\n💾 Saving training data...")
        save_training_data_csv(training_data, results_dir, timestamp)

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