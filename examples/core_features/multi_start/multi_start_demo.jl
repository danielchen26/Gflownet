# Multi-Start GFlowNet Demonstration
# Shows how multiple initial states with learned partition functions work

using GFlowNet
using Random
using Statistics
using Plots

println("🚀 Multi-Start GFlowNet Demonstration\n")
println("Multiple initial states, each with its own partition function Z(s₀).")
println("The model learns which initial states lead to better rewards.\n")

# Set random seed
Random.seed!(42)

# Create multiple initial states at different corners of the grid
initial_states = [
    GridState(1, 1, false),   # Bottom-left
    GridState(5, 1, false),   # Bottom-right
    GridState(1, 5, false),   # Top-left
    GridState(5, 5, false)    # Top-right
]

actions = [MoveRight(), MoveUp(), MoveLeft(), MoveDown(), Terminate()]

println("Creating multi-start GFlowNet with $(length(initial_states)) initial states...")
model = create_multi_start_gflownet(
    initial_states,
    actions,
    state_dim = 3,
    hidden_dim = 64,
    learning_rate = 0.001
)

# Configure training
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 500,
    batch_size = 32,
    validation_frequency = 50
)

# Track statistics during training
initial_state_usage = zeros(Int, length(initial_states))
initial_state_rewards = [Float64[] for _ in 1:length(initial_states)]
log_z_history = [Float64[] for _ in 1:length(initial_states)]

# Custom callback to track progress
function track_progress(model, history, iteration)
    if iteration % 10 == 0
        for i in 1:length(initial_states)
            push!(log_z_history[i], model.log_partition_functions[i])
        end
    end
end

println("\nTraining multi-start GFlowNet...")
history = train_gflownet(model, config; verbose=true)

# Analyze results
println("\n📊 Analyzing multi-start behavior...")

# Sample many trajectories to see distribution
n_samples = 1000
trajectories_with_idx = [sample_trajectory(model) for _ in 1:n_samples]

# Count usage and rewards per initial state
for (traj, idx) in trajectories_with_idx
    initial_state_usage[idx] += 1
    terminal_reward = reward(traj.states[end])
    push!(initial_state_rewards[idx], terminal_reward)
end

# Compute statistics
final_probs = get_initial_state_distribution(model)

println("\nInitial State Statistics:")
for i in 1:length(initial_states)
    usage_pct = 100 * initial_state_usage[i] / n_samples
    avg_reward = mean(initial_state_rewards[i])
    state = initial_states[i]
    
    println("  State $i ($(state.x),$(state.y)):")
    println("    - Learned P(s₀): $(round(final_probs[i], digits=3))")
    println("    - Actual usage: $(round(usage_pct, digits=1))%")
    println("    - Avg reward: $(round(avg_reward, digits=3))")
    println("    - Final log Z: $(round(model.log_partition_functions[i], digits=3))")
end

# Create visualizations
println("\n📈 Creating visualizations...")

# Plot 1: Log Z evolution
p1 = plot(title = "Log Partition Functions Evolution",
    xlabel = "Iteration (×10)",
    ylabel = "log Z(s₀)",
    legend = :right)

for i in 1:length(initial_states)
    plot!(p1, log_z_history[i], 
        label = "State $i ($(initial_states[i].x),$(initial_states[i].y))",
        lw = 2)
end

# Plot 2: Initial state distribution
p2 = bar(1:length(initial_states), final_probs,
    xlabel = "Initial State",
    ylabel = "Probability",
    title = "Learned Initial State Distribution",
    label = nothing,
    color = :viridis)

# Plot 3: Average rewards per initial state
avg_rewards = [mean(rewards) for rewards in initial_state_rewards]
p3 = bar(1:length(initial_states), avg_rewards,
    xlabel = "Initial State",
    ylabel = "Average Reward",
    title = "Average Reward by Initial State",
    label = nothing,
    color = :plasma)

# Plot 4: Terminal state heatmap
terminal_positions = [(t.states[end].x, t.states[end].y) for (t, _) in trajectories_with_idx]
terminal_counts = zeros(5, 5)
for (x, y) in terminal_positions
    terminal_counts[x, y] += 1
end

p4 = heatmap(terminal_counts / n_samples,
    xlabel = "X",
    ylabel = "Y",
    title = "Terminal State Distribution",
    color = :thermal)

# Combine plots
final_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (1000, 800))
savefig(final_plot, "multi_start_results.png")

println("\n✅ Results saved to multi_start_results.png")

# Key insights
println("\n🔑 Key Insights:")
println("1. Each initial state learns its own partition function Z(s₀)")
println("2. P(s₀) = Z(s₀) / Σ Z(s₀') - states with better rewards get higher probability")
println("3. The model automatically discovers which starting positions are advantageous")
println("4. This enables efficient exploration from multiple starting configurations")

# Demonstrate sampling behavior
println("\n🎯 Sampling demonstration:")
for _ in 1:5
    traj, idx = sample_trajectory(model)
    start_state = traj.states[1]
    end_state = traj.states[end]
    r = reward(end_state)
    println("  Started at ($(start_state.x),$(start_state.y)), ended at ($(end_state.x),$(end_state.y)), reward: $r")
end

println("\n🎉 Multi-start GFlowNet demonstration complete!")