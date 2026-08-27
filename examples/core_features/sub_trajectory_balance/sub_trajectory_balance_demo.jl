# Sub-Trajectory Balance Training Example
# Demonstrates the SUB_TRAJECTORY_BALANCE objective for more stable training

using GFlowNet
using Statistics
using Plots
using Printf

println("=== Sub-Trajectory Balance (STB) Training Demo ===\n")

# Configuration
GRID_SIZE = 6
N_ITERATIONS = 200
BATCH_SIZE = 32
SUB_TRAJECTORY_LENGTH = 4

println("Configuration:")
println("  Grid size: $GRID_SIZE x $GRID_SIZE")
println("  Training iterations: $N_ITERATIONS")
println("  Batch size: $BATCH_SIZE")
println("  Sub-trajectory length: $SUB_TRAJECTORY_LENGTH")

# Create three models to compare objectives
println("\nCreating models...")
model_tb = create_grid_world_gflownet(
    grid_size=GRID_SIZE,
    hidden_dim=64,
    learning_rate=0.01
)

model_stb = create_grid_world_gflownet(
    grid_size=GRID_SIZE,
    hidden_dim=64,
    learning_rate=0.01
)

model_db = create_grid_world_gflownet(
    grid_size=GRID_SIZE,
    hidden_dim=64,
    learning_rate=0.01,
    include_backward=true  # Required for DB
)

# Training configurations
config_tb = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    n_iterations=N_ITERATIONS,
    batch_size=BATCH_SIZE,
    validation_frequency=10
)

config_stb = TrainingConfig(
    objective=SUB_TRAJECTORY_BALANCE,
    n_iterations=N_ITERATIONS,
    batch_size=BATCH_SIZE,
    validation_frequency=10,
    sub_trajectory_length=SUB_TRAJECTORY_LENGTH
)

config_db = TrainingConfig(
    objective=DETAILED_BALANCE,
    n_iterations=N_ITERATIONS,
    batch_size=BATCH_SIZE,
    validation_frequency=10
)

# Train models
println("\n1. Training with TRAJECTORY_BALANCE...")
history_tb = train_gflownet(model_tb, config_tb; verbose=false)

println("\n2. Training with SUB_TRAJECTORY_BALANCE...")
history_stb = train_gflownet(model_stb, config_stb; verbose=false)

println("\n3. Training with DETAILED_BALANCE...")
history_db = train_gflownet(model_db, config_db; verbose=false)

# Analyze training dynamics
println("\n=== Training Analysis ===")

# Loss statistics
for (name, history) in [("TB", history_tb), ("STB", history_stb), ("DB", history_db)]
    losses = filter(!isnan, history.losses)
    if !isempty(losses)
        initial_loss = losses[1]
        final_loss = losses[end]
        reduction = (1 - final_loss/initial_loss) * 100
        
        println("\n$name Training:")
        println("  Initial loss: $(round(initial_loss, digits=4))")
        println("  Final loss: $(round(final_loss, digits=4))")
        println("  Reduction: $(round(reduction, digits=1))%")
        println("  Loss variance: $(round(var(losses), digits=6))")
    end
end

# Sample trajectories and analyze
println("\n=== Trajectory Analysis ===")

function analyze_trajectories(model, name)
    trajectories = [sample_trajectory(model) for _ in 1:1000]
    
    # Length distribution
    lengths = [length(t.states) for t in trajectories]
    avg_length = mean(lengths)
    
    # Terminal state distribution
    terminals = [t.states[end] for t in trajectories]
    unique_terminals = unique(terminals)
    
    # Reward distribution
    rewards = [reward(t.states[end]) for t in trajectories]
    avg_reward = mean(rewards)
    
    println("\n$name Model:")
    println("  Average trajectory length: $(round(avg_length, digits=2))")
    println("  Unique terminal states: $(length(unique_terminals))")
    println("  Average reward: $(round(avg_reward, digits=4))")
    println("  Reward std: $(round(std(rewards), digits=4))")
    
    return trajectories, rewards
end

traj_tb, rewards_tb = analyze_trajectories(model_tb, "TB")
traj_stb, rewards_stb = analyze_trajectories(model_stb, "STB")
traj_db, rewards_db = analyze_trajectories(model_db, "DB")

# Visualize training curves
println("\n=== Creating Visualizations ===")

# Plot 1: Training losses
p1 = plot(title="Training Loss Comparison", xlabel="Iteration", ylabel="Loss", legend=:topright)
plot!(p1, history_tb.losses, label="Trajectory Balance", alpha=0.7, linewidth=2)
plot!(p1, history_stb.losses, label="Sub-Trajectory Balance", alpha=0.7, linewidth=2)
plot!(p1, history_db.losses, label="Detailed Balance", alpha=0.7, linewidth=2)

# Plot 2: Loss variance over windows
window_size = 20
function compute_windowed_variance(losses, window)
    variances = Float64[]
    for i in window:length(losses)
        window_losses = losses[max(1, i-window+1):i]
        if length(window_losses) > 1
            push!(variances, var(window_losses))
        end
    end
    return variances
end

var_tb = compute_windowed_variance(history_tb.losses, window_size)
var_stb = compute_windowed_variance(history_stb.losses, window_size)
var_db = compute_windowed_variance(history_db.losses, window_size)

p2 = plot(title="Loss Variance (window=$window_size)", xlabel="Iteration", ylabel="Variance", legend=:topright)
plot!(p2, window_size:length(history_tb.losses), var_tb, label="TB", alpha=0.7)
plot!(p2, window_size:length(history_stb.losses), var_stb, label="STB", alpha=0.7)
plot!(p2, window_size:length(history_db.losses), var_db, label="DB", alpha=0.7)

# Plot 3: Reward distributions
p3 = histogram(rewards_tb, alpha=0.5, label="TB", bins=20, normalize=true, title="Reward Distributions")
histogram!(p3, rewards_stb, alpha=0.5, label="STB", bins=20, normalize=true)
histogram!(p3, rewards_db, alpha=0.5, label="DB", bins=20, normalize=true)
xlabel!(p3, "Reward")
ylabel!(p3, "Frequency")

# Plot 4: Trajectory length distributions
lengths_tb = [length(t.states) for t in traj_tb]
lengths_stb = [length(t.states) for t in traj_stb]
lengths_db = [length(t.states) for t in traj_db]

p4 = histogram(lengths_tb, alpha=0.5, label="TB", bins=10:2:30, normalize=true, title="Trajectory Length Distributions")
histogram!(p4, lengths_stb, alpha=0.5, label="STB", bins=10:2:30, normalize=true)
histogram!(p4, lengths_db, alpha=0.5, label="DB", bins=10:2:30, normalize=true)
xlabel!(p4, "Trajectory Length")
ylabel!(p4, "Frequency")

# Combine plots
final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
savefig(final_plot, joinpath(@__DIR__, "sub_trajectory_balance_comparison.png"))  # next to the example, not the launch CWD
println("\nPlots saved to: sub_trajectory_balance_comparison.png")

# Demonstrate sub-trajectory extraction
println("\n=== Sub-Trajectory Extraction Example ===")

# Take a sample trajectory
sample_traj = traj_stb[1]
println("\nSample trajectory length: $(length(sample_traj.states))")
println("States: $([(s.x, s.y) for s in sample_traj.states])")

# Show sub-trajectories that would be considered
println("\nSub-trajectories (length ≤ $SUB_TRAJECTORY_LENGTH):")
# `local` + a name that does not shadow Base.
#
# This was `count = 0` at top level with `count += 1` inside the nested loop, which
# is an ambiguous soft-scope assignment: at top level Julia treats the loop body
# assignment as creating a fresh LOCAL each iteration, so the read failed with
# `UndefVarError: count not defined in local scope`. The name also shadowed
# `Base.count`. Wrapping the whole thing in a `let` gives one binding the loops can
# actually update.
let shown = 0
    for start_idx in 1:length(sample_traj.states)-1
        for end_idx in start_idx+1:min(start_idx+SUB_TRAJECTORY_LENGTH, length(sample_traj.states))
            sub_states = sample_traj.states[start_idx:end_idx]
            shown += 1
            println("  $shown. States $start_idx-$end_idx: $([(s.x, s.y) for s in sub_states])")

            if shown >= 10  # Limit output
                println("  ... (further sub-trajectories omitted)")
                break
            end
        end
        shown >= 10 && break
    end
end

# Mathematical interpretation
println("\n=== Mathematical Interpretation ===")

println("\nSub-Trajectory Balance enforces flow conservation on partial paths:")
println("  For sub-trajectory s_i → s_j:")
println("  ∏_{k=i}^{j-1} P_F(s_{k+1}|s_k) × F(s_i) = F(s_j)")
println("")
println("Benefits over Trajectory Balance:")
println("  1. More frequent learning signals (O(T²) vs O(T))")
println("  2. Better credit assignment for intermediate states")
println("  3. Lower variance in gradient estimates")
println("  4. Faster convergence in practice")
println("")
println("Trade-offs:")
println("  - Higher computational cost per trajectory")
println("  - Requires flow computation at intermediate states")
println("  - May over-emphasize short paths if sub_length is too small")

println("\n=== Demo Complete ===")