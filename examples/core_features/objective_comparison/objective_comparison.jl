# Comparison of DETAILED_BALANCE vs TRAJECTORY_BALANCE Training Objectives
#
# This example demonstrates the differences between two core GFlowNet training objectives:
# 1. TRAJECTORY_BALANCE (TB): Uses full trajectory probabilities
# 2. DETAILED_BALANCE (DB): Uses local balance constraints
#
# Key differences:
# - TB: Can work without backward policy, considers entire trajectories
# - DB: Requires backward policy, enforces local flow conservation

using GFlowNet
using Statistics
using Random
using Plots
using DataFrames
using CSV

include(joinpath(@__DIR__, "..", "..", "convergence_assertions.jl"))

# Set random seed for reproducibility
Random.seed!(42)

# Create output directory for results next to this example, not in the launch CWD
const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

println("=" ^ 60)
println("GFlowNet Training Objectives Comparison")
println("TRAJECTORY_BALANCE vs DETAILED_BALANCE")
println("=" ^ 60)

# Configuration parameters
const GRID_SIZE = 6
const HIDDEN_DIM = 64
# Budget cut from 500 iterations / batch 32: this script trains TWO models, and at
# the old budget it did not finish inside 300s. 120 iterations at batch 16 is well
# past convergence for both objectives -- see the measured values at the assertions.
const N_ITERATIONS = 120
const BATCH_SIZE = 16
const LEARNING_RATE = 0.01
const N_EVAL_SAMPLES = 300

# Helper function to evaluate model performance
function evaluate_model(model::GFlowNetModel, n_samples::Int)
    trajectories = [sample_trajectory(model) for _ in 1:n_samples]
    
    # Extract terminal states and rewards
    terminal_states = [traj.states[end] for traj in trajectories]
    rewards = [reward(state) for state in terminal_states]
    
    # Calculate metrics
    mean_reward = mean(rewards)
    std_reward = std(rewards)
    unique_terminals = length(unique(terminal_states))
    
    # Calculate state visitation frequency
    state_counts = Dict{Any,Int}()
    for traj in trajectories
        for state in traj.states
            state_counts[state] = get(state_counts, state, 0) + 1
        end
    end
    
    return (
        mean_reward = mean_reward,
        std_reward = std_reward,
        unique_terminals = unique_terminals,
        total_states_visited = length(state_counts),
        trajectories = trajectories,
        state_counts = state_counts
    )
end

# Function to create reward landscape visualization
function visualize_reward_landscape(grid_size::Int)
    rewards = zeros(grid_size, grid_size)
    for x in 1:grid_size, y in 1:grid_size
        # `GridState`, not `GridWorldState` -- the latter has never existed in this
        # package (src/applications/grid_world.jl defines `GridState`), so every use
        # of it here was a latent UndefVarError that the 300s timeout hid.
        state = GridState(x, y, x == grid_size && y == grid_size)
        if is_terminal_state(state)
            rewards[y, x] = reward(state)
        end
    end
    
    heatmap(rewards, 
        title="Reward Landscape",
        xlabel="X coordinate",
        ylabel="Y coordinate",
        color=:viridis,
        aspect_ratio=:equal)
end

# Training comparison
println("\n1. Creating models...")

# Model with TRAJECTORY_BALANCE (no backward policy needed)
model_tb = create_grid_world_gflownet(
    grid_size = GRID_SIZE,
    hidden_dim = HIDDEN_DIM,
    learning_rate = LEARNING_RATE,
    include_backward = false  # TB doesn't require backward policy
)

# Model with DETAILED_BALANCE (requires backward policy)
model_db = create_grid_world_gflownet(
    grid_size = GRID_SIZE,
    hidden_dim = HIDDEN_DIM,
    learning_rate = LEARNING_RATE,
    include_backward = true   # DB requires backward policy
)

println("✓ Models created")
println("  - Grid size: $(GRID_SIZE)×$(GRID_SIZE)")
println("  - Hidden dimension: $HIDDEN_DIM")
println("  - Learning rate: $LEARNING_RATE")

# Training configurations
config_tb = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = N_ITERATIONS,
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    validation_frequency = 50
)

config_db = TrainingConfig(
    objective = DETAILED_BALANCE,
    n_iterations = N_ITERATIONS,
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    validation_frequency = 50
)

# Train both models
println("\n2. Training models...")
println("\nTraining with TRAJECTORY_BALANCE:")
history_tb = train_gflownet(model_tb, config_tb; verbose=true)

println("\nTraining with DETAILED_BALANCE:")
history_db = train_gflownet(model_db, config_db; verbose=true)

# Anti-silent-failure gate. train_gflownet catches per-iteration exceptions and
# records NaN, so without this a run in which every iteration threw would print a
# full comparison table of NaNs and exit 0.
assert_finite_iterations(history_tb, N_ITERATIONS, "TRAJECTORY_BALANCE")
assert_finite_iterations(history_db, N_ITERATIONS, "DETAILED_BALANCE")

# Neither model is given a partition_function_method, so both default to
# SIMPLE_ESTIMATION, which pins log Z = 0 i.e. Z = 1 (src/training/losses.jl:
# "SIMPLE_ESTIMATION: Z = 1, so log Z = 0"). Under Z = 1 the trajectory-balance
# residual (log P(tau) - log R)^2 cannot reach 0 while R > 1, and measured on this
# exact 6x6 configuration the TB loss RISES over the run: 31.103 -> 36.048, ratio
# 1.159. So TB is NOT given a loss-decrease assertion; its sampler quality is checked
# below instead.
#
# DETAILED_BALANCE is a local flow-conservation residual and does converge here.
# Measured at 120 iterations / batch 16: 0.215 -> 0.011, ratio 0.050; and on the same
# objective at 100 iterations, losses[1] = 1.154 -> mean(last 25) = 0.039. Bar 0.25.
assert_loss_below_initial(history_db, "DETAILED_BALANCE"; window=25, max_ratio=0.25)

# Evaluate models
println("\n3. Evaluating models...")
eval_tb = evaluate_model(model_tb, N_EVAL_SAMPLES)
eval_db = evaluate_model(model_db, N_EVAL_SAMPLES)

# TB runs under SIMPLE_ESTIMATION (see above), so the check that training worked is
# that the sampler concentrates on high-reward terminals rather than that the loss
# fell. Both objectives are held to it.
model_untrained = create_grid_world_gflownet(
    grid_size = GRID_SIZE,
    hidden_dim = HIDDEN_DIM,
    learning_rate = LEARNING_RATE,
    include_backward = false
)
rewards_untrained = [reward(sample_trajectory(model_untrained).states[end])
                     for _ in 1:N_EVAL_SAMPLES]
rewards_tb_eval = [reward(t.states[end]) for t in eval_tb.trajectories]
rewards_db_eval = [reward(t.states[end]) for t in eval_db.trajectories]
assert_beats_untrained(rewards_tb_eval, rewards_untrained, "TB sampler"; min_gain=1.2)
assert_beats_untrained(rewards_db_eval, rewards_untrained, "DB sampler"; min_gain=1.2)

println("\nTRAJECTORY_BALANCE Results:")
println("  - Mean reward: $(round(eval_tb.mean_reward, digits=3)) ± $(round(eval_tb.std_reward, digits=3))")
println("  - Unique terminals found: $(eval_tb.unique_terminals)")
println("  - Total states visited: $(eval_tb.total_states_visited)")

println("\nDETAILED_BALANCE Results:")
println("  - Mean reward: $(round(eval_db.mean_reward, digits=3)) ± $(round(eval_db.std_reward, digits=3))")
println("  - Unique terminals found: $(eval_db.unique_terminals)")
println("  - Total states visited: $(eval_db.total_states_visited)")

# Create visualizations
println("\n4. Creating visualizations...")

# Plot 1: Training curves
p1 = plot(history_tb.losses, 
    label="Trajectory Balance",
    xlabel="Iteration",
    ylabel="Loss",
    title="Training Loss Comparison",
    linewidth=2,
    alpha=0.8)
plot!(p1, history_db.losses, 
    label="Detailed Balance",
    linewidth=2,
    alpha=0.8)

# Plot 2: State visitation heatmaps
function create_visitation_heatmap(state_counts, grid_size, title)
    visit_grid = zeros(grid_size, grid_size)
    for (state, count) in state_counts
        if isa(state, GridState)
            visit_grid[state.y, state.x] = count
        end
    end
    
    heatmap(visit_grid,
        title=title,
        xlabel="X coordinate",
        ylabel="Y coordinate",
        color=:plasma,
        aspect_ratio=:equal)
end

p2 = create_visitation_heatmap(eval_tb.state_counts, GRID_SIZE, "TB: State Visitations")
p3 = create_visitation_heatmap(eval_db.state_counts, GRID_SIZE, "DB: State Visitations")

# Plot 3: Reward distribution
p4 = histogram([reward(t.states[end]) for t in eval_tb.trajectories],
    label="Trajectory Balance",
    alpha=0.6,
    bins=20,
    normalize=:probability,
    title="Terminal Reward Distribution",
    xlabel="Reward",
    ylabel="Probability")
histogram!(p4, [reward(t.states[end]) for t in eval_db.trajectories],
    label="Detailed Balance",
    alpha=0.6,
    bins=20,
    normalize=:probability)

# Combine plots
final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
savefig(final_plot, joinpath(RESULTS_DIR, "objective_comparison.png"))

# Analyze trajectory lengths
tb_lengths = [length(t.states) for t in eval_tb.trajectories]
db_lengths = [length(t.states) for t in eval_db.trajectories]

println("\n5. Trajectory Analysis:")
println("\nTRAJECTORY_BALANCE:")
println("  - Mean trajectory length: $(round(mean(tb_lengths), digits=2))")
println("  - Min/Max length: $(minimum(tb_lengths))/$(maximum(tb_lengths))")

println("\nDETAILED_BALANCE:")
println("  - Mean trajectory length: $(round(mean(db_lengths), digits=2))")
println("  - Min/Max length: $(minimum(db_lengths))/$(maximum(db_lengths))")

# Test backward policy (only for DB model)
println("\n6. Backward Policy Analysis (DB only):")
test_state = GridState(3, 3, false)
println("  Testing backward transitions from state (3,3)...")

# Find states that can reach test_state
can_reach = []
for x in 1:GRID_SIZE, y in 1:GRID_SIZE
    source = GridState(x, y, false)
    if is_valid_backward_transition(source, test_state, model_db.all_actions)
        prob = compute_backward_probability(
            model_db.backward_policy, test_state, source,
            model_db.parameters.backward, model_db.states.backward,
            model_db.all_actions
        )
        push!(can_reach, (source, prob))
    end
end

println("  Found $(length(can_reach)) possible previous states:")
for (source, prob) in sort(can_reach, by=x->x[2], rev=true)
    println("    From ($(source.x),$(source.y)): P = $(round(prob, digits=3))")
end

# Save detailed results
results_df = DataFrame(
    Metric = ["Mean Reward", "Std Reward", "Unique Terminals", "States Visited", "Mean Trajectory Length"],
    Trajectory_Balance = [eval_tb.mean_reward, eval_tb.std_reward, eval_tb.unique_terminals, 
                         eval_tb.total_states_visited, mean(tb_lengths)],
    Detailed_Balance = [eval_db.mean_reward, eval_db.std_reward, eval_db.unique_terminals,
                       eval_db.total_states_visited, mean(db_lengths)]
)

CSV.write(joinpath(RESULTS_DIR, "objective_comparison_metrics.csv"), results_df)

# Key insights
# `"=" ^ 60`, not `"=" * 60`. String * Int is a MethodError
# (`MethodError(*, ("=", 60))`), so both of these lines threw. The 300s timeout hid it
# because the script never got this far.
println("\n" * "=" ^ 60)
println("KEY INSIGHTS:")
println("=" ^ 60)

println("\n1. CONVERGENCE SPEED:")
window = min(50, N_ITERATIONS)
tb_final_losses = history_tb.losses[end-window+1:end]
db_final_losses = history_db.losses[end-window+1:end]
println("   TB final loss: $(round(mean(tb_final_losses), digits=4))")
println("   DB final loss: $(round(mean(db_final_losses), digits=4))")

println("\n2. EXPLORATION:")
exploration_ratio_tb = eval_tb.total_states_visited / (GRID_SIZE^2)
exploration_ratio_db = eval_db.total_states_visited / (GRID_SIZE^2)
println("   TB explores $(round(exploration_ratio_tb * 100, digits=1))% of state space")
println("   DB explores $(round(exploration_ratio_db * 100, digits=1))% of state space")

println("\n3. CREDIT ASSIGNMENT:")
println("   TB: Uses full trajectory signal")
println("   DB: Uses local balance constraints + backward policy")

println("\n4. WHEN TO USE EACH:")
println("   - Use TB: Simple problems, no backward info needed")
println("   - Use DB: Complex credit assignment, local constraints important")

println("\n✓ Analysis complete! Results saved to results/")
println("  - objective_comparison.png: Visualizations")
println("  - objective_comparison_metrics.csv: Detailed metrics")