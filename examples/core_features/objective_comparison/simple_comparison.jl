# Simple comparison of TB vs DB training objectives
# This is a minimal example focusing on the key differences

using GFlowNet
using Statistics

println("Simple TB vs DB Comparison\n")

# Create two identical models except for backward policy
model_tb = create_grid_world_gflownet(
    grid_size = 4,
    hidden_dim = 32,
    include_backward = false  # TB doesn't need backward policy
)

model_db = create_grid_world_gflownet(
    grid_size = 4,
    hidden_dim = 32,
    include_backward = true   # DB requires backward policy
)

# Train with different objectives
println("Training with TRAJECTORY_BALANCE...")
config_tb = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 100,
    batch_size = 16
)
history_tb = train_gflownet(model_tb, config_tb; verbose=false)
println("Final TB loss: $(round(history_tb.losses[end], digits=4))")

println("\nTraining with DETAILED_BALANCE...")
config_db = TrainingConfig(
    objective = DETAILED_BALANCE,
    n_iterations = 100,
    batch_size = 16
)
history_db = train_gflownet(model_db, config_db; verbose=false)
println("Final DB loss: $(round(history_db.losses[end], digits=4))")

# Sample and compare
println("\nSampling 100 trajectories from each...")
traj_tb = [sample_trajectory(model_tb) for _ in 1:100]
traj_db = [sample_trajectory(model_db) for _ in 1:100]

# Compare rewards
rewards_tb = [reward(t.states[end]) for t in traj_tb]
rewards_db = [reward(t.states[end]) for t in traj_db]

println("\nResults:")
println("TB - Mean reward: $(round(mean(rewards_tb), digits=3))")
println("DB - Mean reward: $(round(mean(rewards_db), digits=3))")

# Key difference: DB can answer "how did I get here?"
println("\nKey difference - Backward policy (DB only):")
test_state = GridState(3, 3, false)
println("For state (3,3), DB can compute P(previous state | current state)")

# This only works for DB model
if !isnothing(model_db.backward_policy)
    println("✓ DB model has backward policy for credit assignment")
else
    println("✗ DB model missing backward policy")
end

if isnothing(model_tb.backward_policy)
    println("✓ TB model correctly has no backward policy")
end