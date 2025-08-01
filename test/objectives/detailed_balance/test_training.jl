using GFlowNet

# Create model
model = create_grid_world_gflownet(
    grid_size=3,
    hidden_dim=32,
    learning_rate=0.01,
    include_backward=true
)

# Configure training
config = TrainingConfig(
    objective=DETAILED_BALANCE,
    n_iterations=3,
    batch_size=4,
    learning_rate=0.01
)

println("Testing DETAILED_BALANCE training...")

# Train
history = train_gflownet(model, config; verbose=true)

println("\nTraining history:")
println("Losses: $(history.losses)")
println("Valid losses: $(filter(!isnan, history.losses))")

# Test a single step manually
println("\nTesting single training step...")
trajectories = [sample_trajectory(model) for _ in 1:4]

try
    loss = compute_trajectory_loss(model, trajectories, model.parameters, config)
    println("✓ Single step loss: $loss")
catch e
    println("✗ Error in single step: $e")
end