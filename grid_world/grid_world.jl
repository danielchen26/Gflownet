# Visualize results
println("Visualizing results...")

# Create output directory if it doesn't exist
output_dir = "."

# Plot loss curve
losses = get_metric(logger, "loss")
loss_plot = visualize_training_progress(losses)
savefig(loss_plot, joinpath(output_dir, "grid_world_loss.png"))

# Sample and visualize trajectories
n_samples = 10
sampled_trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:n_samples]
grid_plot = visualize_grid(sampled_trajectories)
savefig(grid_plot, joinpath(output_dir, "grid_world_paths.png"))

# Plot reward distribution
reward_plot = visualize_reward_distribution(model, 100)
savefig(reward_plot, joinpath(output_dir, "grid_world_rewards.png"))

println("Example completed. Results saved to $output_dir/grid_world_*.png") 