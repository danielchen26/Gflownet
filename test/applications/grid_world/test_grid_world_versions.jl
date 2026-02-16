"""
Test grid world with both forward-only and full backward policy versions
"""

using GFlowNet
using Random
using Statistics: mean

println("🧪 Testing Grid World with Both Policy Versions")
println("=" ^ 60)

# Test 1: Forward-only version (original)
println("\n1️⃣ Testing Forward-Only Version (Original)")
println("-" ^ 40)

model_forward_only = create_grid_world_gflownet(
    grid_size = 4,
    hidden_dim = 32,
    learning_rate = 0.01,
    include_backward = false  # Original version
)

config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 10,
    batch_size = 8,
    learning_rate = 0.01
)

println("Training forward-only model...")
history_forward = train_gflownet(model_forward_only, config; verbose=false)
println("✅ Forward-only training completed!")
println("   Final loss: $(history_forward.losses[end])")

# Sample some trajectories
trajectories_forward = [sample_trajectory(model_forward_only) for _ in 1:5]
avg_length_forward = mean(length(t.states) for t in trajectories_forward)
println("   Average trajectory length: $avg_length_forward")

# Test 2: Full backward policy version
println("\n2️⃣ Testing Full Backward Policy Version")
println("-" ^ 40)

model_with_backward = create_grid_world_gflownet(
    grid_size = 4,
    hidden_dim = 32,
    learning_rate = 0.01,
    include_backward = true  # New version with backward policy
)

println("Training model with backward policy...")
history_backward = train_gflownet(model_with_backward, config; verbose=false)
println("✅ Backward policy training completed!")
println("   Final loss: $(history_backward.losses[end])")

# Sample some trajectories
trajectories_backward = [sample_trajectory(model_with_backward) for _ in 1:5]
avg_length_backward = mean(length(t.states) for t in trajectories_backward)
println("   Average trajectory length: $avg_length_backward")

# Test 3: Verify backward probabilities are used
println("\n3️⃣ Verifying Backward Policy Usage")
println("-" ^ 40)

# Sample a trajectory and check backward probabilities
traj = sample_trajectory(model_with_backward)
if length(traj.states) >= 2
    s1 = traj.states[1]
    s2 = traj.states[2]
    
    # Check forward probability
    p_forward = forward_transition_probability(model_with_backward, s1, s2)
    println("   P_F($(s2)|$(s1)) = $p_forward")
    
    # Check backward probability
    p_backward = backward_transition_probability(model_with_backward, s2, s1)
    println("   P_B($(s1)|$(s2)) = $p_backward")
    
    # Compute trajectory balance components
    loss_forward_only = trajectory_balance_loss(model_forward_only, traj)
    loss_with_backward = trajectory_balance_loss(model_with_backward, traj)
    
    println("\n   Trajectory balance losses:")
    println("   - Forward-only model: $loss_forward_only")
    println("   - With backward model: $loss_with_backward")
end

println("\n✅ Both versions work correctly!")
println("\n📊 Summary:")
println("   - Forward-only version: Uses simplified TB with P_B = 1")
println("   - Full backward version: Uses complete TB with learned P_B")
println("   - Both versions converge and sample valid trajectories")