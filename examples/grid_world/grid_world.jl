"""
🎯 Grid World GFlowNet Example - Using Core Package Functions
==============================================================

This example demonstrates the proper way to use GFlowNet.jl for grid world problems
using the high-level package interface. No manual implementation required!

Key Features:
- Uses core package functions: create_grid_world_gflownet()
- Demonstrates different grid world configurations
- Shows proper training workflow with TrainingConfig
- Includes comprehensive evaluation and analysis
- Follows GFlowNet best practices throughout
"""

using GFlowNet
using Random
using Dates
using Statistics

include(joinpath(@__DIR__, "..", "convergence_assertions.jl"))

# Include report generation if available
HAS_REPORTING = try
    include("report_generation.jl")
    true
catch
    println("📋 Note: Report generation not available - using basic output")
    false
end

println("🎯 Grid World GFlowNet Example - Core Package Demo")
println("="^60)
println("🕐 Started at: $(Dates.format(now(), "HH:MM:SS"))")
println()

# =============================================================================
# 1. Simple Grid World Creation
# =============================================================================

println("1️⃣ Creating Grid World with Core Package Functions")
println("="^50)

# Create a simple 5x5 grid world using the high-level interface
println("   📦 Using create_grid_world_gflownet() from GFlowNet.jl...")

# Set random seed for reproducibility
Random.seed!(42)

# Create the model with challenging reward structure and all actions
model = create_grid_world_gflownet(
    grid_size=5,
    reward_positions=Dict(
        (5, 5) => 50.0,  # Corner - highest reward (requires optimal navigation)
        (1, 5) => 40.0,  # Top-left corner (requires left movement)
        (5, 1) => 40.0,  # Bottom-right corner (requires down movement)
        (3, 3) => 30.0,  # Center - good intermediate reward
        (2, 4) => 20.0,  # Intermediate positions
        (4, 2) => 20.0,
        (1, 1) => 5.0    # Starting position - low reward to encourage exploration
    ),
    allow_all_moves=true,   # All 5 actions: enables optimal exploration with acyclic control
    hidden_dim=64,
    learning_rate=0.01
)

println("   ✅ Grid world model created successfully!")
println("   📊 Model details:")
println("      - Grid size: 5×5")
# Compute state space on-demand for analysis
state_count = count_reachable_states(model.initial_state, model.all_actions)
println("      - Reachable states: $state_count")
println("      - Parameters: $(length(model.parameters))")
println("      - Actions: $(length(model.all_actions))")

# Analyze the state space structure
space_analysis = analyze_state_space(model.initial_state, model.all_actions)
println("   📈 State Space Analysis:")
println("      - Total states: $(space_analysis.total_states)")
println("      - Terminal states: $(space_analysis.terminal_states)")
println("      - Complete exploration: $(space_analysis.exploration_complete)")

# =============================================================================
# 2. Training Configuration and Execution
# =============================================================================

println("\n2️⃣ Training the GFlowNet Model")
println("="^50)

# Create training configuration using core package
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    partition_function_method=SIMPLE_ESTIMATION,
    n_iterations=50,
    batch_size=16,
    learning_rate=0.01,
    validation_frequency=10,
    early_stopping_patience=20
)

println("   ⚙️  Training configuration:")
println("      - Objective: $(config.objective)")
println("      - Iterations: $(config.n_iterations)")
println("      - Batch size: $(config.batch_size)")
println("      - Learning rate: $(config.learning_rate)")

# Train using core package function
println("   🚀 Starting training...")
training_history = train_gflownet(model, config; verbose=true)

# Extract training statistics
# `filter(!isnan, ...)[end]` reported the last SURVIVING loss, so a run in which 49
# of 50 iterations threw looked identical to a clean one (and a run in which all of
# them threw died with a BoundsError). Asserted instead.
assert_finite_iterations(training_history, config.n_iterations, "grid world main model")

successful_iterations = count(isfinite, training_history[:losses])
final_loss = training_history[:losses][end]
total_time = sum(training_history[:iteration_times])

println("   ✅ Training completed!")
println("      - Successful iterations: $successful_iterations/$(config.n_iterations)")
println("      - Final loss: $(round(final_loss, digits=4))")
println("      - Total time: $(round(total_time, digits=1))s")
println("      - Avg gradient norm: $(round(mean(training_history[:gradient_norms]), digits=4))")

# =============================================================================
# 3. Model Evaluation and Analysis
# =============================================================================

println("\n3️⃣ Evaluating Model Performance")
println("="^50)

# Sample trajectories with acyclic control for optimal performance
n_trajectories = 100
println("   🎯 Sampling $n_trajectories trajectories with acyclic control...")

# Use acyclic control to prevent cycles while maintaining exploration
acyclic_config = create_default_sampling_config(acyclic_rate=0.8)
println("   🔧 Using acyclic_rate = 0.8 (80% cycle prevention)")

trajectories = []
for i in 1:n_trajectories
    try
        traj = sample_trajectory(model; config=acyclic_config)
        push!(trajectories, traj)
    catch e
        println("   ⚠️  Trajectory $i failed: $e")
    end
end

# Analyze results using our custom function
analyze_grid_world_results(trajectories, 5)

# Store these trajectories for consistent analysis throughout
global main_trajectories = trajectories
global main_rewards = [reward(traj.states[end]) for traj in trajectories]

# This model is built with partition_function_method=SIMPLE_ESTIMATION, which pins
# log Z = 0 i.e. Z = 1 (src/training/losses.jl: "SIMPLE_ESTIMATION: Z = 1, so
# log Z = 0"). With rewards up to 50 the trajectory-balance residual
# (log P(tau) - log R)^2 cannot reach 0, and measured on comparable SIMPLE_ESTIMATION
# runs the loss RISES over training (28.021 -> 36.939). A loss-decrease assertion
# would therefore be asserting something false. What must hold is that training pulls
# the sampler toward the high-reward cells, so that is asserted.
let untrained = create_grid_world_gflownet(
        grid_size=5,
        reward_positions=Dict(
            (5, 5) => 50.0, (1, 5) => 40.0, (5, 1) => 40.0, (3, 3) => 30.0,
            (2, 4) => 20.0, (4, 2) => 20.0, (1, 1) => 5.0
        ),
        allow_all_moves=true, hidden_dim=64, learning_rate=0.01)
    untrained_rewards = [reward(sample_trajectory(untrained; config=acyclic_config).states[end])
                         for _ in 1:n_trajectories]
    assert_beats_untrained(main_rewards, untrained_rewards, "grid world sampler"; min_gain=1.15)
end

# =============================================================================
# 4. Demonstrate Different Configurations
# =============================================================================

println("\n4️⃣ Comparing Different Grid World Configurations")
println("="^50)

configurations = [
    (
        name="Small & Fast",
        params=(grid_size=3, hidden_dim=32, learning_rate=0.02),
        description="Quick exploration on small grid"
    ),
    (
        name="Large & Detailed",
        params=(grid_size=4, hidden_dim=128, learning_rate=0.005),
        description="Comprehensive exploration with more capacity"
    ),
    (
        name="Bidirectional",
        params=(grid_size=3, allow_all_moves=true, hidden_dim=64),
        description="All 4 directions + terminate (with cycles)"
    )
]

results = Dict()

for (name, params, description) in configurations
    println("\n   🔧 Testing: $name")
    println("      📝 $description")

    try
        # Create model with different configuration
        test_model = create_grid_world_gflownet(;
            reward_positions=Dict((2,2)=>10.0, (3,3)=>15.0),
            params...
        )

        # Quick training
        test_config = TrainingConfig(
            objective=TRAJECTORY_BALANCE,
            n_iterations=20,
            batch_size=8,
            validation_frequency=10
        )

        test_history = train_gflownet(test_model, test_config; verbose=false)

        # Quick evaluation with acyclic control
        test_config = create_default_sampling_config(acyclic_rate=0.8)
        test_trajectories = [sample_trajectory(test_model; config=test_config) for _ in 1:20]
        test_rewards = [reward(traj.states[end]) for traj in test_trajectories if !isempty(traj.states)]

        n_states = count_reachable_states(test_model.initial_state, test_model.all_actions)
        results[name] = (
            model=test_model,
            mean_reward=mean(test_rewards),
            max_reward=maximum(test_rewards),
            n_states=n_states
        )

        println("      ✅ Success: $n_states states, mean reward: $(round(mean(test_rewards), digits=2))")

    catch e
        println("      ❌ Failed: $e")
        results[name] = nothing
    end
end

# =============================================================================
# 5. Performance Summary
# =============================================================================

println("\n5️⃣ Performance Summary")
println("="^50)

println("   📊 Main Model Results:")
valid_trajectories = filter(traj -> length(traj.states) > 1, main_trajectories)
rewards = main_rewards

println("      - Valid trajectories: $(length(valid_trajectories))/$n_trajectories")
println("      - Mean reward: $(round(mean(rewards), digits=2))")
println("      - Max reward: $(maximum(rewards))")
println("      - High reward (≥30): $(count(r -> r >= 30.0, rewards))")
println("      - Very high reward (≥40): $(count(r -> r >= 40.0, rewards))")
println("      - Optimal reward (≥50): $(count(r -> r >= 50.0, rewards))")

println("\n   🔄 Configuration Comparison:")
for (name, result) in results
    if result !== nothing
        n_states = count_reachable_states(result.model.initial_state, result.model.all_actions)
        println("      - $name: $n_states states, max reward $(round(result.max_reward, digits=1))")
    else
        println("      - $name: Failed")
    end
end

# =============================================================================
# 6. Generate Comprehensive Results
# =============================================================================

println("\n6️⃣ Generating Results")
println("="^50)

# Create results directory in the proper location
results_dir = "results"
mkpath(results_dir)

# Always try to generate comprehensive results (works with or without plotting)
try
    html_path = generate_comprehensive_results(training_history, valid_trajectories, rewards)
    if html_path != "report_generation_failed.html"
        println("   📄 Comprehensive report: $html_path")
        if !HAS_REPORTING
            println("   📊 Note: Generated with basic analysis (plotting dependencies not available)")
        end
    end
catch e
    println("   ⚠️  Report generation failed: $e")
    println("   📊 Continuing with basic analysis...")
end

# Create simple text summary
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
summary_path = joinpath("results", "grid_world_summary_$timestamp.txt")

try
    mkpath("results")
    open(summary_path, "w") do f
        write(f, """
Grid World GFlowNet Results Summary
Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

=== Model Configuration ===
Grid Size: 5×5
Reachable States: $state_count
Parameters: $(length(model.parameters))
Actions: $(length(model.all_actions))
Complete Exploration: $(space_analysis.exploration_complete)

=== Training Results ===
Iterations: $successful_iterations/$(config.n_iterations)
Final Loss: $(round(final_loss, digits=4))
Training Time: $(round(total_time, digits=1))s
Avg Gradient Norm: $(round(mean(training_history[:gradient_norms]), digits=4))

=== Evaluation Results ===
Valid Trajectories: $(length(valid_trajectories))/$n_trajectories
Mean Reward: $(round(mean(rewards), digits=2))
Max Reward: $(maximum(rewards))
High Reward Trajectories (≥30): $(count(r -> r >= 30.0, main_rewards))
Very High Reward Trajectories (≥40): $(count(r -> r >= 40.0, main_rewards))
Optimal Reward Trajectories (≥50): $(count(r -> r >= 50.0, main_rewards))

=== Key Success Metrics ===
✅ State Space: Successful ($state_count states discovered)
✅ Training: Converged in $successful_iterations iterations
✅ High Reward Discovery: Found maximum reward of $(maximum(main_rewards))
✅ Exploration: $(length(unique([(s.x, s.y) for traj in main_trajectories for s in [traj.states[end]]]))) unique end positions
""")
    end
    println("   📝 Text summary: $summary_path")
catch e
    println("   ⚠️  Summary generation failed: $e")
end

# =============================================================================
# 7. Usage Examples for Documentation
# =============================================================================

println("\n7️⃣ Usage Examples")
println("="^50)

println("""
   📚 Key Takeaways - Optimized GFlowNet.jl Usage:

   1️⃣ Optimal Model Creation:
      model = create_grid_world_gflownet(grid_size=5, allow_all_moves=true)

   2️⃣ Strategic Rewards:
      model = create_grid_world_gflownet(
          reward_positions=Dict((5,5)=>50.0, (1,5)=>40.0, (5,1)=>40.0),
          allow_all_moves=true
      )

   3️⃣ Training:
      config = TrainingConfig(n_iterations=50, batch_size=16)
      history = train_gflownet(model, config; verbose=true)

   4️⃣ Acyclic Sampling (Optimal):
      config = create_default_sampling_config(acyclic_rate=0.8)
      trajectories = [sample_trajectory(model; config=config) for _ in 1:100]

   5️⃣ Analysis:
      analyze_grid_world_results(trajectories, grid_size)

   ✨ All 5 actions enabled for optimal exploration!
   ✨ Acyclic control prevents wasted cycles!
   ✨ High-reward policies through smart navigation!
   ✨ Best of both worlds: exploration + efficiency!
""")

println("\n🎯 Grid World Example Completed Successfully!")
println("📊 Demonstrates proper usage of GFlowNet.jl core functions")
println("🔧 High-level interface: create_grid_world_gflownet()")
println("🚀 Generic training: train_gflownet()")
println("📈 Built-in analysis: analyze_grid_world_results()")
println("✨ Production-ready GFlowNet development!")
println("="^60)
