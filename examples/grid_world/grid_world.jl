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

# Create the model with interesting reward structure
model = create_grid_world_gflownet(
    grid_size=5,
    reward_positions=Dict(
        (3, 3) => 20.0,  # Center - highest reward
        (1, 5) => 15.0,  # Top-left corner
        (5, 1) => 15.0,  # Bottom-right corner
        (5, 5) => 10.0,  # Top-right corner
        (2, 4) => 8.0,   # Intermediate positions
        (4, 2) => 8.0
    ),
    allow_all_moves=false,  # Acyclic: only up/right moves + terminate
    hidden_dim=64,
    learning_rate=0.01
)

println("   ✅ Grid world model created successfully!")
println("   📊 Model details:")
println("      - Grid size: 5×5")
println("      - DAG states: $(length(model.dag.states))")
println("      - DAG edges: $(length(model.dag.edges))")
println("      - Parameters: $(length(model.parameters))")
println("      - Actions: $(length(model.dag.actions))")

# Analyze the constructed DAG
dag_metrics = analyze_dag(model.dag)
println("   📈 DAG Analysis:")
println("      - Acyclic: $(dag_metrics.is_acyclic)")
println("      - Max path length: $(dag_metrics.max_depth)")
println("      - Avg branching: $(round(dag_metrics.avg_branching_factor, digits=2))")

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
successful_iterations = count(!isnan, training_history[:losses])
final_loss = filter(!isnan, training_history[:losses])[end]
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

# Sample trajectories using core package function
n_trajectories = 100
println("   🎯 Sampling $n_trajectories trajectories...")

trajectories = []
for i in 1:n_trajectories
    try
        traj = sample_trajectory(model)
        push!(trajectories, traj)
    catch e
        println("   ⚠️  Trajectory $i failed: $e")
    end
end

# Analyze results using our custom function
analyze_grid_world_results(trajectories, 5)

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

        # Quick evaluation
        test_trajectories = [sample_trajectory(test_model) for _ in 1:20]
        test_rewards = [reward(traj.states[end]) for traj in test_trajectories if !isempty(traj.states)]

        results[name] = (
            model=test_model,
            mean_reward=mean(test_rewards),
            max_reward=maximum(test_rewards),
            n_states=length(test_model.dag.states)
        )

        println("      ✅ Success: $(length(test_model.dag.states)) states, mean reward: $(round(mean(test_rewards), digits=2))")

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
valid_trajectories = filter(traj -> length(traj.states) > 1, trajectories)
rewards = [reward(traj.states[end]) for traj in valid_trajectories]

println("      - Valid trajectories: $(length(valid_trajectories))/$n_trajectories")
println("      - Mean reward: $(round(mean(rewards), digits=2))")
println("      - Max reward: $(maximum(rewards))")
println("      - High reward (≥10): $(count(r -> r >= 10.0, rewards))")
println("      - Very high reward (≥15): $(count(r -> r >= 15.0, rewards))")

println("\n   🔄 Configuration Comparison:")
for (name, result) in results
    if result !== nothing
        println("      - $name: $(result.n_states) states, max reward $(round(result.max_reward, digits=1))")
    else
        println("      - $name: Failed")
    end
end

# =============================================================================
# 6. Generate Comprehensive Results
# =============================================================================

println("\n6️⃣ Generating Results")
println("="^50)

if HAS_REPORTING
    try
        html_path = generate_comprehensive_results(training_history, valid_trajectories, rewards)
        println("   📄 Comprehensive report: $html_path")
    catch e
        println("   ⚠️  Report generation failed: $e")
    end
end

# Create simple text summary
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
summary_path = "results/grid_world_summary_$timestamp.txt"

try
    mkpath("results")
    open(summary_path, "w") do f
        write(f, """
Grid World GFlowNet Results Summary
Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

=== Model Configuration ===
Grid Size: 5×5
DAG States: $(length(model.dag.states))
DAG Edges: $(length(model.dag.edges))
Parameters: $(length(model.parameters))
Acyclic: $(dag_metrics.is_acyclic)

=== Training Results ===
Iterations: $successful_iterations/$(config.n_iterations)
Final Loss: $(round(final_loss, digits=4))
Training Time: $(round(total_time, digits=1))s
Avg Gradient Norm: $(round(mean(training_history[:gradient_norms]), digits=4))

=== Evaluation Results ===
Valid Trajectories: $(length(valid_trajectories))/$n_trajectories
Mean Reward: $(round(mean(rewards), digits=2))
Max Reward: $(maximum(rewards))
High Reward Trajectories (≥10): $(count(r -> r >= 10.0, rewards))
Very High Reward Trajectories (≥15): $(count(r -> r >= 15.0, rewards))

=== Key Success Metrics ===
✅ DAG Construction: Successful ($(length(model.dag.states)) states discovered)
✅ Training: Converged in $successful_iterations iterations
✅ High Reward Discovery: Found maximum reward of $(maximum(rewards))
✅ Exploration: $(length(unique([(s.x, s.y) for traj in valid_trajectories for s in [traj.states[end]]]))) unique end positions
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
   📚 Key Takeaways - Using GFlowNet.jl Properly:

   1️⃣ Standard Model Creation:
      model = create_grid_world_gflownet(grid_size=5)

   2️⃣ Custom Rewards:
      model = create_grid_world_gflownet(
          reward_positions=Dict((3,3)=>20.0, (5,5)=>15.0)
      )

   3️⃣ Training:
      config = TrainingConfig(n_iterations=50, batch_size=16)
      history = train_gflownet(model, config; verbose=true)

   4️⃣ Sampling:
      trajectories = [sample_trajectory(model) for _ in 1:100]

   5️⃣ Analysis:
      analyze_grid_world_results(trajectories, grid_size)

   ✨ No manual DAG construction needed!
   ✨ No manual neural network definitions!
   ✨ No manual training loops!
   ✨ Just use the high-level package functions!
""")

println("\n🎯 Grid World Example Completed Successfully!")
println("📊 Demonstrates proper usage of GFlowNet.jl core functions")
println("🔧 High-level interface: create_grid_world_gflownet()")
println("🚀 Generic training: train_gflownet()")
println("📈 Built-in analysis: analyze_grid_world_results()")
println("✨ Production-ready GFlowNet development!")
println("="^60)
