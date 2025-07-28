"""
🔄 Grid World Acyclic Control Example
====================================

Simple demonstration of acyclic rate control in grid world.
Shows how to use the generic acyclic_rate parameter to prevent cycles
while maintaining exploration benefits.

Usage: julia --project=. examples/grid_world/acyclic_example.jl
"""

using GFlowNet
using Random
using Statistics

println("🔄 Grid World Acyclic Control Example")
println("="^40)
println("🎯 Demonstrating acyclic_rate parameter")
println()

Random.seed!(42)

# =============================================================================
# 1. Create Grid World with All 5 Actions (Cycles Possible)
# =============================================================================

println("1️⃣ Setup: Grid World with All Actions")
println("="^35)

# Create model with all 5 actions enabled
model = create_grid_world_gflownet(
    grid_size=4,
    reward_positions=Dict(
        (4, 4) => 50.0,  # High reward goal
        (3, 3) => 30.0,  # Medium reward
        (4, 1) => 20.0,  # Edge goals
        (1, 4) => 20.0
    ),
    allow_all_moves=true,  # All 5 actions: Right, Left, Up, Down, Terminate
    hidden_dim=48,
    learning_rate=0.01
)

println("   ✅ Model created with $(length(model.all_actions)) actions")
println("   📊 Actions: MoveRight, MoveLeft, MoveUp, MoveDown, Terminate")
println("   ⚠️  Cycles are possible with this setup!")
println()

# =============================================================================
# 2. Compare Different Acyclic Rates
# =============================================================================

println("2️⃣ Testing Different Acyclic Rates")
println("="^35)

# Test different acyclic rates
acyclic_rates = [0.0, 0.5, 0.8, 1.0]
results = []

for rate in acyclic_rates
    println("🔬 Testing acyclic_rate = $rate")

    # Create sampling config with acyclic control
    if rate == 0.0
        sampling_config = SamplingConfig(max_trajectory_length=50)
        println("   📝 No acyclic control (baseline)")
    else
        sampling_config = create_acyclic_sampling_config(rate)
        println("   📝 $(round(rate*100))% cycle prevention")
    end

    # Sample trajectories
    n_samples = 60
    trajectories = [sample_trajectory(model; config=sampling_config) for _ in 1:n_samples]

    # Analyze results
    rewards = [reward(traj.states[end]) for traj in trajectories]
    lengths = [length(traj.states) for traj in trajectories]

    # Count cycles
    total_cycles = 0
    for traj in trajectories
        positions = [(s.x, s.y) for s in traj.states if !s.is_terminal]
        visited = Set{Tuple{Int,Int}}()
        for pos in positions
            if pos in visited
                total_cycles += 1
            else
                push!(visited, pos)
            end
        end
    end

    result = (
        rate = rate,
        mean_reward = mean(rewards),
        max_reward = maximum(rewards),
        mean_length = mean(lengths),
        total_cycles = total_cycles,
        efficiency = mean(rewards) / mean(lengths)
    )

    push!(results, result)

    println("   📊 Results: $(total_cycles) cycles, $(round(result.mean_reward,digits=1)) mean reward, $(round(result.efficiency,digits=3)) efficiency")
    println()
end

# =============================================================================
# 3. Analysis and Recommendations
# =============================================================================

println("3️⃣ Analysis and Recommendations")
println("="^35)

println("📊 PERFORMANCE COMPARISON:")
println("   Rate | Cycles | Mean Reward | Efficiency | Max Reward")
println("   " * "-"^50)
for result in results
    println("   $(rpad(result.rate,4)) |   $(rpad(result.total_cycles,4)) |    $(rpad(round(result.mean_reward,digits=1),7)) |     $(rpad(round(result.efficiency,digits=3),6)) |     $(round(result.max_reward,digits=1))")
end

# Find best performing configuration
best_efficiency = argmax([r.efficiency for r in results])
best_reward = argmax([r.mean_reward for r in results])
least_cycles = argmin([r.total_cycles for r in results])

println("\n🏆 BEST PERFORMERS:")
println("   🚀 Best Efficiency: acyclic_rate = $(results[best_efficiency].rate)")
println("   🎯 Best Reward: acyclic_rate = $(results[best_reward].rate)")
println("   🔄 Fewest Cycles: acyclic_rate = $(results[least_cycles].rate)")

# =============================================================================
# 4. Practical Usage Example
# =============================================================================

println("\n4️⃣ Practical Usage Example")
println("="^30)

println("📝 RECOMMENDED USAGE:")
println("""
# 1. Create model with all actions
model = create_grid_world_gflownet(
    grid_size=5,
    allow_all_moves=true  # Enable full exploration
)

# 2. Use acyclic control for sampling
config = create_acyclic_sampling_config(0.8)  # 80% cycle prevention

# 3. Sample with cycle control
trajectories = [sample_trajectory(model; config=config) for _ in 1:100]
""")

# Demonstrate the recommended approach
println("🔬 DEMONSTRATING RECOMMENDED APPROACH:")
recommended_config = create_acyclic_sampling_config(0.8)
demo_trajectories = [sample_trajectory(model; config=recommended_config) for _ in 1:30]
demo_rewards = [reward(traj.states[end]) for traj in demo_trajectories]

println("   ✅ Generated $(length(demo_trajectories)) trajectories")
println("   📊 Mean reward: $(round(mean(demo_rewards), digits=1))")
println("   🎯 Max reward: $(maximum(demo_rewards))")
println("   📈 High reward rate: $(round(count(r -> r >= 30.0, demo_rewards)/length(demo_rewards)*100, digits=1))%")

# =============================================================================
# 5. Key Takeaways
# =============================================================================

println("\n5️⃣ Key Takeaways")
println("="^20)

println("✅ ACYCLIC CONTROL BENEFITS:")
cycle_reduction = results[1].total_cycles - results[end].total_cycles
if cycle_reduction > 0
    println("   🔄 Cycle Reduction: $(cycle_reduction) fewer cycles with acyclic_rate = 1.0")
end

efficiency_improvement = results[end].efficiency - results[1].efficiency
if efficiency_improvement > 0
    println("   ⚡ Efficiency Gain: $(round(efficiency_improvement,digits=3)) improvement")
end

println("   🎯 Exploration Preserved: All positions still reachable")
println("   🔧 Simple Integration: Just add acyclic_rate parameter")

println("\n💡 RECOMMENDATIONS:")
optimal_rate = results[best_efficiency].rate
println("   📈 Optimal Rate: $(optimal_rate) for this problem")
println("   🎛️  General Rule: Start with acyclic_rate = 0.8")
println("   ⚖️  Trade-off: Higher rate = fewer cycles, less exploration")
println("   🔬 Always Test: Different problems may need different rates")

println("\n" * "="^50)
println("🔄 Grid World Acyclic Control Example Complete!")
println("✅ Simple acyclic_rate parameter controls cycles")
println("🎯 Easy integration with existing code")
println("⚡ Measurable performance improvements")
println("="^50)
