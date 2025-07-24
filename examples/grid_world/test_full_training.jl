#!/usr/bin/env julia

# Full training test for improved grid world
include("grid_world.jl")

try
    println("🔧 Creating GFlowNet model...")
    model = create_grid_world_gflownet(false)
    
    println("🚀 Full Training Run (150 iterations)...")
    results = train_grid_gflownet(model, 150, 16, true)
    
    if !isempty(results.losses)
        println("\n📊 Final Training Results:")
        println("   Final loss: $(round(results.losses[end], digits=3))")
        println("   Final mean reward: $(round(results.rewards_mean[end], digits=2))")
        println("   Final high-value rate: $(round(100*results.high_reward_rates[end], digits=1))%")
        println("   Final path length: $(round(results.path_lengths[end], digits=1))")
        
        # Test final performance
        println("\n🧪 Testing final performance (50 trajectories)...")
        final_rewards = Float64[]
        final_positions = Tuple{Int,Int}[]
        
        for i in 1:50
            traj = sample_trajectory_with_exploration(model, 0.1)  # Low exploration
            reward = GFlowNet.reward(traj.states[end])
            pos = (traj.states[end].x, traj.states[end].y)
            push!(final_rewards, reward)
            push!(final_positions, pos)
        end
        
        high_reward_count = count(r -> r >= 5.0, final_rewards)
        optimal_count = count(r -> r == 10.0, final_rewards)
        
        println("   Mean reward: $(round(mean(final_rewards), digits=2))")
        println("   High-value rate (R≥5.0): $(round(100*high_reward_count/50, digits=1))%")
        println("   Optimal rate (R=10.0): $(round(100*optimal_count/50, digits=1))%")
        
        # Show position distribution
        position_counts = Dict{Tuple{Int,Int}, Int}()
        for pos in final_positions
            position_counts[pos] = get(position_counts, pos, 0) + 1
        end
        
        sorted_positions = sort(collect(position_counts), by=x->x[2], rev=true)
        println("   Top 5 target positions:")
        for (i, (pos, count)) in enumerate(sorted_positions[1:min(5, length(sorted_positions))])
            reward = get(Dict(REWARD_POSITIONS), pos, 1.0)
            percentage = round(100*count/50, digits=1)
            println("     $i. Position $pos: $(count) times ($(percentage)%) → Reward = $reward")
        end
        
        # Success assessment
        if optimal_count >= 3  # 6% optimal rate
            println("\n🎉 SUCCESS: Achieved strong optimal targeting!")
        elseif high_reward_count >= 10  # 20% high-value rate
            println("\n✅ GOOD: Achieved target high-value performance!")
        elseif high_reward_count >= 5  # 10% high-value rate
            println("\n📈 PROGRESS: Showing clear learning progress!")
        else
            println("\n📚 LEARNING: Still building reward understanding")
        end
    else
        println("❌ Training failed - no results")
    end
    
catch e
    println("Error: $e")
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
