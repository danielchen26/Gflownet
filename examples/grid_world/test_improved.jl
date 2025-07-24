#!/usr/bin/env julia

# Test script for improved grid world training
include("grid_world.jl")

try
    println("🔧 Creating improved GFlowNet model...")
    model = create_grid_world_gflownet(false)
    
    println("🔍 Validating components...")
    validation_results, all_passed = validate_gflownet_components(model, true)
    
    if !all_passed
        println("❌ Validation failed")
        exit(1)
    end
    
    println("🧪 Testing exploration sampling...")
    for i in 1:5
        traj = sample_trajectory_with_exploration(model, 0.8)
        println("   Trajectory $i: $(length(traj.states)) states, final reward: $(GFlowNet.reward(traj.states[end]))")
    end
    
    println("🚀 Testing improved training (20 iterations)...")
    results = train_grid_gflownet(model, 20, 8, true)
    
    if !isempty(results.losses)
        println("✅ Improved training test passed!")
        println("   Final loss: $(round(results.losses[end], digits=3))")
        println("   Final mean reward: $(round(results.rewards_mean[end], digits=2))")
        println("   Final high-value rate: $(round(100*results.high_reward_rates[end], digits=1))%")
        println("   Final path length: $(round(results.path_lengths[end], digits=1))")
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
