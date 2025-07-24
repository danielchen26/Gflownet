#!/usr/bin/env julia

# Quick test script for grid world training
include("grid_world.jl")

try
    println("Creating model...")
    model = create_grid_world_gflownet(false)
    
    println("Testing short training run...")
    results = train_grid_gflownet(model, 5, 4, true)
    
    if !isempty(results.losses)
        println("✅ Training test passed!")
        println("   Final loss: $(round(results.losses[end], digits=3))")
        println("   Mean reward: $(round(results.rewards_mean[end], digits=2))")
    else
        println("❌ Training failed - no results")
    end
catch e
    println("Error: $e")
end
