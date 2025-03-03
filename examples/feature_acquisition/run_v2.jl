#!/usr/bin/env julia

println("Starting Feature Acquisition Process (Version 2)...")
println("Working directory: $(pwd())")

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)
println("Project activated successfully")

# Create output directory for version 2
output_dir = "v2_results"
mkpath(output_dir)

try
    println("Loading version 2 feature acquisition file...")
    include("main_v2.jl")
    println("Feature acquisition file loaded successfully")

    println("Starting feature acquisition process...")
    # Run feature acquisition with specific parameters
    model, best_strategies = run_feature_acquisition(
        num_features=10,
        num_experiments=10,
        max_steps=5,
        cost_per_measurement=0.1,
        n_iterations=100,
        batch_size=16,
        output_prefix="v2",
        include_ground_truth=true
    )
    println("Feature acquisition completed successfully")

    println("Saving results...")
    # Set the global figs_dir variable before including visualization.jl
    global figs_dir = joinpath(@__DIR__, output_dir)
    println("Output directory set to: $figs_dir")
    
    # Save results and generate visualizations
    include("visualization.jl")
    
    println("Feature acquisition process completed successfully!")

catch e
    println("An error occurred during execution:")
    println(e)
    println(stacktrace(catch_backtrace()))
end

println("\nCompleted!") 