#!/usr/bin/env julia

println("Starting Feature Acquisition Process (Version 1)...")
println("Working directory: $(pwd())")

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)
println("Project activated successfully")

# Create output directory for version 1
output_dir = "v1_results"
mkpath(output_dir)

try
    println("Loading version 1 feature acquisition file...")
    include("main_v1.jl")
    println("Feature acquisition file loaded successfully")

    println("Starting feature acquisition process...")
    # Run feature acquisition
    model, best_strategies = main()
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