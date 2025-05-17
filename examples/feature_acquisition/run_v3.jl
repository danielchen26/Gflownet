#!/usr/bin/env julia

println("Starting Feature Acquisition Process (Version 3)...")
println("Working directory: $(pwd())")

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)
println("Project activated successfully")

# Create output directory
output_dir = "v3_results"
isdir(output_dir) || mkdir(output_dir)

println("Loading version 3 feature acquisition file...")
try
    # Instead of include, we'll use a separate module to avoid name conflicts
    module V3FeatureAcquisition
        using Random
        const rng = Random.MersenneTwister(42)
        
        # Load the main file
        include("main_v3.jl")
        
        # Export function we need
        export run_feature_acquisition_v3
    end
    
    # Run the feature acquisition with specific parameters
    results = V3FeatureAcquisition.run_feature_acquisition_v3(
        num_features=10,  # Total number of features
        num_experiments=10,  # Number of experiments to evaluate
        max_steps=5,  # Maximum number of steps (measurements)
        cost_per_measurement=0.1,  # Cost per measurement
        n_iterations=100,  # Number of training iterations
        batch_size=16,  # Batch size for training
        observation_ratio=0.2,  # 20% of features initially observed
        output_prefix="v3",  # Prefix for output files
        include_ground_truth=true  # Include ground truth in plots
    )
    
    println("\nFeature acquisition process completed successfully!")
catch e
    println("Error during feature acquisition: $e")
    rethrow(e)
end

println("\nCompleted!") 