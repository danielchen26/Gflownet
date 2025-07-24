#!/usr/bin/env julia

println("Starting Feature Acquisition Process (Version 2)...")
println("Working directory: $(pwd())")

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)
println("Project activated successfully")

# Create output directory
rm("v2_results", force=true, recursive=true)
mkdir("v2_results")

# Load the main file
include("main_v2.jl")

# Run feature acquisition directly
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

# Set global variables for visualization
global figs_dir = "v2_results"
global experiment_values = global_experiment_values

# Include visualization script
println("Running visualization...")
include("visualization.jl")

println("Version 2 completed successfully!") 