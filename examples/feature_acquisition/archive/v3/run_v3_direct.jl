#!/usr/bin/env julia

println("Starting Feature Acquisition Process (Version 3)...")
println("Working directory: $(pwd())")

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)
println("Project activated successfully")

# Create output directory
rm("v3_results", force=true, recursive=true)
mkdir("v3_results")

# Load the main file
include("main_v3.jl")

# Run feature acquisition directly
results = run_feature_acquisition_v3(
    num_features=10,
    num_experiments=10,
    max_steps=5,
    cost_per_measurement=0.1,
    n_iterations=100,
    batch_size=16,
    observation_ratio=0.2,
    output_prefix="v3"
)

# Set global variables for visualization
global figs_dir = "v3_results"
global model = results["model"]
global analysis = results["metrics"]
global experiment_values = global_experiment_values

# Include visualization script
println("Running visualization...")
include("visualization.jl")

println("Version 3 completed successfully!") 