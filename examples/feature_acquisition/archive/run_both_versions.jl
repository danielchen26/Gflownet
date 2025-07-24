#!/usr/bin/env julia

println("Starting Feature Acquisition Process (Both Versions)...")
println("Working directory: $(pwd())")

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)
println("Project activated successfully")

# Clean up any previous results
println("Cleaning up previous results directories...")
rm("v2_results", force=true, recursive=true)
rm("v3_results", force=true, recursive=true)
println("Clean-up completed")

# Run Version 2
println("\n=========================================")
println("RUNNING VERSION 2 (All features unmeasured)")
println("=========================================\n")

# Run v2 in a separate process
run(`julia -e '
    println("Starting v2 in separate process...")
    cd(raw"$(pwd())")
    using Pkg
    Pkg.activate(raw"$(pwd())")
    
    include("main_v2.jl")
    
    # Create output directory
    output_dir = "v2_results"
    isdir(output_dir) || mkdir(output_dir)
    
    # Run feature acquisition
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
    global figs_dir = output_dir
    global experiment_values = global_experiment_values
    
    # Include visualization script
    include("visualization.jl")
    
    println("v2 completed successfully!")
'`)

# Run Version 3
println("\n=========================================")
println("RUNNING VERSION 3 (20% features pre-measured)")
println("=========================================\n")

# Run v3 in a separate process
run(`julia -e '
    println("Starting v3 in separate process...")
    cd(raw"$(pwd())")
    using Pkg
    Pkg.activate(raw"$(pwd())")
    
    include("main_v3.jl")
    
    # Create output directory
    output_dir = "v3_results"
    isdir(output_dir) || mkdir(output_dir)
    
    # Run feature acquisition
    results = run_feature_acquisition_v3(
        num_features=10,
        num_experiments=10,
        max_steps=5,
        cost_per_measurement=0.1,
        n_iterations=100,
        batch_size=16,
        observation_ratio=0.2,
        output_prefix="v3",
        include_ground_truth=true
    )
    
    # Set global variables for visualization
    global figs_dir = output_dir
    global model = results["model"]
    global analysis = results["metrics"]
    global experiment_values = global_experiment_values
    
    # Include visualization script
    include("visualization.jl")
    
    println("v3 completed successfully!")
'`)

println("\nBoth versions completed!")
println("Version 2 results in: v2_results/")
println("Version 3 results in: v3_results/")
println("\nMain differences:")
println("- Version 2: All features start unmeasured")
println("- Version 3: 20% of features are pre-measured")
println("Both versions now use the same structured data generation approach") 