#!/usr/bin/env julia

println("Starting cleanup process...")

# Get the directory of the current script
script_dir = @__DIR__

# Define paths to clean
v2_results_dir = joinpath(script_dir, "v2_results")
v3_results_dir = joinpath(script_dir, "v3_results")
temp_files = ["simple_plot.jl", "test_feature_selection.jl"]

# Clean v2_results directory
if isdir(v2_results_dir)
    println("Cleaning v2_results directory...")
    rm(v2_results_dir, recursive=true, force=true)
end
mkdir(v2_results_dir)
println("Created fresh v2_results directory")

# Clean v3_results directory
if isdir(v3_results_dir)
    println("Cleaning v3_results directory...")
    rm(v3_results_dir, recursive=true, force=true)
end
mkdir(v3_results_dir)
println("Created fresh v3_results directory")

# Remove temporary files
for file in temp_files
    filepath = joinpath(script_dir, file)
    if isfile(filepath)
        println("Removing temporary file: $file")
        rm(filepath, force=true)
    end
end

println("Cleanup completed successfully") 