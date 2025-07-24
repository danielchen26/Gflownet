#!/usr/bin/env julia

println("=== RUNNING BOTH VERSIONS OF FEATURE ACQUISITION ===")

# Get the directory of the current script
script_dir = @__DIR__

# First clean up the results directories
include(joinpath(script_dir, "cleanup.jl"))

# Run Version 2
println("\n\n=== RUNNING VERSION 2 ===\n")
include(joinpath(script_dir, "run_v2.jl"))

# Run Version 3
println("\n\n=== RUNNING VERSION 3 ===\n")
include(joinpath(script_dir, "run_v3.jl"))

println("\n=== BOTH VERSIONS COMPLETED SUCCESSFULLY ===")
println("Results are available in:")
println("- v2_results directory")
println("- v3_results directory") 