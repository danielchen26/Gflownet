#!/usr/bin/env julia

# Load the Graphs package which is required
using Graphs

# Define the source directory
src_dir = joinpath(dirname(dirname(abspath(@__FILE__))), "src")

# Helper function to test including a file
function test_include(filename)
    filepath = joinpath(src_dir, filename)
    println("Testing include of: $filepath")
    try
        include(filepath)
        println("✅ Successfully included $filename")
        return true
    catch e
        println("❌ Error including $filename:")
        println(e)
        return false
    end
end

# Try including each file individually
files_to_test = [
    "types.jl",
    "directed_acyclic_graph.jl",
    "flow_networks.jl",
    "utils/utils.jl",
    "policies/forward_policy.jl",
    "policies/backward_policy.jl",
    "training/flow_matching.jl",
    "training/detailed_balance.jl",
    "training/trajectory_balance.jl"
]

for file in files_to_test
    if !test_include(file)
        println("Stopping tests due to failure.")
        break
    end
end

println("File testing completed.") 