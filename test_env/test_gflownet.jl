#!/usr/bin/env julia

# Add the parent directory to the load path so we can import GFlowNet
push!(LOAD_PATH, dirname(dirname(abspath(@__FILE__))))

# Try to load only the Graphs package
println("Loading Graphs package...")
using Graphs

# Try to include the types.jl file directly to check the SimpleGraph import
println("Attempting to include types.jl file...")
include(joinpath(dirname(dirname(abspath(@__FILE__))), "src", "types.jl"))

# If we get here, the file loaded successfully
println("types.jl file loaded successfully!") 