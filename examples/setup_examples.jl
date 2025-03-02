#!/usr/bin/env julia

# Script to set up all example environments
# Run this first before trying the examples

println("Setting up GFlowNet example environments...")

examples = [
    "grid_world",
    "molecule_design",
    "causal_discovery",
    "active_learning",
    "feature_acquisition"
]

using Pkg

# Get the main package directory
main_dir = dirname(@__DIR__)
println("Main package directory: $main_dir")

# Now set up each example
successful_setups = 0
failed_setups = 0

for example in examples
    example_dir = joinpath(@__DIR__, example)
    
    if !isdir(example_dir)
        println("\n⚠️  Warning: Example directory $example not found, skipping.")
        global failed_setups += 1
        continue
    end
    
    println("\n[$example] Setting up environment...")
    
    try
        # Activate the example environment
        Pkg.activate(example_dir)
        println("[$example] Activated environment in $example_dir")
        
        # Remove any existing GFlowNet reference that might cause conflicts
        try
            Pkg.rm("GFlowNet")
            println("[$example] Removed existing GFlowNet reference")
        catch e
            # It's fine if the package wasn't there
            println("[$example] Note: GFlowNet was not previously added (this is normal)")
        end
        
        # Ensure GFlowNet is properly linked as a development dependency
        println("[$example] Linking GFlowNet from $main_dir...")
        Pkg.develop(path=main_dir)
        
        # Install dependencies
        println("[$example] Installing dependencies...")
        Pkg.instantiate()
        
        println("[$example] ✅ Setup completed successfully!")
        global successful_setups += 1
    catch e
        println("[$example] ❌ Error during setup: $(typeof(e))")
        println("[$example] Error details: $e")
        println("[$example] Continuing with next example...")
        global failed_setups += 1
    end
end

println("\n=== Setup Summary ===")
println("$successful_setups examples set up successfully")
if failed_setups > 0
    println("$failed_setups examples failed to set up")
end

println("\nTo run an example:")
println("1. Navigate to its directory: cd examples/example_name")
println("2. Run the example: julia example_name.jl") 

if successful_setups > 0
    println("\nExamples ready to run:")
    # Map of example directory names to their main script files
    example_files = Dict(
        "grid_world" => "grid_world.jl",
        "molecule_design" => "molecule_example.jl",
        "causal_discovery" => "causal_discovery.jl",
        "active_learning" => "active_learning.jl",
        "feature_acquisition" => "feature_acquisition.jl"
    )
    
    for example in examples
        example_dir = joinpath(@__DIR__, example)
        if isdir(example_dir)
            if haskey(example_files, example) && isfile(joinpath(example_dir, example_files[example]))
                println("  cd examples/$example && julia $(example_files[example])")
            end
        end
    end
end 