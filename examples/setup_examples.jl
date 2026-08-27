#!/usr/bin/env julia

# Script to set up all example environments
# Run this first before trying the examples

println("Setting up GFlowNet example environments...")

# Discover every example environment instead of hardcoding a list. The old
# hardcoded list omitted all seven core_features/* environments, which is
# exactly why five of them carried a fabricated GFlowNet UUID for so long --
# this script never touched them, so nobody ever saw them fail to resolve.
examples = String[]
const _EXAMPLES_ROOT = @__DIR__   # bind it: `x == @__DIR__ && y` makes the macro swallow `&& y`
for (root, _, files) in walkdir(_EXAMPLES_ROOT)
    root == _EXAMPLES_ROOT && continue      # the examples/ root has no Project.toml
    occursin("archive", root) && continue   # frozen v1/v2/v3 lineage, not runnable
    "Project.toml" in files || continue
    push!(examples, relpath(root, _EXAMPLES_ROOT))
end
sort!(examples)
println("Discovered $(length(examples)) example environment(s): ", join(examples, ", "))

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
        
        # Just instantiate. Do NOT Pkg.rm + Pkg.develop.
        #
        # That pair was destructive: Pkg.develop REWRITES the example's
        # Project.toml, deleting the `[sources]` table (and its comments) and
        # replacing the resolution with an ABSOLUTE dev path in the manifest. The
        # manifests are gitignored on purpose -- one resolved on a different Julia
        # version cannot be instantiated here -- so the absolute path is all that
        # remained, and it pointed at a volume that no longer exists:
        #   /Volumes/chetianc/Documents/Decision_science_codes/Gflownet
        # which is exactly why active_learning, causal_discovery and
        # molecule_design all failed with
        #   ArgumentError: Package GFlowNet [2d7ca041-...] is required but does not
        #   seem to be installed
        #
        # Each example Project.toml now carries `[sources] GFlowNet = {path="../.."}`,
        # which is relative, version-control friendly and resolved by instantiate
        # alone. Running the old sequence would silently delete that again.
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
        "feature_acquisition" => "main.jl"
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