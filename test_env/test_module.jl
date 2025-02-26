#!/usr/bin/env julia

# Add the parent directory to the load path
push!(LOAD_PATH, dirname(dirname(abspath(@__FILE__))))

# Try to load the GFlowNet module
println("Loading GFlowNet module...")
try
    using GFlowNet
    println("✅ Successfully loaded GFlowNet module")
    
    # List the exported symbols
    println("\nExported symbols:")
    for name in names(GFlowNet)
        println("  - $name")
    end
    
catch e
    println("❌ Error loading GFlowNet module:")
    println(e)
    
    # Try to diagnose the issue
    println("\nAttempting to diagnose the issue...")
    
    # Check if we can load individual components
    println("\nTesting basic imports:")
    try
        using Graphs
        println("✅ Graphs package loaded successfully")
    catch e
        println("❌ Error loading Graphs package: $e")
    end
    
    try 
        include(joinpath(dirname(dirname(abspath(@__FILE__))), "src", "types.jl"))
        println("✅ types.jl loaded successfully")
    catch e
        println("❌ Error loading types.jl: $e")
    end
end 