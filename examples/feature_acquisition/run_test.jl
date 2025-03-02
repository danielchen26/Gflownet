#!/usr/bin/env julia

# Simple test runner for the feature acquisition example
# This script will run the main feature acquisition code and capture output to a log file

# Import required modules
using Dates

# Set working directory to the script directory
cd(@__DIR__)

# Open a log file
log_file = open("feature_acquisition_test.log", "w")

# Redirect standard output and error to the log file
original_stdout = stdout
original_stderr = stderr
redirect_stdout(log_file)
redirect_stderr(log_file)

println("=== Feature Acquisition Test Started at $(now()) ===")
println("Working directory: $(pwd())")

try
    # Include and run the main script
    println("Loading feature_acquisition.jl...")
    include("feature_acquisition.jl")
    
    # Run the main function from the script
    println("Running main() function...")
    main()
    
    println("Test completed successfully!")
catch e
    println("Error running feature acquisition example:")
    println(e)
    println("Stack trace:")
    println(stacktrace(catch_backtrace()))
end

println("=== Feature Acquisition Test Completed at $(now()) ===")

# Restore the original stdout and stderr
redirect_stdout(original_stdout)
redirect_stderr(original_stderr)

# Close the log file
close(log_file)

# Print a message to the console
println("Feature acquisition test completed. See feature_acquisition_test.log for results.") 