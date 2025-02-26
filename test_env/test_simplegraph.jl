#!/usr/bin/env julia

# Test script to verify that SimpleGraph can be loaded from the Graphs package
using Graphs

# Create a simple graph
g = SimpleGraph(5)  # Create a graph with 5 vertices

# Add some edges
add_edge!(g, 1, 2)
add_edge!(g, 2, 3)
add_edge!(g, 3, 4)
add_edge!(g, 4, 5)
add_edge!(g, 5, 1)

# Print the graph
println("Number of vertices: ", nv(g))
println("Number of edges: ", ne(g))
println("Graph structure: ", g)

println("SimpleGraph works correctly!") 