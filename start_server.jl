#!/usr/bin/env julia
# Quick start script for the unified visualization server
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
include("src/utils/visualization/api/unified_server.jl")
start_real_training_server(port=8080)
