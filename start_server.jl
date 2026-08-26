#!/usr/bin/env julia
# The single entry point for the Oxygen backend.
#
# Honors PORT and HOST from the environment so the container CMD does not need
# to re-implement this file inline (Dockerfile:38 used to do exactly that,
# leaving two launch paths to keep in sync by hand).
#
# Defaults are local-development friendly: 127.0.0.1:8080. The container sets
# HOST=0.0.0.0.
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
include(joinpath(@__DIR__, "src", "utils", "visualization", "api", "unified_server.jl"))

port = parse(Int, get(ENV, "PORT", "8080"))
host = get(ENV, "HOST", "127.0.0.1")

@info "Starting GFlowNet API server" host port
start_real_training_server(port = port, host = host)
