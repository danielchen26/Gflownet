# GFlowNet Training Visualization Server
# Julia backend for the interactive training dashboard

FROM julia:1.12-bookworm

# Set working directory
WORKDIR /app

# Set Julia environment variables for production
# Use generic CPU target to avoid architecture warnings on different platforms
ENV JULIA_CPU_TARGET="generic"
ENV JULIA_NUM_THREADS=2
ENV JULIA_PKG_PRECOMPILE_AUTO=0

# Install the General package registry (not included in base Docker image)
RUN julia -e 'using Pkg; Pkg.Registry.add("General")'

# Copy Project.toml (Manifest.toml is gitignored, so resolve from scratch)
COPY Project.toml ./

# Resolve and install Julia dependencies
RUN julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Copy source code
COPY src/ src/
COPY start_server.jl .

# Precompile dependencies to reduce cold start
RUN julia --project=. -e '\
    using Pkg; Pkg.precompile(); \
    @info "Precompilation complete"'

# Railway sets PORT env var; default to 8080
ENV PORT=8080
EXPOSE 8080

# Start the server
CMD ["julia", "--project=.", "-e", "port = parse(Int, get(ENV, \"PORT\", \"8080\")); push!(LOAD_PATH, joinpath(@__DIR__, \"src\")); include(\"src/utils/visualization/api/unified_server.jl\"); start_real_training_server(port=port, host=\"0.0.0.0\")"]
