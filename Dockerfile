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

# Manifest.toml is untracked by policy (.gitignore), so resolve from scratch.
# CondaPkg.toml is REQUIRED: without it PythonCall builds an empty conda env,
# RDKitBridge.init_rdkit! fails at unified_server.jl:36, RDKIT_AVAILABLE stays
# false, and every molecular route degrades -- the image can serve only
# grid_world. Worse, rdkit_bridge.jl:9 is included OUTSIDE the try/catch at
# :35-42, so a PythonCall failure takes the whole server down.
COPY Project.toml CondaPkg.toml ./

# Resolve and install Julia dependencies
RUN julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Copy source code
COPY src/ src/
COPY start_server.jl .

# Precompile dependencies to reduce cold start
RUN julia --project=. -e '\
    using Pkg; Pkg.precompile(); \
    @info "Precompilation complete"'

# Railway sets PORT; start_server.jl reads PORT and HOST from the environment.
ENV PORT=8080
ENV HOST=0.0.0.0
EXPOSE 8080

# Start the server. Do NOT inline this: start_server.jl is the single launch
# path, and duplicating it here is how the two drifted apart before.
CMD ["julia", "--project=.", "start_server.jl"]
