# Shared setup for molecular generation tests
#
# The fragment-based molecular types (MolState, FragmentAction, etc.) are defined
# in src/applications/molecular_generation.jl, which is normally loaded by the
# visualization server (unified_server.jl), NOT by the core GFlowNet module.
#
# This setup file includes molecular_generation.jl so that tests can access
# those types without running the full server.

using GFlowNet
using Random

# Include molecular_generation.jl once (guard against double-inclusion)
if !@isdefined(MolState)
    const _MOLGEN_PATH = joinpath(@__DIR__, "..", "..", "..", "src", "applications", "molecular_generation.jl")
    include(_MOLGEN_PATH)
end
