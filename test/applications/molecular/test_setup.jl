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

# Shared RDKit gate and expected-value constants. Loading this reports WHY
# chemistry assertions are skipped -- previously they were skipped silently, so
# test_reward_function.jl and test_diversity.jl ran zero real assertions and
# still looked green.
#
# The include is guarded: without the guard every molecular test file replaced
# module MolecularFixture, which printed "WARNING: replacing module
# MolecularFixture" plus "ignoring conflicting import of
# MolecularFixture.rdkit_reason into Main" (so Main kept the FIRST module's
# binding) and re-entered _load_rdkit once per file.
if !@isdefined(MolecularFixture)
    include(joinpath(@__DIR__, "..", "..", "fixtures", "molecular.jl"))
end
using .MolecularFixture: RDKIT_AVAILABLE, EXPECTED_FRAGMENT_COUNT, rdkit_reason

if RDKIT_AVAILABLE
    @info "Molecular tests: RDKit available — chemistry assertions will run"
else
    @warn "Molecular tests: RDKit assertions SKIPPED" reason = rdkit_reason()
end
