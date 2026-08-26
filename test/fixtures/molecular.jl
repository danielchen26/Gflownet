# test/fixtures/molecular.jl
#
# Single source of truth for molecular test preconditions.
#
# Pkg.test() must pass with no Python, no conda environment and no network, so
# RDKit is opt-in: set GFLOWNET_TEST_RDKIT=true to load the bridge and run the
# assertions that need real chemistry. When it is not set, the guarded testsets
# report @test_skip AND the reason is logged, so a skipped chemistry suite can
# never again look like a passing one.
#
# Before this file existed, three different guard idioms were in use for the
# same condition -- isdefined(Main, :RDKitBridge), @isdefined(RDKitBridge), and
# inline @isdefined -- and `const _rdkit_available` was defined twice into Main
# by test_diversity.jl and test_docking.jl, colliding in a single session.
module MolecularFixture

export RDKIT_AVAILABLE, EXPECTED_FRAGMENT_COUNT, rdkit_reason

const _REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

"""
Number of fragments the checked-in BRICS library is expected to contain.

Asserted by test_fragment_joining.jl and test_integration.jl; kept here so the
two cannot drift apart. Confirmed against the server's own startup check, which
logs "All 50 fragments validated successfully".
"""
const EXPECTED_FRAGMENT_COUNT = 50

function _load_rdkit()
    if get(ENV, "GFLOWNET_TEST_RDKIT", "false") != "true"
        return (false, "GFLOWNET_TEST_RDKIT is not \"true\"")
    end
    try
        Base.include(Main, joinpath(_REPO_ROOT, "src", "utils", "visualization",
                                    "python", "rdkit_bridge.jl"))
        # invokelatest is REQUIRED: Base.include defines init_rdkit! in a newer
        # world age than the one this function is executing in, so a direct call
        # fails with "method too new to be called from this world context".
        Base.invokelatest(Main.RDKitBridge.init_rdkit!)
        return (true, "loaded")
    catch e
        return (false, "rdkit_bridge failed to load: $(sprint(showerror, e))")
    end
end

const _state = _load_rdkit()

"""Whether RDKit-backed assertions can run in this session."""
const RDKIT_AVAILABLE = _state[1]

"""Human-readable explanation of why RDKit is or is not available."""
rdkit_reason() = _state[2]

end # module
