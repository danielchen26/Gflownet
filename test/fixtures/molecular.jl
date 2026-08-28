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
    # Idempotence is REQUIRED here, not a nicety. Every molecular test file
    # includes test_setup.jl, which includes this file, so a full run rebuilt
    # this module once per file. Re-running Base.include on rdkit_bridge.jl
    # REPLACES Main.RDKitBridge with a fresh, UNINITIALIZED module, while the
    # toplevel eval that is still building this module keeps resolving
    # Main.RDKitBridge to the previous, already-initialized copy -- so
    # init_rdkit! early-returned on the old module and every molecular file
    # after the first one died with "RDKitBridge not initialized! Call
    # init_rdkit!() first." Observed with GFLOWNET_TEST_RDKIT=true: 32 failures
    # and 17 errors across the group, none of them real. Reuse the bridge that
    # is already loaded instead of replacing it.
    if Base.invokelatest(isdefined, Main, :RDKitBridge)
        try
            # init_rdkit! is a no-op when the bridge is already initialized; the
            # closure runs at the latest world age so the global read resolves
            # to the live module.
            Base.invokelatest(() -> Main.RDKitBridge.init_rdkit!())
            return (true, "reusing Main.RDKitBridge already loaded in this session")
        catch e
            return (false, "already-loaded RDKitBridge failed to initialize: " *
                           "$(sprint(showerror, e))")
        end
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
