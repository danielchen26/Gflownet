# Tests for fragment joining operations
# Prerequisite E: Verify place_first_fragment and join_fragment produce valid SMILES

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

# Was `@isdefined(RDKitBridge)` -- one of the three competing guard idioms that
# test/fixtures/molecular.jl was written to replace.
const _rdkit_avail_joining = RDKIT_AVAILABLE

@testset "Fragment Joining" begin

    if _rdkit_avail_joining
        @testset "place_first_fragment" begin
            starter = FRAGMENT_LIBRARY[42]
            empty_state = MolState("", Int[], 0, false, zeros(Float32, 1024))
            new_state = GFlowNet.apply_action(starter, empty_state)
            @test !isempty(new_state.smiles)
            @test new_state.n_fragments == 1
            @test !new_state.is_terminated
            @test length(new_state.fingerprint) == 1024
        end

        @testset "join_fragment to molecule" begin
            starter = FRAGMENT_LIBRARY[42]
            empty_state = MolState("", Int[], 0, false, zeros(Float32, 1024))
            state1 = GFlowNet.apply_action(starter, empty_state)
            # Was `if !isempty(state1.attachment_points)`, which made the whole
            # testset report 0 tests whenever fragment placement regressed --
            # observed as "join_fragment to molecule | 0 tests" in a session
            # where RDKit had silently become uninitialized. FRAGMENT_LIBRARY[42]
            # is 1,3-disubstituted benzene, so it leaves attachment points [0, 6].
            @test state1.attachment_points == [0, 6]

            benzene = FRAGMENT_LIBRARY[1]
            state2 = GFlowNet.apply_action(benzene, state1)
            @test !isempty(state2.smiles)
            @test state2.n_fragments == 2
            @test !state2.is_terminated
            # One attachment point consumed by the join.
            @test length(state2.attachment_points) == length(state1.attachment_points) - 1
        end

        @testset "terminate action" begin
            starter = FRAGMENT_LIBRARY[42]
            empty_state = MolState("", Int[], 0, false, zeros(Float32, 1024))
            state1 = GFlowNet.apply_action(starter, empty_state)
            terminate = TerminateMolAction()
            final_state = GFlowNet.apply_action(terminate, state1)
            @test final_state.is_terminated
            @test isempty(final_state.attachment_points)
            @test final_state.n_fragments == state1.n_fragments
        end

        @testset "all starter fragments produce valid SMILES" begin
            empty_state = MolState("", Int[], 0, false, zeros(Float32, 1024))
            for frag in FRAGMENT_LIBRARY
                if frag.fragment_id >= 41
                    new_state = GFlowNet.apply_action(frag, empty_state)
                    @test !isempty(new_state.smiles)
                    @test new_state.n_fragments == 1
                end
            end
        end
    else
        @warn "Fragment JOINING assertions SKIPPED (place_first_fragment, \
               join_fragment, terminate, starter-fragment validity)" reason = rdkit_reason()
        @test_skip "fragment joining requires RDKit"
    end

    @testset "fragment library validation" begin
        @test length(FRAGMENT_LIBRARY) == EXPECTED_FRAGMENT_COUNT

        # All fragments should have unique IDs
        ids = [f.fragment_id for f in FRAGMENT_LIBRARY]
        @test length(unique(ids)) == EXPECTED_FRAGMENT_COUNT
        @test minimum(ids) == 1
        @test maximum(ids) == EXPECTED_FRAGMENT_COUNT

        # All fragments should have non-empty SMILES
        for frag in FRAGMENT_LIBRARY
            @test !isempty(frag.fragment_smiles)
            @test !isempty(frag.fragment_name)
        end
    end
end
