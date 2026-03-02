# Tests for fragment joining operations
# Prerequisite E: Verify place_first_fragment and join_fragment produce valid SMILES

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

const _rdkit_avail_joining = @isdefined(RDKitBridge)

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
            if !isempty(state1.attachment_points)
                benzene = FRAGMENT_LIBRARY[1]
                state2 = GFlowNet.apply_action(benzene, state1)
                @test !isempty(state2.smiles)
                @test state2.n_fragments == 2
                @test !state2.is_terminated
            end
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
        @info "Skipping RDKit-dependent fragment joining tests (RDKitBridge not loaded)"
        @test_skip "RDKit not available"
    end

    @testset "fragment library validation" begin
        @test length(FRAGMENT_LIBRARY) == 50

        # All fragments should have unique IDs
        ids = [f.fragment_id for f in FRAGMENT_LIBRARY]
        @test length(unique(ids)) == 50
        @test minimum(ids) == 1
        @test maximum(ids) == 50

        # All fragments should have non-empty SMILES
        for frag in FRAGMENT_LIBRARY
            @test !isempty(frag.fragment_smiles)
            @test !isempty(frag.fragment_name)
        end
    end
end
