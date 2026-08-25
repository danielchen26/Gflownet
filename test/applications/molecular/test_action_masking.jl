# Tests for action masking in molecular generation
# Prerequisite E: Verify is_applicable behavior for all action types

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

@testset "Action Masking" begin

    @testset "FragmentAction applicability" begin
        # Empty state: only starters (id >= 41) should be applicable
        empty_state = MolState("", Int[], 0, false, zeros(Float32, 1024))

        for frag in FRAGMENT_LIBRARY
            applicable = GFlowNet.is_applicable(frag, empty_state)
            if frag.fragment_id >= 41
                @test applicable
            else
                @test !applicable
            end
        end
    end

    @testset "FragmentAction with existing molecule" begin
        # State with attachment points: non-starters should be applicable
        state = MolState("c1ccc([*])cc1", [3], 1, false, zeros(Float32, 1024))

        # Non-starter with attachment point → applicable
        benzene = FRAGMENT_LIBRARY[1]  # id=1, benzene
        @test GFlowNet.is_applicable(benzene, state)

        # Starter fragments should NOT be applicable to non-empty state
        starter = FRAGMENT_LIBRARY[41]  # id=41, disubstituted benzene
        @test !GFlowNet.is_applicable(starter, state)
    end

    @testset "FragmentAction no attachments" begin
        # State with no attachment points: no fragments should be applicable
        no_attach = MolState("c1ccccc1", Int[], 1, false, zeros(Float32, 1024))

        for frag in FRAGMENT_LIBRARY
            if frag.fragment_id < 41  # Non-starters
                @test !GFlowNet.is_applicable(frag, no_attach)
            end
        end
    end

    @testset "FragmentAction max fragments" begin
        # State at max fragments: no fragments should be applicable
        max_state = MolState("c1ccccc1", [0], MAX_FRAGMENTS, false, zeros(Float32, 1024))

        for frag in FRAGMENT_LIBRARY
            @test !GFlowNet.is_applicable(frag, max_state)
        end
    end

    @testset "TerminateMolAction applicability" begin
        terminate = TerminateMolAction()

        # Can't terminate empty molecule
        empty_state = MolState("", Int[], 0, false, zeros(Float32, 1024))
        @test !GFlowNet.is_applicable(terminate, empty_state)

        # Can terminate after at least 1 fragment
        has_frag = MolState("c1ccccc1", [0], 1, false, zeros(Float32, 1024))
        @test GFlowNet.is_applicable(terminate, has_frag)

        # Can't terminate already terminated
        terminated = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
        @test !GFlowNet.is_applicable(terminate, terminated)
    end

    @testset "terminated state blocks all actions" begin
        terminated = MolState("c1ccccc1", [0], 1, true, zeros(Float32, 1024))

        for frag in FRAGMENT_LIBRARY
            @test !GFlowNet.is_applicable(frag, terminated)
        end
        @test !GFlowNet.is_applicable(TerminateMolAction(), terminated)
    end
end
