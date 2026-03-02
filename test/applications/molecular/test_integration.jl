# Integration tests for molecular GFlowNet
# Prerequisite E: End-to-end training smoke test

using Test
using Random

include(joinpath(@__DIR__, "test_setup.jl"))

@testset "Molecular GFlowNet Integration" begin

    @testset "model creation defaults" begin
        model = create_molecular_gflownet(rng=Random.MersenneTwister(42))

        @test model isa GFlowNetModel
        @test length(model.all_actions) == 51  # 50 fragments + 1 terminate
        @test model.initial_state isa MolState
        @test model.initial_state.smiles == ""
        @test !isnothing(model.forward_policy)
    end

    @testset "model creation with config" begin
        config = Dict("use_brics_labels" => false, "use_preferences" => false)
        model = create_molecular_gflownet(mol_config=config, rng=Random.MersenneTwister(42))
        @test model isa GFlowNetModel
    end

    if @isdefined(RDKitBridge)
        @testset "trajectory sampling" begin
            model = create_molecular_gflownet(rng=Random.MersenneTwister(42))
            traj = GFlowNet.sample_trajectory(model)
            @test traj isa GFlowNet.Trajectory
            @test length(traj.states) >= 2
            @test length(traj.actions) == length(traj.states) - 1
            @test traj.states[1].smiles == ""
            @test GFlowNet.is_terminal_state(traj.states[end])
        end
    else
        @info "Skipping trajectory sampling test (RDKitBridge not loaded)"
        @test_skip "RDKit not available"
    end

    @testset "state equality and hashing" begin
        s1 = MolState("c1ccccc1", [0, 1], 1, false, zeros(Float32, 1024))
        s2 = MolState("c1ccccc1", [0, 1], 1, false, zeros(Float32, 1024))
        s3 = MolState("c1ccccc1", [0], 1, false, zeros(Float32, 1024))

        @test s1 == s2
        @test hash(s1) == hash(s2)
        @test s1 != s3  # Different attachment points
    end

    @testset "action equality" begin
        a1 = FragmentAction(1, "c1ccc([*])cc1", "benzene")
        a2 = FragmentAction(1, "c1ccc([*])cc1", "benzene")
        a3 = FragmentAction(2, "c1ccnc([*])c1", "pyridine")

        @test a1 == a2
        @test a1 != a3

        t1 = TerminateMolAction()
        t2 = TerminateMolAction()
        @test t1 == t2
    end
end
