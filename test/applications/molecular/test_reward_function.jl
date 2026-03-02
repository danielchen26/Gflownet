# Tests for reward function and multi-objective dispatch
# Prerequisites B+C: Verify dual-signature reward and objective normalization

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

const _rdkit_avail_reward = @isdefined(RDKitBridge)

@testset "Reward Function" begin

    if !_rdkit_avail_reward
        @info "Skipping reward function tests (RDKitBridge not loaded — reward() requires RDKit)"
        @test_skip "RDKit not available"
    else
        @testset "single-objective reward basics" begin
            state = MolState("c1ccccc1", Int[], 1, false, zeros(Float32, 1024))
            @test GFlowNet.reward(state) == 0.0

            empty_state = MolState("", Int[], 0, true, zeros(Float32, 1024))
            @test GFlowNet.reward(empty_state) == 1e-4

            valid_state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
            r = GFlowNet.reward(valid_state)
            @test r > 0.0
            @test r <= 1.0
            @test isfinite(r)
        end

        @testset "compute_all_objectives" begin
            state = MolState("c1ccccc1", Int[], 1, false, zeros(Float32, 1024))
            @test isempty(compute_all_objectives(state))

            valid_state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
            objs = compute_all_objectives(valid_state)
            @test length(objs) == 4
            for obj in objs
                @test 0.0 <= obj <= 1.0
            end
        end

        @testset "multi-objective reward dispatch" begin
            state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
            objs = compute_all_objectives(state)
            if !isempty(objs)
                w_equal = [0.25, 0.25, 0.25, 0.25]
                r_equal = GFlowNet.reward(state, w_equal)
                @test r_equal > 0.0
                @test isfinite(r_equal)

                w_qed = [1.0, 0.0, 0.0, 0.0]
                r_qed = GFlowNet.reward(state, w_qed)
                @test r_qed ≈ objs[1] atol=1e-4

                w_sa = [0.0, 1.0, 0.0, 0.0]
                r_sa = GFlowNet.reward(state, w_sa)
                @test r_sa ≈ objs[2] atol=1e-4
            end
        end

        @testset "reward backward compatibility" begin
            state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
            r_single = GFlowNet.reward(state)
            objs = compute_all_objectives(state)
            if !isempty(objs)
                expected = (objs[1]^0.4) * (objs[2]^0.3) * (objs[3]^0.2) * (objs[4]^0.1)
                expected = max(expected, 1e-4)
                @test r_single ≈ expected atol=1e-6
                r_multi = GFlowNet.reward(state, [0.4, 0.3, 0.2, 0.1])
                @test r_multi > 0.0
            end
        end

        @testset "objective normalization ranges" begin
            aspirin_state = MolState("CC(=O)Oc1ccccc1C(=O)O", Int[], 3, true, zeros(Float32, 1024))
            objs = compute_all_objectives(aspirin_state)
            if !isempty(objs)
                @test 0.0 < objs[1] < 1.0
                @test objs[2] > 0.3
                @test 0.0 < objs[3] <= 1.0
                @test 0.0 < objs[4] <= 1.0
            end
        end
    end
end
