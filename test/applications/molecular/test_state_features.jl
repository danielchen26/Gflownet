# Tests for molecular state feature computation
# Prerequisite E: Verify state_to_features produces correct-length vectors

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

@testset "Molecular State Features" begin

    @testset "compute_state_dim" begin
        # Baseline
        @test compute_state_dim() == 1042
        @test compute_state_dim(Dict()) == 1042

        # With BRICS labels (+16)
        @test compute_state_dim(Dict("use_brics_labels" => true)) == 1058

        # With MOGFN preferences (+64)
        @test compute_state_dim(Dict("use_preferences" => true)) == 1106

        # With custom preference embed dim
        @test compute_state_dim(Dict("use_preferences" => true, "preference_embed_dim" => 32)) == 1074

        # Combined BRICS + MOGFN
        @test compute_state_dim(Dict("use_brics_labels" => true, "use_preferences" => true)) == 1122

        # STATE_DIM backward compat
        @test STATE_DIM == 1042
    end

    @testset "state_to_features output shape" begin
        # Empty initial state
        empty_state = MolState("", Int[], 0, false, zeros(Float32, 1024))
        features = GFlowNet.state_to_features(empty_state)
        @test length(features) == STATE_DIM
        @test length(features) == 1042
        @test eltype(features) == Float32

        # All values should be finite
        @test all(isfinite, features)
    end

    @testset "state_to_features content" begin
        # State with known fingerprint
        fp = zeros(Float32, 1024)
        fp[1] = 1.0f0
        fp[100] = 1.0f0
        state = MolState("test", [0, 1, 2], 3, false, fp)

        features = GFlowNet.state_to_features(state)

        # First 1024 elements are the fingerprint
        @test features[1] == 1.0f0
        @test features[100] == 1.0f0
        @test features[2] == 0.0f0

        # [1025] normalized attachment count = 3/16
        @test features[1025] ≈ Float32(3/16)

        # [1026] normalized fragment count = 3/8
        @test features[1026] ≈ Float32(3/8)

        # [1027:1042] one-hot for attachment slots
        @test features[1027] == 1.0f0  # slot 1 (3 attachments)
        @test features[1028] == 1.0f0  # slot 2
        @test features[1029] == 1.0f0  # slot 3
        @test features[1030] == 0.0f0  # slot 4 (no attachment)
    end

    @testset "terminated state features" begin
        state = MolState("c1ccccc1", Int[], 1, true, ones(Float32, 1024))
        features = GFlowNet.state_to_features(state)
        @test length(features) == STATE_DIM

        # No attachments → all one-hot slots = 0
        @test all(features[1027:1042] .== 0.0f0)
    end
end
