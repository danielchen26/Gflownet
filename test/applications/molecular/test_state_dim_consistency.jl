using Test
using GFlowNet
include(joinpath(@__DIR__, "test_setup.jl"))

@testset "state dimensions derive from their components" begin
    @testset "fragment dimension already has one source of truth" begin
        @test STATE_DIM == compute_state_dim()
        @test STATE_DIM > 0
    end

    @testset "reaction dimension tracks its components" begin
        # 1049 = 1024 fingerprint bits + 17 reaction one-hot + 8 scalar features.
        # That arithmetic used to live only in a comment, so changing n_reactions
        # silently desynced the network input width from the real feature width.
        @test reaction_state_dim() == 1024 + 17 + 8
        @test reaction_state_dim(n_reactions = 20) == 1024 + 20 + 8
        @test reaction_state_dim(fp_dim = 2048) == 2048 + 17 + 8
        @test reaction_state_dim(n_scalar_features = 12) == 1024 + 17 + 12
    end

    @testset "model input layer is sized from the same formula" begin
        # ForwardPolicy wraps the Lux.Chain in its single field `model`.
        for n in (17, 20)
            model = create_reaction_gflownet(n_reactions = n)
            first_layer = model.forward_policy.model.layers[1]
            @test first_layer.in_dims == reaction_state_dim(n_reactions = n)
        end
    end
end
