# Tests for MOGFN Multi-Objective GFlowNet (Gap 5)
# Validates: preference encoder, Z(w) network, Dirichlet sampling,
# scalarized reward, MOGFN model creation, and backward compatibility

using Test
using Random
using Statistics
using Lux

include(joinpath(@__DIR__, "test_setup.jl"))

@testset "MOGFN Multi-Objective GFlowNet" begin

    @testset "Dirichlet preference sampling" begin
        rng_state = Random.seed!(42)

        # Basic properties: sum to 1, all positive
        for _ in 1:20
            w = sample_preference(4)
            @test length(w) == 4
            @test all(w .>= 0.0)
            @test sum(w) ≈ 1.0 atol=1e-10
        end

        # Different objective counts
        for k in [2, 3, 4, 5, 8]
            w = sample_preference(k)
            @test length(w) == k
            @test sum(w) ≈ 1.0 atol=1e-10
        end

        # Custom alpha
        w_focused = sample_preference(4; alpha=0.1)  # Concentrated (sparse)
        @test sum(w_focused) ≈ 1.0 atol=1e-10
        @test all(w_focused .>= 0.0)

        w_uniform = sample_preference(4; alpha=10.0)  # Diffuse (near-uniform)
        @test sum(w_uniform) ≈ 1.0 atol=1e-10
        @test all(w_uniform .>= 0.0)

        # Statistical test: with high alpha, samples should be near-uniform
        samples = [sample_preference(4; alpha=100.0) for _ in 1:100]
        mean_w = mean(samples)
        for i in 1:4
            @test mean_w[i] ≈ 0.25 atol=0.05
        end
    end

    @testset "preference encoder produces different outputs" begin
        rng = Random.MersenneTwister(42)
        n_objectives = 4
        embed_dim = 64

        encoder, ps, st = GFlowNet.create_preference_encoder(n_objectives, embed_dim, rng)

        # Two different preference vectors
        w1 = Float32[0.7, 0.1, 0.1, 0.1]
        w2 = Float32[0.1, 0.7, 0.1, 0.1]
        w3 = Float32[0.25, 0.25, 0.25, 0.25]

        # Forward through encoder
        embed1, _ = Lux.apply(encoder, reshape(w1, :, 1), ps, st)
        embed2, _ = Lux.apply(encoder, reshape(w2, :, 1), ps, st)
        embed3, _ = Lux.apply(encoder, reshape(w3, :, 1), ps, st)

        # All embeddings should have correct dimension
        @test size(embed1) == (embed_dim, 1)
        @test size(embed2) == (embed_dim, 1)

        # Different inputs should produce different outputs
        @test embed1 ≠ embed2
        @test embed1 ≠ embed3
        @test embed2 ≠ embed3

        # Same input should produce same output (deterministic)
        embed1b, _ = Lux.apply(encoder, reshape(w1, :, 1), ps, st)
        @test embed1 ≈ embed1b atol=1e-6
    end

    @testset "Z(w) network produces different values" begin
        rng = Random.MersenneTwister(42)
        embed_dim = 64

        z_net, ps, st = GFlowNet.create_z_network(embed_dim, rng)

        # Different embeddings should give different Z values
        e1 = Float32.(randn(rng, embed_dim, 1))
        e2 = Float32.(randn(rng, embed_dim, 1))

        z1, _ = Lux.apply(z_net, e1, ps, st)
        z2, _ = Lux.apply(z_net, e2, ps, st)

        @test size(z1) == (1, 1)  # Scalar output
        @test size(z2) == (1, 1)
        @test z1[1] ≠ z2[1]  # Different embeddings → different Z

        # Same embedding → same Z
        z1b, _ = Lux.apply(z_net, e1, ps, st)
        @test z1[1] ≈ z1b[1] atol=1e-6
    end

    @testset "MOGFN model creation" begin
        rng = Random.MersenneTwister(42)

        model = create_mogfn_molecular_gflownet(rng=rng)

        @test model isa GFlowNetModel
        @test !isnothing(model.preference_encoder)
        @test !isnothing(model.z_network)
        @test !isnothing(model.forward_policy)
        @test model.initial_state isa MolState
        @test length(model.all_actions) >= 51  # fragments + terminate
    end

    @testset "MOGFN model creation with custom params" begin
        rng = Random.MersenneTwister(42)

        model = create_mogfn_molecular_gflownet(
            hidden_dim=128,
            n_objectives=3,
            preference_dim=32,
            rng=rng
        )

        @test model isa GFlowNetModel
        @test !isnothing(model.preference_encoder)
        @test !isnothing(model.z_network)
    end

    # Trajectory sampling and reward tests require RDKit. This used to read
    # `@isdefined(RDKitBridge)` -- the third guard idiom for one condition, and
    # the one test/fixtures/molecular.jl exists to replace.
    if RDKIT_AVAILABLE
        @testset "MOGFN backward compatibility" begin
            rng = Random.MersenneTwister(42)
            model = create_molecular_gflownet(rng=rng)
            @test model isa GFlowNetModel
            @test isnothing(model.preference_encoder)
            @test isnothing(model.z_network)
            traj = GFlowNet.sample_trajectory(model)
            @test traj isa GFlowNet.Trajectory
            @test length(traj.states) >= 2
        end

        @testset "MOGFN trajectory sampling" begin
            rng = Random.MersenneTwister(42)
            model = create_mogfn_molecular_gflownet(rng=rng)
            w = [0.4, 0.3, 0.2, 0.1]
            traj = GFlowNet.sample_mogfn_trajectory(model, w)
            @test traj isa GFlowNet.Trajectory
            @test length(traj.states) >= 2
            @test length(traj.actions) == length(traj.states) - 1
            @test traj.states[1].smiles == ""
            @test GFlowNet.is_terminal_state(traj.states[end])
            w2 = [0.1, 0.1, 0.1, 0.7]
            traj2 = GFlowNet.sample_mogfn_trajectory(model, w2)
            @test traj2 isa GFlowNet.Trajectory
        end

        @testset "scalarized reward with preference vectors" begin
            state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
            objs = compute_all_objectives(state)
            # Was `if !isempty(objs)` around the whole body, so an empty
            # objective vector would have deleted every assertion here silently.
            @test length(objs) >= 4

            for i in 1:4
                w = zeros(4)
                w[i] = 1.0
                r = GFlowNet.reward(state, w)
                @test r ≈ objs[i] atol=1e-4
            end
            w = [0.5, 0.3, 0.1, 0.1]
            r = GFlowNet.reward(state, w)
            expected = sum(w .* objs[1:4])
            @test r ≈ expected atol=1e-4
            for _ in 1:10
                w = sample_preference(4)
                r = GFlowNet.reward(state, w)
                @test r > 0.0
                @test isfinite(r)
            end
        end
    else
        @warn "MOGFN trajectory-sampling and reward assertions SKIPPED" reason = rdkit_reason()
        @test_skip "MOGFN trajectory/reward assertions require RDKit"
    end

    @testset "MULTI_OBJECTIVE_TB enum exists" begin
        @test isdefined(GFlowNet, :MULTI_OBJECTIVE_TB)
        @test MULTI_OBJECTIVE_TB isa GFlowNet.TrainingObjective
    end

    @testset "TrainingConfig MOGFN fields" begin
        # Default config should have MOGFN fields with defaults
        config = TrainingConfig(
            objective=MULTI_OBJECTIVE_TB,
            n_iterations=10,
            batch_size=4,
            learning_rate=0.001
        )

        @test config.objective == MULTI_OBJECTIVE_TB
        @test config.mogfn_n_objectives == 4
        @test config.mogfn_preference_dim == 64
        @test config.mogfn_dirichlet_alpha == 1.0

        # Custom values
        config2 = TrainingConfig(
            objective=MULTI_OBJECTIVE_TB,
            n_iterations=10,
            batch_size=4,
            learning_rate=0.001,
            mogfn_n_objectives=3,
            mogfn_preference_dim=32,
            mogfn_dirichlet_alpha=0.5
        )

        @test config2.mogfn_n_objectives == 3
        @test config2.mogfn_preference_dim == 32
        @test config2.mogfn_dirichlet_alpha == 0.5
    end

    @testset "GFlowNetModel new fields" begin
        rng = Random.MersenneTwister(42)

        # MOGFN model has encoder and z_network
        mogfn = create_mogfn_molecular_gflownet(rng=rng)
        @test hasfield(typeof(mogfn), :preference_encoder)
        @test hasfield(typeof(mogfn), :z_network)
        @test !isnothing(mogfn.preference_encoder)
        @test !isnothing(mogfn.z_network)

        # Standard model fields are nothing
        standard = create_molecular_gflownet(rng=rng)
        @test isnothing(standard.preference_encoder)
        @test isnothing(standard.z_network)
    end

    @testset "gamma samplers for non-unit Dirichlet" begin
        # Both live gamma samplers, checked by name. The old version of this
        # testset asked `isdefined(GFlowNet, :_sample_gamma)` -- which is false,
        # because _sample_gamma is defined in
        # src/applications/molecular_generation.jl:710 and loaded into Main by
        # test_setup.jl, not into the GFlowNet module -- and so skipped forever.
        # It then wrapped the assertions in a bare try/catch that turned any
        # failure into another skip.
        #
        # GFlowNet._mogfn_sample_gamma (src/training/losses.jl:1024) is the
        # second, independent copy used inside the MOGFN loss. Both are checked
        # here so the pair cannot drift.
        # Seeded: both samplers draw from the global RNG, and the mean check
        # below is a 4-sigma bound -- deterministic beats a 1-in-2600 flake.
        Random.seed!(20260828)

        for sampler in (_sample_gamma, GFlowNet._mogfn_sample_gamma)
            # alpha = 1.0: Gamma(1,1) = Exponential(1)
            # alpha < 1.0: Gamma(a) = Gamma(a+1) * U^(1/a) boost
            # alpha > 1.0: Marsaglia & Tsang squeeze
            for alpha in (1.0, 0.5, 2.5)
                draws = [sampler(alpha) for _ in 1:200]
                @test all(>(0.0), draws)
                @test all(isfinite, draws)
                # Gamma(alpha, 1) has mean alpha. 200 draws put the standard
                # error at sqrt(alpha/200), so 4 s.e. is a loose but non-vacuous
                # bound that a wrong-shape sampler fails.
                @test abs(Statistics.mean(draws) - alpha) < 4 * sqrt(alpha / 200)
            end
        end
    end
end
