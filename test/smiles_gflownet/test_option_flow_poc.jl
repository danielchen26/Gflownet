#!/usr/bin/env julia

using Test
using Random
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "src", "training", "option_flow_dataset.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_model.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_loss.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_training.jl"))

@testset "Option-Flow v0 POC" begin
    @testset "utility normalization" begin
        probs, informative = normalize_option_utilities([0.0, 1.0, 3.0]; epsilon=1.0e-6)
        @test informative
        @test length(probs) == 3
        @test isapprox(sum(probs), 1.0f0; atol=1.0f-6)
        @test probs[3] > probs[2] > probs[1]

        zero_probs, zero_info = normalize_option_utilities([0.0, 0.0, 0.0]; epsilon=1.0e-6)
        @test !zero_info
        @test all(isapprox.(zero_probs, fill(1.0f0 / 3.0f0, 3); atol=1.0f-5))
    end

    @testset "catalog construction and validation" begin
        catalog = make_option_flow_catalog(
            "toy",
            UInt64(1),
            Float32[0.1, 0.2],
            [Float32[1, 0, 0], Float32[0, 1, 0], Float32[0, 0, 1]],
            [0.1, 0.2, 0.7],
        )
        @test validate_option_flow_catalog(catalog)
        @test option_flow_state_dim(catalog) == 2
        @test option_flow_option_dim(catalog) == 3
        @test option_flow_input_dim(catalog) == 5
        X = option_flow_input_matrix(catalog)
        @test size(X) == (5, 3)
    end

    @testset "model probabilities and loss" begin
        catalogs = synthetic_option_flow_catalogs(n_catalogs=8, n_candidates=5, state_dim=4, option_dim=6, seed=3)
        config = create_option_flow_mlp(option_flow_input_dim(catalogs[1]); hidden_dim=16, second_hidden_dim=8)
        params = init_option_flow_params(config; rng=MersenneTwister(3))
        probs = option_flow_probs(params, catalogs[1])
        @test length(probs) == length(catalogs[1].candidates)
        @test isapprox(sum(probs), 1.0f0; atol=1.0f-5)
        ce = catalog_cross_entropy_loss(params, catalogs[1])
        @test isfinite(ce)
        @test ce > 0
        residuals = flow_residual_diagnostics(params, catalogs[1])
        @test haskey(residuals, "mean_abs_residual")
        @test residuals["mean_abs_residual"] >= 0
    end

    @testset "grouped split" begin
        catalogs = synthetic_option_flow_catalogs(n_catalogs=12, n_candidates=4, seed=4)
        train, val = grouped_split_option_flow_catalogs(catalogs; validation_fraction=0.25, seed=4)
        train_ids = Set(c.snapshot_id for c in train)
        val_ids = Set(c.snapshot_id for c in val)
        @test !isempty(train)
        @test !isempty(val)
        @test isempty(intersect(train_ids, val_ids))
    end

    @testset "training improves synthetic catalog objective" begin
        catalogs = synthetic_option_flow_catalogs(n_catalogs=60, n_candidates=6, state_dim=8, option_dim=10, seed=5)
        train, val = grouped_split_option_flow_catalogs(catalogs; validation_fraction=0.25, seed=5)
        model_config = create_option_flow_mlp(option_flow_input_dim(catalogs[1]); hidden_dim=64, second_hidden_dim=32)
        init_params = init_option_flow_params(model_config; rng=MersenneTwister(5))
        initial_val = evaluate_option_flow_model(init_params, val)

        result = train_option_flow_model(catalogs; config=OptionFlowTrainingConfig(
            n_epochs=160,
            learning_rate=0.015,
            hidden_dim=64,
            second_hidden_dim=32,
            validation_fraction=0.25,
            seed=5,
            verbose=false,
        ))
        final_val = result["val_metrics"]
        @test final_val["mean_ce"] < initial_val["mean_ce"]
        @test final_val["mean_ce"] < final_val["mean_uniform_ce"]
        @test final_val["mean_top_quartile_mass"] > final_val["mean_uniform_top_quartile_mass"]
        @test final_val["mean_rank_correlation"] > 0.1
        @test final_val["mean_entropy"] > 0.1
    end
end

println("Option-Flow v0 POC tests completed.")
