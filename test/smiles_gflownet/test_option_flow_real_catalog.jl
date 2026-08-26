using Test
using Serialization

const REAL_TEST_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REAL_TEST_ROOT, "src", "training", "option_flow_dataset.jl"))
include(joinpath(REAL_TEST_ROOT, "src", "training", "option_flow_model.jl"))
include(joinpath(REAL_TEST_ROOT, "src", "training", "option_flow_loss.jl"))
include(joinpath(REAL_TEST_ROOT, "src", "training", "option_flow_training.jl"))
include(joinpath(REAL_TEST_ROOT, "src", "training", "option_flow_real_catalog.jl"))

function _fake_episode(task, phase, idx; config="cfg_a", source_bonus=0.0, budget=220.0, top1=0.70, top10=0.55)
    util = 0.05 * idx + source_bonus
    return Dict{String,Any}(
        "task_name" => task,
        "phase" => phase,
        "config_name" => config,
        "snapshot_id" => string("snapshot-", idx),
        "episode_id" => string(task, "-", phase, "-", idx),
        "run_index" => idx % 3 + 1,
        "episode_index" => idx,
        "segment_index" => 0,
        "budget_remaining_before" => budget + idx,
        "budget_remaining_after" => budget + idx - 5,
        "calls_before" => 10 + idx,
        "calls_after" => 15 + idx,
        "calls_used" => 5,
        "frontier_size_before" => 8 + idx,
        "frontier_size_after" => 10 + idx,
        "frontier_before_summary" => Dict{String,Any}(
            "top1" => top1,
            "top10_mean" => top10,
            "size" => 8 + idx,
            "n_scaffolds" => 4 + idx % 3,
            "graph_unique_count" => 0,
        ),
        "frontier_after_summary" => Dict{String,Any}(
            "top1" => top1 + 0.01 * idx,
            "top10_mean" => top10 + 0.005 * idx,
            "size" => 10 + idx,
            "n_scaffolds" => 5 + idx % 3,
            "graph_unique_count" => 0,
        ),
        "frontier_gain_sum" => util,
        "frontier_gain_max" => util / 2,
        "delta_top1_max" => 0.01 * idx,
        "delta_top10_mean_max" => 0.005 * idx,
        "best_reward" => top1 + 0.01 * idx,
        "commits_applied" => idx,
        "step_count" => idx + 1,
        "unique_parent_count" => 1,
        "unique_basin_count" => 1,
        "enters_topk" => idx % 2 == 0,
        "improved_topk" => idx % 2 == 0,
        "no_valid_proposal" => false,
        "budget_cap_reached" => false,
        "horizon_exhausted" => true,
        "stagnation_or_low_value" => false,
        "terminated_commit" => false,
        "stop_reason" => "horizon_exhausted",
        "best_smiles" => "CCO",
        "chosen_parent_count" => 1,
    )
end

@testset "Option-Flow real artifact catalog" begin
    mktempdir() do dir
        artifact_a = joinpath(dir, "checkpoints", "truth_sprint_stage_b_f015_truth_tasksharded", "shard_tb_he_full_locked-g1", "he_artifacts", "tb_he_full_locked", "qed", "run1")
        artifact_b = joinpath(dir, "checkpoints", "level3_shape_then_tb", "artifacts", "learned_shape_then_tb", "qed", "run1", "shape")
        mkpath(artifact_a)
        mkpath(artifact_b)
        eps_a = [_fake_episode("qed", "warmup", i; config="cfg_a", source_bonus=0.01) for i in 1:4]
        eps_b = [_fake_episode("qed", "warmup", i + 4; config="cfg_b", source_bonus=0.20) for i in 1:4]
        serialize(joinpath(artifact_a, "he_episode_summary.jls"), eps_a)
        serialize(joinpath(artifact_a, "he_capacity_summary.jls"), Dict("episode_summaries" => eps_a))
        serialize(joinpath(artifact_b, "he_episode_summary.jls"), eps_b)
        serialize(joinpath(artifact_b, "he_capacity_summary.jls"), Dict("episode_summaries" => eps_b))

        roots = [joinpath(dir, "checkpoints")]
        files = discover_he_summary_files(roots)
        @test length(files) == 2
        rows, loaded_files, errors = load_he_summary_rows(roots)
        @test length(loaded_files) == 2
        @test isempty(errors)
        @test length(rows) == 8
        @test all(haskey(r, "source_family") for r in rows)

        audit = option_flow_real_artifact_audit(roots)
        @test audit["summary_files"] == 2
        @test audit["episode_count"] == 8
        @test audit["repeated_task_snapshot_groups"] == 0
        @test audit["capacity_status"]["sample_deserializes"] == true

        catalogs, enc, stats = build_summary_proxy_catalogs(rows;
            grouping=:task_phase_budget100,
            feature_mode=:leak_free,
            min_candidates=3,
            max_candidates=12,
            seed=3,
        )
        @test stats["n_catalogs"] >= 1
        @test enc.feature_mode == :leak_free
        @test all(validate_option_flow_catalog(c) for c in catalogs)
        @test all(c.informative for c in catalogs)
        @test all(catalog_has_option_feature_variation(c) for c in catalogs)

        typed_rows = [merge(copy(r), Dict{String,Any}("typed_path_features" => fill(Float32(0.1), OPTION_FLOW_TYPED_PATH_FEATURE_DIM))) for r in rows]
        typed_catalogs, typed_enc, _ = build_summary_proxy_catalogs(typed_rows;
            grouping=:task_phase_budget100,
            feature_mode=:typed_path,
            min_candidates=3,
            max_candidates=12,
            seed=3,
        )
        @test typed_enc.feature_mode == :typed_path
        @test option_flow_option_dim(typed_catalogs[1]) > option_flow_option_dim(catalogs[1])

        leaky_catalogs, leaky_enc, _ = build_summary_proxy_catalogs(rows;
            grouping=:task_phase_budget100,
            feature_mode=:leaky_upper,
            min_candidates=3,
            max_candidates=12,
            seed=3,
        )
        @test leaky_enc.feature_mode == :leaky_upper
        @test option_flow_option_dim(leaky_catalogs[1]) > option_flow_option_dim(catalogs[1])

        headline = filter_real_headline_catalogs(catalogs)
        @test !isempty(headline)
        uniform = uniform_policy_metrics(headline)
        @test uniform["n_catalogs"] == length(headline)
        @test isfinite(uniform["mean_expected_utility"])

        result = train_option_flow_model(headline; config=OptionFlowTrainingConfig(
            n_epochs=8,
            learning_rate=0.01,
            hidden_dim=16,
            second_hidden_dim=8,
            validation_fraction=0.25,
            seed=5,
            verbose=false,
        ))
        metrics = evaluate_real_option_flow_model(result["params"], result["val_catalogs"])
        @test haskey(metrics, "mean_expected_utility_lift")
        @test haskey(metrics, "greedy_model")
        @test haskey(e1_summary_proxy_gate(metrics), "pass")

        prior = evaluate_metadata_prior_baseline(result["train_catalogs"], result["val_catalogs"]; metadata_key="config_name")
        @test prior["metadata_key"] == "config_name"
    end
end
