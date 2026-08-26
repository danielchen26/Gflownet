# Tests for PMO Benchmark Runner
#
# Tests the 23-task PMO evaluation protocol, AUC computation,
# and benchmark report generation.

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

# Include oracle and PMO modules
if !@isdefined(OracleBridge)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
end
if !@isdefined(OracleManager)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
end
if !@isdefined(PMOResult)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))
end

@testset "PMO Benchmark" begin

    @testset "PMO task list" begin
        @test length(PMO_23_TASKS) == 23
        @test "drd2" in PMO_23_TASKS
        @test "gsk3b" in PMO_23_TASKS
        @test "jnk3" in PMO_23_TASKS
        @test "qed" in PMO_23_TASKS
        @test "celecoxib_rediscovery" in PMO_23_TASKS
        @test "zaleplon_mpo" in PMO_23_TASKS
    end

    @testset "SOTA scores" begin
        @test haskey(SOTA_SCORES, "Genetic GFN")
        @test SOTA_SCORES["Genetic GFN"] == 16.2
        @test haskey(SOTA_SCORES, "REINVENT")
        @test SOTA_SCORES["REINVENT"] == 15.2
        @test haskey(SOTA_SCORES, "Mol GA")
        @test SOTA_SCORES["Mol GA"] == 15.7
    end

    @testset "PMOResult struct" begin
        result = PMOResult("drd2", 0.75, 0.95, 0.88, 0.65, 10000, 5000)
        @test result.task_name == "drd2"
        @test result.auc_top10 == 0.75
        @test result.top1 == 0.95
        @test result.top10_mean == 0.88
        @test result.diversity == 0.65
        @test result.n_oracle_calls == 10000
        @test result.unique_molecules == 5000
        @test result.oracle_call_breakdown == Dict{String,Int}()
        @test result.provenance_summary == Dict{String,Any}()
        @test result.artifact_paths == Dict{String,String}()
        @test result.diagnostics_summary == Dict{String,Any}()

        rich = PMOResult(
            "drd2", 0.80, 0.97, 0.90, 0.66, 3000, 1200,
            Dict("seed" => 120, "frontier_bootstrap" => 32, "model" => 2068, "ga" => 500, "he_warmup" => 140, "he_interleaved" => 140, "total" => 3000),
            Dict("topk_source_fractions" => Dict("tb" => 0.4, "ga" => 0.3, "he" => 0.2, "seed" => 0.05, "bootstrap" => 0.05))
        )
        @test rich.oracle_call_breakdown["ga"] == 500
        @test rich.provenance_summary["topk_source_fractions"]["he"] == 0.2
        @test rich.artifact_paths == Dict{String,String}()
        @test rich.diagnostics_summary == Dict{String,Any}()
    end

    @testset "PMOBenchmarkReport struct" begin
        results = [
            PMOResult("drd2", 0.75, 0.95, 0.88, 0.65, 10000, 5000),
            PMOResult("gsk3b", 0.70, 0.90, 0.85, 0.60, 10000, 4000),
        ]
        report = PMOBenchmarkReport(results, 1.45, 5, 10000)
        @test report.total_score == 1.45
        @test report.n_runs == 5
        @test report.budget_per_task == 10000
        @test length(report.task_results) == 2
    end

    @testset "benchmark_results_to_dict" begin
        results = [PMOResult(
            "drd2", 0.75, 0.95, 0.88, 0.65, 10000, 5000,
            Dict("seed" => 100, "frontier_bootstrap" => 32, "model" => 9268, "ga" => 600, "total" => 10000),
            Dict("topk_source_fractions" => Dict("tb" => 0.5, "ga" => 0.3, "he" => 0.05, "seed" => 0.1, "bootstrap" => 0.05)),
            Dict("episode_summary" => "/tmp/he_episode_summary.jls"),
            Dict("run_capacity" => Dict("episode_count" => 4, "total_he_calls" => 180))
        )]
        report = PMOBenchmarkReport(results, 0.75, 5, 10000)

        dict = benchmark_results_to_dict(report)
        @test haskey(dict, "tasks")
        @test haskey(dict, "total_score")
        @test haskey(dict, "n_runs")
        @test haskey(dict, "budget_per_task")
        @test haskey(dict, "sota_comparison")

        @test dict["total_score"] == 0.75
        @test dict["n_runs"] == 5
        @test length(dict["tasks"]) == 1

        task = dict["tasks"][1]
        @test task["task_name"] == "drd2"
        @test task["auc_top10"] == 0.75
        @test task["top1"] == 0.95
        @test task["oracle_call_breakdown"]["ga"] == 600
        @test task["provenance_summary"]["topk_source_fractions"]["tb"] == 0.5
        @test task["artifact_paths"]["episode_summary"] == "/tmp/he_episode_summary.jls"
        @test task["diagnostics_summary"]["run_capacity"]["episode_count"] == 4
    end

    @testset "Stage B prime provenance helpers" begin
        frontier = MolecularFrontierBuffer(32)
        add_to_frontier!(frontier, "CCO"; reward=0.7, source=:model, operator=:sample)
        add_to_frontier!(frontier, "CCN"; reward=0.8, source=:ga, operator=:mutation)
        add_to_frontier!(frontier, "CCC"; reward=0.9, source=:edit, operator=:mutate)
        add_to_frontier!(frontier, "CCF"; reward=0.6, source=:warmup, operator=:crossover)
        add_to_frontier!(frontier, "CCCl"; reward=0.5, source=:seed, operator=:seed)
        add_to_frontier!(frontier, "CCBr"; reward=0.55, source=:bootstrap, operator=:bootstrap)

        summary = _pmo_provenance_summary(frontier)
        @test summary["overall_source_counts"]["tb"] == 1
        @test summary["overall_source_counts"]["ga"] == 1
        @test summary["overall_source_counts"]["he"] == 2
        @test summary["overall_source_counts"]["seed"] == 1
        @test summary["overall_source_counts"]["bootstrap"] == 1
        @test isapprox(summary["topk_source_fractions"]["he"], 2 / 6; atol=1e-8)
    end

    @testset "Stage B prime HE guardrails" begin
        @test_nowarn _assert_heuristic_he_config(HierarchicalEditConfig())
        @test_throws ErrorException _assert_heuristic_he_config(HierarchicalEditConfig(; use_learned_parent=true))
        @test_throws ErrorException _assert_heuristic_he_config(HierarchicalEditConfig(; allow_fragment_ops=true))
        @test_throws ErrorException _assert_heuristic_he_config(HierarchicalEditConfig(; operators=[:mutate, :add_fragment]))
    end

    @testset "AUC top-10 computation edge cases" begin
        # Empty manager
        configs = [OracleConfig("test", 1.0)]
        mgr = OracleManager(configs, 100, 0, Dict{String,Dict{String,Float64}}(), true)
        @test compute_auc_top10(mgr) == 0.0

        # Single checkpoint
        _update_top_scores!(mgr, 0.9)
        _record_auc_checkpoint!(mgr)
        @test compute_auc_top10(mgr) == 0.9

        # Multiple checkpoints — monotonically increasing
        _update_top_scores!(mgr, 0.95)
        _record_auc_checkpoint!(mgr)
        auc = compute_auc_top10(mgr)
        @test auc > 0.0
        @test auc <= 1.0
    end

    @testset "budget-driven termination" begin
        configs = [OracleConfig("test", 1.0)]
        mgr = OracleManager(configs, 10)  # tiny budget

        # Simulate budget consumption
        for i in 1:10
            mgr.calls_used += 1
        end

        @test budget_exhausted(mgr)
        @test budget_remaining(mgr) == 0

        # evaluate_molecules! should skip when budget exhausted
        # (no error, just returns early)
        @test budget_exhausted(mgr)
    end
end
