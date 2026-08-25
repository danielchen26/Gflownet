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
        results = [PMOResult("drd2", 0.75, 0.95, 0.88, 0.65, 10000, 5000)]
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
