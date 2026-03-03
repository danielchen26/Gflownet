# Tests for Oracle Integration (PMO Target-Specific Molecular Optimization)
#
# Tests oracle_bridge, oracle_manager, and their integration with
# compute_all_objectives and the reward system.

using Test
using Statistics: mean

include(joinpath(@__DIR__, "test_setup.jl"))

# Include RDKitBridge (needed by compute_all_objectives)
if !@isdefined(RDKitBridge)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
end

# Include oracle modules (normally loaded by unified_server.jl)
if !@isdefined(OracleBridge)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
end
if !@isdefined(OracleManager)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
end

# Include MolecularAdapter (normally loaded by unified_server.jl)
if !@isdefined(AbstractDomainAdapter)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization", "core", "adapters.jl"))
end
if !@isdefined(MolecularAdapter)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization", "domains", "molecular.jl"))
end

@testset "Oracle Integration" begin

    @testset "OracleConfig struct" begin
        config = OracleConfig("DRD2", 0.4)
        @test config.name == "DRD2"
        @test config.weight == 0.4
    end

    @testset "OracleManager creation" begin
        configs = [OracleConfig("DRD2", 0.4), OracleConfig("GSK3B", 0.3)]
        mgr = OracleManager(configs, 10000, 0, Dict{String,Dict{String,Float64}}(), false)

        @test mgr.budget == 10000
        @test mgr.calls_used == 0
        @test isempty(mgr.cache)
        @test !mgr.benchmark_mode
        @test budget_remaining(mgr) == 10000
        @test !budget_exhausted(mgr)
    end

    @testset "OracleManager cache operations" begin
        configs = [OracleConfig("DRD2", 1.0)]
        mgr = OracleManager(configs, 100)

        # lookup_score returns 0.5 (neutral) for uncached
        @test lookup_score(mgr, "c1ccccc1", "DRD2") == 0.5

        # Manual cache update
        mgr.cache["c1ccccc1"] = Dict("DRD2" => 0.85)
        @test lookup_score(mgr, "c1ccccc1", "DRD2") == 0.85

        # Unknown oracle returns neutral
        @test lookup_score(mgr, "c1ccccc1", "UNKNOWN") == 0.5
    end

    @testset "OracleManager budget tracking" begin
        configs = [OracleConfig("DRD2", 1.0)]
        mgr = OracleManager(configs, 5)  # tiny budget

        # Simulate budget usage
        mgr.calls_used = 3
        @test budget_remaining(mgr) == 2
        @test !budget_exhausted(mgr)

        mgr.calls_used = 5
        @test budget_remaining(mgr) == 0
        @test budget_exhausted(mgr)

        mgr.calls_used = 7  # over budget
        @test budget_remaining(mgr) == 0
    end

    @testset "OracleManager get_objective_names" begin
        configs = [OracleConfig("DRD2", 0.4), OracleConfig("GSK3B", 0.3), OracleConfig("JNK3", 0.3)]
        mgr = OracleManager(configs, 10000)

        names = get_objective_names(mgr)
        @test names == ["DRD2", "GSK3B", "JNK3"]

        weights = get_objective_weights(mgr)
        @test weights == [0.4, 0.3, 0.3]
    end

    @testset "OracleManager get_status" begin
        configs = [OracleConfig("DRD2", 0.4)]
        mgr = OracleManager(configs, 10000)
        mgr.calls_used = 500
        mgr.cache["smi1"] = Dict("DRD2" => 0.9)
        mgr.cache["smi2"] = Dict("DRD2" => 0.3)

        status = get_status(mgr)
        @test status["configured"] == ["DRD2"]
        @test status["budget_used"] == 500
        @test status["budget_total"] == 10000
        @test status["budget_remaining"] == 9500
        @test status["cache_size"] == 2
        @test status["benchmark_mode"] == false
    end

    @testset "AUC top-10 tracking" begin
        configs = [OracleConfig("test", 1.0)]
        mgr = OracleManager(configs, 10000, 0, Dict{String,Dict{String,Float64}}(), true)

        # Add scores
        for s in [0.5, 0.8, 0.3, 0.9, 0.7]
            _update_top_scores!(mgr, s)
        end

        # Top scores should be sorted descending, max 10
        @test mgr.top_scores[1] == 0.9
        @test mgr.top_scores[2] == 0.8
        @test length(mgr.top_scores) == 5

        # Record checkpoint
        _record_auc_checkpoint!(mgr)
        @test length(mgr.auc_checkpoints) == 1
        @test mgr.auc_checkpoints[1] ≈ mean([0.9, 0.8, 0.7, 0.5, 0.3])

        # AUC with single checkpoint
        auc = compute_auc_top10(mgr)
        @test auc > 0.0
    end

    @testset "benchmark_mode reward returns oracle-only objectives" begin
        configs = [OracleConfig("DRD2", 0.5), OracleConfig("GSK3B", 0.5)]
        mgr = OracleManager(configs, 10000, 0, Dict{String,Dict{String,Float64}}(), true)

        # Pre-populate cache
        mgr.cache["c1ccccc1"] = Dict("DRD2" => 0.1, "GSK3B" => 0.2)

        state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
        objs = compute_all_objectives(state; oracle_mgr=mgr)

        # In benchmark_mode, should return ONLY oracle scores
        @test length(objs) == 2
        @test objs[1] == 0.1  # DRD2
        @test objs[2] == 0.2  # GSK3B
    end

    # Initialize RDKit for tests that call compute_all_objectives in normal mode
    RDKitBridge.init_rdkit!()

    @testset "normal mode appends oracle scores to base objectives" begin
        configs = [OracleConfig("DRD2", 0.5)]
        mgr = OracleManager(configs, 10000, 0, Dict{String,Dict{String,Float64}}(), false)

        # Pre-populate cache
        mgr.cache["c1ccccc1"] = Dict("DRD2" => 0.85)

        state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))

        # Without oracle_mgr: base 4 objectives
        objs_base = compute_all_objectives(state)

        # With oracle_mgr: base 4 + 1 oracle
        objs_with = compute_all_objectives(state; oracle_mgr=mgr)

        @test length(objs_with) == length(objs_base) + 1
        @test objs_with[end] == 0.85  # DRD2 score appended
    end

    @testset "compute_all_objectives backward compatibility" begin
        # Without oracle_mgr (existing behavior unchanged)
        state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
        objs = compute_all_objectives(state)
        @test length(objs) >= 4
        @test all(0.0 .<= objs .<= 1.0)

        # Non-terminal returns empty
        non_terminal = MolState("c1ccccc1", Int[], 1, false, zeros(Float32, 1024))
        @test isempty(compute_all_objectives(non_terminal))
    end

    @testset "OracleBridge module structure" begin
        # Test module constants (doesn't require Python)
        @test length(OracleBridge.PMO_23_TASKS) == 23
        @test "drd2" in OracleBridge.PMO_23_TASKS
        @test "gsk3b" in OracleBridge.PMO_23_TASKS
        @test "jnk3" in OracleBridge.PMO_23_TASKS

        @test length(OracleBridge.BIOACTIVITY_ORACLES) == 3

        available = OracleBridge.get_all_available_oracles()
        @test haskey(available, "bioactivity")
        @test haskey(available, "pmo_tasks")
        @test haskey(available, "all")
    end

    @testset "MolecularAdapter oracle_manager field" begin
        # Without oracle manager
        adapter1 = MolecularAdapter(8, FRAGMENT_LIBRARY, Dict[])
        @test adapter1.oracle_manager === nothing

        # With oracle manager
        configs = [OracleConfig("DRD2", 1.0)]
        mgr = OracleManager(configs, 10000)
        adapter2 = MolecularAdapter(8, FRAGMENT_LIBRARY, Dict[], mgr)
        @test adapter2.oracle_manager !== nothing
        @test adapter2.oracle_manager.budget == 10000
    end
end
