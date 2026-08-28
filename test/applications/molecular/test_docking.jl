# Tests for Docking-Based Reward Integration (Gap 2)
# Validates: sigmoid normalization, objective integration, target config,
# proxy model API, and backward compatibility

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

# RDKit availability comes from the single gate in test/fixtures/molecular.jl
# (loaded by test_setup.jl above), which also logs WHY it is unavailable. The
# name is file-local because test_diversity.jl defines its own; both used to
# write `const _rdkit_available` into Main.
const _rdkit_avail_docking = RDKIT_AVAILABLE

@testset "Docking Integration" begin

    # All eight testsets below are RDKit-only. @info is not enough: a skipped
    # suite must be as visible as a failing one, and must carry the reason.
    if !_rdkit_avail_docking
        @warn "ALL 8 docking testsets SKIPPED" reason = rdkit_reason()
    end

    @testset "sigmoid normalization" begin
        if !_rdkit_avail_docking
            @test_skip "RDKitBridge not available"
        else
            # Center point should map to 0.5
            @test RDKitBridge.sigmoid_normalize(-6.0) ≈ 0.5 atol=1e-6

            # More negative (better binder) → higher score
            @test RDKitBridge.sigmoid_normalize(-10.0) > 0.5
            @test RDKitBridge.sigmoid_normalize(-12.0) > RDKitBridge.sigmoid_normalize(-10.0)

            # Less negative (weaker binder) → lower score
            @test RDKitBridge.sigmoid_normalize(-2.0) < 0.5
            @test RDKitBridge.sigmoid_normalize(0.0) < RDKitBridge.sigmoid_normalize(-2.0)

            # All values should be in (0, 1)
            for score in [-15.0, -10.0, -6.0, -3.0, 0.0, 5.0]
                normalized = RDKitBridge.sigmoid_normalize(score)
                @test 0.0 < normalized < 1.0
            end

            # Monotonically decreasing (more negative = better)
            scores = [0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0]
            normalized = [RDKitBridge.sigmoid_normalize(s) for s in scores]
            for i in 1:length(normalized)-1
                @test normalized[i] < normalized[i+1]
            end

            # Custom center and scale
            @test RDKitBridge.sigmoid_normalize(-8.0; center=-8.0, scale=1.0) ≈ 0.5 atol=1e-6
        end
    end

    @testset "docking target configuration" begin
        if !_rdkit_avail_docking
            @test_skip "RDKitBridge not available"
        else
            # Targets should be loaded from JSON
            targets = RDKitBridge.get_docking_targets()
            # May be empty if data/targets/targets.json doesn't exist in test env
            # but the API should not error
            @test targets isa Dict

            # has_docking_target depends on whether targets were loaded
            @test RDKitBridge.has_docking_target() isa Bool
            @test RDKitBridge.get_docking_target() isa String
        end
    end

    @testset "docking availability flags" begin
        if !_rdkit_avail_docking
            @test_skip "RDKitBridge not available"
        else
            @test RDKitBridge.is_docking_available() isa Bool
            @test RDKitBridge.is_proxy_available() isa Bool
        end
    end

    @testset "compute_all_objectives length" begin
        if !_rdkit_avail_docking
            @test_skip "RDKitBridge not available"
        else
            state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
            objs = compute_all_objectives(state)
            # Was `if !isempty(objs)`, which would have hidden the whole check.
            @test !isempty(objs)
            if RDKitBridge.has_docking_target() && RDKitBridge.is_proxy_available()
                @test length(objs) == 5
            else
                @test length(objs) == 4
            end
            for (i, obj) in enumerate(objs)
                @test 0.0 <= obj <= 1.0
            end
        end
    end

    @testset "reward backward compatibility with docking" begin
        if !_rdkit_avail_docking
            @test_skip "RDKitBridge not available"
        else
            state = MolState("c1ccccc1", Int[], 1, true, zeros(Float32, 1024))
            r_single = GFlowNet.reward(state)
            @test r_single > 0.0
            @test isfinite(r_single)

            w4 = [0.25, 0.25, 0.25, 0.25]
            r_multi = GFlowNet.reward(state, w4)
            @test r_multi > 0.0
            @test isfinite(r_multi)

            objs = compute_all_objectives(state)
            # The 5-objective weight vector only applies when a docking target
            # AND a trained proxy are configured; without them the reward is a
            # 4-vector and a 5-weight call is genuinely wrong. Say which branch
            # ran instead of silently taking neither.
            if length(objs) == 5
                w5 = [0.2, 0.15, 0.1, 0.1, 0.45]
                r5 = GFlowNet.reward(state, w5)
                @test r5 > 0.0
                @test isfinite(r5)
            else
                @test length(objs) == 4
                @info "5-objective docking reward not exercised: no docking target / proxy" n_objectives = length(objs)
            end
        end
    end

    @testset "proxy dock fallback" begin
        if !_rdkit_avail_docking
            @test_skip "RDKitBridge not available"
        else
            # proxy_dock returns the neutral 0.5 only while the proxy is
            # untrained; if one ever IS trained in the test environment, assert
            # the normalized-score contract instead of skipping silently.
            score = RDKitBridge.proxy_dock("c1ccccc1")
            if RDKitBridge.is_proxy_available()
                @test 0.0 < score < 1.0
                @info "docking proxy IS trained; asserted range instead of the 0.5 fallback" score
            else
                @test score ≈ 0.5 atol=1e-6
            end
        end
    end

    @testset "DockingResult struct" begin
        if !_rdkit_avail_docking
            @test_skip "RDKitBridge not available"
        else
            result = RDKitBridge.DockingResult("c1ccccc1", -8.5, 3, 1500, "")
            @test result.smiles == "c1ccccc1"
            @test result.affinity_kcal == -8.5
            @test result.n_poses == 3
            @test result.runtime_ms == 1500
            @test isempty(result.error)

            # Error result
            err_result = RDKitBridge.DockingResult("bad", 0.0, 0, 0, "Invalid SMILES")
            @test !isempty(err_result.error)
        end
    end

    @testset "set_docking_target! validation" begin
        if !_rdkit_avail_docking
            @test_skip "RDKitBridge not available"
        else
            # Setting an unknown target should fail gracefully
            result = RDKitBridge.set_docking_target!("nonexistent_target_xyz")
            @test result == false

            # Original target should be preserved
            original = RDKitBridge.get_docking_target()
            RDKitBridge.set_docking_target!("nonexistent_target_xyz")
            @test RDKitBridge.get_docking_target() == original
        end
    end
end
