# Unit tests for Gap 3: Fragment Library Expansion
# Tests FragmentMetadata, BRICS compatibility, dynamic loading, and backward compatibility

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

@testset "Fragment Library Expansion (Gap 3)" begin

    @testset "FragmentMetadata construction" begin
        # Default metadata (no-arg constructor)
        meta = FragmentMetadata()
        @test isempty(meta.brics_labels)
        @test meta.n_attachments == 1
        @test meta.category == "unknown"
        @test !meta.is_starter

        # Full metadata
        meta2 = FragmentMetadata([3, 5], 2, 8, "ring", true)
        @test meta2.brics_labels == [3, 5]
        @test meta2.n_attachments == 2
        @test meta2.heavy_atoms == 8
        @test meta2.category == "ring"
        @test meta2.is_starter
    end

    @testset "FragmentAction backward compatibility" begin
        # 3-arg constructor (legacy)
        action = FragmentAction(1, "c1ccc([*])cc1", "benzene")
        @test action.fragment_id == 1
        @test action.fragment_smiles == "c1ccc([*])cc1"
        @test action.fragment_name == "benzene"
        @test action.metadata.category == "unknown"  # Default metadata

        # 4-arg constructor (with metadata)
        meta = FragmentMetadata([5, 6], 2, 10, "starter", true)
        action2 = FragmentAction(41, "c1ccc([*])c([*])c1", "disubstituted_benzene", meta)
        @test action2.metadata.is_starter
        @test action2.metadata.brics_labels == [5, 6]
    end

    @testset "MolState backward compatibility" begin
        # 5-arg constructor (legacy, no attachment_labels)
        fp = zeros(Float32, 1024)
        state = MolState("c1ccccc1", [0, 1], 1, false, fp)
        @test isempty(state.attachment_labels)

        # 6-arg constructor (with attachment_labels)
        state2 = MolState("c1ccccc1", [0, 1], [5, 6], 1, false, fp)
        @test state2.attachment_labels == [5, 6]
    end

    @testset "BRICS compatibility" begin
        # Known compatible pairs
        @test is_brics_compatible(1, 3) == true
        @test is_brics_compatible(3, 1) == true  # Symmetric
        @test is_brics_compatible(1, 5) == true
        @test is_brics_compatible(5, 6) == true

        # Known incompatible pairs
        @test is_brics_compatible(1, 2) == false
        @test is_brics_compatible(2, 3) == false

        # Zero label = any (legacy compatibility)
        @test is_brics_compatible(0, 5) == true
        @test is_brics_compatible(3, 0) == true
        @test is_brics_compatible(0, 0) == true
    end

    @testset "is_applicable with metadata" begin
        fp = zeros(Float32, 1024)

        # Empty state — only starters allowed
        empty_state = MolState("", Int[], Int[], 0, false, fp)

        # Legacy action (no metadata, fragment_id >= 41 = starter)
        legacy_starter = FragmentAction(41, "c1ccc([*])c([*])c1", "starter")
        @test GFlowNet.is_applicable(legacy_starter, empty_state) == true

        # Legacy action (fragment_id < 41 = not starter)
        legacy_nonstarter = FragmentAction(1, "c1ccc([*])cc1", "benzene")
        @test GFlowNet.is_applicable(legacy_nonstarter, empty_state) == false

        # Metadata-based starter
        meta_starter = FragmentMetadata([5, 6], 2, 10, "starter", true)
        action_starter = FragmentAction(100, "c1ccc([*])c([*])c1", "test_starter", meta_starter)
        @test GFlowNet.is_applicable(action_starter, empty_state) == true

        # Metadata-based non-starter
        meta_ring = FragmentMetadata([5], 1, 6, "ring", false)
        action_ring = FragmentAction(101, "c1ccc([*])cc1", "test_ring", meta_ring)
        @test GFlowNet.is_applicable(action_ring, empty_state) == false

        # Non-empty state with attachment points — non-starters allowed
        nonempty_state = MolState("c1ccccc1", [0], Int[], 1, false, fp)
        @test GFlowNet.is_applicable(action_ring, nonempty_state) == true
        # Starters NOT allowed on non-empty state
        @test GFlowNet.is_applicable(action_starter, nonempty_state) == false

        # BRICS-aware compatibility check
        state_with_labels = MolState("c1ccccc1", [0], [5], 1, false, fp)
        compatible_frag = FragmentMetadata([6], 1, 4, "ring", false)
        action_compat = FragmentAction(102, "c1cc([*])co1", "furan", compatible_frag)
        @test GFlowNet.is_applicable(action_compat, state_with_labels) == true

        incompatible_frag = FragmentMetadata([2], 1, 4, "ring", false)
        action_incompat = FragmentAction(103, "c1cc([*])co1", "furan2", incompatible_frag)
        @test GFlowNet.is_applicable(action_incompat, state_with_labels) == false
    end

    @testset "compute_state_dim with BRICS" begin
        @test compute_state_dim(Dict()) == 1042
        @test compute_state_dim(Dict("use_brics_labels" => true)) == 1058
        @test compute_state_dim(Dict("use_preferences" => true)) == 1106
        @test compute_state_dim(Dict("use_brics_labels" => true, "use_preferences" => true)) == 1122
    end

    @testset "state_to_features with BRICS labels" begin
        fp = zeros(Float32, 1024)

        # Without labels — should be 1042
        state = MolState("c1ccccc1", [0], Int[], 1, false, fp)
        features = GFlowNet.state_to_features(state)
        @test length(features) == 1042

        # With labels — should be 1058 (1042 + 16 BRICS label encoding)
        state_brics = MolState("c1ccccc1", [0], [5], 1, false, fp)
        features_brics = GFlowNet.state_to_features(state_brics)
        @test length(features_brics) == 1058
        # First BRICS label slot should be 5/16
        @test features_brics[1043] ≈ 5.0f0 / 16.0f0
    end

    @testset "legacy FRAGMENT_LIBRARY" begin
        @test length(FRAGMENT_LIBRARY) == EXPECTED_FRAGMENT_COUNT
        # First fragment should be benzene
        @test FRAGMENT_LIBRARY[1].fragment_name == "benzene"
        # Last fragment should be a starter
        @test FRAGMENT_LIBRARY[EXPECTED_FRAGMENT_COUNT].fragment_name == "2,3-disubstituted_thiophene"
    end
end
