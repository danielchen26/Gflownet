# Tests for Reaction-Based Molecular Generation (Gap 4)
# Validates: reaction types, state features, reward function, apply_reaction,
# synthesis route visualization, domain adapter, and factory function

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

# The reaction domain lives in the visualization layer, so `using GFlowNet` does
# not bring it in. It does NOT need a running server: core/adapters.jl plus
# domains/reaction_molecular.jl is the entire dependency chain (the same order
# unified_server.jl uses at src/utils/visualization/api/unified_server.jl:13 and
# :26), and test_setup.jl has already loaded molecular_generation.jl.
#
# Before this, 22 of the 24 testsets below were `@test_skip "... requires server
# module"`: every reaction type, feature-vector, adapter and registry assertion
# in the file verified nothing. Only the two create_reaction_gflownet testsets
# at the bottom ever ran.
const _REACTION_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "src",
                                        "utils", "visualization"))
if !@isdefined(AbstractDomainAdapter)
    include(joinpath(_REACTION_SRC, "core", "adapters.jl"))
end
if !@isdefined(ReactionMolState)
    include(joinpath(_REACTION_SRC, "domains", "reaction_molecular.jl"))
end

# Four testsets below reach real chemistry: reward() and state_to_features() call
# RDKitBridge.compute_mol_properties for a non-empty intermediate
# (reaction_molecular.jl:83, :105), and apply_reaction/get_domain_config call
# RDKitBridge.get_reaction_templates (:145, :261). Those stay gated -- loudly.
if !RDKIT_AVAILABLE
    @warn """Reaction CHEMISTRY assertions SKIPPED: "state_to_features content", \
             "reward function" (terminal branch), "reward monotonicity with steps", \
             "apply_reaction with invalid reaction_id", "get_domain_config". \
             Everything else in this file still runs.""" reason = rdkit_reason()
end

@testset "Reaction Constraints" begin

    # =========================================================================
    # Reaction types (src/utils/visualization/domains/reaction_molecular.jl)
    # =========================================================================

    @testset "ReactionMolState construction" begin
        # Empty initial state
        state = create_initial_reaction_state()
        @test state isa ReactionMolState
        @test isempty(state.intermediates)
        @test isempty(state.reaction_history)
        @test state.n_steps == 0
        @test state.is_terminated == false
        @test length(state.fingerprint) == 1024
        @test all(state.fingerprint .== 0.0f0)

        # State with data
        state2 = ReactionMolState(
            ["c1ccccc1", "CCO"],
            [1, 3],
            2,
            false,
            ones(Float32, 1024),
        )
        @test length(state2.intermediates) == 2
        @test state2.intermediates[end] == "CCO"
        @test state2.reaction_history == [1, 3]
        @test state2.n_steps == 2
    end

    @testset "ReactionAction construction" begin
        # Normal reaction action
        action = ReactionAction(1, "c1ccccc1", "CCO")
        @test action.reaction_id == 1
        @test action.reactant1_smiles == "c1ccccc1"
        @test action.reactant2_smiles == "CCO"

        # Unimolecular reaction
        unimol = ReactionAction(5, "c1ccccc1", "")
        @test unimol.reaction_id == 5
        @test isempty(unimol.reactant2_smiles)

        # Terminate action
        @test TERMINATE_REACTION.reaction_id == 0
        @test isempty(TERMINATE_REACTION.reactant1_smiles)
        @test isempty(TERMINATE_REACTION.reactant2_smiles)
    end

    @testset "ReactionAction equality and hashing" begin
        a1 = ReactionAction(1, "CC", "O")
        a2 = ReactionAction(1, "CC", "O")
        a3 = ReactionAction(2, "CC", "O")

        @test a1 == a2
        @test a1 != a3
        @test hash(a1) == hash(a2)
        @test hash(a1) != hash(a3)

        # Use in Set
        s = Set([a1, a2, a3])
        @test length(s) == 2
    end

    @testset "ReactionMolState equality and hashing" begin
        s1 = ReactionMolState(["CC"], [1], 1, false, zeros(Float32, 1024))
        s2 = ReactionMolState(["CC"], [1], 1, false, ones(Float32, 1024))
        s3 = ReactionMolState(["CCO"], [1], 1, false, zeros(Float32, 1024))

        # Equality ignores fingerprint (based on intermediates + n_steps + is_terminated)
        @test s1 == s2
        @test s1 != s3
    end

    @testset "state_to_features dimensions" begin
        state = create_initial_reaction_state()
        features = GFlowNet.state_to_features(state)

        @test length(features) == REACTION_STATE_DIM
        @test length(features) == 1049  # 1024 FP + 17 rxn one-hot + 8 scalars
        @test eltype(features) == Float32
        @test all(isfinite, features)
    end

    @testset "state_to_features content" begin
        # Needs RDKit: a non-empty intermediate sends state_to_features into
        # RDKitBridge.compute_mol_properties (reaction_molecular.jl:83).
        if !RDKIT_AVAILABLE
            @test_skip "state_to_features on a non-empty state requires RDKit"
        else
            fp = zeros(Float32, 1024)
            fp[1] = 1.0f0
            fp[512] = 1.0f0

            state = ReactionMolState(
                ["c1ccccc1"],
                [3],
                1,
                false,
                fp,
            )

            features = GFlowNet.state_to_features(state)

            # Fingerprint region (1:1024)
            @test features[1] == 1.0f0
            @test features[512] == 1.0f0
            @test features[2] == 0.0f0

            # One-hot reaction region (1025:1041) — last reaction = 3
            @test features[1024 + 3] == 1.0f0  # reaction 3 active
            @test features[1024 + 1] == 0.0f0  # reaction 1 inactive
            @test features[1024 + 17] == 0.0f0 # reaction 17 inactive

            # Scalar features
            offset = 1024 + N_REACTIONS
            @test features[offset + 1] ≈ Float32(1 / MAX_REACTION_STEPS)  # Progress
            @test features[offset + 2] ≈ Float32(1.0)                      # N intermediates
            @test features[offset + 3] ≈ Float32(0.0)                      # Terminal flag (not terminated)
        end
    end

    @testset "state_to_features empty state" begin
        state = create_initial_reaction_state()
        features = GFlowNet.state_to_features(state)

        # All fingerprint should be zero
        @test all(features[1:1024] .== 0.0f0)

        # No reaction history → no one-hot
        @test all(features[1025:1041] .== 0.0f0)

        # Progress = 0
        offset = 1024 + N_REACTIONS
        @test features[offset + 1] == 0.0f0
        @test features[offset + 2] == 0.0f0
    end

    @testset "is_terminal_state" begin
        non_terminal = ReactionMolState(["CC"], [1], 1, false, zeros(Float32, 1024))
        terminal = ReactionMolState(["CC"], [1], 1, true, zeros(Float32, 1024))

        @test GFlowNet.is_terminal_state(non_terminal) == false
        @test GFlowNet.is_terminal_state(terminal) == true
    end

    @testset "reward function" begin
        # These two branches return before touching RDKit
        # (reaction_molecular.jl:101-102), so they are always checked.
        non_terminal = ReactionMolState(["CC"], [1], 1, false, zeros(Float32, 1024))
        @test GFlowNet.reward(non_terminal) == 0.0

        # Terminal state with empty intermediates → minimum reward
        empty_terminal = ReactionMolState(String[], Int[], 0, true, zeros(Float32, 1024))
        @test GFlowNet.reward(empty_terminal) == 1e-4

        # Terminal with a real molecule needs compute_mol_properties.
        if !RDKIT_AVAILABLE
            @test_skip "reward() on a terminal molecule requires RDKit"
        else
            state = ReactionMolState(["c1ccccc1"], [1], 1, true, zeros(Float32, 1024))
            r = GFlowNet.reward(state)
            @test r > 0.0
            @test r <= 1.0
            @test isfinite(r)
        end
    end

    @testset "reward monotonicity with steps" begin
        if !RDKIT_AVAILABLE
            @test_skip "reward() on a terminal molecule requires RDKit"
        else
            # Shorter routes should have higher step_bonus
            state1 = ReactionMolState(["c1ccccc1"], [1], 1, true, zeros(Float32, 1024))
            state5 = ReactionMolState(["c1ccccc1"], [1, 2, 3, 4, 5], 5, true, zeros(Float32, 1024))

            r1 = GFlowNet.reward(state1)
            r5 = GFlowNet.reward(state5)

            # state1 has fewer steps → higher step_bonus → should have higher reward
            # (assuming same molecule)
            @test r1 >= r5
        end
    end

    @testset "apply_reaction termination" begin
        # reaction_id == 0 returns before reaching RDKitBridge
        # (reaction_molecular.jl:134-143), so this is always checked.
        state = ReactionMolState(["CC"], [1], 1, false, zeros(Float32, 1024))
        terminated = apply_reaction(state, TERMINATE_REACTION)

        @test terminated.is_terminated == true
        @test terminated.intermediates == state.intermediates
        @test terminated.reaction_history == state.reaction_history
        @test terminated.n_steps == state.n_steps

        # Original state unchanged
        @test state.is_terminated == false
    end

    @testset "apply_reaction with invalid reaction_id" begin
        # Needs RDKit: the template lookup goes through
        # RDKitBridge.get_reaction_templates (reaction_molecular.jl:145).
        if !RDKIT_AVAILABLE
            @test_skip "apply_reaction template lookup requires RDKit"
        else
            state = create_initial_reaction_state()
            action = ReactionAction(999, "CC", "")  # Invalid ID

            result = apply_reaction(state, action)
            # Should return original state unchanged
            @test result.intermediates == state.intermediates
            @test result.n_steps == state.n_steps
        end
    end

    @testset "constants" begin
        @test MAX_REACTION_STEPS == 5
        @test N_REACTIONS == 17
        @test REACTION_STATE_DIM == 1049
    end

    # =========================================================================
    # ReactionMolecularAdapter (visualization domain adapter)
    # =========================================================================

    @testset "ReactionMolecularAdapter construction" begin
        adapter = ReactionMolecularAdapter()
        @test adapter.max_steps == MAX_REACTION_STEPS
        @test isempty(adapter.generated_molecules)

        adapter2 = ReactionMolecularAdapter(3, Dict[])
        @test adapter2.max_steps == 3
    end

    @testset "get_domain_config" begin
        # Needs RDKit: get_domain_config enumerates the reaction templates
        # (reaction_molecular.jl:261).
        if !RDKIT_AVAILABLE
            @test_skip "get_domain_config enumerates reaction templates, requires RDKit"
        else
            adapter = ReactionMolecularAdapter()
            config = get_domain_config(adapter)

            @test config["domain_type"] == "reaction_molecule"
            @test config["max_steps"] == MAX_REACTION_STEPS
            @test config["state_dim"] == REACTION_STATE_DIM
            @test haskey(config, "n_reactions")
            @test haskey(config, "reactions")
        end
    end

    @testset "get_renderer_name" begin
        adapter = ReactionMolecularAdapter()
        @test get_renderer_name(adapter) == "ReactionMolecularRenderer"
    end

    @testset "domain registry methods" begin
        adapter = ReactionMolecularAdapter()

        @test get_domain_id(adapter) == "reaction_molecule"
        @test !isempty(get_domain_description(adapter))
        @test "Synthesis" in get_domain_tags(adapter)
        @test is_builtin_domain(adapter) == true
        @test is_popular_domain(adapter) == false
    end

    @testset "config schema and validation" begin
        adapter = ReactionMolecularAdapter()
        schema = get_config_schema(adapter)

        @test schema["type"] == "object"
        @test haskey(schema["properties"], "max_steps")
        @test schema["properties"]["max_steps"]["default"] == 5

        # Valid config
        valid, err = validate_config(adapter, Dict("max_steps" => 3))
        @test valid == true
        @test err === nothing

        # Invalid config
        valid, err = validate_config(adapter, Dict("max_steps" => 0))
        @test valid == false
        @test err !== nothing

        valid, err = validate_config(adapter, Dict("max_steps" => 100))
        @test valid == false
    end

    @testset "create_from_config" begin
        adapter = create_from_config(ReactionMolecularAdapter, Dict("max_steps" => 3))
        @test adapter isa ReactionMolecularAdapter
        @test adapter.max_steps == 3
        @test isempty(adapter.generated_molecules)

        # Default config
        adapter2 = create_from_config(ReactionMolecularAdapter, Dict())
        @test adapter2.max_steps == MAX_REACTION_STEPS
    end

    @testset "state_to_viz_data" begin
        adapter = ReactionMolecularAdapter()
        state = ReactionMolState(["c1ccccc1", "CCO"], [1, 3], 2, false, zeros(Float32, 1024))

        viz = state_to_viz_data(adapter, state)
        @test viz["intermediates"] == ["c1ccccc1", "CCO"]
        @test viz["reaction_history"] == [1, 3]
        @test viz["n_steps"] == 2
        @test viz["is_terminated"] == false
        @test viz["primary_smiles"] == "CCO"
    end

    @testset "state_to_viz_data empty state" begin
        adapter = ReactionMolecularAdapter()
        state = create_initial_reaction_state()

        viz = state_to_viz_data(adapter, state)
        @test viz["primary_smiles"] == ""
        @test isempty(viz["intermediates"])
    end

    @testset "compute_domain_metrics empty" begin
        adapter = ReactionMolecularAdapter()
        # The `model` argument is typed ::GFlowNetModel, so the previous
        # `nothing` here would have thrown MethodError had the testset ever run.
        model = create_reaction_gflownet()
        metrics = compute_domain_metrics(adapter, model, Trajectory[])
        @test metrics["n_molecules"] == 0
    end

    # =========================================================================
    # Tests using exported symbols (always available from GFlowNet)
    # =========================================================================

    @testset "create_reaction_gflownet factory" begin
        model = create_reaction_gflownet()
        @test model isa GFlowNet.GFlowNetModel
        @test model.forward_policy !== nothing
    end

    @testset "create_reaction_gflownet custom params" begin
        model = create_reaction_gflownet(hidden_dim=128, learning_rate=0.0005)
        @test model isa GFlowNet.GFlowNetModel
    end
end
