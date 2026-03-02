# Tests for Reaction-Based Molecular Generation (Gap 4)
# Validates: reaction types, state features, reward function, apply_reaction,
# synthesis route visualization, domain adapter, and factory function

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

# Reaction types (ReactionMolState, ReactionAction, TERMINATE_REACTION, etc.) and
# adapter types (ReactionMolecularAdapter, get_domain_config, etc.) are defined in
# the visualization/server layer, not in the core GFlowNet module.
# Only create_reaction_gflownet is exported from GFlowNet.

const _reaction_types_available = isdefined(Main, :ReactionMolState)
const _adapter_available = isdefined(Main, :ReactionMolecularAdapter)

if !_reaction_types_available
    @info "Skipping reaction type tests (ReactionMolState not available - requires server module)"
end

@testset "Reaction Constraints" begin

    # =========================================================================
    # Tests requiring ReactionMolState and related types (server module only)
    # =========================================================================

    @testset "ReactionMolState construction" begin
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
        else
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
    end

    @testset "ReactionAction construction" begin
        if !_reaction_types_available
            @test_skip "ReactionAction not available (requires server module)"
        else
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
    end

    @testset "ReactionAction equality and hashing" begin
        if !_reaction_types_available
            @test_skip "ReactionAction not available (requires server module)"
        else
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
    end

    @testset "ReactionMolState equality and hashing" begin
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
        else
            s1 = ReactionMolState(["CC"], [1], 1, false, zeros(Float32, 1024))
            s2 = ReactionMolState(["CC"], [1], 1, false, ones(Float32, 1024))
            s3 = ReactionMolState(["CCO"], [1], 1, false, zeros(Float32, 1024))

            # Equality ignores fingerprint (based on intermediates + n_steps + is_terminated)
            @test s1 == s2
            @test s1 != s3
        end
    end

    @testset "state_to_features dimensions" begin
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
        else
            state = create_initial_reaction_state()
            features = GFlowNet.state_to_features(state)

            @test length(features) == REACTION_STATE_DIM
            @test length(features) == 1049  # 1024 FP + 17 rxn one-hot + 8 scalars
            @test eltype(features) == Float32
            @test all(isfinite, features)
        end
    end

    @testset "state_to_features content" begin
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
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
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
        else
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
    end

    @testset "is_terminal_state" begin
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
        else
            non_terminal = ReactionMolState(["CC"], [1], 1, false, zeros(Float32, 1024))
            terminal = ReactionMolState(["CC"], [1], 1, true, zeros(Float32, 1024))

            @test GFlowNet.is_terminal_state(non_terminal) == false
            @test GFlowNet.is_terminal_state(terminal) == true
        end
    end

    @testset "reward function" begin
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
        else
            # Non-terminal state should return 0.0
            non_terminal = ReactionMolState(["CC"], [1], 1, false, zeros(Float32, 1024))
            @test GFlowNet.reward(non_terminal) == 0.0

            # Terminal state with empty intermediates → minimum reward
            empty_terminal = ReactionMolState(String[], Int[], 0, true, zeros(Float32, 1024))
            @test GFlowNet.reward(empty_terminal) == 1e-4

            # Terminal with known molecule
            state = ReactionMolState(["c1ccccc1"], [1], 1, true, zeros(Float32, 1024))
            r = GFlowNet.reward(state)
            @test r > 0.0
            @test r <= 1.0
            @test isfinite(r)
        end
    end

    @testset "reward monotonicity with steps" begin
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
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
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
        else
            state = ReactionMolState(["CC"], [1], 1, false, zeros(Float32, 1024))
            terminated = apply_reaction(state, TERMINATE_REACTION)

            @test terminated.is_terminated == true
            @test terminated.intermediates == state.intermediates
            @test terminated.reaction_history == state.reaction_history
            @test terminated.n_steps == state.n_steps

            # Original state unchanged
            @test state.is_terminated == false
        end
    end

    @testset "apply_reaction with invalid reaction_id" begin
        if !_reaction_types_available
            @test_skip "ReactionMolState not available (requires server module)"
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
        if !_reaction_types_available
            @test_skip "Reaction constants not available (requires server module)"
        else
            @test MAX_REACTION_STEPS == 5
            @test N_REACTIONS == 17
            @test REACTION_STATE_DIM == 1049
        end
    end

    # =========================================================================
    # Tests requiring ReactionMolecularAdapter (viz/server module only)
    # =========================================================================

    @testset "ReactionMolecularAdapter construction" begin
        if !_adapter_available
            @test_skip "ReactionMolecularAdapter not available (requires server module)"
        else
            adapter = ReactionMolecularAdapter()
            @test adapter.max_steps == MAX_REACTION_STEPS
            @test isempty(adapter.generated_molecules)

            adapter2 = ReactionMolecularAdapter(3, Dict[])
            @test adapter2.max_steps == 3
        end
    end

    @testset "get_domain_config" begin
        if !_adapter_available
            @test_skip "ReactionMolecularAdapter not available (requires server module)"
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
        if !_adapter_available
            @test_skip "ReactionMolecularAdapter not available (requires server module)"
        else
            adapter = ReactionMolecularAdapter()
            @test get_renderer_name(adapter) == "ReactionMolecularRenderer"
        end
    end

    @testset "domain registry methods" begin
        if !_adapter_available
            @test_skip "ReactionMolecularAdapter not available (requires server module)"
        else
            adapter = ReactionMolecularAdapter()

            @test get_domain_id(adapter) == "reaction_molecule"
            @test !isempty(get_domain_description(adapter))
            @test "Synthesis" in get_domain_tags(adapter)
            @test is_builtin_domain(adapter) == true
            @test is_popular_domain(adapter) == false
        end
    end

    @testset "config schema and validation" begin
        if !_adapter_available
            @test_skip "ReactionMolecularAdapter not available (requires server module)"
        else
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
    end

    @testset "create_from_config" begin
        if !_adapter_available
            @test_skip "ReactionMolecularAdapter not available (requires server module)"
        else
            adapter = create_from_config(ReactionMolecularAdapter, Dict("max_steps" => 3))
            @test adapter isa ReactionMolecularAdapter
            @test adapter.max_steps == 3
            @test isempty(adapter.generated_molecules)

            # Default config
            adapter2 = create_from_config(ReactionMolecularAdapter, Dict())
            @test adapter2.max_steps == MAX_REACTION_STEPS
        end
    end

    @testset "state_to_viz_data" begin
        if !_adapter_available || !_reaction_types_available
            @test_skip "Reaction types not available (requires server module)"
        else
            adapter = ReactionMolecularAdapter()
            state = ReactionMolState(["c1ccccc1", "CCO"], [1, 3], 2, false, zeros(Float32, 1024))

            viz = state_to_viz_data(adapter, state)
            @test viz["intermediates"] == ["c1ccccc1", "CCO"]
            @test viz["reaction_history"] == [1, 3]
            @test viz["n_steps"] == 2
            @test viz["is_terminated"] == false
            @test viz["primary_smiles"] == "CCO"
        end
    end

    @testset "state_to_viz_data empty state" begin
        if !_adapter_available || !_reaction_types_available
            @test_skip "Reaction types not available (requires server module)"
        else
            adapter = ReactionMolecularAdapter()
            state = create_initial_reaction_state()

            viz = state_to_viz_data(adapter, state)
            @test viz["primary_smiles"] == ""
            @test isempty(viz["intermediates"])
        end
    end

    @testset "compute_domain_metrics empty" begin
        if !_adapter_available
            @test_skip "ReactionMolecularAdapter not available (requires server module)"
        else
            adapter = ReactionMolecularAdapter()
            metrics = compute_domain_metrics(adapter, nothing, Trajectory[])
            @test metrics["n_molecules"] == 0
        end
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
