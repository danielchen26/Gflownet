using Test
using GFlowNet
using Random
using Statistics
using Lux
using Optimisers

# Deterministic length-based reward (no RDKit dependency)
function length_reward(smiles::String)::Float64
    if isempty(smiles) || length(smiles) < 2
        return 0.0
    end
    # Reward based on length: prefer molecules of length 10-30
    len = length(smiles)
    if len < 5 || len > 100
        return 0.01
    end
    # Gaussian-like reward peaked at length 20
    return exp(-((len - 20)^2) / (2 * 10^2))
end

@testset "QGFN Integration" begin
    rng = Random.MersenneTwister(42)

    # Create a small model for testing
    vocab_size = 50
    hidden_dim = 64
    embed_dim = 32
    n_layers = 2

    # Build vocab from a minimal set
    smiles_list = ["C", "CC", "CCC", "O", "N", "c1ccccc1", "C=O", "C(=O)O",
                   "CCO", "CCN", "CCCC", "c1ccc(O)cc1", "CC(=O)O", "CC=O"]
    vocab = SMILESVocabulary()
    ds = prepare_zinc_dataset(vocab, smiles_list)

    model, params, states = create_smiles_policy(;
        vocab_size=vocab.size, hidden_dim=hidden_dim, embed_dim=embed_dim, n_layers=n_layers
    )

    @testset "Transition Collection" begin
        # Sample with transition collection
        smiles, tokens, log_prob, transitions = sample_smiles_autoregressive(
            model, params, states, vocab;
            max_length=50, temperature=1.0, collect_transitions=true, constrained=false
        )

        @test length(transitions) > 0
        @test length(transitions) == length(tokens) - 1  # One per generated token (START has none)

        # Check transition structure
        t = transitions[1]
        @test haskey(t, :hidden)
        @test haskey(t, :action)
        @test haskey(t, :next_hidden)
        @test haskey(t, :is_terminal)

        @test length(t.hidden) == hidden_dim  # Hidden state from last GRU layer
        @test t.action isa Integer
        @test length(t.next_hidden) == hidden_dim

        # Last transition should be terminal (END token)
        last_t = transitions[end]
        @test last_t.is_terminal == (tokens[end] == GFlowNet.END_TOKEN)
    end

    @testset "Fill Transitions with Reward" begin
        buffer = QTrainingBuffer(1000)

        # Sample and fill buffer
        smiles, tokens, log_prob, transitions = sample_smiles_autoregressive(
            model, params, states, vocab;
            max_length=50, temperature=1.0, collect_transitions=true, constrained=false
        )

        reward = length_reward(smiles)
        log_r = log(max(reward, 1e-8))
        fill_transitions_with_reward!(buffer, transitions, log_r)

        # Check buffer was filled
        @test length(buffer.hidden_states) == length(transitions)
        @test length(buffer.actions) == length(transitions)
        @test length(buffer.rewards) == length(transitions)

        # Non-terminal transitions should have reward 0
        for i in 1:(length(transitions) - 1)
            if !transitions[i].is_terminal
                @test buffer.rewards[i] == 0.0
            end
        end

        # Terminal transition should have log_reward
        terminal_idx = findfirst(t -> t.is_terminal, transitions)
        if !isnothing(terminal_idx)
            @test buffer.rewards[terminal_idx] ≈ log_r
        end
    end

    @testset "Collect and Fill Q-Buffer" begin
        buffer = QTrainingBuffer(5000)

        n_valid, mean_r, smiles_list_out = collect_and_fill_q_buffer!(
            buffer, model, params, states, vocab,
            length_reward;
            n_samples=20, max_length=50
        )

        # Should have collected some transitions
        @test length(buffer.hidden_states) > 0
        @test n_valid >= 0
        @test mean_r >= 0.0
        @test length(smiles_list_out) == n_valid
    end

    @testset "Q-Training End-to-End" begin
        # Create Q-function
        q_net, q_ps, q_st = create_q_function(hidden_dim, vocab.size; rng=rng)
        q_opt = Optimisers.Adam(1e-3)
        q_opt_state = Optimisers.setup(q_opt, q_ps)

        # Fill buffer with transitions
        buffer = QTrainingBuffer(5000)
        for _ in 1:10
            collect_and_fill_q_buffer!(
                buffer, model, params, states, vocab,
                length_reward;
                n_samples=10, max_length=50
            )
        end

        @test length(buffer.hidden_states) > 0

        # Train Q-function for a few steps
        losses = Float64[]
        for _ in 1:5
            loss, q_ps, q_opt_state = train_q_function!(
                q_net, q_ps, q_st, buffer, q_opt_state;
                batch_size=16, gamma=0.99
            )
            push!(losses, loss)
        end

        # Loss should be finite
        @test all(isfinite, losses)
    end

    @testset "Q-Masking During Sampling" begin
        # Create and minimally train a Q-function
        q_net, q_ps, q_st = create_q_function(hidden_dim, vocab.size; rng=rng)

        # Sample WITHOUT Q-masking
        smi1, _, _ = sample_smiles_autoregressive(
            model, params, states, vocab;
            max_length=50, temperature=1.0, constrained=false
        )

        # Sample WITH Q-masking (p=0.5)
        smi2, _, _ = sample_smiles_autoregressive(
            model, params, states, vocab;
            max_length=50, temperature=1.0, constrained=false,
            q_net=q_net, q_params=q_ps, q_states=q_st, p_quantile=0.5
        )

        # Both should produce something (may be empty for untrained model)
        @test smi1 isa String
        @test smi2 isa String

        # p=0 should be equivalent to no Q-masking
        smi3, _, _ = sample_smiles_autoregressive(
            model, params, states, vocab;
            max_length=50, temperature=1.0, constrained=false,
            q_net=q_net, q_params=q_ps, q_states=q_st, p_quantile=0.0
        )
        @test smi3 isa String
    end

    @testset "Boosting Ensemble" begin
        ensemble = BoostedGFlowNet()

        @test n_rounds(ensemble) == 0
        @test should_continue_boosting(ensemble; max_rounds=3)

        # Oracle caching (use SMILES with length ≥ 5 so length_reward > 0)
        r, cached = cached_oracle_call(ensemble, "CCCCCCO", length_reward)
        @test !cached
        @test r > 0.0

        r2, cached2 = cached_oracle_call(ensemble, "CCCCCCO", length_reward)
        @test cached2
        @test r2 == r

        # Residual reward — first round gives full reward
        res = compute_residual_reward(0.5, ensemble, "newmol")
        @test res == 0.5  # No checkpoints → full reward

        # Add a round
        add_boosting_round!(ensemble, params, states, 1.0, 100)
        @test n_rounds(ensemble) == 1

        # After first round, cached molecules get residual
        res2 = compute_residual_reward(0.5, ensemble, "CCCCCCO")
        @test res2 < 0.5  # Reduced by coverage

        # Novel molecule still gets full reward
        res3 = compute_residual_reward(0.5, ensemble, "novel_mol")
        @test res3 == 0.5

        # Ensemble sampling
        smi, _, _, round_idx = sample_from_ensemble(
            ensemble, model, vocab;
            max_length=50, temperature=1.0
        )
        @test round_idx == 1  # Only one round
        @test smi isa String

        # Stats
        stats = get_ensemble_stats(ensemble)
        @test stats["n_rounds"] == 1
        @test stats["total_molecules"] >= 1
    end

    @testset "Boosting Stop Criteria" begin
        ensemble = BoostedGFlowNet()

        # Not started → continue
        @test should_continue_boosting(ensemble; max_rounds=3)

        # Add rounds with diminishing discoveries
        add_boosting_round!(ensemble, params, states, 5.0, 100)
        @test should_continue_boosting(ensemble; max_rounds=3, min_new_molecules=10)

        add_boosting_round!(ensemble, params, states, 2.0, 50)
        @test should_continue_boosting(ensemble; max_rounds=3, min_new_molecules=10)

        # Max rounds reached
        add_boosting_round!(ensemble, params, states, 1.0, 30)
        @test !should_continue_boosting(ensemble; max_rounds=3, min_new_molecules=10)

        # Few new molecules → stop
        ensemble2 = BoostedGFlowNet()
        add_boosting_round!(ensemble2, params, states, 5.0, 5)
        @test !should_continue_boosting(ensemble2; max_rounds=5, min_new_molecules=10)

        # Very small Z → stop
        ensemble3 = BoostedGFlowNet()
        add_boosting_round!(ensemble3, params, states, -15.0, 100)
        @test !should_continue_boosting(ensemble3; max_rounds=5, min_new_molecules=10)
    end

    @testset "Ensemble Weighted Sampling" begin
        ensemble = BoostedGFlowNet()

        # Add two rounds with different log_Z
        add_boosting_round!(ensemble, params, states, 10.0, 100)  # High Z
        ps2 = deepcopy(params)
        add_boosting_round!(ensemble, ps2, states, 1.0, 50)  # Low Z

        # Sample many times and check distribution
        round_counts = zeros(Int, 2)
        for _ in 1:100
            _, _, _, idx = sample_from_ensemble(ensemble, model, vocab;
                max_length=30, temperature=1.0)
            round_counts[idx] += 1
        end

        # Round 1 (log_Z=10) should be sampled much more than round 2 (log_Z=1)
        @test round_counts[1] > round_counts[2]
    end
end
