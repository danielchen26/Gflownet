using Test
using GFlowNet

@testset "SMILES State and Actions" begin

    @testset "Initial state" begin
        state = create_initial_smiles_state(max_length=100, vocab_size=80)
        @test state.tokens == [START_TOKEN]
        @test state.max_length == 100
        @test state.vocab_size == 80
        @test !GFlowNet.is_terminal_state(state)
    end

    @testset "Action creation" begin
        actions = create_smiles_actions(80)
        @test length(actions) == 78  # vocab_size - 2 (skip PAD and START)
        @test actions[1].token_idx == END_TOKEN  # END is first valid action
    end

    @testset "Action applicability" begin
        state = create_initial_smiles_state(max_length=100, vocab_size=80)

        # END action should be applicable
        end_action = SMILESTokenAction(END_TOKEN)
        @test GFlowNet.is_applicable(end_action, state)

        # PAD and START should not be applicable
        pad_action = SMILESTokenAction(PAD_TOKEN)
        start_action = SMILESTokenAction(START_TOKEN)
        @test !GFlowNet.is_applicable(pad_action, state)
        @test !GFlowNet.is_applicable(start_action, state)

        # Regular token should be applicable
        regular_action = SMILESTokenAction(5)
        @test GFlowNet.is_applicable(regular_action, state)
    end

    @testset "Apply action" begin
        state = create_initial_smiles_state(max_length=100, vocab_size=80)
        action = SMILESTokenAction(5)

        new_state = GFlowNet.apply_action(action, state)

        # Original state should be unchanged (functional)
        @test length(state.tokens) == 1
        # New state should have the appended token
        @test length(new_state.tokens) == 2
        @test new_state.tokens == [START_TOKEN, 5]
    end

    @testset "Terminal state — END token" begin
        state = create_initial_smiles_state(max_length=100, vocab_size=80)
        end_action = SMILESTokenAction(END_TOKEN)
        terminal = GFlowNet.apply_action(end_action, state)
        @test GFlowNet.is_terminal_state(terminal)
    end

    @testset "Terminal state — max length" begin
        state = create_initial_smiles_state(max_length=3, vocab_size=80)
        action = SMILESTokenAction(5)

        s1 = GFlowNet.apply_action(action, state)    # length 2
        @test !GFlowNet.is_terminal_state(s1)

        s2 = GFlowNet.apply_action(action, s1)        # length 3 = max_length
        @test GFlowNet.is_terminal_state(s2)
    end

    @testset "No actions on terminal state" begin
        state = create_initial_smiles_state(max_length=100, vocab_size=80)
        end_action = SMILESTokenAction(END_TOKEN)
        terminal = GFlowNet.apply_action(end_action, state)

        # No action should be applicable on a terminal state
        for token_idx in 0:79
            action = SMILESTokenAction(token_idx)
            @test !GFlowNet.is_applicable(action, terminal)
        end
    end

    @testset "State equality and hashing" begin
        s1 = SMILESState([1, 5, 10], 100, 80)
        s2 = SMILESState([1, 5, 10], 100, 80)
        s3 = SMILESState([1, 5, 11], 100, 80)

        @test s1 == s2
        @test s1 != s3
        @test hash(s1) == hash(s2)
        @test hash(s1) != hash(s3)  # Very unlikely to collide
    end

    @testset "State to features" begin
        state = create_initial_smiles_state(max_length=100, vocab_size=80)
        features = GFlowNet.state_to_features(state)
        @test length(features) == 80
        @test features[START_TOKEN + 1] == 1.0f0  # One-hot at START
        @test sum(features) ≈ 1.0f0
    end

    @testset "Reward function" begin
        state = create_initial_smiles_state(max_length=100, vocab_size=80)
        @test GFlowNet.reward(state) == 0.0  # Non-terminal

        terminal = GFlowNet.apply_action(SMILESTokenAction(END_TOKEN), state)
        @test GFlowNet.reward(terminal) == 1.0  # Terminal placeholder
    end

    @testset "SMILES conversion" begin
        vocab = SMILESVocabulary()
        smiles = "CCO"
        state = smiles_to_state(vocab, smiles; max_length=100)
        recovered = state_to_smiles(state, vocab)
        @test recovered == smiles
    end
end
