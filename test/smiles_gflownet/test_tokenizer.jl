using Test
using GFlowNet

@testset "SMILES Tokenizer" begin
    vocab = SMILESVocabulary()

    @testset "Vocabulary basics" begin
        @test length(vocab) > 50  # Should have at least 50 tokens
        @test has_token(vocab, "[PAD]")
        @test has_token(vocab, "[START]")
        @test has_token(vocab, "[END]")
        @test has_token(vocab, "C")
        @test has_token(vocab, "N")
        @test has_token(vocab, "O")
        @test has_token(vocab, "Cl")
        @test has_token(vocab, "Br")
    end

    @testset "Special token indices" begin
        @test PAD_TOKEN == 0
        @test START_TOKEN == 1
        @test END_TOKEN == 2
    end

    @testset "Tokenize simple SMILES" begin
        tokens = tokenize_smiles("CCO")
        @test tokens == ["C", "C", "O"]

        tokens = tokenize_smiles("C=O")
        @test tokens == ["C", "=", "O"]

        tokens = tokenize_smiles("ClC")
        @test tokens == ["Cl", "C"]

        tokens = tokenize_smiles("BrC")
        @test tokens == ["Br", "C"]
    end

    @testset "Tokenize aromatic SMILES" begin
        tokens = tokenize_smiles("c1ccccc1")
        @test tokens == ["c", "1", "c", "c", "c", "c", "c", "1"]
    end

    @testset "Tokenize bracket atoms" begin
        tokens = tokenize_smiles("[C@@H](O)F")
        @test length(tokens) >= 4
        @test tokens[1] == "[C@@H]"
    end

    @testset "Encode/decode roundtrip" begin
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "C=CC=C"]

        for smiles in smiles_list
            indices = encode(vocab, smiles; add_special_tokens=true)
            decoded = decode(vocab, indices; strip_special=true)
            @test decoded == smiles
        end
    end

    @testset "Encode with special tokens" begin
        indices = encode(vocab, "CCO"; add_special_tokens=true)
        @test indices[1] == START_TOKEN
        @test indices[end] == END_TOKEN
        @test length(indices) == 5  # START + C + C + O + END
    end

    @testset "Encode without special tokens" begin
        indices = encode(vocab, "CCO"; add_special_tokens=false)
        @test indices[1] != START_TOKEN
        @test indices[end] != END_TOKEN
        @test length(indices) == 3  # C + C + O
    end

    @testset "Pad sequence" begin
        indices = [1, 2, 3]
        padded = pad_sequence(indices, 6)
        @test length(padded) == 6
        @test padded[1:3] == indices
        @test padded[4:6] == [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]

        # Truncation
        truncated = pad_sequence(indices, 2)
        @test length(truncated) == 2
        @test truncated == [1, 2]
    end

    @testset "Batch encode" begin
        smiles_list = ["CCO", "CCCC", "c1ccccc1"]
        batch = batch_encode(vocab, smiles_list; max_length=20)
        @test size(batch, 2) == 3
        @test size(batch, 1) == 20
        @test batch[1, 1] == START_TOKEN  # First token of first SMILES
    end

    @testset "Dynamic vocabulary expansion" begin
        initial_size = length(vocab)
        # Encoding a novel token should expand vocab
        idx = get_or_add_token!(vocab, "[Zn]")
        @test length(vocab) == initial_size + 1
        @test idx == initial_size
        @test has_token(vocab, "[Zn]")
    end
end
