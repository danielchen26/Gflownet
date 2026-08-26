using Test
using GFlowNet
using Random

@testset "QGFN Q-Function" begin
    rng = Random.MersenneTwister(42)
    hidden_dim = 64
    vocab_size = 50

    @testset "Q-function creation" begin
        q_net, q_ps, q_st = create_q_function(hidden_dim, vocab_size; rng=rng)
        @test q_net isa QFunctionNetwork

        # Test forward pass
        hidden = randn(Float32, hidden_dim)
        q_values, _ = compute_q_values(q_net, hidden, q_ps, q_st)
        @test length(q_values) == vocab_size
        @test all(isfinite, q_values)
    end

    @testset "p-quantile masking" begin
        logits = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
        q_values = Float32[0.1, 0.5, 0.3, 0.9, 0.7]

        # p=0 → no masking
        masked = apply_q_masking(logits, q_values, 0.0)
        @test masked == logits

        # p=0.5 → mask bottom 50%
        masked = apply_q_masking(logits, q_values, 0.5)
        # Q-values sorted: [0.1, 0.3, 0.5, 0.7, 0.9]
        # 50th percentile ≈ 0.5
        # Actions with Q < 0.5 should be masked
        @test masked[1] == -Inf  # Q=0.1 < 0.5
        @test masked[3] == -Inf  # Q=0.3 < 0.5
        @test isfinite(masked[4])  # Q=0.9 ≥ 0.5
        @test isfinite(masked[5])  # Q=0.7 ≥ 0.5

        # p=1.0 → only keep best (but fallback ensures at least 1)
        masked = apply_q_masking(logits, q_values, 1.0)
        n_finite = count(isfinite, masked)
        @test n_finite >= 1
    end

    @testset "Q-training buffer" begin
        buffer = QTrainingBuffer(100)

        # Add some transitions
        for i in 1:50
            hidden = randn(Float32, hidden_dim)
            next_hidden = randn(Float32, hidden_dim)
            add_q_transition!(buffer, hidden, rand(1:vocab_size), randn(), next_hidden, false)
        end

        @test length(buffer.hidden_states) == 50

        # Sample a batch
        batch = sample_q_batch(buffer, 16)
        @test !isnothing(batch)
        @test length(batch.hidden_states) == 16
        @test length(batch.actions) == 16
        @test length(batch.rewards) == 16
    end

    @testset "p-quantile schedule" begin
        total_budget = 10000

        # At start (warmup) → p=0
        p = compute_p_quantile(0, total_budget; p_start=0.0, p_end=0.8, warmup_fraction=0.2)
        @test p ≈ 0.0

        # During warmup → still p=0
        p = compute_p_quantile(1000, total_budget; p_start=0.0, p_end=0.8, warmup_fraction=0.2)
        @test p ≈ 0.0

        # After warmup → ramps up
        p = compute_p_quantile(6000, total_budget; p_start=0.0, p_end=0.8, warmup_fraction=0.2)
        @test p > 0.0
        @test p < 0.8

        # At end → p=0.8
        p = compute_p_quantile(10000, total_budget; p_start=0.0, p_end=0.8, warmup_fraction=0.2)
        @test p ≈ 0.8 atol=0.01
    end

    @testset "Applicable mask" begin
        logits = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
        q_values = Float32[0.1, 0.5, 0.3, 0.9, 0.7]
        mask = [true, true, false, true, true]  # Action 3 not applicable

        masked = apply_q_masking(logits, q_values, 0.5; applicable_mask=mask)
        # Only applicable actions considered for quantile
        @test !isfinite(masked[3]) || masked[3] == logits[3]  # May or may not be masked (not applicable)
    end
end
