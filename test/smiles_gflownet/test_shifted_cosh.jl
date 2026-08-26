using Test
using GFlowNet

@testset "Shifted-Cosh TB Loss" begin

    @testset "MSE baseline" begin
        @test apply_tb_loss(0.0, :mse) == 0.0
        @test apply_tb_loss(1.0, :mse) == 1.0
        @test apply_tb_loss(-2.0, :mse) == 4.0
        @test apply_tb_loss(3.0, :mse) == 9.0
    end

    @testset "Shifted-Cosh at zero" begin
        # g(0) = 2(cosh(0) - 1) = 2(1 - 1) = 0
        @test apply_tb_loss(0.0, :shifted_cosh) ≈ 0.0 atol=1e-10
    end

    @testset "Shifted-Cosh symmetry" begin
        # g(δ) = g(-δ) since cosh is symmetric
        for delta in [0.5, 1.0, 2.0, 5.0, 10.0]
            @test apply_tb_loss(delta, :shifted_cosh) ≈ apply_tb_loss(-delta, :shifted_cosh) atol=1e-8
        end
    end

    @testset "Shifted-Cosh vs MSE" begin
        # For small δ, shifted-cosh ≈ δ² (Taylor expansion: cosh(δ) ≈ 1 + δ²/2)
        # So g(δ) = 2(cosh(δ) - 1) ≈ δ²
        @test apply_tb_loss(0.1, :shifted_cosh) ≈ apply_tb_loss(0.1, :mse) atol=0.01
        @test apply_tb_loss(0.01, :shifted_cosh) ≈ apply_tb_loss(0.01, :mse) atol=0.0001

        # For large δ, shifted-cosh >> MSE (exponential growth)
        @test apply_tb_loss(5.0, :shifted_cosh) > apply_tb_loss(5.0, :mse)
        @test apply_tb_loss(10.0, :shifted_cosh) > apply_tb_loss(10.0, :mse)
    end

    @testset "Hybrid linear extension" begin
        threshold = 15.0

        # At threshold, should match pure shifted-cosh
        pure_at_threshold = 2.0 * (cosh(threshold) - 1.0)
        @test apply_tb_loss(threshold, :shifted_cosh; threshold=threshold) ≈ pure_at_threshold atol=1e-6

        # Beyond threshold, growth should be linear (not exponential)
        val_at_16 = apply_tb_loss(16.0, :shifted_cosh; threshold=threshold)
        val_at_17 = apply_tb_loss(17.0, :shifted_cosh; threshold=threshold)
        val_at_18 = apply_tb_loss(18.0, :shifted_cosh; threshold=threshold)

        # Linear growth means equal increments
        diff1 = val_at_17 - val_at_16
        diff2 = val_at_18 - val_at_17
        @test diff1 ≈ diff2 atol=1e-6

        # And the gradient should be 2*sinh(threshold), which is finite
        expected_gradient = 2.0 * sinh(threshold)
        @test isfinite(expected_gradient)
        @test diff1 ≈ expected_gradient atol=1e-6
    end

    @testset "Numerical stability for large δ" begin
        # Should NOT overflow or return NaN/Inf for large values
        for delta in [20.0, 50.0, 100.0, 500.0]
            val = apply_tb_loss(delta, :shifted_cosh; threshold=15.0)
            @test isfinite(val)
            @test val > 0
        end
    end

    @testset "Custom threshold" begin
        # Small threshold
        val_small = apply_tb_loss(5.0, :shifted_cosh; threshold=3.0)
        # Beyond threshold=3, should be in linear regime
        val_at_3 = 2.0 * (cosh(3.0) - 1.0)
        gradient_at_3 = 2.0 * sinh(3.0)
        expected = val_at_3 + gradient_at_3 * (5.0 - 3.0)
        @test val_small ≈ expected atol=1e-6
    end

    @testset "Mode-covering property" begin
        # For positive δ (model underestimates reward):
        # shifted-cosh gradient = sinh(δ) grows with δ
        # This means it penalizes missed modes more than MSE
        for delta in [1.0, 2.0, 5.0]
            cosh_loss = apply_tb_loss(delta, :shifted_cosh)
            mse_loss = apply_tb_loss(delta, :mse)
            if delta > 2.0
                @test cosh_loss > mse_loss  # Stronger penalty for large errors
            end
        end
    end

    @testset "Invalid loss type" begin
        @test_throws ErrorException apply_tb_loss(1.0, :invalid_type)
    end

    @testset "TrainingConfig with SHIFTED_COSH_TB" begin
        config = TrainingConfig(
            objective=SHIFTED_COSH_TB,
            n_iterations=10,
            batch_size=4,
        )
        @test config.loss_type == :shifted_cosh
        @test config.cosh_delta_threshold == 15.0
    end

    @testset "Gradient clipping config" begin
        config = TrainingConfig(
            gradient_clip_norm=0.5,
            n_iterations=10,
            batch_size=4,
        )
        @test config.gradient_clip_norm == 0.5
    end
end
