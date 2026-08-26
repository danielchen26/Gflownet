using Test
using GFlowNet
using GFlowNet: compute_trajectory_loss

# Create model
model = create_grid_world_gflownet(
    grid_size=3,
    hidden_dim=32,
    learning_rate=0.01,
    include_backward=true
)

# Configure training
config = TrainingConfig(
    objective=DETAILED_BALANCE,
    n_iterations=3,
    batch_size=4,
    learning_rate=0.01
)

@testset "DETAILED_BALANCE training" begin
    history = train_gflownet(model, config; verbose=false)

    # Training must produce one loss per iteration, and at least one of them
    # must be a real number. Previously this file only printed history.losses.
    @test length(history.losses) == config.n_iterations
    @test any(!isnan, history.losses)

    # A single manual step must compute a finite loss. This was wrapped in
    # try/catch that printed "✗ Error in single step" and let the file pass,
    # so a broken DETAILED_BALANCE objective reported green.
    trajectories = [sample_trajectory(model) for _ in 1:4]
    loss = compute_trajectory_loss(model, trajectories, model.parameters, config)
    @test isfinite(loss)
end