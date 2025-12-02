# Test Training Infrastructure
# Tests for training configuration and execution

using Test
using GFlowNet
using Random
using Optimisers

@testset "Training Infrastructure" begin
    @testset "Training Configuration" begin
        # Test default configuration creation
        config = GFlowNet.create_default_config()
        @test config isa TrainingConfig
        @test config.objective isa TrainingObjective
        @test config.n_iterations == 1000
        @test config.batch_size == 32
        @test config.learning_rate > 0
        
        # Test custom configuration using keyword arguments
        custom_config = TrainingConfig(
            objective=DETAILED_BALANCE,
            partition_function_method=SIMPLE_ESTIMATION,
            optimization_method=ADAM,
            n_iterations=100,
            batch_size=16,
            learning_rate=0.01,
            entropy_weight=0.0,
            parameter_regularization=0.0,
            gradient_clip_norm=1.0,
            temperature=1.0,
            exploration_noise=0.0,
            validation_frequency=10,
            checkpoint_frequency=50,
            early_stopping_patience=10,
            verbose=false
        )
        
        @test custom_config.objective == DETAILED_BALANCE
        @test custom_config.n_iterations == 100
        @test custom_config.batch_size == 16
        @test custom_config.learning_rate == 0.01
        @test custom_config.validation_frequency == 10
    end
    
    @testset "Training Objectives" begin
        # Test that all objectives are defined
        @test TRAJECTORY_BALANCE isa TrainingObjective
        @test DETAILED_BALANCE isa TrainingObjective
        @test FLOW_MATCHING isa TrainingObjective
    end
    
    @testset "Optimizer Options" begin
        # Test optimizer types
        @test ADAM isa OptimizationMethod
        @test RMSPROP isa OptimizationMethod
        @test SGD isa OptimizationMethod
        @test ADAMW isa OptimizationMethod
        
        # Test optimizer creation
        optimizer = GFlowNet.create_optimizer(ADAM, 0.01)
        @test optimizer isa Optimisers.Adam
        
        optimizer = GFlowNet.create_optimizer(SGD, 0.1)
        @test optimizer isa Optimisers.Descent
    end
    
    @testset "Sampling Configuration" begin
        # Test default sampling config
        config = GFlowNet.SamplingConfig()
        @test config.strategy isa GFlowNet.SamplingStrategy
        @test config.temperature == 1.0
        @test config.acyclic_rate == 0.0
        
        # Test custom sampling config (epsilon field doesn't exist)
        custom = GFlowNet.SamplingConfig(
            strategy=GREEDY_SAMPLING,
            temperature=0.5,
            max_trajectory_length=50,
            acyclic_rate=0.0
        )
        @test custom.strategy == GREEDY_SAMPLING
        @test custom.temperature == 0.5
        @test custom.max_trajectory_length == 50
    end
    
    @testset "Training State Management" begin
        # Test training state initialization
        state = GFlowNet.TrainingState()
        @test state.iteration == 0
        @test state.best_loss == Inf
        
        # Test metrics tracking
        metrics = GFlowNet.TrainingMetrics()
        @test isempty(metrics.loss_components)
        @test isempty(metrics.gradient_norms)
        @test metrics.flow_conservation_score == 0.0
    end
    
    @testset "Simple Training Run" begin
        # Create a minimal model
        model = create_grid_world_gflownet(
            grid_size=2,  # Very small grid
            reward_positions=Dict((2, 2) => 1.0),
            hidden_dim=4,  # Small network
            learning_rate=0.1
        )
        
        # Create minimal training config using keyword arguments
        config = TrainingConfig(
            objective=TRAJECTORY_BALANCE,
            partition_function_method=SIMPLE_ESTIMATION,
            optimization_method=ADAM,
            n_iterations=5,  # Just a few iterations
            batch_size=2,    # Small batch
            learning_rate=0.1,
            entropy_weight=0.0,
            parameter_regularization=0.0,
            gradient_clip_norm=1.0,
            temperature=1.0,
            exploration_noise=0.0,
            validation_frequency=10,
            checkpoint_frequency=50,
            early_stopping_patience=10,
            verbose=false
        )
        
        # Run training
        history = train_gflownet(model, config; verbose=false)
        
        @test history isa GFlowNet.TrainingHistory
        @test length(history.losses) == 5
        @test all(isfinite, history.losses)
        
        # Check that we can sample after training
        trajectory = GFlowNet.sample_trajectory(model)
        @test trajectory isa GFlowNet.Trajectory
        @test !isempty(trajectory.states)
    end
    
    @testset "Trajectory Sampling" begin
        # Create model
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=8
        )
        
        # Test single trajectory sampling
        traj = GFlowNet.sample_trajectory(model)
        @test traj isa GFlowNet.Trajectory
        @test traj.states[1] == model.initial_state
        @test GFlowNet.is_terminal_state(traj.states[end])
        @test length(traj.actions) == length(traj.states) - 1
        
        # Test batch sampling
        trajectories = GFlowNet.sample_trajectory_batch(model, 10)
        @test length(trajectories) == 10
        @test all(t -> t isa GFlowNet.Trajectory, trajectories)
        @test all(t -> !isempty(t.states), trajectories)
        
        # Test different sampling strategies
        greedy_config = GFlowNet.SamplingConfig(strategy=GREEDY_SAMPLING)
        greedy_traj = GFlowNet.sample_trajectory(model; config=greedy_config)
        @test greedy_traj isa GFlowNet.Trajectory
        
        temp_config = GFlowNet.SamplingConfig(
            strategy=TEMPERATURE_SAMPLING,
            temperature=0.5
        )
        temp_traj = GFlowNet.sample_trajectory(model; config=temp_config)
        @test temp_traj isa GFlowNet.Trajectory
    end
    
    @testset "Configuration Helpers" begin
        # Test fast config
        fast_config = GFlowNet.create_fast_config()
        @test fast_config isa TrainingConfig
        @test fast_config.n_iterations <= 100
        
        # Test robust config
        robust_config = GFlowNet.create_robust_config()
        @test robust_config isa TrainingConfig
        @test robust_config.n_iterations >= 1000
        
        # Test that configs have valid parameters
        for config in [fast_config, robust_config]
            @test config.learning_rate > 0
            @test config.batch_size > 0
            @test config.gradient_clip_norm > 0
        end
    end
end