# Test suite for Multi-Start GFlowNets
# Tests multiple initial states with per-state partition functions

using Test
using GFlowNet
using Random
using Statistics
using ComponentArrays

# Import specific types
using GFlowNet: GridState, MoveRight, MoveUp, MoveLeft, MoveDown, Terminate
using GFlowNet: MultiStartGFlowNetModel, sample_initial_state, get_initial_state_distribution

@testset "Multi-Start GFlowNet Tests" begin
    Random.seed!(42)
    
    @testset "Model Creation" begin
        # Create multiple initial states
        initial_states = [
            GridState(1, 1, false),
            GridState(3, 3, false),
            GridState(5, 5, false)
        ]
        
        all_actions = [MoveRight(), MoveUp(), MoveLeft(), MoveDown(), Terminate()]
        
        # Create multi-start model
        model = create_multi_start_gflownet(
            initial_states,
            all_actions,
            state_dim = 3,
            hidden_dim = 32
        )
        
        @test isa(model, MultiStartGFlowNetModel)
        @test length(model.initial_states) == 3
        @test length(model.log_partition_functions) == 3
        @test all(model.log_partition_functions .== 0.0)  # Default initialization
        
        # Test with custom initialization
        model2 = create_multi_start_gflownet(
            initial_states,
            all_actions,
            state_dim = 3,
            hidden_dim = 32,
            initialize_log_z = 1.0
        )
        
        @test all(model2.log_partition_functions .== 1.0)
    end
    
    @testset "Initial State Selection" begin
        initial_states = [
            GridState(1, 1, false),
            GridState(2, 2, false),
            GridState(3, 3, false)
        ]
        
        model = create_multi_start_gflownet(
            initial_states,
            [MoveRight(), MoveUp(), Terminate()],
            state_dim = 3,
            hidden_dim = 16
        )
        
        # Set different log Z values
        model.log_partition_functions = [0.0, 2.0, 1.0]  # Middle state most likely
        model.parameters = ComponentArray(
            forward = model.parameters.forward,
            flow = model.parameters.flow,
            log_Z = model.log_partition_functions
        )
        
        # Sample many times and check distribution
        counts = zeros(Int, 3)
        for _ in 1:1000
            state, idx = sample_initial_state(model)
            counts[idx] += 1
        end
        
        # Check that middle state (highest log Z) is sampled most
        @test argmax(counts) == 2
        
        # Check probability distribution
        probs = get_initial_state_distribution(model)
        @test length(probs) == 3
        @test sum(probs) ≈ 1.0
        @test argmax(probs) == 2
    end
    
    @testset "Trajectory Sampling" begin
        initial_states = [
            GridState(1, 1, false),
            GridState(2, 2, false)
        ]
        
        model = create_multi_start_gflownet(
            initial_states,
            [MoveRight(), MoveUp(), MoveLeft(), MoveDown(), Terminate()],
            state_dim = 3,
            hidden_dim = 32
        )
        
        # Sample trajectories
        trajectories_with_idx = [sample_trajectory(model) for _ in 1:10]
        
        # Check that we get valid trajectories
        for (traj, idx) in trajectories_with_idx
            @test isa(traj, Trajectory)
            @test 1 ≤ idx ≤ 2
            @test !isempty(traj.states)
            @test traj.states[1] == initial_states[idx]
            @test is_terminal_state(traj.states[end])
        end
        
        # Check that both initial states are used
        indices = [idx for (_, idx) in trajectories_with_idx]
        @test 1 in indices
        @test 2 in indices
    end
    
    @testset "Training Integration" begin
        # Create a simple multi-start grid world
        initial_states = [
            GridState(1, 1, false),
            GridState(3, 3, false)
        ]
        
        model = create_multi_start_gflownet(
            initial_states,
            [MoveRight(), MoveUp(), MoveLeft(), MoveDown(), Terminate()],
            state_dim = 3,
            hidden_dim = 32,
            learning_rate = 0.01
        )
        
        # Configure training
        config = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            n_iterations = 50,
            batch_size = 16,
            validation_frequency = 10
        )
        
        # Train model
        history = train_gflownet(model, config; verbose=false)
        
        @test length(history.losses) == 50
        @test all(isfinite.(filter(!isnan, history.losses)))
        
        # Check that log Z values have changed
        initial_log_z = zeros(2)
        final_log_z = model.log_partition_functions
        @test any(initial_log_z .!= final_log_z)
        
        # Check final distribution makes sense
        probs = get_initial_state_distribution(model)
        @test all(0 .≤ probs .≤ 1)
        @test sum(probs) ≈ 1.0
    end
    
    @testset "Loss Computation" begin
        initial_states = [
            GridState(1, 1, false),
            GridState(2, 2, false)
        ]
        
        model = create_multi_start_gflownet(
            initial_states,
            [MoveRight(), MoveUp(), Terminate()],
            state_dim = 3,
            hidden_dim = 16
        )
        
        # Create test trajectories
        traj1 = Trajectory(
            [GridState(1, 1, false), GridState(2, 1, false), GridState(2, 1, true)],
            [MoveRight(), Terminate()]
        )
        
        traj2 = Trajectory(
            [GridState(2, 2, false), GridState(3, 2, false), GridState(3, 2, true)],
            [MoveRight(), Terminate()]
        )
        
        trajectories_with_idx = [(traj1, 1), (traj2, 2)]
        
        # Test loss computation
        config = TrainingConfig(objective = TRAJECTORY_BALANCE)
        
        # This should work without errors
        loss = GFlowNet.compute_trajectory_loss_multi_start(
            model, trajectories_with_idx, model.parameters, config
        )
        
        @test isfinite(loss)
        @test loss ≥ 0
    end
    
    @testset "Backward Compatibility" begin
        # Single initial state should still work with regular API
        single_model = create_gflownet(
            GridState(1, 1, false),
            [MoveRight(), MoveUp(), Terminate()],
            state_dim = 3,
            hidden_dim = 32
        )
        
        @test isa(single_model, GFlowNetModel)
        @test !isa(single_model, MultiStartGFlowNetModel)
        
        # Can train normally
        config = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            n_iterations = 10,
            batch_size = 8
        )
        
        history = train_gflownet(single_model, config; verbose=false)
        @test length(history.losses) == 10
    end
    
    @testset "Different Training Objectives" begin
        # This testset used to train FLOW_MATCHING and DETAILED_BALANCE and assert only
        # `length(history.losses) == 20`. Twenty NaN satisfy that, and twenty NaN is exactly
        # what both produced: no `forward_transition_probability` or `flow` method existed
        # for MultiStartGFlowNetModel, so every iteration threw a MethodError,
        # train_gflownet caught it and pushed NaN. Measured 0 of 20 finite losses for both.
        # The suite reported two objectives as working when neither had run a single step.
        #
        # All three work now, and the assertions are the two properties the old length check
        # could not see: every iteration FINITE, and every gradient norm NON-ZERO. The
        # second one matters on its own -- DETAILED_BALANCE passed the finiteness check while
        # still computing P_F from `model.parameters` instead of the differentiated `params`,
        # which made the loss independent of what was being optimised and left the gradient
        # at exactly 0.0. A finite loss with a zero gradient is not training.
        initial_states = [
            GridState(1, 1, false),
            GridState(2, 2, false)
        ]

        # TWO action sets, because they are not interchangeable and the original single
        # cyclic set hid that.
        #
        # acyclic: MoveRight/MoveUp only, so the state graph is a monotone lattice.
        # cyclic:  all four moves, so the start state acquires parents and the backward
        #          chain is never absorbed at s_0.
        acyclic = [MoveRight(), MoveUp(), Terminate()]
        cyclic  = [MoveRight(), MoveUp(), MoveLeft(), MoveDown(), Terminate()]

        # On an ACYCLIC graph all three objectives train, and the assertions are the two
        # properties a length check cannot see: every iteration FINITE, and every gradient
        # norm NON-ZERO. The second matters on its own -- DETAILED_BALANCE passed finiteness
        # while still reading P_F from `model.parameters` instead of the differentiated
        # `params`, which left the gradient at exactly 0.0. A finite loss with a zero
        # gradient is not training.
        for obj in (TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING),
            include_backward in (false, true)

            model = create_multi_start_gflownet(initial_states, acyclic;
                                                state_dim = 3, hidden_dim = 32,
                                                include_backward = include_backward)
            history = train_gflownet(model,
                TrainingConfig(objective = obj, n_iterations = 20, batch_size = 8);
                verbose = false)

            @test length(history.losses) == 20
            @test count(isfinite, history.losses) == 20
            @test count(g -> isfinite(g) && g > 0, history.gradient_norms) == 20
        end

        # On a CYCLIC graph the three objectives SPLIT, and that split is the finding.
        #
        # TB carries its partition function as a learned parameter, so cycles are absorbed
        # into log_Z and it trains. DB and FM need a finite F(s), which they obtain from the
        # conservation recursion -- and on a cyclic graph that recursion has no solution:
        # the backward transfer matrix has spectral radius exactly 1, sigma_min(I - W') is
        # 1.5e-16, and the partial sums for sum_tau P_B(tau|x) diverge as
        # 2.0/12.0/49.5/249.5/999.5 at horizons 10/50/200/1000/4000 where the value must be
        # 1. So they must refuse, and before the cycle guard they crashed with
        # StackOverflowError which the training loop recorded as NaN.
        #
        # This testset previously used the cyclic set for ALL SIX combinations and asserted
        # they all train. Four of them cannot.
        model_tb_cyclic = create_multi_start_gflownet(initial_states, cyclic;
                                                     state_dim = 3, hidden_dim = 32)
        history_tb_cyclic = train_gflownet(model_tb_cyclic,
            TrainingConfig(objective = TRAJECTORY_BALANCE, n_iterations = 20, batch_size = 8);
            verbose = false)
        @test count(isfinite, history_tb_cyclic.losses) == 20

        for obj in (DETAILED_BALANCE, FLOW_MATCHING)
            @test_throws ArgumentError train_gflownet(
                create_multi_start_gflownet(initial_states, cyclic;
                                            state_dim = 3, hidden_dim = 32),
                TrainingConfig(objective = obj, n_iterations = 5, batch_size = 4);
                verbose = false)
        end

        # An objective with no multi-start branch must be refused up front, not discovered
        # as NaN inside the loop -- the loop converts any throw into a NaN entry and reports
        # the run as complete.
        @test_throws ArgumentError train_gflownet(
            create_multi_start_gflownet(initial_states, acyclic;
                                        state_dim = 3, hidden_dim = 32),
            TrainingConfig(objective = SUB_TRAJECTORY_BALANCE, n_iterations = 5,
                           batch_size = 4); verbose = false)
    end
end

println("\n✅ All multi-start GFlowNet tests passed!")