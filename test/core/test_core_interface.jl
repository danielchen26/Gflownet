# Test Core Interface Functions
# Tests for the high-level GFlowNet interface

using Test
using GFlowNet
using Random

@testset "Core Interface" begin
    @testset "State and Action Interface" begin
        # Use grid world as a test domain
        state = GridState(2, 3, false)
        action = MoveRight()
        
        # Test required interface methods exist
        @test hasmethod(GFlowNet.state_to_features, (typeof(state),))
        @test hasmethod(GFlowNet.is_terminal_state, (typeof(state),))
        @test hasmethod(GFlowNet.reward, (typeof(state),))
        @test hasmethod(GFlowNet.is_applicable, (typeof(action), typeof(state)))
        @test hasmethod(GFlowNet.apply_action, (typeof(action), typeof(state)))
    end
    
    @testset "Model Creation Interface" begin
        # Test create_gflownet function
        initial_state = GridState(1, 1, false)
        actions = [MoveRight(), MoveUp(), MoveLeft(), MoveDown(), Terminate()]
        
        model = GFlowNet.create_gflownet(
            initial_state,
            actions;
            state_dim=3,
            hidden_dim=16,
            learning_rate=0.01,
            rng=Random.default_rng()
        )
        
        @test model isa GFlowNet.GFlowNetModel
        @test model.initial_state == initial_state
        @test model.all_actions == actions
        # Note: The model doesn't store state_dim directly
    end
    
    # These two testsets used to be `@test_broken false` with a comment saying
    # "This test is removed because forward_action_probabilities,
    # sample_forward_action and forward_probability don't exist as model
    # methods". They do exist -- as POLICY methods, taking (policy, state,
    # actions, parameters, states); the note recorded a signature mismatch and
    # then deleted the coverage. `@test_broken false` can never fail, so nothing
    # would ever have reopened it.
    @testset "Policy Functions" begin
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=8
        )
        s = model.initial_state

        probs = GFlowNet.forward_action_probabilities(
            model.forward_policy, s, model.all_actions,
            model.parameters.forward, model.states.forward
        )
        @test length(probs) == length(model.all_actions)
        @test all(>=(0.0f0), probs)
        @test sum(probs) ≈ 1.0 atol=1e-5

        # P_F must put exactly zero mass on inapplicable actions -- at (1,1) of a
        # 3x3 grid that is Terminate(), and leaking mass there is how a policy
        # silently samples an illegal transition.
        for (i, a) in enumerate(model.all_actions)
            if !GFlowNet.is_applicable(a, s)
                @test probs[i] == 0.0f0
            else
                @test probs[i] > 0.0f0
            end
        end

        # sample_forward_action returns (action, probability) -- note the
        # docstring in src/core/policies.jl:135 still says (action, index).
        action, p = GFlowNet.sample_forward_action(
            model.forward_policy, s, model.all_actions,
            model.parameters.forward, model.states.forward;
            rng=Random.MersenneTwister(0)
        )
        @test GFlowNet.is_applicable(action, s)
        idx = findfirst(==(action), model.all_actions)
        @test p ≈ probs[idx] atol=1e-6

        # forward_probability must agree with the full distribution it is a
        # single entry of.
        @test GFlowNet.forward_probability(
            model.forward_policy, s, action,
            model.parameters.forward, model.states.forward, model.all_actions
        ) ≈ probs[idx] atol=1e-6
    end

    @testset "Flow Functions" begin
        # flow_estimate is a FlowEstimator method, so the model has to be built
        # with include_flow_estimator=true; the old comment claimed the function
        # did not exist.
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=8,
            include_backward=true,
            include_flow_estimator=true
        )
        s = model.initial_state

        f = GFlowNet.flow_estimate(
            model.flow_estimator, s, model.parameters.flow, model.states.flow
        )
        @test f isa Float64
        @test isfinite(f)
        @test f > 0.0  # the estimator exponentiates a log-flow

        # The exact flow functions enumerate the DAG, so they are independent of
        # the (untrained) weights: F(s_0) is the partition function, and the 3x3
        # grid with reward 10 at (3,3) has Z = 19 -- the same constant
        # test/theory/test_reward_proportionality.jl pins by enumeration.
        @test GFlowNet.flow(model, s) ≈ GFlowNet.partition_function(model)
        @test GFlowNet.flow(model, s) ≈ 19.0 atol=1e-6
        @test GFlowNet.compute_recursive_flow(model, s) ≈ GFlowNet.flow(model, s)
        @test GFlowNet.validate_flow_conservation(model, s)
    end
    
    @testset "Trajectory Functions" begin
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=8
        )
        
        # Sample a trajectory
        traj = GFlowNet.sample_trajectory(model)
        
        # Test trajectory structure
        @test traj isa GFlowNet.Trajectory
        @test !isempty(traj.states)
        @test traj.states[1] == model.initial_state
        @test GFlowNet.is_terminal_state(traj.states[end])
        @test length(traj.actions) == length(traj.states) - 1
        
        # Test trajectory validation using the actual function name
        is_valid = GFlowNet.is_valid_trajectory(traj)
        @test is_valid
    end
    
    @testset "State Space Analysis" begin
        # Small grid for complete analysis
        model = create_grid_world_gflownet(grid_size=2, hidden_dim=4)
        
        # Test state space exploration
        reachable_states = GFlowNet.explore_state_space(
            model.initial_state,
            model.all_actions;
            max_states=100
        )
        
        @test model.initial_state in reachable_states
        @test length(reachable_states) > 1
        
        # Test state space analysis
        analysis = GFlowNet.analyze_state_space(
            model.initial_state,
            model.all_actions;
            max_states=100
        )
        
        @test haskey(analysis, :total_states)
        @test haskey(analysis, :terminal_states)
        @test haskey(analysis, :actions_count)
        @test analysis[:total_states] > 0
        @test analysis[:terminal_states] > 0
    end
    
    @testset "Training Interface" begin
        # Test configuration helpers
        default_config = GFlowNet.create_default_config()
        @test default_config isa TrainingConfig
        @test default_config.n_iterations == 1000
        
        fast_config = GFlowNet.create_fast_config()
        @test fast_config.n_iterations < default_config.n_iterations
        
        robust_config = GFlowNet.create_robust_config()
        @test robust_config.n_iterations > default_config.n_iterations
    end
    
    @testset "Validation Functions" begin
        # Was `@test_broken false` with "These specific validation functions
        # don't exist in the current interface". They do:
        # validate_policy_consistency and is_valid_trajectory are both exported
        # from src/core/policies.jl and src/core/trajectories.jl.
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=8
        )

        # Returns nothing and throws ArgumentError on inconsistency.
        @test GFlowNet.validate_policy_consistency(model) === nothing

        equipped = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=8,
            include_backward=true,
            include_flow_estimator=true
        )
        @test GFlowNet.validate_policy_consistency(equipped) === nothing

        # A sampled trajectory must validate, and the validator must be capable
        # of saying no: the states/actions length invariant is enforced at
        # construction, so a malformed trajectory cannot even be built.
        traj = GFlowNet.sample_trajectory(model)
        @test GFlowNet.is_valid_trajectory(traj)
        @test_throws ArgumentError GFlowNet.Trajectory(
            traj.states, traj.actions[1:end-1]
        )
    end
end