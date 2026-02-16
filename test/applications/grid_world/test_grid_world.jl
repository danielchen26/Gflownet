# Test Grid World Application
# Tests for the grid world domain implementation

using Test
using GFlowNet
using Random

@testset "Grid World Tests" begin
    @testset "Grid State and Actions" begin
        # Create a simple grid state
        state = GridState(2, 3, false)
        
        @test state.x == 2
        @test state.y == 3
        @test !state.is_terminal
        @test !GFlowNet.is_terminal_state(state)
        
        # Test state features
        # First set grid config so normalization works correctly
        GFlowNet.GRID_CONFIG[] = (grid_size=5, reward_positions=Dict{Tuple{Int,Int},Float64}())
        
        features = GFlowNet.state_to_features(state)
        @test features isa Vector{Float32}
        @test length(features) == 3
        # Normalized coordinates: (x-1)/(grid_size-1), (y-1)/(grid_size-1)
        @test features[1] ≈ Float32((2-1)/(5-1))  # x normalized
        @test features[2] ≈ Float32((3-1)/(5-1))  # y normalized
        @test features[3] == Float32(0)  # not terminal
        
        # Test actions
        @test MoveRight() isa GridAction
        @test MoveUp() isa GridAction
        @test MoveLeft() isa GridAction
        @test MoveDown() isa GridAction
        @test Terminate() isa GridAction
    end
    
    @testset "Action Applicability" begin
        # Set grid size for testing
        GFlowNet.GRID_CONFIG[] = (grid_size=5, reward_positions=Dict{Tuple{Int,Int},Float64}())
        grid_size = 5
        
        # Corner states
        top_left = GridState(1, 1, false)
        top_right = GridState(grid_size, 1, false)
        bottom_left = GridState(1, grid_size, false)
        bottom_right = GridState(grid_size, grid_size, false)
        center = GridState(3, 3, false)
        
        # Test MoveLeft boundaries (x > 1)
        @test !GFlowNet.is_applicable(MoveLeft(), top_left)      # x=1, can't go left
        @test GFlowNet.is_applicable(MoveLeft(), top_right)      # x=5, can go left
        @test GFlowNet.is_applicable(MoveLeft(), center)         # x=3, can go left
        
        # Test MoveRight boundaries (x < grid_size)
        @test GFlowNet.is_applicable(MoveRight(), top_left)      # x=1, can go right
        @test !GFlowNet.is_applicable(MoveRight(), top_right)    # x=5, can't go right
        @test GFlowNet.is_applicable(MoveRight(), center)        # x=3, can go right
        
        # Test MoveUp boundaries (y < grid_size)
        @test GFlowNet.is_applicable(MoveUp(), top_left)         # y=1, can go up
        @test GFlowNet.is_applicable(MoveUp(), top_right)        # y=1, can go up
        @test !GFlowNet.is_applicable(MoveUp(), bottom_left)     # y=5, can't go up
        @test GFlowNet.is_applicable(MoveUp(), center)           # y=3, can go up
        
        # Test MoveDown boundaries (y > 1)
        @test !GFlowNet.is_applicable(MoveDown(), top_left)      # y=1, can't go down
        @test !GFlowNet.is_applicable(MoveDown(), top_right)     # y=1, can't go down
        @test GFlowNet.is_applicable(MoveDown(), bottom_left)    # y=5, can go down
        @test GFlowNet.is_applicable(MoveDown(), center)         # y=3, can go down
        
        # Terminate is applicable except at (1,1)
        @test !GFlowNet.is_applicable(Terminate(), GridState(1, 1, false))  # Starting position
        @test GFlowNet.is_applicable(Terminate(), center)
        @test GFlowNet.is_applicable(Terminate(), top_right)
        
        # Terminal state - no actions applicable
        terminal = GridState(3, 3, true)
        @test !GFlowNet.is_applicable(MoveRight(), terminal)
        @test !GFlowNet.is_applicable(Terminate(), terminal)
    end
    
    @testset "Action Application" begin
        # Set grid size
        GFlowNet.GRID_CONFIG[] = (grid_size=5, reward_positions=Dict{Tuple{Int,Int},Float64}())
        
        start = GridState(2, 2, false)
        
        # Test movement
        right_state = GFlowNet.apply_action(MoveRight(), start)
        @test right_state.x == 3
        @test right_state.y == 2
        @test !right_state.is_terminal
        
        up_state = GFlowNet.apply_action(MoveUp(), start)
        @test up_state.x == 2
        @test up_state.y == 3
        @test !up_state.is_terminal
        
        left_state = GFlowNet.apply_action(MoveLeft(), start)
        @test left_state.x == 1
        @test left_state.y == 2
        @test !left_state.is_terminal
        
        down_state = GFlowNet.apply_action(MoveDown(), start)
        @test down_state.x == 2
        @test down_state.y == 1
        @test !down_state.is_terminal
        
        # Test termination
        terminal_state = GFlowNet.apply_action(Terminate(), start)
        @test terminal_state.x == 2
        @test terminal_state.y == 2
        @test terminal_state.is_terminal
    end
    
    @testset "Rewards" begin
        # Create reward structure
        reward_positions = Dict((5, 5) => 100.0, (1, 5) => 50.0)
        GFlowNet.GRID_CONFIG[] = (grid_size=5, reward_positions=reward_positions)
        
        # Non-terminal states have no reward
        state = GridState(3, 3, false)
        @test GFlowNet.reward(state) == 0.0
        
        # Terminal states get rewards based on position
        terminal_reward = GridState(5, 5, true)
        @test GFlowNet.reward(terminal_reward) == 100.0
        
        terminal_reward2 = GridState(1, 5, true)
        @test GFlowNet.reward(terminal_reward2) == 50.0
        
        # Terminal state without specific reward gets base reward
        terminal_no_reward = GridState(2, 2, true)
        @test GFlowNet.reward(terminal_no_reward) == 1.0  # base reward
    end
    
    @testset "Grid World Model Creation" begin
        # Test model creation with default parameters
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            learning_rate=0.01,
            hidden_dim=16
        )
        
        @test model isa GFlowNet.GFlowNetModel
        @test model.initial_state == GridState(1, 1, false)
        @test length(model.all_actions) >= 3  # At least move right, move up, and terminate
        
        # Test that we can sample from the model
        trajectory = GFlowNet.sample_trajectory(model)
        @test trajectory isa GFlowNet.Trajectory
        @test !isempty(trajectory.states)
        @test trajectory.states[1] == GridState(1, 1, false)
        @test GFlowNet.is_terminal_state(trajectory.states[end])
    end
    
    @testset "State Equality and Hashing" begin
        state1 = GridState(2, 3, false)
        state2 = GridState(2, 3, false)
        state3 = GridState(2, 3, true)
        state4 = GridState(3, 2, false)
        
        # Test equality
        @test state1 == state2
        @test state1 != state3
        @test state1 != state4
        
        # Test hashing
        @test hash(state1) == hash(state2)
        @test hash(state1) != hash(state3)
        @test hash(state1) != hash(state4)
        
        # Test that states can be used in sets
        state_set = Set([state1, state2, state3, state4])
        @test length(state_set) == 3  # state1 and state2 are the same
    end
    
    @testset "Action Types" begin
        # Test all action types
        actions = [MoveRight(), MoveUp(), MoveLeft(), MoveDown(), Terminate()]
        
        for action in actions
            @test action isa GridAction
        end
        
        # Test action equality
        @test MoveRight() == MoveRight()
        @test MoveUp() != MoveDown()
        @test Terminate() == Terminate()
    end
end