# test/runtests.jl

using Test
using GFlowNet
using Lux
using Random
using Statistics
using Optimisers
using Graphs

@testset "GFlowNet.jl" begin
    @testset "Types" begin
        # Test the abstract types
        @test GFlowNet.AbstractState isa Type
        @test GFlowNet.AbstractAction isa Type
        
        # Test DirectedAcyclicGraph
        @test GFlowNet.DirectedAcyclicGraph isa Type
    end
    
    @testset "Grid World Example" begin
        # Define a simple grid world
        struct GridState <: GFlowNet.AbstractState
            x::Int
            y::Int
            is_terminal::Bool
        end
        
        struct MoveAction <: GFlowNet.AbstractAction
            direction::Symbol # :up, :down, :left, :right
        end
        
        struct TerminateAction <: GFlowNet.AbstractAction end
        
        # Define is_applicable and apply_action methods
        function GFlowNet.is_applicable(action::MoveAction, state::GridState)
            !state.is_terminal
        end
        
        function GFlowNet.is_applicable(action::TerminateAction, state::GridState)
            !state.is_terminal
        end
        
        function GFlowNet.apply_action(action::MoveAction, state::GridState)
            if action.direction == :up
                return GridState(state.x, state.y + 1, false)
            elseif action.direction == :down
                return GridState(state.x, state.y - 1, false)
            elseif action.direction == :left
                return GridState(state.x - 1, state.y, false)
            elseif action.direction == :right
                return GridState(state.x + 1, state.y, false)
            end
        end
        
        function GFlowNet.apply_action(action::TerminateAction, state::GridState)
            return GridState(state.x, state.y, true)
        end
        
        function GFlowNet.state_to_features(state::GridState)
            return Float32[state.x, state.y, state.is_terminal ? 1.0 : 0.0]
        end
        
        function GFlowNet.reward(state::GridState)
            if !state.is_terminal
                return 0.0
            end
            return Float32(state.x + state.y) / 10.0
        end
        
        # Create a simple DAG
        initial_state = GridState(0, 0, false)
        terminal_states = [GridState(x, y, true) for x in 0:2 for y in 0:2]
        terminal_sink = GridState(-1, -1, true)
        actions = [
            MoveAction(:up), MoveAction(:down), MoveAction(:left), MoveAction(:right),
            TerminateAction()
        ]
        
        # Test DAG creation
        dag = GFlowNet.create_dag(initial_state, terminal_states, terminal_sink, actions)
        @test dag isa GFlowNet.DirectedAcyclicGraph
        @test dag.initial_state == initial_state
        @test length(dag.terminal_states) == length(terminal_states)
        @test dag.terminal_sink == terminal_sink
        @test dag.actions == actions
        
        # Test next/previous state functions
        next_states = GFlowNet.get_next_states(dag, initial_state)
        @test length(next_states) > 0
        
        # Test that the graph is properly acyclic
        @test is_acyclic(dag.graph)
    end
    
    @testset "Flow Functions" begin
        # Create simple neural networks for testing
        rng = Random.default_rng()
        
        # Forward policy with a simple network
        forward_policy, ps, st = GFlowNet.create_forward_policy(3, 10, 5, rng)
        @test forward_policy isa GFlowNet.ForwardPolicy
        
        # Flow estimator
        flow_estimator, ps, st = GFlowNet.create_flow_estimator(3, 10, rng)
        @test flow_estimator isa GFlowNet.FlowEstimator
    end
end
