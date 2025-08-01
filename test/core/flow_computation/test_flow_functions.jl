using Test
using GFlowNet
using Random

# Simple test domain for flow computation
struct TestState <: AbstractState
    value::Int
    is_terminal::Bool
end

struct IncrementAction <: AbstractAction
    delta::Int
end

# Implement required interface
GFlowNet.state_to_features(state::TestState)::Vector{Float32} = Float32[state.value, state.is_terminal ? 1.0 : 0.0]
GFlowNet.is_terminal_state(state::TestState)::Bool = state.is_terminal
GFlowNet.reward(state::TestState)::Float64 = state.is_terminal ? Float64(state.value) : 0.0
GFlowNet.is_applicable(action::IncrementAction, state::TestState)::Bool = !state.is_terminal && state.value + action.delta <= 5
GFlowNet.apply_action(action::IncrementAction, state::TestState)::TestState = TestState(state.value + action.delta, state.value + action.delta >= 5)

# Equality for testing
Base.:(==)(s1::TestState, s2::TestState) = s1.value == s2.value && s1.is_terminal == s2.is_terminal
Base.hash(s::TestState, h::UInt) = hash((s.value, s.is_terminal), h)

@testset "Flow Functions" begin
    Random.seed!(42)
    
    # Create simple test model
    initial_state = TestState(0, false)
    all_actions = [IncrementAction(1), IncrementAction(2)]
    
    model = create_gflownet(
        initial_state,
        all_actions;
        state_dim = 2,
        hidden_dim = 16,
        learning_rate = 0.01
    )
    
    @testset "Terminal Flow" begin
        # Test terminal flow computation
        terminal_state = TestState(5, true)
        @test GFlowNet.terminal_flow(terminal_state) == 5.0
        
        # Test error for non-terminal state
        non_terminal = TestState(3, false)
        @test_throws ArgumentError GFlowNet.terminal_flow(non_terminal)
    end
    
    @testset "Recursive Flow Computation" begin
        # Test flow for terminal state
        terminal_state = TestState(5, true)
        flow_val = GFlowNet.compute_recursive_flow(model, terminal_state)
        @test flow_val == 5.0
        
        # Test flow for non-terminal state
        state = TestState(3, false)
        flow_val = GFlowNet.compute_recursive_flow(model, state)
        @test flow_val > 0.0  # Should be positive
        @test isfinite(flow_val)  # Should be finite
        
        # Test flow for initial state
        flow_val = GFlowNet.compute_recursive_flow(model, initial_state)
        @test flow_val > 0.0
        @test isfinite(flow_val)
    end
    
    @testset "Flow Memoization" begin
        # Clear cache
        GFlowNet.clear_flow_cache!()
        
        # Compute flow twice
        state = TestState(2, false)
        flow1 = GFlowNet.compute_recursive_flow_memoized(model, state)
        flow2 = GFlowNet.compute_recursive_flow_memoized(model, state)
        
        @test flow1 == flow2  # Should return cached value
    end
    
    @testset "Edge Flow" begin
        # Test valid edge
        source = TestState(2, false)
        target = TestState(3, false)
        edge_flow_val = GFlowNet.edge_flow(model, source, target)
        @test edge_flow_val >= 0.0  # Non-negative
        @test isfinite(edge_flow_val)
        
        # Test invalid edge (no direct transition)
        source = TestState(1, false)
        target = TestState(4, false)  # Can't jump by 3
        edge_flow_val = GFlowNet.edge_flow(model, source, target)
        @test edge_flow_val == 0.0  # No flow through invalid edge
    end
    
    @testset "Partition Function" begin
        # Partition function should be positive
        Z = partition_function(model)
        @test Z > 0.0
        @test isfinite(Z)
        
        # Should equal flow through initial state
        initial_flow = GFlowNet.flow(model, model.initial_state)
        @test Z ≈ initial_flow
    end
    
    @testset "Flow Conservation" begin
        # Test flow conservation for non-terminal states
        state = TestState(2, false)
        is_conserved = GFlowNet.validate_flow_conservation(model, state; tolerance=0.1)
        @test is_conserved
        
        # Terminal states should always satisfy conservation
        terminal = TestState(5, true)
        is_conserved = GFlowNet.validate_flow_conservation(model, terminal)
        @test is_conserved
    end
    
    @testset "Flow Analysis" begin
        # Test flow analysis output
        state = TestState(3, false)
        analysis = GFlowNet.flow_analysis(model, state)
        
        @test haskey(analysis, :flow_value)
        @test haskey(analysis, :is_terminal)
        @test haskey(analysis, :next_states)
        @test haskey(analysis, :transition_probs)
        @test haskey(analysis, :conservation_check)
        
        @test analysis.flow_value > 0.0
        @test !analysis.is_terminal
        @test length(analysis.next_states) > 0
        @test all(p >= 0 && p <= 1 for p in analysis.transition_probs)
        @test sum(analysis.transition_probs) ≈ 1.0 atol=1e-6
    end
    
    @testset "Different Flow Methods" begin
        # Test recursive flow method
        state = TestState(2, false)
        flow_recursive = GFlowNet.flow(model, state; method=GFlowNet.RECURSIVE_FLOW)
        @test flow_recursive > 0.0
        
        # Direct flow should fail without flow estimator
        if isnothing(model.flow_estimator)
            @test_throws ArgumentError GFlowNet.flow(model, state; method=GFlowNet.DIRECT_FLOW)
        else
            flow_direct = GFlowNet.flow(model, state; method=GFlowNet.DIRECT_FLOW)
            @test flow_direct > 0.0
        end
        
        # Mixed method should work (falls back to recursive)
        flow_mixed = GFlowNet.flow(model, state; method=GFlowNet.MIXED_FLOW)
        @test flow_mixed > 0.0
    end
end

println("Flow function tests completed!")