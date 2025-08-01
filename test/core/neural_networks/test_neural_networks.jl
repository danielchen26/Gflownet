# Test Neural Network Components
# Tests for Lux.jl and ComponentArrays integration with GFlowNet

using Test
using GFlowNet
using ComponentArrays
using Lux
using Random
using NNlib
using Zygote

@testset "Neural Network Integration" begin
    Random.seed!(42)
    
    @testset "Lux.jl Network Creation" begin
        # Create simple neural networks directly
        forward_model = Chain(
            Dense(4, 16, tanh),
            Dense(16, 3)  # 3 actions
        )
        
        flow_model = Chain(
            Dense(4, 8, tanh),
            Dense(8, 1)
        )
        
        # Initialize parameters
        rng = Random.default_rng()
        forward_params, forward_states = Lux.setup(rng, forward_model)
        flow_params, flow_states = Lux.setup(rng, flow_model)
        
        @test forward_params isa NamedTuple
        @test flow_params isa NamedTuple
    end
    
    @testset "ComponentArrays Integration" begin
        # Create networks
        model = Chain(Dense(2, 4, tanh), Dense(4, 1))
        rng = Random.default_rng()
        params, states = Lux.setup(rng, model)
        
        # Convert to ComponentArray
        ca_params = ComponentArray(params)
        
        @test ca_params isa ComponentArray
        @test length(ca_params) > 0
        
        # Test that we can reconstruct the NamedTuple
        reconstructed = NamedTuple(ca_params)
        @test keys(reconstructed) == keys(params)
    end
    
    @testset "Forward Pass" begin
        # Create a simple network
        model = Chain(
            Dense(4, 8, relu),
            Dense(8, 3)
        )
        
        rng = Random.default_rng()
        params, states = Lux.setup(rng, model)
        
        # Test input
        test_features = Float32[1.0, 0.5, -0.3, 0.8]
        
        # Forward pass - Lux expects inputs to be matrices (features x batch)
        input_batch = reshape(test_features, :, 1)
        output, _ = Lux.apply(model, input_batch, params, states)
        
        @test size(output) == (3, 1)
        @test eltype(output) == Float32
        
        # Test softmax
        probs = softmax(output[:, 1])
        @test length(probs) == 3
        @test sum(probs) ≈ 1.0
        @test all(p -> 0 <= p <= 1, probs)
    end
    
    @testset "Gradient Computation" begin
        # Simple network
        model = Chain(Dense(2, 4, tanh), Dense(4, 1))
        rng = Random.default_rng()
        params, states = Lux.setup(rng, model)
        
        # Convert to ComponentArray for easier gradient handling
        ca_params = ComponentArray(params)
        
        # Loss function that doesn't mutate - create matrices differently
        function loss(p)
            # Create matrices without using literal syntax that might cause mutations
            input = hcat([1.0f0, 0.8f0], [0.5f0, -0.2f0])  # 2x2 batch
            target = hcat([0.5f0], [-0.3f0])  # 1x2 target
            
            # Use NamedTuple reconstruction to pass to Lux
            params_nt = NamedTuple(p)
            output, _ = Lux.apply(model, input, params_nt, states)
            return sum((output .- target).^2)
        end
        
        # Compute gradient
        grad = Zygote.gradient(loss, ca_params)[1]
        
        @test grad isa ComponentArray
        @test size(grad) == size(ca_params)
        @test all(isfinite, grad)
    end
    
    @testset "GFlowNet Policy Networks" begin
        # Test the create_forward_policy function
        state_dim = 4
        n_actions = 3
        hidden_dim = 16
        rng = Random.default_rng()
        
        policy_wrapper, params, states = GFlowNet.create_forward_policy(
            state_dim, hidden_dim, n_actions, rng
        )
        
        @test policy_wrapper isa GFlowNet.ForwardPolicy
        @test params isa NamedTuple
        
        # Test that the underlying model works
        test_features = rand(Float32, state_dim, 5)  # batch of 5
        output, _ = Lux.apply(policy_wrapper.model, test_features, params, states)
        
        @test size(output) == (n_actions, 5)
        @test eltype(output) == Float32
        
        # Test probabilities
        probs = softmax(output; dims=1)
        @test all(sum(probs; dims=1) .≈ 1.0)
    end
    
    @testset "Flow Estimator Network" begin
        state_dim = 4
        hidden_dim = 16
        rng = Random.default_rng()
        
        flow_wrapper, params, states = GFlowNet.create_flow_estimator(
            state_dim, hidden_dim, rng
        )
        
        @test flow_wrapper isa GFlowNet.FlowEstimator
        
        # Test forward pass with the underlying model
        test_features = rand(Float32, state_dim, 10)
        output, _ = Lux.apply(flow_wrapper.model, test_features, params, states)
        
        @test size(output) == (1, 10)
        @test eltype(output) == Float32
    end
    
    @testset "Integration with GFlowNet Model" begin
        # Create a simple grid world model to test integration
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3, 3) => 10.0),
            hidden_dim=8,
            learning_rate=0.01
        )
        
        @test model isa GFlowNet.GFlowNetModel
        @test model.forward_policy isa GFlowNet.ForwardPolicy
        @test model.flow_estimator isa GFlowNet.FlowEstimator
        @test model.parameters isa ComponentArray
        
        # Test that we can use the model to compute forward probabilities
        state = model.initial_state
        actions = model.all_actions
        
        # Get action probabilities using the correct function signature
        action_probs = GFlowNet.forward_action_probabilities(
            model.forward_policy, state, actions,
            model.parameters.forward, model.states.forward
        )
        @test length(action_probs) == length(actions)
        @test all(p -> 0 <= p <= 1, action_probs)
        @test sum(action_probs) ≈ 1.0
    end
end