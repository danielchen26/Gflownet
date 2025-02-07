# test/runtests.jl

using Test
using GFlowNet
using Flux
using StatsBase
using StaticArrays

# Load the module which includes the GFlowNetModel, train!, and other functions
include("../src/GFlowNet.jl")

@testset "All Tests" begin
    include("test_model.jl")
    include("test_training.jl")
    include("test_environment.jl")
end

# Test the GFlowNetModel structure
@testset "GFlowNetModel Structure" begin
    input_dim = 10
    hidden_dim = 64
    output_dim = 2
    model = GFlowNetModel(input_dim, hidden_dim, output_dim)

    @test model.encoder isa Chain
    @test model.policy_head isa Chain
    @test length(Flux.params(model)) > 0
end

# Test the training process
@testset "Training Process" begin
    input_dim = 10
    hidden_dim = 64
    output_dim = 2
    model = GFlowNetModel(input_dim, hidden_dim, output_dim)
    data = [(Float32.(rand(input_dim)), Float32.(rand(output_dim))) for _ in 1:10]
    opt = Flux.Adam(0.001)
    loss_fn = (x, y, m) -> Flux.mse(m(x), y)
    epochs = 5

    initial_loss = evaluate_model(model, data)
    train!(model, data, opt, loss_fn, epochs)
    final_loss = evaluate_model(model, data)

    @test final_loss ≤ initial_loss
end

# Test the utility functions
@testset "Utility Functions" begin
    data = [(Float32.(rand(10)), Float32.(rand(2))) for _ in 1:100]
    train_data, val_data = split_data(data, 0.8)

    @test length(train_data) == 80
    @test length(val_data) == 20
end

# Comprehensive test cases
@testset "Core Functionality" begin
    # Create environment
    env = DiscreteEnvironment(
        [Float32[0], Float32[1]],  # states
        1:2,                        # actions
        (s, a) -> Float32[s[1] + a], # transition
        (s, a, s′) -> Float32(a),    # reward
        s -> s[1] > 5               # terminal
    )
    
    # Initialize policies
    forward = ForwardPolicy(1, 4, 2)
    backward = BackwardPolicy(Chain(Dense(1, 4, relu), Dense(4, 2, softmax)))
    
    # Generate trajectory
    traj = generate_trajectory(forward, env)
    @test length(traj.actions) > 0
    
    # Test training
    train!(forward, backward, env, Flux.Adam(0.01), epochs=3)
end

@testset "Training Components" begin
    env = (
        initial_state = () -> Float32(0),
        is_terminal = s -> s > 5,
        step = (s, a) -> (s + a, Float32(a))
    )
    
    @testset "Trajectory Generation" begin
        model = GFlowNetModel(1, 4, 3)
        trajectory = generate_trajectory(model, env)
        
        @test length(trajectory.states) > 0
        @test length(trajectory.actions) == length(trajectory.rewards)
        @test trajectory.states[end] > 5  # Should reach terminal state
    end
end

# Include additional test sets as necessary
