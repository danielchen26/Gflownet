# test/runtests.jl

using Test
using GFlowNet
using Flux
using StatsBase

# Load the module which includes the GFlowNetModel, train!, and other functions
include("../src/GFlowNet.jl")

# Test the GFlowNetModel structure
@testset "GFlowNetModel Structure" begin
    input_dim = 10
    hidden_dim = 64
    output_dim = 2
    model = GFlowNet.GFlowNetModel(input_dim, hidden_dim, output_dim)

    @test model.layer1 isa Flux.Dense
    @test model.layer2 isa Flux.Dense
    @test length(Flux.params(model)) > 0
end

# Test the training process
@testset "Training Process" begin
    input_dim = 10
    hidden_dim = 64
    output_dim = 2
    model = GFlowNet.GFlowNetModel(input_dim, hidden_dim, output_dim)
    data = [(rand(input_dim), rand(output_dim)) for _ in 1:10]
    opt = Flux.ADAM(0.001)
    loss_fn(x, y) = Flux.mse(model(x), y)
    epochs = 5

    initial_loss = GFlowNet.evaluate_model(model, data)
    GFlowNet.train!(model, data, opt, loss_fn, epochs)
    trained_loss = GFlowNet.evaluate_model(model, data)

    @test trained_loss < initial_loss
end

# Test the utility functions
@testset "Utility Functions" begin
    data = [(rand(10), rand(2)) for _ in 1:100]
    train_data, val_data = GFlowNet.split_data(data, 0.8)

    @test length(train_data) == 80
    @test length(val_data) == 20
end

# Comprehensive test cases
@testset "Core Functionality" begin
    @testset "Model Architecture" begin
        model = GFlowNet.GFlowNetModel(10, 64, 5)
        @test size(model.encoder[1].W) == (64, 10)
        @test model.policy_head[end].σ == softmax
    end

    @testset "Forward Pass" begin
        model = GFlowNet.GFlowNetModel(3, 8, 2)
        x = rand(Float32, 3)
        @test size(model(x)) == (2,)
    end
end

@testset "Training Components" begin
    env = (initial_state=()->0, is_terminal=s->s>5, step=(s,a)->(s+a, Float32(a)))
    
    @testset "Trajectory Generation" begin
        model = GFlowNet.GFlowNetModel(1, 4, 3)
        states, actions, rewards = GFlowNet.generate_trajectory(model, env)
        @test length(states) == length(actions) == length(rewards)
    end
end

# Include additional test sets as necessary
