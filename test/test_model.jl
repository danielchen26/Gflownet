@testset "GFlowNet Model" begin
    model = GFlowNetModel(10, 64, 5)
    x = rand(Float32, 10)
    
    @testset "Forward Pass" begin
        fwd, bwd = model(x)
        @test size(fwd) == (5,)
        @test all(0 .≤ fwd .≤ 1)
        @test sum(fwd) ≈ 1.0 atol=1e-5
    end
    
    @testset "Parameter Count" begin
        params = Flux.params(model)
        @test length(params) > 0
    end
end 