using Test
using GFlowNet

# Julia does not error when a module exports a name it never defines. The gap
# only surfaces as UndefVarError at the call site, which is why 13 phantom
# exports survived in src/GFlowNet.jl for so long. This guards against that.
@testset "every exported name is defined" begin
    undefined = Symbol[]
    for name in names(GFlowNet)
        name === :GFlowNet && continue
        isdefined(GFlowNet, name) || push!(undefined, name)
    end
    if !isempty(undefined)
        @info "exported but undefined" count = length(undefined) names = undefined
    end
    @test isempty(undefined)
end
