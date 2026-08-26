# CAFE-GFN Test Suite
# Run all SMILES GFlowNet tests

using Test

@testset "CAFE-GFN SMILES GFlowNet" begin
    include("test_tokenizer.jl")
    include("test_shifted_cosh.jl")
    include("test_smiles_state.jl")
    include("test_qgfn.jl")
    include("test_qgfn_integration.jl")
    include("test_hierarchical_edit_baseline.jl")
    include("test_option_flow_poc.jl")
    include("test_option_flow_real_catalog.jl")
end
