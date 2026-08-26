module GFlowNetUtils

using Optimisers  # Add Optimisers package
using ..GFlowNet  # Import the parent module to access GFlowNetModel

# NOTE: logging.jl and report.jl are included by src/GFlowNet.jl:86,88 at
# package level. They used to be included HERE as well, which compiled
# GFlowNetLogger (logging.jl:9) and ReportData (report.jl:17) into two distinct
# types with the same name -- GFlowNet.GFlowNetLogger !== 
# GFlowNet.GFlowNetUtils.GFlowNetLogger. This module therefore no longer owns
# those names and must not re-export them.
#
# visualization/visualization.jl below is NOT a duplicate: GFlowNet.jl:87
# includes the sibling file utils/visualization.jl, which is a different file.

# Performance utilities
export summarize_performance, time_execution

# Visualization utilities (domain-specific)
export visualize_molecular_state, visualize_causal_graph, visualize_experiment_selection

# Include visualization utilities
include("visualization/visualization.jl")

# Mathematical utilities
"""
    softmax(x::AbstractVector)

Compute softmax probabilities from logits.

# Mathematical Definition
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)

# Arguments
- `x::AbstractVector`: Vector of logits

# Returns
- Vector of probabilities that sum to 1
"""
function softmax(x::AbstractVector)
    # Subtract maximum for numerical stability
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x)
end

export softmax

end # module
