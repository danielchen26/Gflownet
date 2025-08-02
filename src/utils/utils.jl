module GFlowNetUtils

using Optimisers  # Add Optimisers package
using ..GFlowNet  # Import the parent module to access GFlowNetModel

# Logging utilities
export GFlowNetLogger
export log_metric!, log_iteration!, get_metric, get_last_metric, reset!, save_metrics

# Performance utilities
export summarize_performance, time_execution, benchmark_sampling

# Visualization utilities (domain-specific)
export visualize_molecular_state, visualize_causal_graph, visualize_experiment_selection

# Include logging utilities
include("logging.jl")

# Include visualization utilities
include("visualization/visualization.jl")

# Include HTML report system
include("report.jl")

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
