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
include("visualization.jl")

# Include HTML report system
include("report.jl")

end # module
