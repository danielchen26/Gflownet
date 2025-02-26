module GFlowNetUtils

using Optimisers  # Add Optimisers package
using ..GFlowNet  # Import the parent module to access GFlowNetModel

export GFlowNetLogger
export log_metric!, log_iteration!, get_metric, get_last_metric, reset!, save_metrics
export summarize_performance, time_execution, benchmark_sampling

export visualize_dag, visualize_flows, visualize_trajectory
export visualize_reward_distribution, visualize_training_progress
export visualize_molecular_state, visualize_causal_graph, visualize_experiment_selection

# Include logging utilities
include("logging.jl")

# Include visualization utilities
include("visualization.jl")

end # module 