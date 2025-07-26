module GFlowNetUtils

using Optimisers  # Add Optimisers package
using ..GFlowNet  # Import the parent module to access GFlowNetModel

export GFlowNetLogger
export log_metric!, log_iteration!, get_metric, get_last_metric, reset!, save_metrics
export summarize_performance, time_execution, benchmark_sampling

export visualize_dag, visualize_flows, visualize_trajectory
export visualize_reward_distribution, visualize_training_progress
export visualize_molecular_state, visualize_causal_graph, visualize_experiment_selection

# HTML Report System exports
export generate_html_report, save_html_report
export ReportData, add_section!, add_plot!, add_table!, add_metrics!
export create_grid_visualization, create_reward_distribution_plot, create_training_progress_plot

# Include logging utilities
include("logging.jl")

# Include visualization utilities
include("visualization.jl")

# Include HTML report system
include("report.jl")

end # module 