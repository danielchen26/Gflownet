# Domain adapter interface for visualization
# Converts domain-specific GFlowNet data to visualization-friendly formats

using GFlowNet: AbstractState, AbstractAction, GFlowNetModel, Trajectory

"""
    AbstractDomainAdapter

Base interface for converting domain-specific GFlowNet data to
visualization-friendly formats. All domain adapters must implement this.
"""
abstract type AbstractDomainAdapter end

# ============================================
# Required Interface Methods
# ============================================

"""Convert state to JSON-serializable visualization data"""
function state_to_viz_data(adapter::AbstractDomainAdapter, state::AbstractState)::Dict
    error("Not implemented for $(typeof(adapter))")
end

"""Convert trajectory to JSON-serializable visualization data"""
function trajectory_to_viz_data(adapter::AbstractDomainAdapter, traj::Trajectory, id::String)::Dict
    error("Not implemented for $(typeof(adapter))")
end

"""Get domain configuration for frontend"""
function get_domain_config(adapter::AbstractDomainAdapter)::Dict
    error("Not implemented for $(typeof(adapter))")
end

"""Get frontend renderer component name"""
function get_renderer_name(adapter::AbstractDomainAdapter)::String
    error("Not implemented for $(typeof(adapter))")
end

"""Compute domain-specific visualization metrics"""
function compute_domain_metrics(adapter::AbstractDomainAdapter, model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    error("Not implemented for $(typeof(adapter))")
end

"""Compute flow field for visualization (if applicable)"""
function compute_flow_field(adapter::AbstractDomainAdapter, model::GFlowNetModel)::Dict
    # Default: not supported
    return Dict("supported" => false)
end

"""Compute distribution heatmap (if applicable)"""
function compute_distribution_data(adapter::AbstractDomainAdapter, model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    # Default: not supported
    return Dict("supported" => false)
end
