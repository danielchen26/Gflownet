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

# ============================================
# Domain Registry Interface Methods (Phase 1)
# ============================================

"""Get unique domain identifier (e.g., 'grid_world', 'dag', 'sequence')"""
function get_domain_id(adapter::AbstractDomainAdapter)::String
    error("Not implemented for $(typeof(adapter))")
end

"""Get human-readable domain description"""
function get_domain_description(adapter::AbstractDomainAdapter)::String
    error("Not implemented for $(typeof(adapter))")
end

"""Get JSON Schema for domain configuration"""
function get_config_schema(adapter::AbstractDomainAdapter)::Dict
    # Default: empty schema (no configuration required)
    return Dict(
        "type" => "object",
        "properties" => Dict(),
        "required" => String[]
    )
end

"""Validate domain configuration against schema"""
function validate_config(adapter::AbstractDomainAdapter, config::Dict)::Tuple{Bool, Union{String, Nothing}}
    # Default: always valid
    return (true, nothing)
end

"""
    create_from_config(adapter_type::Type{<:AbstractDomainAdapter}, config::Dict)

Factory function to create a domain adapter from configuration.
Each domain adapter must implement this as a static method.
"""
function create_from_config(adapter_type::Type{<:AbstractDomainAdapter}, config::Dict)
    error("create_from_config not implemented for $adapter_type")
end

# ============================================
# Optional Interface Methods
# ============================================

"""
    apply_reward_shaping(adapter, reward_positions)

Apply domain-specific reward shaping to compensate for structural path asymmetry.

Each domain has different structural biases — some terminal states are easier
to reach than others due to the DAG structure. This method scales rewards
to compensate, making all modes equally discoverable during training.

The default implementation returns rewards unchanged. Domain adapters should
override this with their own structural bias compensation.

# Arguments
- `adapter::AbstractDomainAdapter`: The domain adapter
- `reward_positions::Dict`: Mapping from terminal state identifiers to reward values

# Returns
- `Dict`: Modified reward positions with shaping applied
"""
function apply_reward_shaping(adapter::AbstractDomainAdapter, reward_positions::Dict)
    # Default: no shaping (domain doesn't know its structural bias)
    return reward_positions
end

"""Get domain tags for categorization"""
function get_domain_tags(adapter::AbstractDomainAdapter)::Vector{String}
    return String[]
end

"""Check if domain is a built-in domain"""
function is_builtin_domain(adapter::AbstractDomainAdapter)::Bool
    return false
end

"""Check if domain is marked as popular"""
function is_popular_domain(adapter::AbstractDomainAdapter)::Bool
    return false
end
