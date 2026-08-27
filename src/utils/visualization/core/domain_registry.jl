# Domain Registry for GFlowNet Visualization Platform
# Provides centralized registration and discovery of domain adapters
#
# Phase 1: Domain-Agnostic Architecture
# NOTE: adapters.jl is included by unified_server.jl before this file.
# Do NOT include it here to avoid redefining AbstractDomainAdapter.

"""
    DomainRegistry

Global registry for domain adapters. Maintains a mapping from domain IDs
to adapter types, enabling dynamic domain discovery and instantiation.
"""
struct DomainRegistry
    domains::Dict{String, Type{<:AbstractDomainAdapter}}

    DomainRegistry() = new(Dict{String, Type{<:AbstractDomainAdapter}}())
end

# Global singleton registry
const DOMAIN_REGISTRY = DomainRegistry()

"""
    register_domain!(id::String, adapter_type::Type{<:AbstractDomainAdapter})

Register a domain adapter type with the global registry.

# Arguments
- `id`: Unique domain identifier (e.g., "grid_world", "dag", "sequence")
- `adapter_type`: The adapter type to register

# Example
```julia
register_domain!("grid_world", GridWorldAdapter)
```
"""
function register_domain!(id::String, adapter_type::Type{<:AbstractDomainAdapter})
    if haskey(DOMAIN_REGISTRY.domains, id)
        @warn "Overwriting existing domain registration: $id"
    end
    DOMAIN_REGISTRY.domains[id] = adapter_type
    invalidate_domains_cache!()
    @info "Registered domain: $id => $adapter_type"
    return nothing
end

"""
    unregister_domain!(id::String)

Remove a domain from the registry.
"""
function unregister_domain!(id::String)
    if haskey(DOMAIN_REGISTRY.domains, id)
        delete!(DOMAIN_REGISTRY.domains, id)
        invalidate_domains_cache!()
        @info "Unregistered domain: $id"
    end
    return nothing
end

"""
    list_domains()::Vector{Dict}

List all registered domains with their metadata.

# Returns
Vector of dictionaries containing:
- `id`: Domain identifier
- `name`: Human-readable name
- `description`: Domain description
- `renderer`: Frontend renderer component name
- `configSchema`: JSON Schema for configuration
- `isBuiltIn`: Whether this is a built-in domain
- `isPopular`: Whether this domain is marked as popular
- `tags`: Categorization tags
"""
# Memo for list_domains(). The result is a pure function of the registry, which is
# only mutated by register_domain!/unregister_domain! -- both of which run once at
# server startup. Without this, every call rebuilt a temporary instance of all
# three adapters plus six metadata calls and a sort. GET /api/v2/domains paid that
# TWICE per request (it calls list_domains directly and again via
# builtin_domain_count), and get_domain_info is implemented as list_domains plus a
# linear scan. Invalidated explicitly on registry mutation, so the output stays
# byte-identical.
const _DOMAINS_CACHE = Ref{Union{Nothing,Vector{Dict}}}(nothing)

function invalidate_domains_cache!()
    _DOMAINS_CACHE[] = nothing
    return nothing
end

function list_domains()::Vector{Dict}
    cached = _DOMAINS_CACHE[]
    cached === nothing || return cached

    domains = Dict[]

    for (id, adapter_type) in DOMAIN_REGISTRY.domains
        try
            # Create a temporary instance to get metadata
            # For domains requiring config, we create with empty config
            adapter = try
                create_from_config(adapter_type, Dict())
            catch
                # If create_from_config fails, try direct construction
                adapter_type()
            end

            push!(domains, Dict(
                "id" => id,
                "name" => titlecase(replace(id, "_" => " ")),
                "description" => get_domain_description(adapter),
                "renderer" => get_renderer_name(adapter),
                "configSchema" => get_config_schema(adapter),
                "isBuiltIn" => is_builtin_domain(adapter),
                "isPopular" => is_popular_domain(adapter),
                "tags" => get_domain_tags(adapter)
            ))
        catch e
            @warn "Failed to get metadata for domain $id: $e"
            # Still include basic info
            push!(domains, Dict(
                "id" => id,
                "name" => titlecase(replace(id, "_" => " ")),
                "description" => "Domain adapter: $adapter_type",
                "renderer" => "custom",
                "configSchema" => Dict("type" => "object", "properties" => Dict()),
                "isBuiltIn" => false,
                "isPopular" => false,
                "tags" => String[]
            ))
        end
    end

    # Sort by popularity, then by name
    sort!(domains, by = d -> (!d["isPopular"], d["name"]))

    _DOMAINS_CACHE[] = domains
    return domains
end

"""
    get_domain_info(id::String)::Union{Dict, Nothing}

Get detailed information about a specific domain.
"""
function get_domain_info(id::String)::Union{Dict, Nothing}
    if !haskey(DOMAIN_REGISTRY.domains, id)
        return nothing
    end

    domains = list_domains()
    for d in domains
        if d["id"] == id
            return d
        end
    end
    return nothing
end

"""
    has_domain(id::String)::Bool

Check if a domain is registered.
"""
function has_domain(id::String)::Bool
    return haskey(DOMAIN_REGISTRY.domains, id)
end

"""
    create_domain(id::String, config::Dict)::AbstractDomainAdapter

Create a domain adapter instance from the registry.

# Arguments
- `id`: Domain identifier (must be registered)
- `config`: Configuration dictionary

# Returns
A new domain adapter instance

# Throws
- `KeyError` if domain is not registered
- `ArgumentError` if config is invalid
"""
function create_domain(id::String, config::Dict)::AbstractDomainAdapter
    if !haskey(DOMAIN_REGISTRY.domains, id)
        throw(KeyError("Unknown domain: $id. Available domains: $(keys(DOMAIN_REGISTRY.domains))"))
    end

    adapter_type = DOMAIN_REGISTRY.domains[id]

    # Create temporary instance for validation
    temp_adapter = try
        create_from_config(adapter_type, Dict())
    catch
        adapter_type()
    end

    # Validate config
    is_valid, error_msg = validate_config(temp_adapter, config)
    if !is_valid
        throw(ArgumentError("Invalid config for domain $id: $error_msg"))
    end

    # Create adapter from config
    return create_from_config(adapter_type, config)
end

"""
    get_domain_schema(id::String)::Union{Dict, Nothing}

Get the JSON Schema for a domain's configuration.
"""
function get_domain_schema(id::String)::Union{Dict, Nothing}
    if !haskey(DOMAIN_REGISTRY.domains, id)
        return nothing
    end

    adapter_type = DOMAIN_REGISTRY.domains[id]
    temp_adapter = try
        create_from_config(adapter_type, Dict())
    catch
        adapter_type()
    end

    return get_config_schema(temp_adapter)
end

"""
    validate_domain_config(id::String, config::Dict)::Tuple{Bool, Union{String, Nothing}}

Validate configuration for a specific domain.

# Returns
Tuple of (is_valid, error_message)
"""
function validate_domain_config(id::String, config::Dict)::Tuple{Bool, Union{String, Nothing}}
    if !haskey(DOMAIN_REGISTRY.domains, id)
        return (false, "Unknown domain: $id")
    end

    adapter_type = DOMAIN_REGISTRY.domains[id]
    temp_adapter = try
        create_from_config(adapter_type, Dict())
    catch
        adapter_type()
    end

    return validate_config(temp_adapter, config)
end

# ============================================
# Domain Count and Statistics
# ============================================

"""Get count of registered domains"""
function domain_count()::Int
    return length(DOMAIN_REGISTRY.domains)
end

"""Get count of built-in domains"""
function builtin_domain_count()::Int
    count = 0
    for domain_info in list_domains()
        if domain_info["isBuiltIn"]
            count += 1
        end
    end
    return count
end

"""Get all domain IDs"""
function domain_ids()::Vector{String}
    return collect(keys(DOMAIN_REGISTRY.domains))
end
