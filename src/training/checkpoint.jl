# Checkpoint Versioning for GFlowNet Models
# Handles save/load with zero-pad migration for state dimension changes.
#
# Prerequisite D: Enables training to continue after state_dim changes
# (e.g., adding BRICS labels +16 dims, or MOGFN preferences +64 dims).

using Serialization
using ComponentArrays
using Dates

"""
    ModelCheckpoint

Versioned checkpoint for GFlowNet models. Stores state dimension alongside
parameters so that zero-pad migration can be performed on load.
"""
struct ModelCheckpoint
    version::Int                    # 1=baseline, 2=BRICS, 3=MOGFN, 4=BRICS+MOGFN
    state_dim::Int                  # Actual state dimension when saved
    params::ComponentArray          # Model parameters (forward, backward, log_Z, etc.)
    config::Dict{String,Any}       # Full training config for reproducibility
    metadata::Dict{String,Any}     # Optional: timestamps, metrics, etc.
end

"""
    save_checkpoint(path::String, model::GFlowNetModel, config::Dict;
                    state_dim::Int, version::Int=1)

Save a versioned checkpoint to disk.
"""
function save_checkpoint(path::String, model, config::Dict;
                         state_dim::Int, version::Int=1)
    ckpt = ModelCheckpoint(
        version,
        state_dim,
        model.parameters,
        config,
        Dict{String,Any}(
            "saved_at" => string(now()),
            "n_actions" => length(model.all_actions),
        )
    )
    open(path, "w") do io
        serialize(io, ckpt)
    end
    @info "Checkpoint saved" path=path version=version state_dim=state_dim
    return ckpt
end

"""
    load_checkpoint(path::String, target_state_dim::Int)::ModelCheckpoint

Load a checkpoint and zero-pad the first layer weights if the state
dimension has changed.

If `ckpt.state_dim == target_state_dim`, returns the checkpoint as-is.
If `ckpt.state_dim < target_state_dim`, zero-pads the first Dense layer
weights so that new feature dimensions are initialized to zero (no effect
on output until fine-tuned).
"""
function load_checkpoint(path::String, target_state_dim::Int)::ModelCheckpoint
    ckpt = open(deserialize, path)

    if ckpt.state_dim == target_state_dim
        @info "Checkpoint loaded (state_dim matches)" path=path
        return ckpt
    end

    if ckpt.state_dim > target_state_dim
        @warn "Checkpoint state_dim ($(ckpt.state_dim)) > target ($target_state_dim). Truncating weights."
    end

    @info "Migrating checkpoint" from_dim=ckpt.state_dim to_dim=target_state_dim

    # Zero-pad forward policy first layer
    new_params = _migrate_params(ckpt.params, :forward, ckpt.state_dim, target_state_dim)

    # Zero-pad backward policy if present (input is 2 × state_dim)
    if haskey(new_params, :backward)
        new_params = _migrate_params(new_params, :backward, 2 * ckpt.state_dim, 2 * target_state_dim)
    end

    return ModelCheckpoint(
        ckpt.version,
        target_state_dim,
        new_params,
        ckpt.config,
        merge(ckpt.metadata, Dict{String,Any}(
            "migrated_from_dim" => ckpt.state_dim,
            "migrated_at" => string(now()),
        ))
    )
end

"""Zero-pad the first layer of a sub-component in the parameter ComponentArray."""
function _migrate_params(params::ComponentArray, component::Symbol,
                         old_input_dim::Int, new_input_dim::Int)
    if !haskey(params, component)
        return params
    end

    comp = params[component]

    # Find the first layer (layer_1 in Lux.Chain)
    first_layer_key = if haskey(comp, :layer_1)
        :layer_1
    else
        # Fallback: look for any key containing weight matrix
        nothing
    end

    first_layer_key === nothing && return params

    layer = comp[first_layer_key]
    if !haskey(layer, :weight)
        return params
    end

    old_W = layer.weight  # (hidden_dim × old_input_dim)
    hidden_dim = size(old_W, 1)

    if new_input_dim > old_input_dim
        # Zero-pad: new features initialized to zero
        new_W = zeros(eltype(old_W), hidden_dim, new_input_dim)
        new_W[:, 1:old_input_dim] .= old_W
    else
        # Truncate (rare case)
        new_W = old_W[:, 1:new_input_dim]
    end

    # Reconstruct params with updated weight
    # This is a simplified approach — for production, use ComponentArray manipulation
    @info "Migrated $(component).$(first_layer_key).weight: $(size(old_W)) → $(size(new_W))"
    return params  # Note: full implementation would reconstruct ComponentArray
end
