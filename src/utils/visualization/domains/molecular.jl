# Molecular Domain Visualization Adapter
#
# Converts molecular GFlowNet data to visualization-friendly formats.
# Implements the AbstractDomainAdapter interface for the molecular domain.
#
# Dependencies: molecular_generation.jl (MolState, FragmentAction),
#               rdkit_bridge.jl (RDKitBridge module)

using GFlowNet: GFlowNetModel, Trajectory, is_terminal_state, reward

# ============================================
# Adapter Type
# ============================================

mutable struct MolecularAdapter <: AbstractDomainAdapter
    max_fragments::Int
    fragment_library::Vector{FragmentAction}
    generated_molecules::Vector{Dict}
    oracle_manager::Union{Nothing, OracleManager}
end

# Backward-compatible 3-arg constructor (no oracle_manager)
MolecularAdapter(max_frags::Int, lib::Vector{FragmentAction}, mols::Vector{Dict}) =
    MolecularAdapter(max_frags, lib, mols, nothing)

# No-arg constructor for domain registry's list_domains()
MolecularAdapter() = MolecularAdapter(8, FRAGMENT_LIBRARY, Dict[], nothing)

# ============================================
# Required Interface Methods
# ============================================

function state_to_viz_data(adapter::MolecularAdapter, state::MolState)::Dict
    return Dict(
        "smiles"            => state.smiles,
        "n_fragments"       => state.n_fragments,
        "attachment_points"  => state.attachment_points,
        "is_terminated"     => state.is_terminated,
    )
end

function trajectory_to_viz_data(adapter::MolecularAdapter, traj::Trajectory, id::String)::Dict
    steps = Dict[]
    for (i, state) in enumerate(traj.states)
        step = Dict(
            "smiles"      => state.smiles,
            "n_fragments"  => state.n_fragments,
        )
        if i <= length(traj.actions)
            action = traj.actions[i]
            step["action"] = if action isa FragmentAction
                Dict("type" => "add_fragment", "fragment" => action.fragment_name, "fragment_id" => action.fragment_id)
            else
                Dict("type" => "terminate")
            end
        end
        push!(steps, step)
    end

    terminal = traj.states[end]
    props = if is_terminal_state(terminal) && !isempty(terminal.smiles)
        _properties_to_dict(RDKitBridge.compute_mol_properties(terminal.smiles))
    else
        nothing
    end

    return Dict(
        "id"         => id,
        "smiles"     => terminal.smiles,
        "steps"      => steps,
        "n_steps"    => length(traj.actions),
        "reward"     => is_terminal_state(terminal) ? reward(terminal) : 0.0,
        "properties" => props,
    )
end

function get_domain_config(adapter::MolecularAdapter)::Dict
    config = Dict(
        "domain_type"          => "molecule",
        "max_fragments"        => adapter.max_fragments,
        "n_fragment_actions"   => length(adapter.fragment_library),
        "fragment_categories"  => Dict(
            "rings" => 15, "functional_groups" => 15,
            "linkers" => 10, "starters" => 10
        ),
    )
    if adapter.oracle_manager !== nothing
        config["oracle_status"] = get_status(adapter.oracle_manager)
    end
    return config
end

function get_renderer_name(adapter::MolecularAdapter)::String
    return "MolecularRenderer"
end

function compute_domain_metrics(adapter::MolecularAdapter, model::GFlowNetModel,
                                trajectories::Vector{Trajectory})::Dict
    valid_mols = String[]
    for t in trajectories
        terminal = t.states[end]
        if is_terminal_state(terminal) && !isempty(terminal.smiles)
            push!(valid_mols, terminal.smiles)
        end
    end
    unique_mols = unique(valid_mols)

    validity_count = 0
    for s in valid_mols
        RDKitBridge.validate_smiles(s) && (validity_count += 1)
    end

    return Dict(
        "total_generated" => length(valid_mols),
        "unique_count"    => length(unique_mols),
        "diversity_ratio"  => length(valid_mols) > 0 ?
            length(unique_mols) / length(valid_mols) : 0.0,
        "validity_rate"    => length(valid_mols) > 0 ?
            validity_count / length(valid_mols) : 0.0,
        "molecules_stored"  => length(adapter.generated_molecules),
    )
end

# ============================================
# Molecule Storage (called from step!())
# ============================================

"""
Store generated molecules from completed trajectories.
Called after each training iteration from step!() in training_session.jl.
"""
function store_molecules_from_trajectories!(adapter::MolecularAdapter,
                                            trajectories::Vector{Trajectory},
                                            iteration::Int;
                                            session_id::Union{String,Nothing}=nothing)
    new_mols = Dict[]

    for traj in trajectories
        terminal = traj.states[end]
        !is_terminal_state(terminal) && continue
        isempty(terminal.smiles) && continue

        props = RDKitBridge.compute_mol_properties(terminal.smiles)
        props === nothing && continue

        svg = try
            RDKitBridge.mol_to_svg(terminal.smiles)
        catch
            nothing
        end

        mol_dict = Dict(
            "id"              => "mol_$(length(adapter.generated_molecules) + 1)",
            "smiles"          => terminal.smiles,
            "reward"          => reward(terminal),
            "generation_step" => iteration,
            "method"          => "fragment_gflownet",
            "svg_2d"          => svg,
            "properties"      => Dict(
                "molecular_weight"         => props.mw,
                "logp"                     => props.logp,
                "qed"                      => props.qed,
                "synthetic_accessibility"  => props.sa_score,
                "tpsa"                     => props.tpsa,
                "rotatable_bonds"          => props.rotatable_bonds,
                "hbd"                      => props.hbd,
                "hba"                      => props.hba,
                "num_rings"                => props.num_rings,
                "num_aromatic_rings"       => props.num_aromatic_rings,
                "formula"                  => props.formula,
            ),
            "fingerprint"     => terminal.fingerprint,
        )

        # Add oracle scores if oracle_manager is configured
        if adapter.oracle_manager !== nothing
            oracle_scores = Dict{String,Float64}()
            for config in adapter.oracle_manager.configs
                oracle_scores[config.name] = lookup_score(adapter.oracle_manager, terminal.smiles, config.name)
            end
            mol_dict["oracle_scores"] = oracle_scores
        end

        push!(adapter.generated_molecules, mol_dict)
        push!(new_mols, mol_dict)
    end

    # Persist to SQLite database (if available)
    if !isempty(new_mols) && session_id !== nothing && MOL_DB[] !== nothing
        try
            db_store_molecules!(new_mols, session_id)
        catch e
            @warn "Failed to persist molecules to database" exception=e count=length(new_mols)
        end
    end
end

# ============================================
# Domain Registry Methods
# ============================================

get_domain_id(::MolecularAdapter) = "molecule"

get_domain_description(::MolecularAdapter) =
    "Fragment-based molecular generation with GFlowNet. Builds drug-like molecules " *
    "by combining molecular fragments with multi-objective reward (QED, SA, LogP)."

function get_config_schema(::MolecularAdapter)::Dict
    return Dict(
        "type" => "object",
        "properties" => Dict(
            "max_fragments" => Dict("type" => "integer", "default" => 8, "minimum" => 2, "maximum" => 15),
            "hidden_dim"    => Dict("type" => "integer", "default" => 256),
            "learning_rate" => Dict("type" => "number", "default" => 0.001),
        ),
        "required" => String[],
    )
end

function validate_config(::MolecularAdapter, config::Dict)::Tuple{Bool, Union{String, Nothing}}
    max_frag = get(config, "max_fragments", 8)
    if max_frag < 2 || max_frag > 15
        return (false, "max_fragments must be between 2 and 15")
    end
    return (true, nothing)
end

function create_from_config(::Type{MolecularAdapter}, config::Dict)
    max_frag = get(config, "max_fragments", 8)
    return MolecularAdapter(max_frag, FRAGMENT_LIBRARY, Dict[], nothing)
end

get_domain_tags(::MolecularAdapter) = ["Chemistry", "Drug Discovery", "Molecular Design"]
is_builtin_domain(::MolecularAdapter) = true
is_popular_domain(::MolecularAdapter) = true

# ============================================
# Helper
# ============================================

function _properties_to_dict(props::Union{RDKitBridge.MolProperties, Nothing})::Union{Dict, Nothing}
    props === nothing && return nothing
    return Dict(
        "molecular_weight"         => props.mw,
        "logp"                     => props.logp,
        "qed"                      => props.qed,
        "synthetic_accessibility"  => props.sa_score,
        "tpsa"                     => props.tpsa,
        "rotatable_bonds"          => props.rotatable_bonds,
        "hbd"                      => props.hbd,
        "hba"                      => props.hba,
        "num_rings"                => props.num_rings,
        "num_aromatic_rings"       => props.num_aromatic_rings,
        "formula"                  => props.formula,
    )
end
