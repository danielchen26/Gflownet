# Reaction-Based Molecular Domain Adapter (Gap 4)
#
# A separate domain adapter that generates synthesizable molecules
# via reaction templates instead of fragment-based construction.
# Every generated molecule has a valid synthesis route.
#
# Based on: RGFN (NeurIPS 2024), RxnFlow (ICLR 2025)
#
# Design: This is a PARALLEL domain to MolecularAdapter (Gap 3).
# Users choose one or the other at training time.

using GFlowNet: GFlowNetModel, Trajectory, AbstractState, AbstractAction,
                is_terminal_state, reward, reaction_state_dim

# ============================================
# Types
# ============================================

const MAX_REACTION_STEPS = 5
const N_REACTIONS = 17
# Derived from GFlowNet.reaction_state_dim so this cannot drift from the network
# input width that create_reaction_gflownet builds. Was the literal 1049, with
# the arithmetic (1024 FP + 17 rxn one-hot + 8 scalars) only in a comment.
const REACTION_STATE_DIM = reaction_state_dim(; n_reactions = N_REACTIONS)

"""State for reaction-based molecular generation."""
mutable struct ReactionMolState <: AbstractState
    intermediates::Vector{String}       # Current molecules (SMILES)
    reaction_history::Vector{Int}       # Reaction IDs applied so far
    n_steps::Int
    is_terminated::Bool
    fingerprint::Vector{Float32}        # Morgan FP of primary intermediate
end

"""Construct an empty initial state."""
function create_initial_reaction_state()
    ReactionMolState(String[], Int[], 0, false, zeros(Float32, 1024))
end

"""Action: apply a reaction with specified reactants."""
struct ReactionAction <: AbstractAction
    reaction_id::Int        # 1-17 for reactions, 0 for terminate
    reactant1_smiles::String  # First reactant (or intermediate reference)
    reactant2_smiles::String  # Second reactant (empty if unimolecular)
end

"""Termination action."""
const TERMINATE_REACTION = ReactionAction(0, "", "")

# Equality and hashing
Base.:(==)(a::ReactionAction, b::ReactionAction) = a.reaction_id == b.reaction_id && a.reactant1_smiles == b.reactant1_smiles && a.reactant2_smiles == b.reactant2_smiles
Base.hash(a::ReactionAction, h::UInt) = hash(a.reaction_id, hash(a.reactant1_smiles, hash(a.reactant2_smiles, h)))

Base.:(==)(a::ReactionMolState, b::ReactionMolState) = a.intermediates == b.intermediates && a.n_steps == b.n_steps && a.is_terminated == b.is_terminated
Base.hash(s::ReactionMolState, h::UInt) = hash(s.intermediates, hash(s.n_steps, h))

# ============================================
# GFlowNet Interface Implementation
# ============================================

function GFlowNet.state_to_features(state::ReactionMolState)::Vector{Float32}
    features = zeros(Float32, REACTION_STATE_DIM)

    # 1024: Morgan fingerprint of primary intermediate
    features[1:1024] = state.fingerprint

    # 17: one-hot of last reaction applied
    if !isempty(state.reaction_history)
        last_rxn = state.reaction_history[end]
        if 1 <= last_rxn <= N_REACTIONS
            features[1024 + last_rxn] = 1.0f0
        end
    end

    # 8 scalar features
    offset = 1024 + N_REACTIONS
    features[offset + 1] = Float32(state.n_steps / MAX_REACTION_STEPS)  # Progress
    features[offset + 2] = Float32(length(state.intermediates))          # N intermediates
    features[offset + 3] = Float32(state.is_terminated)                 # Terminal flag
    # Remaining 5 slots reserved for functional group indicators
    if !isempty(state.intermediates)
        primary_smi = state.intermediates[end]
        props = RDKitBridge.compute_mol_properties(primary_smi)
        if props !== nothing
            features[offset + 4] = Float32(clamp(props.mw / 500.0, 0.0, 1.0))
            features[offset + 5] = Float32(clamp(props.logp / 5.0, -1.0, 1.0))
            features[offset + 6] = Float32(props.qed)
            features[offset + 7] = Float32(clamp(props.num_rings / 5.0, 0.0, 1.0))
            features[offset + 8] = Float32(clamp(props.rotatable_bonds / 10.0, 0.0, 1.0))
        end
    end

    return features
end

function GFlowNet.is_terminal_state(state::ReactionMolState)::Bool
    return state.is_terminated
end

function GFlowNet.reward(state::ReactionMolState)::Float64
    !state.is_terminated && return 0.0
    isempty(state.intermediates) && return 1e-4

    primary = state.intermediates[end]
    props = RDKitBridge.compute_mol_properties(primary)
    props === nothing && return 1e-4

    qed_score = props.qed
    sa_norm = clamp(1.0 - (props.sa_score - 1.0) / 9.0, 0.0, 1.0)
    logp_score = exp(-0.5 * ((props.logp - 2.5) / 2.5)^2)
    mw_score = exp(-0.5 * ((props.mw - 350.0) / 150.0)^2)

    # Bonus for synthesis feasibility (shorter routes preferred)
    step_bonus = exp(-0.1 * state.n_steps)

    # Bonus for estimated yield
    yield_bonus = 1.0
    templates = RDKitBridge.get_reaction_templates()
    for rxn_id in state.reaction_history
        for t in templates
            if t["id"] == rxn_id
                yield_bonus *= t["yield_estimate"]
                break
            end
        end
    end

    base_reward = (qed_score^0.35) * (sa_norm^0.25) * (logp_score^0.15) * (mw_score^0.1) * (step_bonus^0.1) * (yield_bonus^0.05)
    return max(base_reward, 1e-4)
end

"""Execute a reaction and return the new state."""
function apply_reaction(state::ReactionMolState, action::ReactionAction)::ReactionMolState
    if action.reaction_id == 0
        # Terminate
        return ReactionMolState(
            copy(state.intermediates),
            copy(state.reaction_history),
            state.n_steps,
            true,
            copy(state.fingerprint),
        )
    end

    templates = RDKitBridge.get_reaction_templates()
    template = nothing
    for t in templates
        if t["id"] == action.reaction_id
            template = t
            break
        end
    end

    template === nothing && return state  # Invalid reaction

    reactants = String[]
    push!(reactants, action.reactant1_smiles)
    if !isempty(action.reactant2_smiles)
        push!(reactants, action.reactant2_smiles)
    end

    result = RDKitBridge.execute_reaction(template["smarts"], reactants)
    if !result.valid
        return state  # Reaction failed, state unchanged
    end

    new_intermediates = copy(state.intermediates)
    push!(new_intermediates, result.product_smiles)

    new_fp = RDKitBridge.compute_fingerprint(result.product_smiles)

    return ReactionMolState(
        new_intermediates,
        [state.reaction_history; action.reaction_id],
        state.n_steps + 1,
        false,
        new_fp,
    )
end

# ============================================
# Visualization Adapter
# ============================================

mutable struct ReactionMolecularAdapter <: AbstractDomainAdapter
    max_steps::Int
    generated_molecules::Vector{Dict}
end

ReactionMolecularAdapter() = ReactionMolecularAdapter(MAX_REACTION_STEPS, Dict[])

function state_to_viz_data(adapter::ReactionMolecularAdapter, state::ReactionMolState)::Dict
    return Dict(
        "intermediates"      => state.intermediates,
        "reaction_history"   => state.reaction_history,
        "n_steps"           => state.n_steps,
        "is_terminated"     => state.is_terminated,
        "primary_smiles"    => isempty(state.intermediates) ? "" : state.intermediates[end],
    )
end

function trajectory_to_viz_data(adapter::ReactionMolecularAdapter, traj::Trajectory, id::String)::Dict
    steps = Dict[]
    for (i, state) in enumerate(traj.states)
        step = Dict(
            "intermediates" => state.intermediates,
            "n_steps"       => state.n_steps,
        )
        if i <= length(traj.actions)
            action = traj.actions[i]
            if action isa ReactionAction
                if action.reaction_id == 0
                    step["action"] = Dict("type" => "terminate")
                else
                    templates = RDKitBridge.get_reaction_templates()
                    rxn_name = "Reaction $(action.reaction_id)"
                    for t in templates
                        t["id"] == action.reaction_id && (rxn_name = t["name"])
                    end
                    step["action"] = Dict(
                        "type" => "reaction",
                        "reaction_id" => action.reaction_id,
                        "reaction_name" => rxn_name,
                        "reactants" => [action.reactant1_smiles, action.reactant2_smiles],
                    )
                end
            end
        end
        push!(steps, step)
    end

    terminal = traj.states[end]
    primary_smi = isempty(terminal.intermediates) ? "" : terminal.intermediates[end]

    props = if is_terminal_state(terminal) && !isempty(primary_smi)
        p = RDKitBridge.compute_mol_properties(primary_smi)
        p === nothing ? nothing : Dict(
            "molecular_weight" => p.mw, "logp" => p.logp, "qed" => p.qed,
            "synthetic_accessibility" => p.sa_score, "tpsa" => p.tpsa,
            "hbd" => p.hbd, "hba" => p.hba, "rotatable_bonds" => p.rotatable_bonds,
            "num_rings" => p.num_rings, "num_aromatic_rings" => p.num_aromatic_rings,
            "formula" => p.formula,
        )
    else
        nothing
    end

    return Dict(
        "id"                => id,
        "smiles"            => primary_smi,
        "steps"             => steps,
        "n_steps"           => length(traj.actions),
        "reaction_history"  => terminal.reaction_history,
        "reward"            => is_terminal_state(terminal) ? reward(terminal) : 0.0,
        "properties"        => props,
        "synthesis_route"   => terminal.reaction_history,
        "has_synthesis"     => true,
    )
end

function get_domain_config(adapter::ReactionMolecularAdapter)::Dict
    templates = RDKitBridge.get_reaction_templates()
    return Dict(
        "domain_type"      => "reaction_molecule",
        "max_steps"        => adapter.max_steps,
        "n_reactions"      => length(templates),
        "reactions"        => [Dict("id" => t["id"], "name" => t["name"], "class" => t["class"]) for t in templates],
        "state_dim"        => REACTION_STATE_DIM,
    )
end

function get_renderer_name(adapter::ReactionMolecularAdapter)::String
    return "ReactionMolecularRenderer"
end

function compute_domain_metrics(adapter::ReactionMolecularAdapter, model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    total = length(trajectories)
    total == 0 && return Dict("n_molecules" => 0)

    valid = 0
    total_steps = 0
    total_yield = 0.0
    smiles_set = Set{String}()

    for traj in trajectories
        terminal = traj.states[end]
        if terminal isa ReactionMolState && is_terminal_state(terminal) && !isempty(terminal.intermediates)
            valid += 1
            total_steps += terminal.n_steps
            push!(smiles_set, terminal.intermediates[end])

            # Compute estimated yield from reaction history
            yield_est = 1.0
            templates = RDKitBridge.get_reaction_templates()
            for rxn_id in terminal.reaction_history
                for t in templates
                    t["id"] == rxn_id && (yield_est *= t["yield_estimate"])
                end
            end
            total_yield += yield_est
        end
    end

    return Dict(
        "n_molecules"        => valid,
        "unique_molecules"   => length(smiles_set),
        "avg_steps"          => valid > 0 ? total_steps / valid : 0.0,
        "avg_yield_estimate" => valid > 0 ? total_yield / valid : 0.0,
        "validity_rate"      => total > 0 ? valid / total : 0.0,
    )
end

# ============================================
# Molecule Storage (called from step!())
# ============================================

"""Store generated molecules from completed reaction trajectories."""
function store_molecules_from_trajectories!(adapter::ReactionMolecularAdapter,
                                            trajectories::Vector{Trajectory},
                                            iteration::Int;
                                            session_id::Union{String,Nothing}=nothing)
    for traj in trajectories
        terminal = traj.states[end]
        !(terminal isa ReactionMolState) && continue
        !is_terminal_state(terminal) && continue
        isempty(terminal.intermediates) && continue

        primary_smi = terminal.intermediates[end]
        props = RDKitBridge.compute_mol_properties(primary_smi)
        props === nothing && continue

        svg = try
            RDKitBridge.mol_to_svg(primary_smi)
        catch
            nothing
        end

        # Build synthesis route from reaction history
        templates = RDKitBridge.get_reaction_templates()
        synthesis_steps = Dict[]
        for (step_idx, rxn_id) in enumerate(terminal.reaction_history)
            rxn_name = "Reaction $rxn_id"
            rxn_class = "unknown"
            yield_est = 0.8
            for t in templates
                if t["id"] == rxn_id
                    rxn_name = t["name"]
                    rxn_class = t["class"]
                    yield_est = t["yield_estimate"]
                    break
                end
            end
            push!(synthesis_steps, Dict(
                "step"          => step_idx,
                "reaction_id"   => rxn_id,
                "reaction_name" => rxn_name,
                "reaction_class" => rxn_class,
                "yield_estimate" => yield_est,
                "intermediate"  => step_idx <= length(terminal.intermediates) ? terminal.intermediates[step_idx] : "",
            ))
        end

        mol_dict = Dict(
            "id"              => "rxn_mol_$(length(adapter.generated_molecules) + 1)",
            "smiles"          => primary_smi,
            "reward"          => reward(terminal),
            "generation_step" => iteration,
            "method"          => "reaction_gflownet",
            "svg_2d"          => svg,
            "has_synthesis"   => true,
            "synthesis_route" => synthesis_steps,
            "n_reaction_steps" => terminal.n_steps,
            "properties"      => Dict(
                "molecular_weight"        => props.mw,
                "logp"                    => props.logp,
                "qed"                     => props.qed,
                "synthetic_accessibility" => props.sa_score,
                "tpsa"                    => props.tpsa,
                "rotatable_bonds"         => props.rotatable_bonds,
                "hbd"                     => props.hbd,
                "hba"                     => props.hba,
                "num_rings"               => props.num_rings,
                "num_aromatic_rings"      => props.num_aromatic_rings,
                "formula"                 => props.formula,
            ),
            "fingerprint"     => terminal.fingerprint,
        )

        push!(adapter.generated_molecules, mol_dict)
    end
end

# ============================================
# Domain Registry Methods
# ============================================

get_domain_id(::ReactionMolecularAdapter) = "reaction_molecule"

get_domain_description(::ReactionMolecularAdapter) =
    "Reaction-based molecular generation with GFlowNet. Builds synthesizable molecules " *
    "via validated reaction templates (Suzuki, amide formation, etc.) ensuring every " *
    "generated molecule has a valid synthesis route."

function get_config_schema(::ReactionMolecularAdapter)::Dict
    return Dict(
        "type" => "object",
        "properties" => Dict(
            "max_steps"     => Dict("type" => "integer", "default" => 5, "minimum" => 1, "maximum" => 10),
            "hidden_dim"    => Dict("type" => "integer", "default" => 256),
            "learning_rate" => Dict("type" => "number", "default" => 0.001),
        ),
        "required" => String[],
    )
end

function validate_config(::ReactionMolecularAdapter, config::Dict)::Tuple{Bool, Union{String, Nothing}}
    max_steps = get(config, "max_steps", 5)
    if max_steps < 1 || max_steps > 10
        return (false, "max_steps must be between 1 and 10")
    end
    return (true, nothing)
end

function create_from_config(::Type{ReactionMolecularAdapter}, config::Dict)
    max_steps = get(config, "max_steps", MAX_REACTION_STEPS)
    return ReactionMolecularAdapter(max_steps, Dict[])
end

get_domain_tags(::ReactionMolecularAdapter) = ["Chemistry", "Synthesis", "Reaction Design"]
is_builtin_domain(::ReactionMolecularAdapter) = true
is_popular_domain(::ReactionMolecularAdapter) = false
