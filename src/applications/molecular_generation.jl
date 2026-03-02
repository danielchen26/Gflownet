# Fragment-Based Molecular Generation Domain for GFlowNet
#
# Implements the GFlowNet interface for molecular generation using
# molecular fragments as the action space. Molecules are built by
# iteratively attaching fragments at open attachment points.
#
# Dependencies: RDKitBridge module (for Python/RDKit operations)
# Interface: GFlowNet.{state_to_features, is_terminal_state, reward,
#            is_applicable, apply_action, find_parent_for_action}

using Random
using GFlowNet: AbstractState, AbstractAction, GFlowNetModel
using GFlowNet: ForwardPolicy, PartitionFunctionMethod, LEARNABLE_ESTIMATION

# ============================================
# State Type
# ============================================

"""
    MolState <: AbstractState

A molecular state for fragment-based GFlowNet generation.

Feature vector layout (STATE_DIM = 1042):
  [1:1024]     Morgan fingerprint (global molecular structure)
  [1025]       Normalized attachment point count
  [1026]       Normalized fragment count (generation progress)
  [1027:1042]  One-hot for available attachment point slots
"""
struct MolState <: AbstractState
    smiles::String
    attachment_points::Vector{Int}
    n_fragments::Int
    is_terminated::Bool
    fingerprint::Vector{Float32}  # length = FINGERPRINT_DIM (1024)
end

# ============================================
# Action Types
# ============================================

"""
    FragmentAction <: AbstractAction

Add a molecular fragment to the current molecule.
Fragment SMILES uses [*] to mark attachment atoms.
"""
struct FragmentAction <: AbstractAction
    fragment_id::Int
    fragment_smiles::String
    fragment_name::String
end

"""
    TerminateMolAction <: AbstractAction

Terminate molecular generation — finalize the current molecule.
"""
struct TerminateMolAction <: AbstractAction end

# ============================================
# Constants
# ============================================

const FINGERPRINT_DIM = 1024
const MAX_FRAGMENTS = 8
const MAX_ATTACHMENTS = 16
const STATE_DIM = FINGERPRINT_DIM + 2 + MAX_ATTACHMENTS  # 1042

# ============================================
# Fragment Library (50 fragments)
# All SMILES validated against RDKit at module load time.
# ============================================

const FRAGMENT_LIBRARY = [
    # === Ring Systems (15) ===
    FragmentAction(1,  "c1ccc([*])cc1",          "benzene"),
    FragmentAction(2,  "c1ccnc([*])c1",          "pyridine"),
    FragmentAction(3,  "c1cnc([*])nc1",          "pyrimidine"),
    FragmentAction(4,  "c1cc([*])c[nH]1",        "pyrrole"),
    FragmentAction(5,  "c1cc([*])cs1",           "thiophene"),
    FragmentAction(6,  "c1cc([*])co1",           "furan"),
    FragmentAction(7,  "c1cnc([*])o1",           "oxazole"),
    FragmentAction(8,  "c1cn([*])cn1",           "imidazole"),
    FragmentAction(9,  "C1CCC([*])CC1",          "cyclohexane"),
    FragmentAction(10, "C1CC([*])OC1",           "tetrahydrofuran"),
    FragmentAction(11, "c1ccc2cc([*])ccc2c1",    "naphthalene"),
    FragmentAction(12, "c1ccc2[nH]c([*])cc2c1",  "indole"),
    FragmentAction(13, "C1CC([*])NC1",           "pyrrolidine"),
    FragmentAction(14, "C1CCN([*])CC1",          "piperidine"),
    FragmentAction(15, "C1CN([*])CCO1",          "morpholine"),

    # === Functional Groups (15) ===
    FragmentAction(16, "[*]O",                   "hydroxyl"),
    FragmentAction(17, "[*]N",                   "amine"),
    FragmentAction(18, "[*]C(=O)O",              "carboxyl"),
    FragmentAction(19, "[*]C(=O)N",              "amide"),
    FragmentAction(20, "[*]C#N",                 "nitrile"),
    FragmentAction(21, "[*]S(=O)(=O)N",          "sulfonamide"),
    FragmentAction(22, "[*]F",                   "fluorine"),
    FragmentAction(23, "[*]Cl",                  "chlorine"),
    FragmentAction(24, "[*]C(F)(F)F",            "trifluoromethyl"),
    FragmentAction(25, "[*]OC",                  "methoxy"),
    FragmentAction(26, "[*]N(C)C",               "dimethylamine"),
    FragmentAction(27, "[*]C(=O)",               "carbonyl"),
    FragmentAction(28, "[*]S(=O)(=O)C",          "methylsulfonyl"),
    FragmentAction(29, "[*][N+](=O)[O-]",        "nitro"),
    FragmentAction(30, "[*]C(=O)OC",             "ester"),

    # === Linkers (10) — have 2 attachment points ===
    FragmentAction(31, "[*]C[*]",                "methylene_bridge"),
    FragmentAction(32, "[*]CC[*]",               "ethylene_bridge"),
    FragmentAction(33, "[*]OC[*]",               "ether_bridge"),
    FragmentAction(34, "[*]NC[*]",               "amine_bridge"),
    FragmentAction(35, "[*]C(=O)N[*]",           "amide_bridge"),
    FragmentAction(36, "[*]c1cccc([*])c1",       "meta_phenyl_bridge"),
    FragmentAction(37, "[*]c1ccc([*])cc1",       "para_phenyl_bridge"),
    FragmentAction(38, "[*]C(=O)[*]",            "carbonyl_bridge"),
    FragmentAction(39, "[*]SC[*]",               "thioether_bridge"),
    FragmentAction(40, "[*]NC(=O)[*]",           "reverse_amide_bridge"),

    # === Starter Fragments (10) — have 2 attachment points, used as first fragment ===
    FragmentAction(41, "c1ccc([*])c([*])c1",     "1,2-disubstituted_benzene"),
    FragmentAction(42, "c1cc([*])cc([*])c1",     "1,3-disubstituted_benzene"),
    FragmentAction(43, "c1cc([*])ccc1[*]",       "1,4-disubstituted_benzene"),
    FragmentAction(44, "c1cc([*])nc([*])c1",     "2,6-disubstituted_pyridine"),
    FragmentAction(45, "[*]C[*]",                "methane_core"),
    FragmentAction(46, "[*]CC[*]",               "ethane_core"),
    FragmentAction(47, "c1ccc2c([*])c([*])ccc2c1", "2,3-disubstituted_naphthalene"),
    FragmentAction(48, "C1CC([*])CCC1[*]",       "1,4-disubstituted_cyclohexane"),
    FragmentAction(49, "c1cn([*])c([*])n1",      "1,3-disubstituted_imidazole"),
    FragmentAction(50, "c1cc([*])c([*])s1",      "2,3-disubstituted_thiophene"),
]

# ============================================
# Fragment Validation
# ============================================

"""Validate all fragments in the library against RDKit. Call at startup."""
function validate_fragment_library!()
    for frag in FRAGMENT_LIBRARY
        if !RDKitBridge.validate_smarts(frag.fragment_smiles)
            error("Invalid fragment SMARTS at id=$(frag.fragment_id) '$(frag.fragment_name)': $(frag.fragment_smiles)")
        end
    end
    @info "All $(length(FRAGMENT_LIBRARY)) fragments validated successfully"
end

# ============================================
# Equality & Hashing
# ============================================

# Include attachment_points in equality — same SMILES with different
# available attachments represents different positions in the DAG.
function Base.:(==)(a::MolState, b::MolState)
    return a.smiles == b.smiles &&
           a.is_terminated == b.is_terminated &&
           a.attachment_points == b.attachment_points
end

function Base.hash(s::MolState, h::UInt)
    h = hash(s.smiles, h)
    h = hash(s.is_terminated, h)
    for ap in s.attachment_points
        h = hash(ap, h)
    end
    return h
end

Base.:(==)(a::FragmentAction, b::FragmentAction) = a.fragment_id == b.fragment_id
Base.hash(a::FragmentAction, h::UInt) = hash((:FragmentAction, a.fragment_id), h)

Base.:(==)(::TerminateMolAction, ::TerminateMolAction) = true
Base.hash(::TerminateMolAction, h::UInt) = hash(:TerminateMolAction, h)

# ============================================
# GFlowNet Interface Implementations
# ============================================

"""Convert MolState to feature vector for neural network input.
Uses vcat (no mutation) for Zygote autodiff compatibility."""
function GFlowNet.state_to_features(state::MolState)::Vector{Float32}
    # [1025] Normalized attachment point count
    attach_norm = Float32(length(state.attachment_points)) / Float32(MAX_ATTACHMENTS)
    # [1026] Normalized fragment count (progress)
    frag_norm = Float32(state.n_fragments) / Float32(MAX_FRAGMENTS)
    # [1027:1042] One-hot for available attachment slots
    attach_onehot = Float32[i <= length(state.attachment_points) ? 1.0f0 : 0.0f0
                            for i in 1:MAX_ATTACHMENTS]
    # Concatenate without mutation (Zygote-safe)
    return vcat(state.fingerprint, Float32[attach_norm, frag_norm], attach_onehot)
end

"""Check if molecular generation is complete."""
function GFlowNet.is_terminal_state(state::MolState)::Bool
    return state.is_terminated
end

"""
Compute reward for a terminal molecule.
Uses smooth Gaussian penalties instead of hard thresholds.
Returns > 0 always (required by TB loss for log computation).
"""
function GFlowNet.reward(state::MolState)::Float64
    !state.is_terminated && return 0.0
    isempty(state.smiles) && return 1e-4

    props = RDKitBridge.compute_mol_properties(state.smiles)
    props === nothing && return 1e-4

    qed_score = props.qed
    sa_norm = clamp(1.0 - (props.sa_score - 1.0) / 9.0, 0.0, 1.0)

    # Smooth Gaussian penalties
    logp_score = exp(-0.5 * ((props.logp - 2.5) / 2.5)^2)
    mw_score = exp(-0.5 * ((props.mw - 350.0) / 150.0)^2)

    reward_val = (qed_score^0.4) * (sa_norm^0.3) * (logp_score^0.2) * (mw_score^0.1)
    return max(reward_val, 1e-4)
end

"""Check if a fragment action is applicable from the current state."""
function GFlowNet.is_applicable(action::FragmentAction, state::MolState)::Bool
    state.is_terminated && return false
    state.n_fragments >= MAX_FRAGMENTS && return false

    if isempty(state.smiles)
        # First fragment: only starter fragments (id >= 41)
        return action.fragment_id >= 41
    else
        return !isempty(state.attachment_points)
    end
end

function GFlowNet.is_applicable(action::TerminateMolAction, state::MolState)::Bool
    state.is_terminated && return false
    return state.n_fragments >= 1
end

"""
Apply a fragment action to the current molecular state.
Returns a new MolState (pure functional — no mutation).
Wrapped in try/catch with safe fallback on join failure.
"""
function GFlowNet.apply_action(action::FragmentAction, state::MolState)::MolState
    try
        if isempty(state.smiles)
            new_smiles, new_attachments = RDKitBridge.place_first_fragment(action.fragment_smiles)
        else
            attach_idx = state.attachment_points[1]
            new_smiles, new_attachments = RDKitBridge.join_fragment(
                state.smiles, action.fragment_smiles, attach_idx
            )
        end

        # Validate result
        if isempty(new_smiles) || !RDKitBridge.validate_smiles(new_smiles)
            @warn "Fragment join produced invalid SMILES, returning current state" action=action.fragment_name
            return state
        end

        new_fp = RDKitBridge.compute_fingerprint(new_smiles)
        return MolState(new_smiles, new_attachments, state.n_fragments + 1, false, new_fp)
    catch e
        @warn "apply_action failed" exception=e action=action.fragment_name smiles=state.smiles
        return state
    end
end

function GFlowNet.apply_action(action::TerminateMolAction, state::MolState)::MolState
    final_smiles = RDKitBridge.finalize_smiles(state.smiles)
    final_fp = RDKitBridge.compute_fingerprint(final_smiles)
    return MolState(final_smiles, Int[], state.n_fragments, true, final_fp)
end

# ============================================
# TLM Backward Support
# ============================================

# For molecular generation, we cannot efficiently reverse a fragment join.
# Return nothing → TLM uses trajectory-based backward sampling.
function GFlowNet.find_parent_for_action(state::MolState, action::FragmentAction)
    return nothing
end

function GFlowNet.find_parent_for_action(state::MolState, action::TerminateMolAction)
    return nothing
end

# ============================================
# Model Factory
# ============================================

"""
    create_molecular_gflownet(; kwargs...)

Create a GFlowNet model configured for fragment-based molecular generation.

# Arguments
- `hidden_dim::Int=256`: Hidden layer size (larger due to 1042-dim input)
- `learning_rate::Float64=0.001`: Learning rate
- `include_backward::Bool=false`: Include backward policy (for DB/TLM)
- `include_flow_estimator::Bool=false`: Include flow estimator (for FM/DFO)
- `partition_function_method`: How to handle Z (default: LEARNABLE_ESTIMATION)
- `rng`: Random number generator
"""
function create_molecular_gflownet(;
    hidden_dim::Int = 256,
    learning_rate::Float64 = 0.001,
    include_backward::Bool = false,
    include_flow_estimator::Bool = false,
    partition_function_method::PartitionFunctionMethod = LEARNABLE_ESTIMATION,
    rng = Random.default_rng()
)
    initial_state = MolState("", Int[], 0, false, zeros(Float32, FINGERPRINT_DIM))
    all_actions = AbstractAction[FRAGMENT_LIBRARY..., TerminateMolAction()]

    return GFlowNet.create_gflownet(
        initial_state,
        all_actions;
        state_dim = STATE_DIM,
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        include_backward = include_backward,
        include_flow_estimator = include_flow_estimator,
        partition_function_method = partition_function_method,
        rng = rng
    )
end
