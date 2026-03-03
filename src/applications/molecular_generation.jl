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

Feature vector layout (STATE_DIM = 1042 baseline, 1058 with BRICS):
  [1:1024]     Morgan fingerprint (global molecular structure)
  [1025]       Normalized attachment point count
  [1026]       Normalized fragment count (generation progress)
  [1027:1042]  One-hot for available attachment point slots
  [1043:1058]  BRICS label encoding per attachment slot (if enabled)
"""
struct MolState <: AbstractState
    smiles::String
    attachment_points::Vector{Int}
    attachment_labels::Vector{Int}   # BRICS labels per attachment point (Gap 3)
    n_fragments::Int
    is_terminated::Bool
    fingerprint::Vector{Float32}     # length = FINGERPRINT_DIM (1024)
end

# Backward-compatible 5-arg constructor (no attachment_labels)
MolState(smiles::String, attachment_points::Vector{Int}, n_fragments::Int,
         is_terminated::Bool, fingerprint::Vector{Float32}) =
    MolState(smiles, attachment_points, Int[], n_fragments, is_terminated, fingerprint)

# ============================================
# Action Types
# ============================================

"""
    FragmentMetadata

Metadata for a fragment in the BRICS-aware fragment library (Gap 3).
Used for compatibility checking and categorization.
"""
struct FragmentMetadata
    brics_labels::Vector{Int}   # BRICS environment labels (1-16), empty for legacy
    n_attachments::Int          # Number of attachment points
    heavy_atoms::Int            # Heavy atom count
    category::String            # "ring", "functional_group", "linker", "starter"
    is_starter::Bool            # Can be used as first fragment
end

# Default metadata for legacy fragments (no BRICS labels)
FragmentMetadata() = FragmentMetadata(Int[], 1, 0, "unknown", false)

"""
    FragmentAction <: AbstractAction

Add a molecular fragment to the current molecule.
Fragment SMILES uses [*] to mark attachment atoms.
"""
struct FragmentAction <: AbstractAction
    fragment_id::Int
    fragment_smiles::String
    fragment_name::String
    metadata::FragmentMetadata
end

# Backward-compatible 3-arg constructor (no metadata)
FragmentAction(id::Int, smiles::String, name::String) =
    FragmentAction(id, smiles, name, FragmentMetadata())

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

# Default objective weights for single-objective reward (geometric mean exponents)
const DEFAULT_REWARD_WEIGHTS = Float64[0.4, 0.3, 0.2, 0.1]

"""
    compute_state_dim(config::Dict=Dict())::Int

Compute the state feature dimension based on configuration.
Replaces the former `const STATE_DIM = 1042` to support dynamic dimensions
required by BRICS labels (Gap 3) and preference conditioning (Gap 5).

| Config                      | State Dim | Delta  |
|-----------------------------|-----------|--------|
| Baseline                    | 1042      | —      |
| + BRICS labels              | 1058      | +16    |
| + MOGFN preferences         | 1106      | +64    |
| + BRICS + MOGFN             | 1122      | +80    |
"""
function compute_state_dim(config::Dict=Dict())::Int
    base = FINGERPRINT_DIM + 2 + MAX_ATTACHMENTS  # 1042
    if get(config, "use_brics_labels", false)
        base += 16  # BRICS label encoding
    end
    if get(config, "use_preferences", false)
        base += get(config, "preference_embed_dim", 64)
    end
    return base
end

# Backward-compatible constant for code that still references STATE_DIM
const STATE_DIM = compute_state_dim()  # 1042

# ============================================
# Fragment Library (50 fragments)
# All SMILES validated against RDKit at module load time.
# ============================================

const FRAGMENT_LIBRARY = [
    # === Ring Systems (15) — 1 attachment point ===
    FragmentAction(1,  "c1ccc([*])cc1",          "benzene",       FragmentMetadata(Int[],  1, 6,  "ring", false)),
    FragmentAction(2,  "c1ccnc([*])c1",          "pyridine",      FragmentMetadata(Int[],  1, 6,  "ring", false)),
    FragmentAction(3,  "c1cnc([*])nc1",          "pyrimidine",    FragmentMetadata(Int[],  1, 6,  "ring", false)),
    FragmentAction(4,  "c1cc([*])c[nH]1",        "pyrrole",       FragmentMetadata(Int[],  1, 5,  "ring", false)),
    FragmentAction(5,  "c1cc([*])cs1",           "thiophene",     FragmentMetadata(Int[],  1, 5,  "ring", false)),
    FragmentAction(6,  "c1cc([*])co1",           "furan",         FragmentMetadata(Int[],  1, 5,  "ring", false)),
    FragmentAction(7,  "c1cnc([*])o1",           "oxazole",       FragmentMetadata(Int[],  1, 5,  "ring", false)),
    FragmentAction(8,  "c1cn([*])cn1",           "imidazole",     FragmentMetadata(Int[],  1, 5,  "ring", false)),
    FragmentAction(9,  "C1CCC([*])CC1",          "cyclohexane",   FragmentMetadata(Int[],  1, 6,  "ring", false)),
    FragmentAction(10, "C1CC([*])OC1",           "tetrahydrofuran", FragmentMetadata(Int[], 1, 5, "ring", false)),
    FragmentAction(11, "c1ccc2cc([*])ccc2c1",    "naphthalene",   FragmentMetadata(Int[],  1, 10, "ring", false)),
    FragmentAction(12, "c1ccc2[nH]c([*])cc2c1",  "indole",        FragmentMetadata(Int[],  1, 9,  "ring", false)),
    FragmentAction(13, "C1CC([*])NC1",           "pyrrolidine",   FragmentMetadata(Int[],  1, 5,  "ring", false)),
    FragmentAction(14, "C1CCN([*])CC1",          "piperidine",    FragmentMetadata(Int[],  1, 6,  "ring", false)),
    FragmentAction(15, "C1CN([*])CCO1",          "morpholine",    FragmentMetadata(Int[],  1, 6,  "ring", false)),

    # === Functional Groups (15) — 1 attachment point ===
    FragmentAction(16, "[*]O",                   "hydroxyl",       FragmentMetadata(Int[], 1, 1, "functional_group", false)),
    FragmentAction(17, "[*]N",                   "amine",          FragmentMetadata(Int[], 1, 1, "functional_group", false)),
    FragmentAction(18, "[*]C(=O)O",              "carboxyl",       FragmentMetadata(Int[], 1, 3, "functional_group", false)),
    FragmentAction(19, "[*]C(=O)N",              "amide",          FragmentMetadata(Int[], 1, 3, "functional_group", false)),
    FragmentAction(20, "[*]C#N",                 "nitrile",        FragmentMetadata(Int[], 1, 2, "functional_group", false)),
    FragmentAction(21, "[*]S(=O)(=O)N",          "sulfonamide",    FragmentMetadata(Int[], 1, 4, "functional_group", false)),
    FragmentAction(22, "[*]F",                   "fluorine",       FragmentMetadata(Int[], 1, 1, "functional_group", false)),
    FragmentAction(23, "[*]Cl",                  "chlorine",       FragmentMetadata(Int[], 1, 1, "functional_group", false)),
    FragmentAction(24, "[*]C(F)(F)F",            "trifluoromethyl", FragmentMetadata(Int[], 1, 4, "functional_group", false)),
    FragmentAction(25, "[*]OC",                  "methoxy",        FragmentMetadata(Int[], 1, 2, "functional_group", false)),
    FragmentAction(26, "[*]N(C)C",               "dimethylamine",  FragmentMetadata(Int[], 1, 3, "functional_group", false)),
    FragmentAction(27, "[*]C(=O)",               "carbonyl",       FragmentMetadata(Int[], 1, 2, "functional_group", false)),
    FragmentAction(28, "[*]S(=O)(=O)C",          "methylsulfonyl", FragmentMetadata(Int[], 1, 4, "functional_group", false)),
    FragmentAction(29, "[*][N+](=O)[O-]",        "nitro",          FragmentMetadata(Int[], 1, 3, "functional_group", false)),
    FragmentAction(30, "[*]C(=O)OC",             "ester",          FragmentMetadata(Int[], 1, 4, "functional_group", false)),

    # === Linkers (10) — 2 attachment points ===
    FragmentAction(31, "[*]C[*]",                "methylene_bridge",    FragmentMetadata(Int[], 2, 1, "linker", false)),
    FragmentAction(32, "[*]CC[*]",               "ethylene_bridge",     FragmentMetadata(Int[], 2, 2, "linker", false)),
    FragmentAction(33, "[*]OC[*]",               "ether_bridge",        FragmentMetadata(Int[], 2, 2, "linker", false)),
    FragmentAction(34, "[*]NC[*]",               "amine_bridge",        FragmentMetadata(Int[], 2, 2, "linker", false)),
    FragmentAction(35, "[*]C(=O)N[*]",           "amide_bridge",        FragmentMetadata(Int[], 2, 3, "linker", false)),
    FragmentAction(36, "[*]c1cccc([*])c1",       "meta_phenyl_bridge",  FragmentMetadata(Int[], 2, 6, "linker", false)),
    FragmentAction(37, "[*]c1ccc([*])cc1",       "para_phenyl_bridge",  FragmentMetadata(Int[], 2, 6, "linker", false)),
    FragmentAction(38, "[*]C(=O)[*]",            "carbonyl_bridge",     FragmentMetadata(Int[], 2, 2, "linker", false)),
    FragmentAction(39, "[*]SC[*]",               "thioether_bridge",    FragmentMetadata(Int[], 2, 2, "linker", false)),
    FragmentAction(40, "[*]NC(=O)[*]",           "reverse_amide_bridge", FragmentMetadata(Int[], 2, 3, "linker", false)),

    # === Starter Fragments (10) — 2 attachment points, used as first fragment ===
    FragmentAction(41, "c1ccc([*])c([*])c1",     "1,2-disubstituted_benzene",    FragmentMetadata(Int[], 2, 6,  "starter", true)),
    FragmentAction(42, "c1cc([*])cc([*])c1",     "1,3-disubstituted_benzene",    FragmentMetadata(Int[], 2, 6,  "starter", true)),
    FragmentAction(43, "c1cc([*])ccc1[*]",       "1,4-disubstituted_benzene",    FragmentMetadata(Int[], 2, 6,  "starter", true)),
    FragmentAction(44, "c1cc([*])nc([*])c1",     "2,6-disubstituted_pyridine",   FragmentMetadata(Int[], 2, 6,  "starter", true)),
    FragmentAction(45, "[*]C[*]",                "methane_core",                 FragmentMetadata(Int[], 2, 1,  "starter", true)),
    FragmentAction(46, "[*]CC[*]",               "ethane_core",                  FragmentMetadata(Int[], 2, 2,  "starter", true)),
    FragmentAction(47, "c1ccc2c([*])c([*])ccc2c1", "2,3-disubstituted_naphthalene", FragmentMetadata(Int[], 2, 10, "starter", true)),
    FragmentAction(48, "C1CC([*])CCC1[*]",       "1,4-disubstituted_cyclohexane", FragmentMetadata(Int[], 2, 6, "starter", true)),
    FragmentAction(49, "c1cn([*])c([*])n1",      "1,3-disubstituted_imidazole",  FragmentMetadata(Int[], 2, 5,  "starter", true)),
    FragmentAction(50, "c1cc([*])c([*])s1",      "2,3-disubstituted_thiophene",  FragmentMetadata(Int[], 2, 5,  "starter", true)),
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
# BRICS Compatibility (Gap 3)
# ============================================

"""
BRICS environment compatibility matrix.
Label pairs (L1, L2) that are allowed to form bonds.
Based on RDKit's BRICS decomposition rules.
"""
const BRICS_COMPATIBLE_PAIRS = Set([
    (1, 3), (1, 5), (1, 10),
    (3, 4), (3, 13), (3, 14), (3, 16),
    (4, 5), (4, 11),
    (5, 5), (5, 6), (5, 13), (5, 15), (5, 16),
    (6, 13), (6, 14), (6, 15), (6, 16),
    (7, 7),
    (8, 9), (8, 10), (8, 11), (8, 13), (8, 14), (8, 15), (8, 16),
    (9, 15), (9, 16),
    (10, 13), (10, 14), (10, 15), (10, 16),
    (11, 13), (11, 14), (11, 15), (11, 16),
    (13, 14), (13, 15), (13, 16),
    (14, 15), (14, 16),
    (15, 16),
])

"""Check if two BRICS labels are compatible for bond formation."""
function is_brics_compatible(label1::Int, label2::Int)::Bool
    # Zero label = no BRICS info, allow any connection (legacy compatibility)
    (label1 == 0 || label2 == 0) && return true
    lo, hi = minmax(label1, label2)
    return (lo, hi) in BRICS_COMPATIBLE_PAIRS
end

# ============================================
# Fragment Library Loading (Gap 3)
# ============================================

"""
    load_fragment_library(path::String)::Vector{FragmentAction}

Load a fragment library from a JSON file. Returns a vector of FragmentAction
structs with metadata populated from the JSON data.

Expected JSON format: { "fragments": [{ "id", "smiles", "name", "brics_labels",
    "n_attachments", "heavy_atoms", "category", "is_starter" }, ...] }
"""
function load_fragment_library(path::String)::Vector{FragmentAction}
    !isfile(path) && error("Fragment library not found: $path")
    data = JSON3.read(read(path, String))

    fragments = FragmentAction[]
    for f in data["fragments"]
        meta = FragmentMetadata(
            Int[l for l in get(f, "brics_labels", [])],
            get(f, "n_attachments", 1),
            get(f, "heavy_atoms", 0),
            get(f, "category", "unknown"),
            get(f, "is_starter", false),
        )
        push!(fragments, FragmentAction(
            f["id"],
            f["smiles"],
            get(f, "name", "fragment_$(f["id"])"),
            meta,
        ))
    end

    @info "Loaded fragment library" path=path n_fragments=length(fragments)
    return fragments
end

# ============================================
# Equality & Hashing
# ============================================

# Include attachment_points in equality — same SMILES with different
# available attachments represents different positions in the DAG.
function Base.:(==)(a::MolState, b::MolState)
    return a.smiles == b.smiles &&
           a.is_terminated == b.is_terminated &&
           a.attachment_points == b.attachment_points &&
           a.attachment_labels == b.attachment_labels
end

function Base.hash(s::MolState, h::UInt)
    h = hash(s.smiles, h)
    h = hash(s.is_terminated, h)
    for ap in s.attachment_points
        h = hash(ap, h)
    end
    for al in s.attachment_labels
        h = hash(al, h)
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
Uses vcat (no mutation) for Zygote autodiff compatibility.
When BRICS labels are present, appends a 16-dim label encoding."""
function GFlowNet.state_to_features(state::MolState)::Vector{Float32}
    # [1025] Normalized attachment point count
    attach_norm = Float32(length(state.attachment_points)) / Float32(MAX_ATTACHMENTS)
    # [1026] Normalized fragment count (progress)
    frag_norm = Float32(state.n_fragments) / Float32(MAX_FRAGMENTS)
    # [1027:1042] One-hot for available attachment slots
    attach_onehot = Float32[i <= length(state.attachment_points) ? 1.0f0 : 0.0f0
                            for i in 1:MAX_ATTACHMENTS]

    base_features = vcat(state.fingerprint, Float32[attach_norm, frag_norm], attach_onehot)

    # Gap 3: BRICS label encoding (16 dims, normalized to [0,1])
    if !isempty(state.attachment_labels)
        brics_encoding = Float32[
            i <= length(state.attachment_labels) ?
                Float32(state.attachment_labels[i]) / 16.0f0 : 0.0f0
            for i in 1:MAX_ATTACHMENTS
        ]
        return vcat(base_features, brics_encoding)
    end

    return base_features
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

# ============================================
# Objective Normalization (Prerequisite C)
# ============================================

"""
    compute_all_objectives(state::MolState; oracle_mgr=nothing)::Vector{Float64}

Compute all individual objectives normalized to [0,1] range.
Used by MOGFN (Gap 5) for preference-conditioned scalarization
and by the docking reward (Gap 2) for consistent normalization.

Returns: [qed, sa_norm, logp_score, mw_score] — all in [0,1].
When a docking target is configured: [qed, sa, logp, mw, dock_score].
When oracles are configured: appends oracle scores from cache.
In benchmark_mode: returns ONLY oracle scores (PMO-compliant).
"""
function compute_all_objectives(state::MolState; oracle_mgr=nothing)::Vector{Float64}
    !state.is_terminated && return Float64[]
    isempty(state.smiles) && return Float64[]

    # In benchmark_mode, return ONLY oracle scores (PMO compliant)
    if oracle_mgr !== nothing && oracle_mgr.benchmark_mode
        return [lookup_score(oracle_mgr, state.smiles, c.name) for c in oracle_mgr.configs]
    end

    props = RDKitBridge.compute_mol_properties(state.smiles)
    props === nothing && return Float64[]

    # Base objectives (always computed — these are free/fast)
    objectives = Float64[
        props.qed,                                            # Already [0,1]
        clamp(1.0 - (props.sa_score - 1.0) / 9.0, 0.0, 1.0), # SA → [0,1]
        exp(-0.5 * ((props.logp - 2.5) / 2.5)^2),            # LogP → [0,1] Gaussian
        exp(-0.5 * ((props.mw - 350.0) / 150.0)^2),          # MW → [0,1] Gaussian
    ]

    # Gap 2: Append docking score when a target is configured
    if RDKitBridge.has_docking_target() && RDKitBridge.is_proxy_available()
        dock_score = RDKitBridge.proxy_dock(state.smiles)
        push!(objectives, dock_score)
    end

    # Oracle objectives (looked up from pre-computed cache)
    if oracle_mgr !== nothing
        for config in oracle_mgr.configs
            push!(objectives, lookup_score(oracle_mgr, state.smiles, config.name))
        end
    end

    return objectives
end

# ============================================
# Dual-Signature Reward API (Prerequisite B)
# ============================================

"""
    GFlowNet.reward(state::MolState, w::Vector{Float64})::Float64

Multi-objective reward using linear scalarization with preference vector `w`.
Used by MOGFN-PC (Gap 5) for preference-conditioned training.

R(x, w) = Σᵢ wᵢ × Rᵢ(x)

The preference vector `w` should sum to 1.0 and have the same length as
the objectives returned by `compute_all_objectives()`.

The single-argument `reward(state)` method (above) preserves the original
geometric mean formula for backward compatibility.
"""
function GFlowNet.reward(state::MolState, w::Vector{Float64})::Float64
    objectives = compute_all_objectives(state)
    isempty(objectives) && return 1e-4

    # Linear scalarization: R(x,w) = Σ wᵢ × Rᵢ(x)
    reward_val = sum(w .* objectives[1:min(length(w), length(objectives))])
    return max(reward_val, 1e-4)
end

"""
    GFlowNet.reward(state::MolState, w::Vector{Float64}, oracle_mgr)::Float64

Oracle-aware variant that passes oracle_mgr to compute_all_objectives.
Used when oracles are configured for target-specific optimization.
"""
function GFlowNet.reward(state::MolState, w::Vector{Float64}, oracle_mgr)::Float64
    objectives = compute_all_objectives(state; oracle_mgr=oracle_mgr)
    isempty(objectives) && return 1e-4

    reward_val = sum(w .* objectives[1:min(length(w), length(objectives))])
    return max(reward_val, 1e-4)
end

"""Check if a fragment action is applicable from the current state.
Uses metadata-based starter check (Gap 3) with fallback to legacy id-based check."""
function GFlowNet.is_applicable(action::FragmentAction, state::MolState)::Bool
    state.is_terminated && return false
    state.n_fragments >= MAX_FRAGMENTS && return false

    # Determine if this fragment is a starter
    has_metadata = !isempty(action.metadata.category) && action.metadata.category != "unknown"
    is_starter = has_metadata ? action.metadata.is_starter : (action.fragment_id >= 41)

    if isempty(state.smiles)
        # First fragment: only starter fragments
        return is_starter
    else
        # Non-starter fragments only after first fragment
        is_starter && return false
        # Must have attachment points
        isempty(state.attachment_points) && return false
        # BRICS compatibility check (if labels available)
        if !isempty(state.attachment_labels) && !isempty(action.metadata.brics_labels)
            return any(
                is_brics_compatible(mol_label, frag_label)
                for mol_label in state.attachment_labels
                for frag_label in action.metadata.brics_labels
            )
        end
        return true
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
Propagates BRICS attachment labels when available (Gap 3).
"""
function GFlowNet.apply_action(action::FragmentAction, state::MolState)::MolState
    try
        if isempty(state.smiles)
            new_smiles, new_attachments = RDKitBridge.place_first_fragment(action.fragment_smiles)
            # For first fragment, infer labels from fragment metadata
            new_labels = if !isempty(action.metadata.brics_labels) && length(action.metadata.brics_labels) >= length(new_attachments)
                action.metadata.brics_labels[1:length(new_attachments)]
            else
                Int[]
            end
        else
            attach_idx = state.attachment_points[1]
            new_smiles, new_attachments = RDKitBridge.join_fragment(
                state.smiles, action.fragment_smiles, attach_idx
            )
            # Propagate remaining labels: remove the used attachment label,
            # keep rest, and add labels from the new fragment's remaining attachments
            new_labels = Int[]
            if !isempty(state.attachment_labels)
                # Remove the label at position 1 (used for joining), keep the rest
                remaining_mol_labels = length(state.attachment_labels) > 1 ?
                    state.attachment_labels[2:end] : Int[]
                # Fragment may contribute new labels for its remaining attachment points
                frag_new_labels = if action.metadata.n_attachments > 1 && !isempty(action.metadata.brics_labels)
                    action.metadata.brics_labels[2:min(end, action.metadata.n_attachments)]
                else
                    Int[]
                end
                new_labels = vcat(remaining_mol_labels, frag_new_labels)
            end
        end

        # Validate result
        if isempty(new_smiles) || !RDKitBridge.validate_smiles(new_smiles)
            @warn "Fragment join produced invalid SMILES, returning current state" action=action.fragment_name
            return state
        end

        new_fp = RDKitBridge.compute_fingerprint(new_smiles)
        return MolState(new_smiles, new_attachments, new_labels, state.n_fragments + 1, false, new_fp)
    catch e
        @warn "apply_action failed" exception=e action=action.fragment_name smiles=state.smiles
        return state
    end
end

function GFlowNet.apply_action(action::TerminateMolAction, state::MolState)::MolState
    final_smiles = RDKitBridge.finalize_smiles(state.smiles)
    final_fp = RDKitBridge.compute_fingerprint(final_smiles)
    return MolState(final_smiles, Int[], Int[], state.n_fragments, true, final_fp)
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
- `mol_config::Dict=Dict()`: Molecular config for dynamic state dim
  - `"use_brics_labels"`: Enable BRICS label features (+16 dims)
  - `"use_preferences"`: Enable preference embedding (+64 dims)
  - `"preference_embed_dim"`: Preference embedding size (default: 64)
- `fragment_library`: Fragment library to use (default: FRAGMENT_LIBRARY)
- `rng`: Random number generator
"""
function create_molecular_gflownet(;
    hidden_dim::Int = 256,
    learning_rate::Float64 = 0.001,
    include_backward::Bool = false,
    include_flow_estimator::Bool = false,
    partition_function_method::PartitionFunctionMethod = LEARNABLE_ESTIMATION,
    mol_config::Dict = Dict(),
    fragment_library::Vector{FragmentAction} = FRAGMENT_LIBRARY,
    rng = Random.default_rng()
)
    initial_state = MolState("", Int[], 0, false, zeros(Float32, FINGERPRINT_DIM))
    all_actions = AbstractAction[fragment_library..., TerminateMolAction()]

    state_dim = compute_state_dim(mol_config)

    return GFlowNet.create_gflownet(
        initial_state,
        all_actions;
        state_dim = state_dim,
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        include_backward = include_backward,
        include_flow_estimator = include_flow_estimator,
        partition_function_method = partition_function_method,
        rng = rng
    )
end

# ============================================
# MOGFN-PC Model Factory (Gap 5)
# ============================================

"""
    create_mogfn_molecular_gflownet(; kwargs...)

Create a MOGFN-PC (preference-conditioned) GFlowNet for fragment-based molecular generation.

The model conditions on a preference vector w ∈ Δ^K (K-simplex) so that a single
trained model generates molecules optimized for ANY preference weighting at inference time.

# Arguments
- `hidden_dim::Int=256`: Hidden layer size
- `learning_rate::Float64=0.001`: Learning rate
- `n_objectives::Int=4`: Number of objectives (QED, SA, LogP, MW)
- `preference_dim::Int=64`: Preference embedding dimension
- `include_backward::Bool=false`: Include backward policy
- `mol_config::Dict=Dict()`: Molecular config (use_brics_labels, etc.)
- `fragment_library`: Fragment library to use
- `rng`: Random number generator
"""
function create_mogfn_molecular_gflownet(;
    hidden_dim::Int = 256,
    learning_rate::Float64 = 0.001,
    n_objectives::Int = 4,
    preference_dim::Int = 64,
    include_backward::Bool = false,
    mol_config::Dict = Dict(),
    fragment_library::Vector{FragmentAction} = FRAGMENT_LIBRARY,
    rng = Random.default_rng()
)
    initial_state = MolState("", Int[], 0, false, zeros(Float32, FINGERPRINT_DIM))
    all_actions = AbstractAction[fragment_library..., TerminateMolAction()]

    # Base state dim (without preference embedding — that's handled by create_mogfn_gflownet)
    base_state_dim = compute_state_dim(mol_config)

    return GFlowNet.create_mogfn_gflownet(
        initial_state,
        all_actions;
        state_dim = base_state_dim,
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        n_objectives = n_objectives,
        preference_dim = preference_dim,
        include_backward = include_backward,
        rng = rng
    )
end

# ============================================
# Dirichlet Preference Sampling (Gap 5)
# ============================================

"""
    sample_preference(n_objectives::Int; alpha::Float64=1.0)::Vector{Float64}

Sample a preference vector w from Dirichlet(α) distribution.
For α = 1.0, this gives uniform coverage of the K-simplex.

Uses Gamma-based sampling: draw xᵢ ~ Gamma(α, 1), then normalize.
For α = 1.0, Gamma(1,1) = Exp(1), so we use `randexp()`.
"""
function sample_preference(n_objectives::Int; alpha::Float64=1.0)::Vector{Float64}
    if alpha == 1.0
        # Dirichlet(1,...,1) = uniform on simplex
        # Gamma(1,1) = Exponential(1)
        gammas = [randexp() for _ in 1:n_objectives]
    else
        # General Dirichlet via Gamma sampling
        # Gamma(α, 1) can be approximated for non-unit alpha
        # Using acceptance-rejection for general α
        gammas = Float64[]
        for _ in 1:n_objectives
            g = _sample_gamma(alpha)
            push!(gammas, g)
        end
    end
    total = sum(gammas)
    return gammas ./ total
end

"""Sample from Gamma(alpha, 1) distribution using Marsaglia & Tsang's method."""
function _sample_gamma(alpha::Float64)::Float64
    if alpha >= 1.0
        d = alpha - 1.0/3.0
        c = 1.0 / sqrt(9.0 * d)
        while true
            x = randn()
            v = (1.0 + c * x)^3
            if v > 0.0
                u = rand()
                if u < 1.0 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
                    return d * v
                end
            end
        end
    else
        # For alpha < 1, use the relation: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        return _sample_gamma(alpha + 1.0) * rand()^(1.0 / alpha)
    end
end
