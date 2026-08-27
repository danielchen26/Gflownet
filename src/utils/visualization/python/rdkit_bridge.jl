# RDKit Bridge — Julia ↔ Python via PythonCall.jl
#
# Single point of contact between Julia and Python.
# All RDKit molecular operations go through this module.
#
# IMPORTANT: This module uses explicit init_rdkit!() instead of __init__()
# because unified_server.jl loads files via include() not using/import.

module RDKitBridge

using PythonCall
using JSON3

# Python modules (lazy-loaded via init_rdkit!)
const rdkit_chem    = Ref{Py}()
const rdkit_desc    = Ref{Py}()
const rdkit_mold    = Ref{Py}()  # rdMolDescriptors
const rdkit_qed     = Ref{Py}()
const rdkit_allchem  = Ref{Py}()
const rdkit_draw    = Ref{Py}()
const np            = Ref{Py}()
const sascorer_mod  = Ref{Py}()
const _initialized  = Ref{Bool}(false)

# Python helper functions (loaded once)
const _py_place_fragment = Ref{Py}()
const _py_join_fragment  = Ref{Py}()
const _py_finalize       = Ref{Py}()
const _py_batch_fp       = Ref{Py}()
const _py_atom_contrib   = Ref{Py}()

# Gap 2: Docking helpers
const _py_prepare_ligand  = Ref{Py}()
const _py_dock_molecule   = Ref{Py}()
const _py_dock_batch      = Ref{Py}()
const _py_proxy_predict   = Ref{Py}()
const _py_proxy_train     = Ref{Py}()
const _docking_available  = Ref{Bool}(false)
const _proxy_available    = Ref{Bool}(false)

# Docking target configuration
const _docking_target     = Ref{String}("")
const _docking_targets    = Ref{Dict{String,Any}}(Dict{String,Any}())

"""
    init_rdkit!()

Initialize Python/RDKit modules. Must be called once at server startup.
"""
function init_rdkit!()
    _initialized[] && return

    @info "Initializing RDKitBridge..."

    rdkit_chem[]    = pyimport("rdkit.Chem")
    rdkit_desc[]    = pyimport("rdkit.Chem.Descriptors")
    rdkit_mold[]    = pyimport("rdkit.Chem.rdMolDescriptors")
    rdkit_qed[]     = pyimport("rdkit.Chem.QED")
    rdkit_allchem[] = pyimport("rdkit.Chem.AllChem")
    rdkit_draw[]    = pyimport("rdkit.Chem.Draw")
    np[]            = pyimport("numpy")

    # SA Score — try contrib sascorer first, then fallback
    try
        sascorer_mod[] = pyimport("rdkit.Contrib.SA_Score.sascorer")
        @info "SA Score: using rdkit.Contrib.SA_Score.sascorer"
    catch
        try
            sascorer_mod[] = pyimport("sascorer")
            @info "SA Score: using standalone sascorer module"
        catch
            sascorer_mod[] = pybuiltins.None
            @warn "SA Score: no sascorer found, using descriptor-based approximation"
        end
    end

    # Compile Python helper functions once
    _compile_python_helpers!()

    # Gap 2: Try to initialize docking tools (optional)
    try
        _compile_docking_helpers!()
        _docking_available[] = true
        @info "Docking tools (meeko + vina) available"
    catch e
        _docking_available[] = false
        @info "Docking tools not available (optional): $(sprint(showerror, e))"
    end

    # Load target configurations
    _load_docking_targets!()

    _initialized[] = true
    @info "RDKitBridge initialized successfully"
end

function _ensure_init()
    _initialized[] || error("RDKitBridge not initialized! Call init_rdkit!() first.")
end

"""Compile Python helper functions for fragment operations."""
function _compile_python_helpers!()
    # Define all helper functions in Python's __main__ module
    main_mod = pyimport("__main__")
    pyexec("""
from rdkit import Chem
from rdkit.Chem import AllChem

def _place_fragment(frag_smarts):
    mol = Chem.MolFromSmarts(frag_smarts)
    if mol is None:
        return ('', [])

    # Find dummy atom indices ([#0] or [*])
    dummy_indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_indices.append(atom.GetIdx())

    # Try to sanitize; SMARTS-derived mols may need special handling
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # For SMARTS with wildcards, convert to SMILES representation
        pass

    smiles = Chem.MolToSmiles(mol)
    if not smiles:
        return ('', [])

    # Re-parse to find canonical dummy positions
    new_mol = Chem.MolFromSmiles(smiles)
    if new_mol is None:
        return (smiles, dummy_indices)

    new_dummy = []
    for atom in new_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            new_dummy.append(atom.GetIdx())

    return (smiles, new_dummy)

def _join_fragment(mol_smi, frag_smarts, attach_idx):
    mol = Chem.MolFromSmiles(mol_smi)
    frag = Chem.MolFromSmarts(frag_smarts)

    if mol is None or frag is None:
        return (mol_smi, [])

    # Find the dummy atom at attach_idx in mol
    mol_dummy_idx = None
    dummy_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            if dummy_count == attach_idx:
                mol_dummy_idx = atom.GetIdx()
                break
            dummy_count += 1

    if mol_dummy_idx is None:
        return (mol_smi, [])

    # Find first dummy atom in fragment
    frag_dummy_idx = None
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            frag_dummy_idx = atom.GetIdx()
            break

    if frag_dummy_idx is None:
        return (mol_smi, [])

    # Get the neighbor of each dummy (the real atom it's bonded to)
    mol_dummy_atom = mol.GetAtomWithIdx(mol_dummy_idx)
    frag_dummy_atom = frag.GetAtomWithIdx(frag_dummy_idx)

    if len(mol_dummy_atom.GetNeighbors()) == 0 or len(frag_dummy_atom.GetNeighbors()) == 0:
        return (mol_smi, [])

    mol_neighbor = mol_dummy_atom.GetNeighbors()[0].GetIdx()
    frag_neighbor = frag_dummy_atom.GetNeighbors()[0].GetIdx()

    # Combine molecules
    combo = Chem.RWMol(Chem.CombineMols(mol, frag))

    # Adjust fragment indices (shifted by mol atom count)
    n_mol_atoms = mol.GetNumAtoms()
    frag_neighbor_in_combo = frag_neighbor + n_mol_atoms
    frag_dummy_in_combo = frag_dummy_idx + n_mol_atoms

    # Add bond between the real neighbors
    combo.AddBond(mol_neighbor, frag_neighbor_in_combo, Chem.BondType.SINGLE)

    # Remove dummy atoms (remove higher index first to preserve indices)
    dummies_to_remove = sorted([mol_dummy_idx, frag_dummy_in_combo], reverse=True)
    for idx in dummies_to_remove:
        combo.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(combo)
        new_smiles = Chem.MolToSmiles(combo)
    except Exception:
        return (mol_smi, [])

    # Find remaining dummy atoms (attachment points)
    new_mol = Chem.MolFromSmiles(new_smiles)
    if new_mol is None:
        return (new_smiles, [])

    new_dummies = []
    for atom in new_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            new_dummies.append(atom.GetIdx())

    return (new_smiles, new_dummies)

def _finalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi

    rw = Chem.RWMol(mol)

    # Find all dummy atoms
    dummies = []
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummies.append(atom.GetIdx())

    # Remove in reverse order to preserve indices
    for idx in sorted(dummies, reverse=True):
        rw.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return smi

def _batch_fingerprints(smiles_list, nbits=1024, radius=2):
    results = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            results.append(list(fp))
        else:
            results.append([0] * nbits)
    return results

def _atom_attribution(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    # Use Morgan fingerprint bit info for atom contribution
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=bi)
    n_atoms = mol.GetNumAtoms()
    scores = [0.0] * n_atoms
    for bit_id, atom_envs in bi.items():
        for atom_idx, radius in atom_envs:
            if atom_idx < n_atoms:
                scores[atom_idx] += 1.0 / max(len(bi), 1)
    # Normalize to [0, 1]
    max_score = max(scores) if scores else 1.0
    if max_score > 0:
        scores = [s / max_score for s in scores]
    return scores
""", main_mod.__dict__)

    # Capture references to the compiled Python functions
    _py_place_fragment[] = main_mod._place_fragment
    _py_join_fragment[]  = main_mod._join_fragment
    _py_finalize[]       = main_mod._finalize
    _py_batch_fp[]       = main_mod._batch_fingerprints
    _py_atom_contrib[]   = main_mod._atom_attribution
end

# ============================================
# SMILES Validation
# ============================================

function validate_smiles(smiles::String)::Bool
    _ensure_init()
    mol = rdkit_chem[].MolFromSmiles(smiles)
    return !pyis(mol, pybuiltins.None)
end

function validate_smarts(smarts::String)::Bool
    _ensure_init()
    mol = rdkit_chem[].MolFromSmarts(smarts)
    return !pyis(mol, pybuiltins.None)
end

function canonicalize_smiles(smiles::String)::Union{String, Nothing}
    _ensure_init()
    mol = rdkit_chem[].MolFromSmiles(smiles)
    pyis(mol, pybuiltins.None) && return nothing
    return pyconvert(String, rdkit_chem[].MolToSmiles(mol))
end

# ============================================
# Fingerprint Computation
# ============================================

function compute_fingerprint(smiles::String)::Vector{Float32}
    _ensure_init()
    mol = rdkit_chem[].MolFromSmiles(smiles)
    if pyis(mol, pybuiltins.None)
        return zeros(Float32, 1024)
    end
    fp = rdkit_allchem[].GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np[].array(fp)
    return Float32.(pyconvert(Vector{Int}, arr))
end

"""Batch fingerprint computation — single Python call for multiple SMILES."""
function compute_fingerprints_batch(smiles_list::Vector{String})::Vector{Vector{Float32}}
    _ensure_init()
    isempty(smiles_list) && return Vector{Float32}[]

    py_result = _py_batch_fp[](pylist(smiles_list))

    n = length(smiles_list)
    results = Vector{Vector{Float32}}(undef, n)
    for i in 1:n
        results[i] = Float32.(pyconvert(Vector{Int}, py_result[i-1]))
    end
    return results
end

# ============================================
# Property Computation
# ============================================

struct MolProperties
    mw::Float64
    logp::Float64
    qed::Float64
    sa_score::Float64
    tpsa::Float64
    hbd::Int
    hba::Int
    rotatable_bonds::Int
    num_rings::Int
    num_aromatic_rings::Int
    formula::String
end

function compute_mol_properties(smiles::String)::Union{MolProperties, Nothing}
    _ensure_init()
    mol = rdkit_chem[].MolFromSmiles(smiles)
    pyis(mol, pybuiltins.None) && return nothing

    mw    = pyconvert(Float64, rdkit_desc[].MolWt(mol))
    logp  = pyconvert(Float64, rdkit_desc[].MolLogP(mol))
    qed   = pyconvert(Float64, rdkit_qed[].qed(mol))
    tpsa  = pyconvert(Float64, rdkit_desc[].TPSA(mol))
    hbd   = pyconvert(Int, rdkit_desc[].NumHDonors(mol))
    hba   = pyconvert(Int, rdkit_desc[].NumHAcceptors(mol))
    rot   = pyconvert(Int, rdkit_desc[].NumRotatableBonds(mol))
    rings = pyconvert(Int, rdkit_desc[].RingCount(mol))
    arom  = pyconvert(Int, rdkit_desc[].NumAromaticRings(mol))
    form  = pyconvert(String, rdkit_mold[].CalcMolFormula(mol))

    sa = _compute_sa_score(mol)

    return MolProperties(mw, logp, qed, sa, tpsa, hbd, hba, rot, rings, arom, form)
end

function _compute_sa_score(mol::Py)::Float64
    if !pyis(sascorer_mod[], pybuiltins.None)
        return pyconvert(Float64, sascorer_mod[].calculateScore(mol))
    else
        # Fallback: approximate SA from descriptors
        nRings  = pyconvert(Int, rdkit_desc[].RingCount(mol))
        nHeavy  = pyconvert(Int, rdkit_desc[].HeavyAtomCount(mol))
        sa = 1.0 + 0.5 * nRings + 0.02 * nHeavy
        return clamp(sa, 1.0, 10.0)
    end
end

# ============================================
# Fragment Joining
# ============================================

"""Place the first fragment. Returns (smiles, attachment_point_indices)."""
function place_first_fragment(fragment_smiles::String)::Tuple{String, Vector{Int}}
    _ensure_init()

    result = _py_place_fragment[](fragment_smiles)
    smiles = pyconvert(String, result[0])
    attachments = pyconvert(Vector{Int}, result[1])
    return (smiles, attachments)
end

"""Join a fragment to the molecule at the specified attachment point index."""
function join_fragment(mol_smiles::String, frag_smiles::String, attach_idx::Int)::Tuple{String, Vector{Int}}
    _ensure_init()

    result = _py_join_fragment[](mol_smiles, frag_smiles, attach_idx)
    smiles = pyconvert(String, result[0])
    attachments = pyconvert(Vector{Int}, result[1])
    return (smiles, attachments)
end

"""Remove remaining [*] atoms and canonicalize SMILES."""
function finalize_smiles(smiles::String)::String
    _ensure_init()
    result = _py_finalize[](smiles)
    return pyconvert(String, result)
end

# ============================================
# SVG Rendering
# ============================================

function mol_to_svg(smiles::String; width=300, height=200)::Union{String, Nothing}
    _ensure_init()
    mol = rdkit_chem[].MolFromSmiles(smiles)
    pyis(mol, pybuiltins.None) && return nothing
    rdkit_allchem[].Compute2DCoords(mol)
    drawer = rdkit_draw[].MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return pyconvert(String, drawer.GetDrawingText())
end

# ============================================
# Atom Attribution
# ============================================

"""Compute per-atom contribution scores based on Morgan fingerprint bit info."""
function compute_atom_attribution(smiles::String)::Vector{Float64}
    _ensure_init()
    result = _py_atom_contrib[](smiles)
    return pyconvert(Vector{Float64}, result)
end

# ============================================
# ADMET Computation (matches frontend ADMETData interface)
# ============================================

function compute_admet(smiles::String)::Dict
    _ensure_init()
    props = compute_mol_properties(smiles)
    props === nothing && return Dict("error" => "Invalid SMILES: $smiles")

    return Dict(
        "absorption" => Dict(
            "oral_bioavailability" => _estimate_oral_bioavailability(props),
            "caco2_permeability"   => _estimate_caco2(props),
            "pgp_substrate"        => props.mw > 400 && props.hbd > 2,
        ),
        "distribution" => Dict(
            "vd"                     => 0.5 + 0.3 * props.logp,
            "plasma_protein_binding" => clamp(70.0 + 5.0 * props.logp, 50.0, 99.0),
            "bbb_penetration"        => props.tpsa < 90 && props.mw < 450,
        ),
        "metabolism" => Dict(
            "cyp2d6_inhibitor" => props.logp > 3.0 && props.num_aromatic_rings >= 2,
            "cyp3a4_inhibitor" => props.mw > 350 && props.logp > 3.5,
            "half_life_hours"  => clamp(2.0 + props.logp * 1.5 - props.tpsa * 0.02, 0.5, 24.0),
        ),
        "excretion" => Dict(
            "clearance"       => clamp(10.0 + 3.0 * props.logp - 0.05 * props.mw, 1.0, 50.0),
            "renal_excretion" => props.mw < 300 && props.logp < 1.0,
        ),
        "toxicity" => Dict(
            "herg_inhibition"     => props.logp > 3.5 && props.mw > 400,
            "ames_mutagenicity"   => false,
            "hepatotoxicity_risk" => _estimate_hepatotox_risk(props),
        ),
    )
end

function _estimate_oral_bioavailability(p::MolProperties)::Float64
    lipinski = (p.mw <= 500 && p.logp <= 5 && p.hbd <= 5 && p.hba <= 10)
    base = lipinski ? 0.7 : 0.3
    return clamp(base + p.qed * 0.2, 0.0, 1.0)
end

function _estimate_caco2(p::MolProperties)::Float64
    return -5.5 + 0.3 * p.logp - 0.01 * p.tpsa
end

function _estimate_hepatotox_risk(p::MolProperties)::String
    score = 0.0
    p.logp > 3.0 && (score += 1.0)
    p.mw > 400 && (score += 0.5)
    p.num_rings > 4 && (score += 0.5)
    return score >= 1.5 ? "high" : score >= 0.5 ? "medium" : "low"
end

# ============================================
# Chemical Space Projection
# ============================================

function compute_projection(fingerprints::Vector{Vector{Float32}}, method::String)::Vector{Dict}
    _ensure_init()
    isempty(fingerprints) && return Dict[]

    n = length(fingerprints)
    # Need at least 2 points for dimensionality reduction
    n < 2 && return [Dict("x" => 0.0, "y" => 0.0)]

    # Cross the PythonCall boundary O(1) times instead of O(n*1024).
    #
    # This built the input as `np.array(pylist(Float64.(vcat(fingerprints...))))`,
    # which converts EVERY scalar into a separate Python float: n*1024 crossings,
    # 204,800 for n=200 and 1,949,696 for n=1904. It also splatted n arrays into
    # vcat and allocated the matrix three times over (the caller narrows to
    # Float32, this widened back to Float64).
    #
    # A dense Julia Matrix handed to numpy travels through the buffer protocol as
    # a single memcpy. Values, shape and dtype are identical, so the reducer sees
    # byte-identical input and the output is unchanged.
    #
    # NOTE ON LAYOUT: Julia is column-major and numpy expects row-major, so the
    # matrix is built TRANSPOSED as (1024, n) and transposed on the Python side.
    # Building (n, 1024) directly and passing it would silently transpose the data
    # and project garbage.
    mat = Matrix{Float64}(undef, 1024, n)
    @inbounds for j in 1:n
        fp = fingerprints[j]
        length(fp) == 1024 || throw(ArgumentError(
            "fingerprint $j has length $(length(fp)), expected 1024"))
        for i in 1:1024
            mat[i, j] = Float64(fp[i])
        end
    end
    py_arr = np[].asarray(Py(mat)).T

    coords = if method == "umap" && n >= 5
        umap_mod = pyimport("umap")
        n_neighbors = min(15, n - 1)
        # random_state pins the embedding. Without it two identical requests
        # returned DIFFERENT coordinates, so the scatter visibly jittered on every
        # 10 s refetch, and no caching of the result could have been sound.
        reducer = umap_mod.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                                n_components=2, random_state=42)
        reducer.fit_transform(py_arr)
    elseif method == "tsne" && n >= 5
        sklearn_manifold = pyimport("sklearn.manifold")
        perplexity = min(30.0, Float64(n - 1))
        reducer = sklearn_manifold.TSNE(n_components=2, perplexity=perplexity,
                                        random_state=42)
        reducer.fit_transform(py_arr)
    else  # pca (default)
        sklearn_decomp = pyimport("sklearn.decomposition")
        reducer = sklearn_decomp.PCA(n_components=2)
        reducer.fit_transform(py_arr)
    end

    # One conversion of the whole (n, 2) result instead of 4n scalar crossings.
    c = pyconvert(Matrix{Float64}, np[].asarray(coords, dtype=np[].float64))
    points = Vector{Dict}(undef, n)
    @inbounds for i in 1:n
        points[i] = Dict("x" => c[i, 1], "y" => c[i, 2])
    end
    return points
end

# ============================================
# Gap 3: BRICS-Aware Fragment Joining
# ============================================

"""
    join_fragment_brics(mol_smiles, frag_smiles, attach_idx, mol_label, frag_label)

Join a fragment to a molecule with BRICS label awareness.
Returns (new_smiles, new_positions, new_labels) where new_labels
are the BRICS labels for each remaining attachment point.
"""
function join_fragment_brics(mol_smiles::String, frag_smiles::String,
                             attach_idx::Int, mol_label::Int, frag_label::Int)
    _ensure_init()

    # Use standard join — BRICS compatibility already checked in is_applicable
    new_smiles, new_positions = join_fragment(mol_smiles, frag_smiles, attach_idx)

    # Parse remaining dummy atoms for their isotope labels (BRICS convention)
    new_labels = Int[]
    if !isempty(new_smiles)
        try
            mol = rdkit_chem[].MolFromSmiles(new_smiles)
            if !pyis(mol, pybuiltins.None)
                for atom in mol.GetAtoms()
                    if pyconvert(Int, atom.GetAtomicNum()) == 0
                        isotope = pyconvert(Int, atom.GetIsotope())
                        push!(new_labels, isotope)
                    end
                end
            end
        catch
            # Fall back to empty labels
        end
    end

    return new_smiles, new_positions, new_labels
end

# ============================================
# Gap 1: Tanimoto Diversity Metrics
# ============================================

"""
    compute_tanimoto_matrix(fps::Vector{Vector{Float32}})::Matrix{Float64}

Compute pairwise Tanimoto similarity matrix from fingerprint vectors.
Uses RDKit DataStructs.BulkTanimotoSimilarity for C++-optimized computation.
For N molecules, returns N×N symmetric similarity matrix.
"""
function compute_tanimoto_matrix(fps::Vector{Vector{Float32}})::Matrix{Float64}
    _ensure_init()
    n = length(fps)
    n == 0 && return Matrix{Float64}(undef, 0, 0)

    py_ds = pyimport("rdkit.DataStructs")
    py_fps = Py[]

    # Convert Float32 vectors to RDKit ExplicitBitVect
    for fp in fps
        ebv = py_ds.ExplicitBitVect(length(fp))
        for (i, bit) in enumerate(fp)
            if bit > 0.5f0
                ebv.SetBit(i - 1)  # 0-indexed
            end
        end
        push!(py_fps, ebv)
    end

    # Compute full pairwise matrix
    sim_matrix = zeros(Float64, n, n)
    for i in 1:n
        sim_matrix[i, i] = 1.0
        if i < n
            # BulkTanimotoSimilarity returns similarities for fp[i] vs all fps[i+1:end]
            remaining = pylist(py_fps[i:end])
            sims = py_ds.BulkTanimotoSimilarity(py_fps[i], remaining)
            for j in 1:length(py_fps)-i+1
                sim_val = pyconvert(Float64, sims[j-1])
                sim_matrix[i, i+j-1] = sim_val
                sim_matrix[i+j-1, i] = sim_val
            end
        end
    end
    return sim_matrix
end

"""
    compute_diversity_stats(fps::Vector{Vector{Float32}})::Dict

Compute diversity statistics from fingerprint vectors.
Returns: mean_pairwise_tanimoto, internal_diversity_1, internal_diversity_2,
         nearest_neighbor distances.
"""
function compute_diversity_stats(fps::Vector{Vector{Float32}})::Dict
    _ensure_init()
    n = length(fps)
    n < 2 && return Dict(
        "mean_pairwise" => 0.0,
        "internal_diversity_1" => 1.0,
        "internal_diversity_2" => 1.0,
        "min_nn_distance" => 0.0,
        "max_nn_distance" => 0.0,
        "median_nn_distance" => 0.0,
        "n_molecules" => n,
    )

    # For large sets, sample to avoid O(n²) explosion
    if n > 1000
        sample_idx = sort(randperm(n)[1:1000])
        sample_fps = fps[sample_idx]
    else
        sample_fps = fps
    end

    sim_matrix = compute_tanimoto_matrix(sample_fps)
    ns = size(sim_matrix, 1)

    # Mean pairwise Tanimoto (excluding diagonal)
    total_sim = 0.0
    n_pairs = 0
    nn_distances = Float64[]

    for i in 1:ns
        min_dist = 1.0  # distance = 1 - similarity
        for j in 1:ns
            if i != j
                total_sim += sim_matrix[i, j]
                n_pairs += 1
                dist = 1.0 - sim_matrix[i, j]
                min_dist = min(min_dist, dist)
            end
        end
        push!(nn_distances, min_dist)
    end

    mean_sim = n_pairs > 0 ? total_sim / n_pairs : 0.0
    sort!(nn_distances)

    # IntDiv1 = 1 - mean_pairwise_tanimoto
    # IntDiv2 = 1 - sqrt(mean(sim²))
    total_sim_sq = 0.0
    for i in 1:ns, j in 1:ns
        if i != j
            total_sim_sq += sim_matrix[i, j]^2
        end
    end
    mean_sim_sq = n_pairs > 0 ? total_sim_sq / n_pairs : 0.0

    return Dict(
        "mean_pairwise" => mean_sim,
        "internal_diversity_1" => 1.0 - mean_sim,
        "internal_diversity_2" => 1.0 - sqrt(mean_sim_sq),
        "min_nn_distance" => isempty(nn_distances) ? 0.0 : nn_distances[1],
        "max_nn_distance" => isempty(nn_distances) ? 0.0 : nn_distances[end],
        "median_nn_distance" => isempty(nn_distances) ? 0.0 : nn_distances[div(length(nn_distances)+1, 2)],
        "n_molecules" => n,
    )
end

"""
    compute_scaffold_diversity(smiles_list::Vector{String})::Dict

Compute Bemis-Murcko scaffold diversity from a list of SMILES.
Returns: n_unique_scaffolds, scaffold_entropy, scaffold_distribution.
"""
function compute_scaffold_diversity(smiles_list::Vector{String})::Dict
    _ensure_init()
    isempty(smiles_list) && return Dict(
        "n_unique_scaffolds" => 0,
        "scaffold_entropy" => 0.0,
        "scaffold_distribution" => Dict{String,Int}(),
    )

    murcko = pyimport("rdkit.Chem.Scaffolds.MurckoScaffold")
    scaffold_counts = Dict{String,Int}()

    for smi in smiles_list
        mol = rdkit_chem[].MolFromSmiles(smi)
        pyis(mol, pybuiltins.None) && continue

        try
            core = murcko.GetScaffoldForMol(mol)
            scaffold_smi = pyconvert(String, rdkit_chem[].MolToSmiles(core))
            scaffold_counts[scaffold_smi] = get(scaffold_counts, scaffold_smi, 0) + 1
        catch
            # Some molecules may not have a Murcko scaffold
            continue
        end
    end

    n_unique = length(scaffold_counts)
    total = sum(values(scaffold_counts))

    # Shannon entropy
    entropy = 0.0
    if total > 0 && n_unique > 1
        for count in values(scaffold_counts)
            p = count / total
            if p > 0
                entropy -= p * log2(p)
            end
        end
    end

    return Dict(
        "n_unique_scaffolds" => n_unique,
        "scaffold_entropy" => entropy,
        "scaffold_distribution" => scaffold_counts,
    )
end

"""
    compute_nearest_neighbors(fps::Vector{Vector{Float32}}, k::Int=5)::Vector{Dict}

Compute k-nearest neighbors for each molecule based on Tanimoto similarity.
Returns vector of {id, neighbor_id, similarity} for top-k pairs.
"""
function compute_nearest_neighbors(fps::Vector{Vector{Float32}}, ids::Vector{String}; k::Int=5)::Vector{Dict}
    _ensure_init()
    n = length(fps)
    n < 2 && return Dict[]

    sim_matrix = compute_tanimoto_matrix(fps)
    results = Dict[]

    for i in 1:n
        # Get similarities to all other molecules
        sims = [(j, sim_matrix[i, j]) for j in 1:n if j != i]
        sort!(sims, by=x -> -x[2])  # Sort by descending similarity

        for (j, sim) in sims[1:min(k, length(sims))]
            push!(results, Dict(
                "id" => ids[i],
                "neighbor_id" => ids[j],
                "similarity" => sim,
            ))
        end
    end

    return results
end

# ============================================
# Gap 2: Docking-Based Reward Integration
# ============================================

"""Load docking target configurations from data/targets/targets.json."""
function _load_docking_targets!()
    targets_file = joinpath(@__DIR__, "..", "..", "..", "..", "data", "targets", "targets.json")
    if isfile(targets_file)
        try
            data = JSON3.read(read(targets_file, String))
            for target in data.targets
                _docking_targets[][target.id] = Dict{String,Any}(
                    "id" => target.id,
                    "name" => target.name,
                    "pdb_id" => target.pdb_id,
                    "center" => collect(target.center),
                    "box_size" => collect(target.box_size),
                    "exhaustiveness" => target.exhaustiveness,
                    "description" => get(target, :description, ""),
                    "known_inhibitor_smiles" => get(target, :known_inhibitor_smiles, ""),
                )
            end
            default = get(data, :default_target, "seh")
            if haskey(_docking_targets[], default)
                _docking_target[] = default
            end
            @info "Loaded $(length(_docking_targets[])) docking targets" default=_docking_target[]
        catch e
            @warn "Failed to load docking targets: $e"
        end
    end
end

"""Compile Python docking helper functions (meeko + vina)."""
function _compile_docking_helpers!()
    main_mod = pyimport("__main__")

    pyexec("""
import os, sys, json, time
from rdkit import Chem
from rdkit.Chem import AllChem

# Try importing docking tools
try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina
    _DOCKING_OK = True
except ImportError:
    _DOCKING_OK = False

# Proxy model (sklearn MLP, trained on fingerprint → docking score)
_proxy_model = None
_proxy_scaler = None

def _prepare_ligand(smiles):
    \"\"\"Convert SMILES to PDBQT string via meeko.\"\"\"
    if not _DOCKING_OK:
        return {'error': 'Docking tools not installed'}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': f'Invalid SMILES: {smiles}'}

    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception as e:
        return {'error': f'3D embedding failed: {str(e)}'}

    try:
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        if not mol_setups:
            return {'error': 'Meeko preparation failed'}
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            return {'error': f'PDBQT conversion failed: {error_msg}'}
        return {'pdbqt': pdbqt_string}
    except Exception as e:
        return {'error': f'Ligand preparation failed: {str(e)}'}

def _dock_molecule(smiles, receptor_path, center, box_size, exhaustiveness=8, n_poses=5):
    \"\"\"Dock a single molecule against a target receptor.\"\"\"
    if not _DOCKING_OK:
        return {'error': 'Docking tools not installed'}

    start_time = time.time()

    # Prepare ligand
    lig_result = _prepare_ligand(smiles)
    if 'error' in lig_result:
        return lig_result

    if not os.path.isfile(receptor_path):
        return {'error': f'Receptor file not found: {receptor_path}'}

    try:
        v = Vina(sf_name='vina', verbosity=0)
        v.set_receptor(receptor_path)
        v.set_ligand_from_string(lig_result['pdbqt'])
        v.compute_vina_maps(center=center, box_size=box_size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        energies = v.energies(n_poses=n_poses)
        best_affinity = float(energies[0][0])  # kcal/mol, more negative = better

        poses = []
        for i, e in enumerate(energies):
            poses.append({
                'rank': i + 1,
                'affinity_kcal': float(e[0]),
                'rmsd_lb': float(e[1]) if len(e) > 1 else 0.0,
                'rmsd_ub': float(e[2]) if len(e) > 2 else 0.0,
            })

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            'smiles': smiles,
            'affinity_kcal': best_affinity,
            'poses': poses,
            'n_poses': len(poses),
            'runtime_ms': elapsed_ms,
        }
    except Exception as e:
        return {'error': f'Docking failed: {str(e)}'}

def _dock_batch(smiles_list, receptor_path, center, box_size, exhaustiveness=8):
    \"\"\"Dock multiple molecules sequentially.\"\"\"
    results = []
    for smi in smiles_list:
        result = _dock_molecule(smi, receptor_path, center, box_size, exhaustiveness, n_poses=1)
        results.append(result)
    return results

def _proxy_predict_single(fp_vector):
    \"\"\"Predict docking score from fingerprint using proxy model.\"\"\"
    global _proxy_model, _proxy_scaler
    if _proxy_model is None:
        return {'error': 'Proxy model not trained'}
    import numpy as np
    fp = np.array(fp_vector).reshape(1, -1)
    if _proxy_scaler is not None:
        fp = _proxy_scaler.transform(fp)
    score = float(_proxy_model.predict(fp)[0])
    return {'predicted_affinity': score}

def _proxy_predict_batch(fp_vectors):
    \"\"\"Predict docking scores for multiple molecules.\"\"\"
    global _proxy_model, _proxy_scaler
    if _proxy_model is None:
        return {'error': 'Proxy model not trained'}
    import numpy as np
    fps = np.array(fp_vectors)
    if _proxy_scaler is not None:
        fps = _proxy_scaler.transform(fps)
    scores = _proxy_model.predict(fps).tolist()
    return {'predicted_affinities': scores}

def _proxy_train(fingerprints, docking_scores, hidden_dims=(256, 128)):
    \"\"\"Train proxy MLP on (fingerprint, docking_score) pairs.\"\"\"
    global _proxy_model, _proxy_scaler
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    X = np.array(fingerprints)
    y = np.array(docking_scores)

    _proxy_scaler = StandardScaler()
    X_scaled = _proxy_scaler.fit_transform(X)

    _proxy_model = MLPRegressor(
        hidden_layer_sizes=hidden_dims,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    _proxy_model.fit(X_scaled, y)

    # Return training metrics
    train_score = _proxy_model.score(X_scaled, y)
    predictions = _proxy_model.predict(X_scaled)
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))

    return {
        'r2_score': float(train_score),
        'rmse': rmse,
        'n_samples': len(y),
        'best_loss': float(_proxy_model.best_loss_) if hasattr(_proxy_model, 'best_loss_') else None,
    }
""", main_mod.__dict__)

    _py_prepare_ligand[] = main_mod._prepare_ligand
    _py_dock_molecule[]  = main_mod._dock_molecule
    _py_dock_batch[]     = main_mod._dock_batch
    _py_proxy_predict[]  = main_mod._proxy_predict_single
    _py_proxy_train[]    = main_mod._proxy_train

    # Verify docking tools are actually available
    docking_ok = pyconvert(Bool, main_mod._DOCKING_OK)
    if !docking_ok
        error("meeko/vina not installed")
    end
end

# --- Julia API for Docking ---

"""Check if docking tools are available."""
is_docking_available() = _docking_available[]

"""Check if a proxy model is trained and ready."""
is_proxy_available() = _proxy_available[]

"""Get current docking target ID."""
get_docking_target() = _docking_target[]

"""Check if a docking target is configured."""
has_docking_target() = !isempty(_docking_target[])

"""Get all available docking targets."""
function get_docking_targets()::Dict{String,Any}
    return _docking_targets[]
end

"""Set the active docking target by ID."""
function set_docking_target!(target_id::String)
    if haskey(_docking_targets[], target_id)
        _docking_target[] = target_id
        @info "Docking target set to: $target_id"
        return true
    else
        @warn "Unknown docking target: $target_id"
        return false
    end
end

"""Get the receptor PDBQT file path for a target."""
function _get_receptor_path(target_id::String)::String
    return joinpath(@__DIR__, "..", "..", "..", "..", "data", "targets", target_id, "receptor.pdbqt")
end

struct DockingResult
    smiles::String
    affinity_kcal::Float64
    n_poses::Int
    runtime_ms::Int
    error::String
end

"""
    dock_molecule(smiles, target_id) → DockingResult

Dock a molecule against a target using AutoDock Vina.
"""
function dock_molecule(smiles::String, target_id::String="")::DockingResult
    _ensure_init()
    !_docking_available[] && return DockingResult(smiles, 0.0, 0, 0, "Docking tools not available")

    tid = isempty(target_id) ? _docking_target[] : target_id
    isempty(tid) && return DockingResult(smiles, 0.0, 0, 0, "No docking target configured")

    target = get(_docking_targets[], tid, nothing)
    target === nothing && return DockingResult(smiles, 0.0, 0, 0, "Unknown target: $tid")

    receptor_path = _get_receptor_path(tid)

    result = _py_dock_molecule[](
        smiles,
        receptor_path,
        pylist(target["center"]),
        pylist(target["box_size"]),
        target["exhaustiveness"],
    )

    if pyhasattr(result, "keys") && pyconvert(Bool, pycontains(result, "error"))
        err = pyconvert(String, result["error"])
        return DockingResult(smiles, 0.0, 0, 0, err)
    end

    return DockingResult(
        smiles,
        pyconvert(Float64, result["affinity_kcal"]),
        pyconvert(Int, result["n_poses"]),
        pyconvert(Int, result["runtime_ms"]),
        "",
    )
end

"""
    dock_batch(smiles_list, target_id) → Vector{DockingResult}

Dock multiple molecules against a target.
"""
function dock_batch(smiles_list::Vector{String}, target_id::String="")::Vector{DockingResult}
    _ensure_init()
    results = DockingResult[]
    for smi in smiles_list
        push!(results, dock_molecule(smi, target_id))
    end
    return results
end

"""
    proxy_dock(smiles, target_id) → Float64

Predict docking score using the proxy model (~<1ms).
Returns normalized score in [0, 1] for use as a reward objective.
Falls back to 0.5 (neutral) if proxy is not trained.
"""
function proxy_dock(smiles::String, target_id::String="")::Float64
    _ensure_init()
    !_proxy_available[] && return 0.5

    fp = compute_fingerprint(smiles)
    result = _py_proxy_predict[](pylist(Float64.(fp)))

    if pyconvert(Bool, pycontains(result, "error"))
        return 0.5
    end

    raw_score = pyconvert(Float64, result["predicted_affinity"])
    return sigmoid_normalize(raw_score)
end

"""
    proxy_dock_batch(fps) → Vector{Float64}

Predict docking scores for multiple fingerprints using proxy model.
"""
function proxy_dock_batch(fps::Vector{Vector{Float32}})::Vector{Float64}
    _ensure_init()
    !_proxy_available[] && return fill(0.5, length(fps))

    py_fps = pylist([pylist(Float64.(fp)) for fp in fps])
    result = _py_proxy_predict[](py_fps)  # Will error — use batch variant
    # For now, use sequential
    return [proxy_dock_from_fp(fp) for fp in fps]
end

"""Predict docking score from pre-computed fingerprint."""
function proxy_dock_from_fp(fp::Vector{Float32})::Float64
    !_proxy_available[] && return 0.5

    result = _py_proxy_predict[](pylist(Float64.(fp)))
    if pyconvert(Bool, pycontains(result, "error"))
        return 0.5
    end

    raw_score = pyconvert(Float64, result["predicted_affinity"])
    return sigmoid_normalize(raw_score)
end

"""
    train_proxy!(fingerprints, docking_scores) → Dict

Train the proxy docking model on (fingerprint, score) pairs.
"""
function train_proxy!(fingerprints::Vector{Vector{Float32}},
                      docking_scores::Vector{Float64})::Dict
    _ensure_init()
    length(fingerprints) != length(docking_scores) && error("Mismatched lengths")

    py_fps = pylist([pylist(Float64.(fp)) for fp in fingerprints])
    py_scores = pylist(docking_scores)

    result = _py_proxy_train[](py_fps, py_scores)
    _proxy_available[] = true

    return Dict(
        "r2_score" => pyconvert(Float64, result["r2_score"]),
        "rmse" => pyconvert(Float64, result["rmse"]),
        "n_samples" => pyconvert(Int, result["n_samples"]),
    )
end

"""
    sigmoid_normalize(score; center=-6.0, scale=2.0) → Float64

Map Vina docking score to [0, 1].
Vina scores: ~-12 (strong binder) to ~0 (no binding).
center=-6.0 maps to 0.5; more negative → higher score.
"""
function sigmoid_normalize(score::Float64; center::Float64=-6.0, scale::Float64=2.0)::Float64
    return 1.0 / (1.0 + exp(-(score - center) / scale))
end

# ============================================
# Gap 4: Reaction Engine for Synthesizable Molecules
# ============================================

const _py_execute_reaction = Ref{Py}()
const _py_check_reactant   = Ref{Py}()
const _py_compute_rascore   = Ref{Py}()
const _reaction_engine_available = Ref{Bool}(false)

# Reaction templates loaded from JSON
const _reaction_templates = Ref{Vector{Dict{String,Any}}}(Dict{String,Any}[])

"""Load reaction templates from data/reactions/reaction_templates.json."""
function load_reaction_templates!()
    templates_file = joinpath(@__DIR__, "..", "..", "..", "..", "data", "reactions", "reaction_templates.json")
    if isfile(templates_file)
        try
            data = JSON3.read(read(templates_file, String))
            _reaction_templates[] = [Dict{String,Any}(
                "id" => r.id,
                "name" => r.name,
                "class" => r.class,
                "smarts" => r.smarts,
                "n_reactants" => r.n_reactants,
                "yield_estimate" => r.yield_estimate,
                "functional_groups" => collect(r.functional_groups),
            ) for r in data.reactions]
            @info "Loaded $(length(_reaction_templates[])) reaction templates"
        catch e
            @warn "Failed to load reaction templates: $e"
        end
    end
end

"""Initialize reaction engine Python functions."""
function init_reaction_engine!()
    _ensure_init()

    main_mod = pyimport("__main__")
    pyexec("""
from rdkit import Chem
from rdkit.Chem import AllChem
import json

# Reaction template cache
_rxn_cache = {}

def _init_reaction(smarts):
    \"\"\"Parse and cache a reaction SMARTS.\"\"\"
    if smarts not in _rxn_cache:
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is not None:
            _rxn_cache[smarts] = rxn
    return _rxn_cache.get(smarts)

def _execute_reaction(smarts, reactant_smiles_list):
    \"\"\"
    Execute a reaction given SMARTS and reactant SMILES.
    Returns {'product': smiles, 'valid': bool} or {'error': message}.
    \"\"\"
    rxn = _init_reaction(smarts)
    if rxn is None:
        return {'error': f'Invalid reaction SMARTS: {smarts}', 'valid': False}

    reactants = []
    for smi in reactant_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return {'error': f'Invalid reactant SMILES: {smi}', 'valid': False}
        reactants.append(mol)

    try:
        products = rxn.RunReactants(tuple(reactants))
        if not products:
            return {'error': 'No products generated', 'valid': False}

        # Take the first product set, first product molecule
        product = products[0][0]
        Chem.SanitizeMol(product)
        product_smiles = Chem.MolToSmiles(product)
        return {'product': product_smiles, 'valid': True}
    except Exception as e:
        return {'error': str(e), 'valid': False}

def _check_reactant_compatibility(smarts, smiles, reactant_idx=0):
    \"\"\"Check if a molecule is a valid reactant for a reaction at given position.\"\"\"
    rxn = _init_reaction(smarts)
    if rxn is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        return rxn.IsMoleculeReactant(mol, reactant_idx)
    except Exception:
        return False

def _compute_rascore(smiles):
    \"\"\"
    Compute Rapid Assessment (RA) synthesizability score.
    Uses heuristic based on SA score, ring complexity, and stereochemistry.
    Returns value in [0, 1] where 1 = easily synthesizable.
    \"\"\"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    from rdkit.Chem import Descriptors, rdMolDescriptors

    # Heuristic factors
    n_rings = Descriptors.RingCount(mol)
    n_heavy = Descriptors.HeavyAtomCount(mol)
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_rotatable = Descriptors.NumRotatableBonds(mol)

    # Penalize complexity
    ring_penalty = min(n_rings * 0.08, 0.4)
    size_penalty = max(0, (n_heavy - 30) * 0.01)
    chiral_penalty = n_chiral * 0.1
    flexibility_bonus = min(n_rotatable * 0.02, 0.1)

    score = 1.0 - ring_penalty - size_penalty - chiral_penalty + flexibility_bonus
    return max(0.0, min(1.0, score))
""", main_mod.__dict__)

    _py_execute_reaction[] = main_mod._execute_reaction
    _py_check_reactant[]   = main_mod._check_reactant_compatibility
    _py_compute_rascore[]   = main_mod._compute_rascore

    _reaction_engine_available[] = true
    @info "Reaction engine initialized"
end

"""Check if reaction engine is available."""
is_reaction_engine_available() = _reaction_engine_available[]

"""Get loaded reaction templates."""
get_reaction_templates() = _reaction_templates[]

struct ReactionResult
    product_smiles::String
    valid::Bool
    error::String
end

"""
    execute_reaction(smarts, reactant_smiles) → ReactionResult

Execute a chemical reaction given SMARTS pattern and reactant SMILES.
"""
function execute_reaction(smarts::String, reactant_smiles::Vector{String})::ReactionResult
    _ensure_init()
    !_reaction_engine_available[] && return ReactionResult("", false, "Reaction engine not available")

    result = _py_execute_reaction[](smarts, pylist(reactant_smiles))

    if pyconvert(Bool, pycontains(result, "error"))
        return ReactionResult("", false, pyconvert(String, result["error"]))
    end

    return ReactionResult(
        pyconvert(String, result["product"]),
        pyconvert(Bool, result["valid"]),
        "",
    )
end

"""
    check_reactant(smarts, smiles, reactant_idx) → Bool

Check if a molecule is a valid reactant for a given reaction template.
"""
function check_reactant(smarts::String, smiles::String, reactant_idx::Int=0)::Bool
    _ensure_init()
    !_reaction_engine_available[] && return false
    return pyconvert(Bool, _py_check_reactant[](smarts, smiles, reactant_idx))
end

"""
    compute_rascore(smiles) → Float64

Compute Rapid Assessment synthesizability score for a molecule.
Returns value in [0,1] where 1 = easily synthesizable.
"""
function compute_rascore(smiles::String)::Float64
    _ensure_init()
    !_reaction_engine_available[] && return 0.5
    return pyconvert(Float64, _py_compute_rascore[](smiles))
end

end # module RDKitBridge
