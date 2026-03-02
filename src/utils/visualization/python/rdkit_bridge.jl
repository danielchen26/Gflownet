# RDKit Bridge — Julia ↔ Python via PythonCall.jl
#
# Single point of contact between Julia and Python.
# All RDKit molecular operations go through this module.
#
# IMPORTANT: This module uses explicit init_rdkit!() instead of __init__()
# because unified_server.jl loads files via include() not using/import.

module RDKitBridge

using PythonCall

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

    # Convert to flat Python list, then reshape in Python
    flat = vcat(fingerprints...)
    py_arr = np[].array(pylist(Float64.(flat))).reshape(n, 1024)

    coords = if method == "umap" && n >= 5
        umap_mod = pyimport("umap")
        n_neighbors = min(15, n - 1)
        reducer = umap_mod.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2)
        reducer.fit_transform(py_arr)
    elseif method == "tsne" && n >= 5
        sklearn_manifold = pyimport("sklearn.manifold")
        perplexity = min(30.0, Float64(n - 1))
        reducer = sklearn_manifold.TSNE(n_components=2, perplexity=perplexity)
        reducer.fit_transform(py_arr)
    else  # pca (default)
        sklearn_decomp = pyimport("sklearn.decomposition")
        reducer = sklearn_decomp.PCA(n_components=2)
        reducer.fit_transform(py_arr)
    end

    points = Dict[]
    for i in 1:n
        push!(points, Dict(
            "x" => pyconvert(Float64, coords[i-1, 0]),
            "y" => pyconvert(Float64, coords[i-1, 1]),
        ))
    end
    return points
end

end # module RDKitBridge
