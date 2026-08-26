# Genetic Operations for SMILES GFlowNet (Genetic GFN, NeurIPS 2024)
#
# Implements genetic algorithm operations to complement GFlowNet training:
# 1. SMILES Augmentation — randomized SMILES via RDKit (free data multiplication)
# 2. Crossover — fragment exchange between two high-reward molecules
# 3. Mutation — single-atom substitution on a molecule
# 4. Scaffold Diversity Filter — Bemis-Murcko scaffold tracking to prevent mode collapse
#
# All RDKit operations use PythonCall.jl. Functions are designed to fail
# gracefully (return empty/original) when PythonCall is unavailable or
# RDKit operations fail on invalid molecules.

using Random
using PythonCall

# =============================================================================
# SMILES Augmentation (Randomized SMILES)
# =============================================================================

"""
    augment_smiles_rdkit(smiles::String; n_augmentations::Int=4) → Vector{String}

Generate randomized SMILES representations for the same molecule using RDKit.

Each molecule has many valid SMILES representations. Training on multiple
representations of the same molecule (with the same reward) teaches the model
that the underlying molecular graph matters, not the specific string encoding.

This is essentially "free" data augmentation — no oracle calls needed.

Requires PythonCall.jl and RDKit to be available.
Returns the original SMILES if augmentation fails.
"""
function augment_smiles_rdkit(smiles::String; n_augmentations::Int=4)
    augmented = String[smiles]  # Always include canonical

    try
        rdkit = Base.get_extension(Main, :PythonCall) !== nothing ?
            pyimport("rdkit.Chem") :
            pyimport("rdkit.Chem")

        mol = rdkit.MolFromSmiles(smiles)
        if pyisinstance(mol, rdkit.Mol) || mol !== nothing
            for _ in 1:n_augmentations
                try
                    random_smi = string(rdkit.MolToSmiles(mol, doRandom=true))
                    if !isempty(random_smi) && random_smi ∉ augmented
                        push!(augmented, random_smi)
                    end
                catch
                    continue
                end
            end
        end
    catch
        # PythonCall or RDKit not available — return original only
    end

    return augmented
end

"""
    create_augment_fn(vocab) → Function

Create an augmentation function that generates randomized SMILES and
tokenizes them. Returns a function: (smiles) → Vector{(smiles, tokens)}.

This is the recommended way to use augmentation in the training loop.
"""
function create_augment_fn(vocab; n_augmentations::Int=4)
    return function(smiles::String)
        aug_smiles = augment_smiles_rdkit(smiles; n_augmentations=n_augmentations)
        results = Tuple{String, Vector{Int}}[]
        for smi in aug_smiles
            try
                tokens = encode(vocab, smi)
                if length(tokens) >= 2
                    push!(results, (smi, tokens))
                end
            catch
                continue
            end
        end
        return results
    end
end

# =============================================================================
# Crossover — Fragment Exchange Between Molecules
# =============================================================================

"""
    smiles_crossover_rdkit(smiles1::String, smiles2::String) → Vector{String}

Perform BRICS-based fragment crossover between two SMILES molecules.

1. Decompose both molecules into BRICS fragments
2. Randomly exchange one fragment between the molecules
3. Reconstruct and validate resulting molecules

Returns a vector of valid child SMILES (may be empty if crossover fails).

From Genetic GFN: crossover operates on molecular graphs, not SMILES strings.
"""
function smiles_crossover_rdkit(smiles1::String, smiles2::String)
    children = String[]

    try
        Chem = pyimport("rdkit.Chem")
        BRICS = pyimport("rdkit.Chem.BRICS")
        AllChem = pyimport("rdkit.Chem.AllChem")

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 === nothing || mol2 === nothing
            return children
        end

        # BRICS decomposition
        frags1 = collect(BRICS.BRICSDecompose(mol1))
        frags2 = collect(BRICS.BRICSDecompose(mol2))

        if isempty(frags1) || isempty(frags2)
            return children
        end

        # Try recombining fragments from both molecules
        all_frags = vcat(frags1, frags2)
        # Limit to avoid combinatorial explosion
        if length(all_frags) > 10
            all_frags = all_frags[randperm(length(all_frags))[1:10]]
        end

        # Convert fragment strings to mol objects
        frag_mols = []
        for f in all_frags
            fm = Chem.MolFromSmiles(string(f))
            if fm !== nothing
                push!(frag_mols, fm)
            end
        end

        if length(frag_mols) >= 2
            # Try BRICS build (limited attempts)
            try
                # BRICSBuild returns a generator; take first few results
                built = BRICS.BRICSBuild(frag_mols)
                for (i, mol) in enumerate(built)
                    i > 5 && break  # Limit output
                    smi = string(Chem.MolToSmiles(mol))
                    if !isempty(smi) && smi != smiles1 && smi != smiles2
                        # Validate
                        check_mol = Chem.MolFromSmiles(smi)
                        if check_mol !== nothing
                            push!(children, smi)
                        end
                    end
                end
            catch
                # BRICS build can fail on complex fragments
            end
        end
    catch
        # PythonCall or RDKit not available
    end

    return children
end

# =============================================================================
# Mutation — Single-Atom Substitution
# =============================================================================

# Common atom substitutions for drug-like molecules
const ATOM_SUBSTITUTIONS = Dict(
    "C" => ["N", "O", "S"],
    "N" => ["C", "O", "S"],
    "O" => ["N", "S", "C"],
    "S" => ["N", "O", "C"],
    "F" => ["Cl", "Br"],
    "Cl" => ["F", "Br"],
    "Br" => ["F", "Cl"],
)

"""
    smiles_mutate_rdkit(smiles::String; n_mutations::Int=1) → Vector{String}

Mutate a SMILES molecule by randomly substituting atoms.

For each mutation attempt:
1. Pick a random atom in the molecule
2. Replace it with a chemically similar atom
3. Validate the resulting molecule with RDKit

Returns valid mutant SMILES (may be empty if all mutations fail).
"""
function smiles_mutate_rdkit(smiles::String; n_mutations::Int=3)
    mutants = String[]

    try
        Chem = pyimport("rdkit.Chem")

        mol = Chem.MolFromSmiles(smiles)
        if mol === nothing
            return mutants
        end

        n_atoms = pyconvert(Int, mol.GetNumAtoms())
        if n_atoms == 0
            return mutants
        end

        candidate_edits = Tuple{Int,String}[]
        for atom_idx in 0:n_atoms-1
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_symbol = string(atom.GetSymbol())
            subs = get(ATOM_SUBSTITUTIONS, atom_symbol, String[])
            for new_atom in subs
                push!(candidate_edits, (atom_idx, new_atom))
            end
        end
        isempty(candidate_edits) && return mutants

        edit_order = randperm(length(candidate_edits))
        periodic_table = Chem.GetPeriodicTable()

        for edit_idx in edit_order
            try
                atom_idx, new_atom = candidate_edits[edit_idx]
                rw_mol = Chem.RWMol(mol)
                rw_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(
                    pyconvert(Int, periodic_table.GetAtomicNumber(new_atom))
                )
                Chem.SanitizeMol(rw_mol)
                new_smi = string(Chem.MolToSmiles(rw_mol))
                if !isempty(new_smi) && new_smi != smiles && new_smi ∉ mutants
                    check = Chem.MolFromSmiles(new_smi)
                    if check !== nothing
                        push!(mutants, new_smi)
                    end
                end
                length(mutants) >= n_mutations && break
            catch
                continue
            end
        end
    catch
        # PythonCall or RDKit not available
    end

    return mutants
end

# =============================================================================
# Token-Level Mutation (No RDKit Required)
# =============================================================================

"""
    smiles_mutate_tokens(tokens::Vector{Int}, vocab; n_mutations::Int=1) → Vector{Int}

Mutate a SMILES token sequence by randomly replacing atom tokens.
This is a fast alternative to RDKit-based mutation that operates at the
token level. Results are NOT guaranteed to be chemically valid.

Returns a mutated token sequence.
"""
function smiles_mutate_tokens(tokens::Vector{Int}, vocab; n_mutations::Int=1)
    mutated = copy(tokens)
    n_tokens = length(mutated)

    if n_tokens < 3  # Need at least START, atom, END
        return mutated
    end

    # Find indices of atom tokens (not START, END, PAD, or structural tokens)
    structural_tokens = Set{Int}()
    for special in ["(", ")", "=", "#", "/", "\\", "-", "+", ".", "[", "]",
                     "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                     "@", "@@"]
        idx = get(vocab.token_to_idx, special, -1)
        if idx >= 0
            push!(structural_tokens, idx)
        end
    end
    push!(structural_tokens, PAD_TOKEN)
    push!(structural_tokens, START_TOKEN)
    push!(structural_tokens, END_TOKEN)

    # Identify atom token positions
    atom_positions = Int[]
    for i in 2:(n_tokens - 1)  # Skip START and END
        if mutated[i] ∉ structural_tokens
            push!(atom_positions, i)
        end
    end

    if isempty(atom_positions)
        return mutated
    end

    # Collect all atom token IDs for replacement
    atom_token_ids = [t for t in 0:(vocab.size-1) if t ∉ structural_tokens]
    if isempty(atom_token_ids)
        return mutated
    end

    # Perform mutations
    for _ in 1:min(n_mutations, length(atom_positions))
        pos = rand(atom_positions)
        mutated[pos] = rand(atom_token_ids)
    end

    return mutated
end

# =============================================================================
# Scaffold Diversity Filter
# =============================================================================

"""
    ScaffoldFilter

Tracks Bemis-Murcko scaffolds to prevent mode collapse in the replay buffer.
Molecules with over-represented scaffolds are deprioritized.

# Fields
- `scaffold_counts`: Scaffold SMILES → count of molecules with that scaffold
- `max_per_scaffold`: Maximum molecules allowed per scaffold
"""
mutable struct ScaffoldFilter
    scaffold_counts::Dict{String, Int}
    max_per_scaffold::Int

    function ScaffoldFilter(; max_per_scaffold::Int=25)
        new(Dict{String, Int}(), max_per_scaffold)
    end
end

"""
    get_scaffold(smiles::String) → String

Extract the Bemis-Murcko scaffold from a molecule using RDKit.
Returns empty string if extraction fails.
"""
function get_scaffold(smiles::String)
    try
        Chem = pyimport("rdkit.Chem")
        Scaffolds = pyimport("rdkit.Chem.Scaffolds.MurckoScaffold")

        mol = Chem.MolFromSmiles(smiles)
        if mol === nothing
            return ""
        end

        scaffold = Scaffolds.GetScaffoldForMol(mol)
        scaffold_smi = string(Chem.MolToSmiles(scaffold))
        return scaffold_smi
    catch
        return ""
    end
end

"""
    should_add_molecule(filter::ScaffoldFilter, smiles::String) → Bool

Check if a molecule should be added based on scaffold diversity.
Returns true if the molecule's scaffold is under-represented.
"""
function should_add_molecule(filter::ScaffoldFilter, smiles::String)
    scaffold = get_scaffold(smiles)
    if isempty(scaffold)
        return true  # Can't determine scaffold — allow it
    end

    count = get(filter.scaffold_counts, scaffold, 0)
    return count < filter.max_per_scaffold
end

"""
    register_molecule!(filter::ScaffoldFilter, smiles::String)

Register a molecule's scaffold in the filter.
"""
function register_molecule!(filter::ScaffoldFilter, smiles::String)
    scaffold = get_scaffold(smiles)
    if !isempty(scaffold)
        filter.scaffold_counts[scaffold] = get(filter.scaffold_counts, scaffold, 0) + 1
    end
end

"""
    scaffold_diversity_stats(filter::ScaffoldFilter) → Dict

Return statistics about scaffold diversity.
"""
function scaffold_diversity_stats(filter::ScaffoldFilter)
    if isempty(filter.scaffold_counts)
        return Dict("n_scaffolds" => 0, "max_count" => 0, "mean_count" => 0.0)
    end
    counts = values(filter.scaffold_counts)
    return Dict(
        "n_scaffolds" => length(filter.scaffold_counts),
        "max_count" => maximum(counts),
        "mean_count" => mean(collect(counts)),
        "over_represented" => count(c -> c >= filter.max_per_scaffold, counts),
    )
end

# =============================================================================
# Genetic Operation Orchestrator
# =============================================================================

"""
    generate_genetic_molecules(replay_buffer, vocab;
        n_crossover=8, n_mutation=8, n_augmentation=16,
        scaffold_filter=nothing) → Vector{(smiles, tokens)}

Generate new molecules using genetic operations on the replay buffer.

1. Select parent molecules from top of replay buffer
2. Apply crossover, mutation, and augmentation
3. Filter by scaffold diversity
4. Tokenize and return

Returns vector of (smiles, tokens) tuples ready for reward evaluation.
"""
function generate_genetic_molecules(replay_buffer, vocab;
    n_crossover::Int=8, n_mutation::Int=8, n_augmentation::Int=16,
    scaffold_filter=nothing)

    results = Tuple{String, Vector{Int}}[]

    if isempty(replay_buffer)
        return results
    end

    top = get_top_molecules(replay_buffer, 20)

    # --- Augmentation (randomized SMILES) ---
    for i in 1:min(n_augmentation, length(top))
        aug_list = augment_smiles_rdkit(top[i].smiles; n_augmentations=2)
        for smi in aug_list
            canonical_smi = canonicalize_smiles_identity(smi)
            if !isempty(canonical_smi) && canonical_smi != top[i].smiles
                _try_add_result!(results, canonical_smi, vocab, scaffold_filter)
            end
        end
    end

    # --- Crossover ---
    if length(top) >= 2
        for _ in 1:n_crossover
            i, j = rand(1:min(10, length(top)), 2)
            i == j && continue
            children = smiles_crossover_rdkit(top[i].smiles, top[j].smiles)
            for child in children
                _try_add_result!(results, child, vocab, scaffold_filter)
                length(results) >= n_crossover && break
            end
        end
    end

    # --- Mutation ---
    for _ in 1:n_mutation
        parent = top[rand(1:min(10, length(top)))]
        mutants = smiles_mutate_rdkit(parent.smiles; n_mutations=3)
        for mutant in mutants
            _try_add_result!(results, mutant, vocab, scaffold_filter)
        end
    end

    return results
end

"""Try to tokenize and add a SMILES to results, respecting scaffold filter."""
function _try_add_result!(results, smiles, vocab, scaffold_filter)
    # Scaffold diversity check
    if !isnothing(scaffold_filter) && !should_add_molecule(scaffold_filter, smiles)
        return
    end

    try
        tokens = encode(vocab, smiles)
        if length(tokens) >= 2
            push!(results, (smiles, tokens))
        end
    catch
        # Tokenization failed — skip
    end
end

# =============================================================================
# Graph GA (Jensen 2019) — Julia PythonCall Wrapper
# =============================================================================

# Lazy-loaded Python module reference
const _graph_ga_module = Ref{Any}(nothing)

"""
    _get_graph_ga_module() → Python module

Load the graph_ga.py Python module lazily via PythonCall.
"""
function _get_graph_ga_module()
    if _graph_ga_module[] === nothing
        try
            sys = pyimport("sys")
            # Add our training directory to Python path
            ga_dir = joinpath(@__DIR__)
            sys.path.insert(0, ga_dir)
            _graph_ga_module[] = pyimport("graph_ga")
        catch e
            @warn "Failed to load graph_ga.py" exception=e
            return nothing
        end
    end
    return _graph_ga_module[]
end

"""
    graph_ga_crossover_mutate(smiles_list, scores;
        n_crossover=10, n_mutation=10) → Vector{String}

Run one step of Jensen 2019 Graph GA: crossover + mutation on molecular graphs.

Uses tournament selection weighted by scores. Returns valid child SMILES
that are distinct from parents.

Requires PythonCall.jl and RDKit.
"""
function graph_ga_crossover_mutate(smiles_list::Vector{String},
                                    scores::Vector{Float64};
                                    n_crossover::Int=10,
                                    n_mutation::Int=10)::Vector{String}
    ga = _get_graph_ga_module()
    if ga === nothing
        return String[]
    end

    try
        children = ga.graph_ga_step(smiles_list, scores,
                                     n_crossover=n_crossover,
                                     n_mutation=n_mutation)
        return String[string(c) for c in children]
    catch e
        @warn "Graph GA step failed" exception=e
        return String[]
    end
end

"""
    graph_ga_scaffold_crossover(smiles_list, scaffolds, scores;
        target_scaffold=nothing, n_children=10) → Vector{String}

Scaffold-preserving crossover: exchange decorations within scaffold groups.

Only crosses molecules that share the same Bemis-Murcko scaffold,
preserving the core structure while varying substituents.
"""
function graph_ga_scaffold_crossover(smiles_list::Vector{String},
                                      scaffolds::Vector{String},
                                      scores::Vector{Float64};
                                      target_scaffold::Union{Nothing,String}=nothing,
                                      n_children::Int=10)::Vector{String}
    ga = _get_graph_ga_module()
    if ga === nothing
        return String[]
    end

    try
        children = ga.scaffold_preserving_crossover(
            smiles_list, scaffolds, scores,
            target_scaffold=isnothing(target_scaffold) ? nothing : target_scaffold,
            n_children=n_children)
        return String[string(c) for c in children]
    catch e
        @warn "Scaffold-preserving crossover failed" exception=e
        return String[]
    end
end

"""
    generate_graph_ga_molecules(replay_buffer, vocab;
        n_crossover=10, n_mutation=10,
        scaffold_filter=nothing) → Vector{Tuple{String, Vector{Int}}}

Generate new molecules using Jensen 2019 Graph GA on the replay buffer.

Unlike BRICS-based crossover, Graph GA operates directly on molecular graphs:
- Ring crossover: exchange ring systems
- Non-ring crossover: exchange acyclic fragments
- 7 mutation types: insert/delete/change atoms, bonds, rings

Returns vector of (smiles, tokens) tuples ready for reward evaluation.
"""
function generate_graph_ga_molecules(replay_buffer, vocab;
    n_crossover::Int=10, n_mutation::Int=10,
    scaffold_filter=nothing)

    results = Tuple{String, Vector{Int}}[]

    if isempty(replay_buffer)
        return results
    end

    # Get top molecules as parents
    top = get_top_molecules(replay_buffer, min(50, length(replay_buffer)))
    smiles_list = String[m.smiles for m in top]
    scores = Float64[m.reward for m in top]

    # Run Graph GA
    children = graph_ga_crossover_mutate(smiles_list, scores;
        n_crossover=n_crossover, n_mutation=n_mutation)

    # Tokenize and filter
    for child_smi in children
        _try_add_result!(results, child_smi, vocab, scaffold_filter)
    end

    return results
end
