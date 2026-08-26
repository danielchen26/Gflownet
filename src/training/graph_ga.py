"""
Graph-based Genetic Algorithm for Molecular Optimization (Jensen 2019)

Implements graph-level genetic operations on molecular graphs:
- Ring crossover: exchange ring systems between molecules
- Non-ring crossover: exchange non-cyclic fragments
- 7 mutation types on molecular graphs

Reference: Jensen, J.H. "A graph-based genetic algorithm and generative model/Monte Carlo
tree search for the exploration of chemical space." Chem. Sci. 10, 3567-3572 (2019).

Called from Julia via PythonCall.jl.
"""

import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdmolops


# ============================================================================
# Crossover Operations
# ============================================================================

def _get_ring_atoms(mol):
    """Get sets of atoms in each ring."""
    ring_info = mol.GetRingInfo()
    return [set(ring) for ring in ring_info.AtomRings()]


def _ring_crossover(mol1, mol2):
    """
    Ring crossover: exchange a ring system between two molecules.

    1. Find rings in both molecules
    2. Pick a random ring from each
    3. Cut the bonds connecting the ring to the rest
    4. Recombine ring from mol1 with non-ring from mol2
    """
    rings1 = _get_ring_atoms(mol1)
    rings2 = _get_ring_atoms(mol2)

    if not rings1 or not rings2:
        return None

    ring1 = random.choice(rings1)
    ring2 = random.choice(rings2)

    # Find bonds connecting ring to rest of molecule
    def get_ring_bonds(mol, ring_atoms):
        bonds = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if (a1 in ring_atoms) != (a2 in ring_atoms):
                bonds.append(bond.GetIdx())
        return bonds

    bonds1 = get_ring_bonds(mol1, ring1)
    bonds2 = get_ring_bonds(mol2, ring2)

    if not bonds1 or not bonds2:
        return None

    # Fragment both molecules
    try:
        frag1 = Chem.FragmentOnBonds(mol1, bonds1, addDummies=True)
        frag2 = Chem.FragmentOnBonds(mol2, bonds2, addDummies=True)
    except Exception:
        return None

    frags1 = Chem.GetMolFrags(frag1, asMols=True, sanitizeFrags=True)
    frags2 = Chem.GetMolFrags(frag2, asMols=True, sanitizeFrags=True)

    if len(frags1) < 2 or len(frags2) < 2:
        return None

    # Identify ring and non-ring fragments
    def has_ring(frag):
        return frag.GetRingInfo().NumRings() > 0

    ring_frags1 = [f for f in frags1 if has_ring(f)]
    nonring_frags2 = [f for f in frags2 if not has_ring(f)]

    if not ring_frags1 or not nonring_frags2:
        # Try the other direction
        ring_frags2 = [f for f in frags2 if has_ring(f)]
        nonring_frags1 = [f for f in frags1 if not has_ring(f)]
        if not ring_frags2 or not nonring_frags1:
            return None
        ring_frag = random.choice(ring_frags2)
        nonring_frag = random.choice(nonring_frags1)
    else:
        ring_frag = random.choice(ring_frags1)
        nonring_frag = random.choice(nonring_frags2)

    # Recombine using dummy atoms as attachment points
    try:
        combined = AllChem.CombineMols(ring_frag, nonring_frag)
        rw = Chem.RWMol(combined)

        # Find dummy atom pairs and connect them
        dummy_atoms = []
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atom
                dummy_atoms.append(atom.GetIdx())

        if len(dummy_atoms) >= 2:
            # Connect first dummy's neighbor to second dummy's neighbor
            # Then remove dummies
            neighbors = []
            for d in dummy_atoms[:2]:
                atom = rw.GetAtomWithIdx(d)
                n = [x.GetIdx() for x in atom.GetNeighbors() if x.GetAtomicNum() != 0]
                if n:
                    neighbors.append(n[0])

            if len(neighbors) >= 2:
                rw.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
                # Remove dummies in reverse order
                for d in sorted(dummy_atoms, reverse=True):
                    rw.RemoveAtom(d)

                Chem.SanitizeMol(rw)
                smi = Chem.MolToSmiles(rw)
                check = Chem.MolFromSmiles(smi)
                if check is not None:
                    return smi
    except Exception:
        pass

    return None


def _non_ring_crossover(mol1, mol2):
    """
    Non-ring crossover: exchange non-cyclic fragments.

    1. Find acyclic single bonds in both molecules
    2. Fragment each molecule at a random acyclic bond
    3. Recombine fragments from different molecules
    """
    # Find acyclic single bonds suitable for cutting
    def get_acyclic_bonds(mol):
        bonds = []
        for bond in mol.GetBonds():
            if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                # Skip bonds to H or bonds where either atom has only 1 heavy neighbor
                if a1.GetAtomicNum() > 1 and a2.GetAtomicNum() > 1:
                    if a1.GetDegree() > 1 and a2.GetDegree() > 1:
                        bonds.append(bond.GetIdx())
        return bonds

    bonds1 = get_acyclic_bonds(mol1)
    bonds2 = get_acyclic_bonds(mol2)

    if not bonds1 or not bonds2:
        return None

    bond1 = random.choice(bonds1)
    bond2 = random.choice(bonds2)

    try:
        frag1 = Chem.FragmentOnBonds(mol1, [bond1], addDummies=True)
        frag2 = Chem.FragmentOnBonds(mol2, [bond2], addDummies=True)
    except Exception:
        return None

    frags1 = Chem.GetMolFrags(frag1, asMols=True, sanitizeFrags=True)
    frags2 = Chem.GetMolFrags(frag2, asMols=True, sanitizeFrags=True)

    if len(frags1) < 2 or len(frags2) < 2:
        return None

    # Recombine: take one fragment from each parent
    f1 = random.choice(frags1)
    f2 = random.choice(frags2)

    try:
        combined = AllChem.CombineMols(f1, f2)
        rw = Chem.RWMol(combined)

        # Find and connect dummy atoms
        dummy_atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]

        if len(dummy_atoms) >= 2:
            neighbors = []
            for d in dummy_atoms[:2]:
                atom = rw.GetAtomWithIdx(d)
                n = [x.GetIdx() for x in atom.GetNeighbors() if x.GetAtomicNum() != 0]
                if n:
                    neighbors.append(n[0])

            if len(neighbors) >= 2:
                rw.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
                for d in sorted(dummy_atoms, reverse=True):
                    rw.RemoveAtom(d)

                Chem.SanitizeMol(rw)
                smi = Chem.MolToSmiles(rw)
                check = Chem.MolFromSmiles(smi)
                if check is not None:
                    return smi
    except Exception:
        pass

    return None


def crossover(smiles1, smiles2):
    """
    Perform crossover between two SMILES molecules.
    Randomly chooses ring or non-ring crossover.

    Returns: child SMILES string, or None if crossover fails.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return None

    # Size filter: skip very large molecules (>50 heavy atoms)
    if mol1.GetNumHeavyAtoms() > 50 or mol2.GetNumHeavyAtoms() > 50:
        return None

    # Random choice: ring vs non-ring crossover
    if random.random() < 0.5:
        child = _ring_crossover(mol1, mol2)
    else:
        child = _non_ring_crossover(mol1, mol2)

    if child is None:
        # Fallback: try the other type
        if random.random() < 0.5:
            child = _non_ring_crossover(mol1, mol2)
        else:
            child = _ring_crossover(mol1, mol2)

    # Size filter on child
    if child is not None:
        child_mol = Chem.MolFromSmiles(child)
        if child_mol is not None and child_mol.GetNumHeavyAtoms() > 50:
            return None

    return child


# ============================================================================
# Mutation Operations (7 Types)
# ============================================================================

def _insert_atom(rw, atom_idx):
    """Insert a new atom along a bond connected to atom_idx."""
    atom = rw.GetAtomWithIdx(atom_idx)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    if not neighbors:
        return False

    n_idx = random.choice(neighbors)
    bond = rw.GetBondBetweenAtoms(atom_idx, n_idx)
    if bond is None:
        return False

    bond_type = bond.GetBondType()
    rw.RemoveBond(atom_idx, n_idx)

    new_atom = random.choice([6, 7, 8])  # C, N, O
    new_idx = rw.AddAtom(Chem.Atom(new_atom))
    rw.AddBond(atom_idx, new_idx, bond_type)
    rw.AddBond(new_idx, n_idx, Chem.BondType.SINGLE)

    return True


def _delete_atom(rw, atom_idx):
    """Delete an atom and reconnect its neighbors."""
    atom = rw.GetAtomWithIdx(atom_idx)
    if atom.IsInRing():
        return False  # Don't delete ring atoms
    if atom.GetDegree() < 2:
        return False  # Terminal atoms handled differently

    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    if len(neighbors) != 2:
        return False  # Only handle degree-2 atoms

    # Reconnect neighbors
    rw.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
    rw.RemoveAtom(atom_idx)

    return True


def _change_atom(rw, atom_idx):
    """Change an atom to a different element."""
    atom = rw.GetAtomWithIdx(atom_idx)
    current = atom.GetAtomicNum()

    # Possible replacements (drug-like elements)
    replacements = {
        6: [7, 8, 16],        # C -> N, O, S
        7: [6, 8, 16],        # N -> C, O, S
        8: [6, 7, 16],        # O -> C, N, S
        16: [6, 7, 8],        # S -> C, N, O
        9: [17, 35],          # F -> Cl, Br
        17: [9, 35],          # Cl -> F, Br
        35: [9, 17],          # Br -> F, Cl
    }

    options = replacements.get(current, [6, 7, 8])
    new_num = random.choice(options)
    atom.SetAtomicNum(new_num)
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(0)

    return True


def _insert_bond(rw):
    """Add a bond between two non-bonded atoms."""
    n_atoms = rw.GetNumAtoms()
    if n_atoms < 3:
        return False

    # Find pairs of atoms that could be bonded
    for _ in range(10):
        i = random.randint(0, n_atoms - 1)
        j = random.randint(0, n_atoms - 1)
        if i == j:
            continue

        if rw.GetBondBetweenAtoms(i, j) is not None:
            continue

        # Check valence allows new bond
        atom_i = rw.GetAtomWithIdx(i)
        atom_j = rw.GetAtomWithIdx(j)

        try:
            rw.AddBond(i, j, Chem.BondType.SINGLE)
            return True
        except Exception:
            continue

    return False


def _delete_bond(rw):
    """Delete a non-ring bond."""
    bonds = []
    for bond in rw.GetBonds():
        if not bond.IsInRing():
            bonds.append(bond)

    if not bonds:
        return False

    bond = random.choice(bonds)
    rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return True


def _change_bond(rw):
    """Change bond type (single <-> double)."""
    bonds = list(rw.GetBonds())
    if not bonds:
        return False

    bond = random.choice(bonds)
    if bond.GetBondType() == Chem.BondType.SINGLE:
        bond.SetBondType(Chem.BondType.DOUBLE)
    elif bond.GetBondType() == Chem.BondType.DOUBLE:
        bond.SetBondType(Chem.BondType.SINGLE)
    else:
        return False

    return True


def _add_ring(rw, atom_idx):
    """Add a small ring (3-6 atoms) at a position."""
    ring_size = random.choice([3, 4, 5, 6])

    try:
        # Build a ring by adding atoms and connecting back
        start = atom_idx
        prev = start
        ring_atoms = [start]

        for _ in range(ring_size - 1):
            new_atom = Chem.Atom(6)  # Carbon ring
            new_idx = rw.AddAtom(new_atom)
            rw.AddBond(prev, new_idx, Chem.BondType.SINGLE)
            ring_atoms.append(new_idx)
            prev = new_idx

        # Close the ring
        rw.AddBond(prev, start, Chem.BondType.SINGLE)
        return True
    except Exception:
        return False


MUTATION_TYPES = [
    "insert_atom",
    "delete_atom",
    "change_atom",
    "insert_bond",
    "delete_bond",
    "change_bond",
    "add_ring",
]


def mutate(smiles):
    """
    Mutate a SMILES molecule using one of 7 mutation types.

    Returns: mutated SMILES string, or None if mutation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumHeavyAtoms()
    if n_atoms < 2:
        return None

    # Try each mutation type (shuffled) until one succeeds
    mutation_order = list(range(len(MUTATION_TYPES)))
    random.shuffle(mutation_order)

    for mut_idx in mutation_order:
        mut_type = MUTATION_TYPES[mut_idx]

        try:
            rw = Chem.RWMol(Chem.MolFromSmiles(smiles))
            success = False

            if mut_type == "insert_atom":
                atom_idx = random.randint(0, rw.GetNumAtoms() - 1)
                success = _insert_atom(rw, atom_idx)
            elif mut_type == "delete_atom":
                atom_idx = random.randint(0, rw.GetNumAtoms() - 1)
                success = _delete_atom(rw, atom_idx)
            elif mut_type == "change_atom":
                atom_idx = random.randint(0, rw.GetNumAtoms() - 1)
                success = _change_atom(rw, atom_idx)
            elif mut_type == "insert_bond":
                success = _insert_bond(rw)
            elif mut_type == "delete_bond":
                success = _delete_bond(rw)
            elif mut_type == "change_bond":
                success = _change_bond(rw)
            elif mut_type == "add_ring":
                atom_idx = random.randint(0, rw.GetNumAtoms() - 1)
                success = _add_ring(rw, atom_idx)

            if not success:
                continue

            try:
                Chem.SanitizeMol(rw)
                new_smi = Chem.MolToSmiles(rw)
                if new_smi and new_smi != smiles:
                    # Verify
                    check = Chem.MolFromSmiles(new_smi)
                    if check is not None and check.GetNumHeavyAtoms() <= 50:
                        return new_smi
            except Exception:
                continue

        except Exception:
            continue

    return None


# ============================================================================
# Batch Operations for Julia Interface
# ============================================================================

def batch_crossover(smiles_list, n_children, scores=None):
    """
    Generate n_children via crossover from a scored population.

    Args:
        smiles_list: list of SMILES strings
        n_children: number of children to generate
        scores: optional list of scores (for tournament selection)

    Returns: list of valid child SMILES
    """
    if len(smiles_list) < 2:
        return []

    children = []
    attempts = 0
    max_attempts = n_children * 5

    while len(children) < n_children and attempts < max_attempts:
        attempts += 1

        # Tournament selection (select 2 parents)
        if scores is not None and len(scores) == len(smiles_list):
            # Score-weighted selection
            idx1 = _tournament_select(scores)
            idx2 = _tournament_select(scores)
            while idx2 == idx1 and len(smiles_list) > 2:
                idx2 = _tournament_select(scores)
        else:
            idx1 = random.randint(0, len(smiles_list) - 1)
            idx2 = random.randint(0, len(smiles_list) - 1)
            while idx2 == idx1 and len(smiles_list) > 2:
                idx2 = random.randint(0, len(smiles_list) - 1)

        child = crossover(smiles_list[idx1], smiles_list[idx2])
        if child is not None and child not in children:
            children.append(child)

    return children


def batch_mutate(smiles_list, n_mutants, scores=None):
    """
    Generate n_mutants via mutation from a scored population.

    Args:
        smiles_list: list of SMILES strings
        n_mutants: number of mutants to generate
        scores: optional list of scores (for tournament selection)

    Returns: list of valid mutant SMILES
    """
    if len(smiles_list) < 1:
        return []

    mutants = []
    attempts = 0
    max_attempts = n_mutants * 5

    while len(mutants) < n_mutants and attempts < max_attempts:
        attempts += 1

        if scores is not None and len(scores) == len(smiles_list):
            idx = _tournament_select(scores)
        else:
            idx = random.randint(0, len(smiles_list) - 1)

        child = mutate(smiles_list[idx])
        if child is not None and child not in mutants:
            mutants.append(child)

    return mutants


def _tournament_select(scores, k=3):
    """Tournament selection: pick best of k random candidates."""
    n = len(scores)
    candidates = random.sample(range(n), min(k, n))
    best = max(candidates, key=lambda i: scores[i])
    return best


def graph_ga_step(smiles_list, scores, n_crossover=10, n_mutation=10):
    """
    One step of Graph GA: generate children via crossover + mutation.

    This is the main entry point called from Julia.

    Args:
        smiles_list: list of parent SMILES
        scores: list of parent scores (same order)
        n_crossover: number of crossover children
        n_mutation: number of mutation children

    Returns: list of unique valid SMILES (children)
    """
    children = set()

    # Crossover
    xover = batch_crossover(smiles_list, n_crossover, scores)
    children.update(xover)

    # Mutation
    muts = batch_mutate(smiles_list, n_mutation, scores)
    children.update(muts)

    # Remove parents from children
    parent_set = set(smiles_list)
    children = [s for s in children if s not in parent_set]

    return children


def scaffold_preserving_crossover(smiles_list, scaffolds, scores,
                                   target_scaffold=None, n_children=10):
    """
    Scaffold-preserving crossover: only exchange decorations, keeping scaffold intact.

    Groups molecules by scaffold, then crosses decorations within each group.

    Args:
        smiles_list: list of SMILES
        scaffolds: list of scaffold SMILES (same order)
        scores: list of scores (same order)
        target_scaffold: if provided, focus on this scaffold
        n_children: number of children to generate

    Returns: list of child SMILES
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    # Group molecules by scaffold
    scaffold_groups = {}
    for smi, scaf, score in zip(smiles_list, scaffolds, scores):
        if scaf not in scaffold_groups:
            scaffold_groups[scaf] = []
        scaffold_groups[scaf].append((smi, score))

    # Focus on target scaffold if specified
    if target_scaffold and target_scaffold in scaffold_groups:
        groups_to_process = {target_scaffold: scaffold_groups[target_scaffold]}
    else:
        # Process scaffolds with at least 2 molecules
        groups_to_process = {k: v for k, v in scaffold_groups.items()
                             if len(v) >= 2}

    if not groups_to_process:
        return []

    children = []
    for scaffold, mols in groups_to_process.items():
        if len(mols) < 2:
            continue

        smis = [m[0] for m in mols]
        scrs = [m[1] for m in mols]

        # Use non-ring crossover within scaffold group
        # (ring crossover would destroy the scaffold)
        for _ in range(n_children):
            idx1 = _tournament_select(scrs, k=2)
            idx2 = _tournament_select(scrs, k=2)
            if idx1 == idx2:
                continue

            mol1 = Chem.MolFromSmiles(smis[idx1])
            mol2 = Chem.MolFromSmiles(smis[idx2])
            if mol1 is None or mol2 is None:
                continue

            child = _non_ring_crossover(mol1, mol2)
            if child and child not in children:
                # Verify scaffold is preserved
                try:
                    child_mol = Chem.MolFromSmiles(child)
                    if child_mol is not None:
                        child_scaffold = Chem.MolToSmiles(
                            MurckoScaffold.GetScaffoldForMol(child_mol))
                        if child_scaffold == scaffold:
                            children.append(child)
                except Exception:
                    pass

            if len(children) >= n_children:
                break

        if len(children) >= n_children:
            break

    return children[:n_children]
