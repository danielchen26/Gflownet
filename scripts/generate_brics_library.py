#!/usr/bin/env python3
"""
Generate a curated BRICS fragment library from well-known drug molecules.

This script uses RDKit's BRICS decomposition to extract fragments from a
predefined set of ~100 drug-like molecules, filters and deduplicates them,
categorizes them by BRICS labels, and exports the result to JSON.

Usage:
    python generate_brics_library.py [--output PATH] [--min-heavy 3] [--max-heavy 25]

Requirements:
    - rdkit (pip install rdkit or conda install -c conda-forge rdkit)

Output format:
    JSON file with fields: version, source, n_fragments, fragments[], compatibility_matrix{}
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import BRICS, AllChem, Descriptors, rdMolDescriptors
except ImportError:
    print("ERROR: RDKit is required. Install with:")
    print("  pip install rdkit")
    print("  or: conda install -c conda-forge rdkit")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Curated drug molecule SMILES (~100 well-known drugs)
# ---------------------------------------------------------------------------
DRUG_MOLECULES: List[Tuple[str, str]] = [
    # Analgesics / Anti-inflammatory
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
    ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("indomethacin", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1"),
    ("piroxicam", "CN1C(C(=O)Nc2ccccn2)=C(O)c2ccccc2S1(=O)=O"),
    ("meloxicam", "CN1C(C(=O)Nc2nccs2)=C(O)c2ccccc2S1(=O)=O"),
    ("celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),

    # CNS / Psychotropics
    ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    ("fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
    ("sertraline", "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"),
    ("escitalopram", "Fc1ccc(C2(CCCN(C)C)OCc3cc(C#N)ccc32)cc1"),
    ("paroxetine", "Fc1ccc(C2CCNCC2COc2ccc3c(c2)OCO3)cc1"),
    ("venlafaxine", "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1"),
    ("duloxetine", "CNCC(Oc1cccc2ccccc12)c1cccs1"),
    ("bupropion", "CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1"),
    ("aripiprazole", "Clc1cccc(N2CCN(CCCCOc3ccc4c(c3)CCC(=O)N4)CC2)c1Cl"),
    ("quetiapine", "OCCOCCN1CCN(c2c3ccc(Cl)cc3Sc3ccccc32)CC1"),
    ("olanzapine", "Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1"),
    ("risperidone", "Cc1nc2n(c1CC(=O)N1CCC(c3noc4cc(F)ccc34)CC1)CCCC2"),
    ("carbamazepine", "NC(=O)N1c2ccccc2C=Cc2ccccc21"),
    ("phenytoin", "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1"),
    ("lamotrigine", "Nc1nnc(-c2cccc(Cl)c2Cl)c(N)n1"),
    ("gabapentin", "NCC1(CC(=O)O)CCCCC1"),

    # Cardiovascular
    ("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"),
    ("rosuvastatin", "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/C(O)CC(O)CC(=O)O"),
    ("simvastatin", "CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21"),
    ("losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1"),
    ("valsartan", "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1)C(C(=O)O)C(C)C"),
    ("amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl"),
    ("metoprolol", "COCCc1ccc(OCC(O)CNC(C)C)cc1"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("lisinopril", "NCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O"),
    ("enalapril", "CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O"),
    ("ramipril", "CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1C2CCCC2CC1C(=O)O"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("clopidogrel", "COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1"),
    ("rivaroxaban", "O=C(NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1)c1ccc(Cl)s1"),
    ("apixaban", "COc1ccc(-n2nc(C(N)=O)c3c2C(=O)N(c2ccc(N4CCOCC4)cc2)CC3)cc1"),
    ("digoxin", "CC1OC(OC2CCC3(C)C(CCC4C3CC(O)C3(C)C(C5=CC(=O)OC5)CCC43)C2)CC(O)C1O"),

    # Antidiabetics
    ("metformin", "CN(C)C(=N)NC(=N)N"),
    ("glipizide", "Cc1cnc(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NC3CCCCC3)cc2)cn1"),
    ("pioglitazone", "O=C1NC(=O)C(Cc2ccc(OCCc3ncccc3)cc2)S1"),
    ("sitagliptin", "N[C@@H](CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F"),
    ("dapagliflozin", "OC1OC(CO)C(O)C(O)C1c1cc(Cc2ccc(OCC)cc2)ccc1Cl"),
    ("empagliflozin", "OCC1OC(c2cc(Cc3ccc4c(c3)OCC4)ccc2Cl)C(O)C(O)C1O"),
    ("canagliflozin", "OCC1OC(c2cc(-c3ccc(F)cc3)c(Cc3ccc(S(=O)(=O)C4CCCC4)s3)s2)C(O)C(O)C1O"),

    # GI / Antacids
    ("omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
    ("pantoprazole", "COc1ccnc(CS(=O)c2nc3cc(OC(F)F)ccc3[nH]2)c1OC"),
    ("lansoprazole", "Cc1c(OCC(F)(F)F)ccnc1CS(=O)c1nc2ccccc2[nH]1"),
    ("ranitidine", "CNC(/N=C/[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1"),
    ("famotidine", "NC(N)=Nc1nc(CSCCC(=N)NS(N)(=O)=O)cs1"),

    # Antibiotics
    ("ciprofloxacin", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
    ("levofloxacin", "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23"),
    ("amoxicillin", "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"),
    ("azithromycin", "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O"),
    ("doxycycline", "O=C1C2=C(O)c3c(O)cccc3C(O)(C)C2CC2C(N(C)C)C(O)=C(C(N)=O)C(=O)C12"),
    ("metronidazole", "Cc1ncc([N+](=O)[O-])n1CCO"),
    ("trimethoprim", "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC"),
    ("clindamycin", "CCCC1CC(C(=O)NC(C(O)C(=O)O)C(O)CO)N(C)C1"),
    ("vancomycin_frag", "Oc1cc(O)c(Cl)c(O)c1"),  # simplified fragment

    # Antivirals
    ("acyclovir", "Nc1nc(=O)c2ncn(COCCO)c2[nH]1"),
    ("oseltamivir", "CCOC(=O)C1=CC(OC(CC)CC)C(NC(C)=O)C(N)C1"),
    ("remdesivir_core", "c1nc(N)c2ncnn2c1"),  # simplified adenine core

    # Respiratory
    ("montelukast", "CC(C)(O)c1ccccc1CCC(SCC1(CC(=O)O)CC1)c1cccc(/C=C/c2ccc3ccc(Cl)cc3n2)c1"),
    ("salbutamol", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"),
    ("budesonide", "CCC(=O)OC1(C(C)CC)CC2C3CCC4=CC(=O)C=CC4(C)C3C(O)CC2(C)C1=O"),
    ("fluticasone_frag", "OC(F)(F)C(F)c1ccccc1"),  # simplified fragment
    ("theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    ("tiotropium_frag", "c1cc(C(O)(c2cccs2)c2cccs2)cs1"),  # simplified fragment

    # Antidepressants / Anxiolytics (additional)
    ("mirtazapine", "CN1CCN2c3ccccc3Cc3cccnc3C2C1"),
    ("trazodone", "Clc1cccc(N2CCN(CCCN3C(=O)c4ccccc4N=C3)CC2)c1"),
    ("buspirone", "O=C1CC2(CCCC2)CC(=O)N1CCCCN1CCN(c2ncccn2)CC1"),
    ("zolpidem", "Cc1ccc2nc(-c3ccc(C)cn3)c(CC(=O)N(C)C)c2c1C"),
    ("alprazolam", "Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2"),

    # Migraine / Neuro
    ("sumatriptan", "CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1"),
    ("rizatriptan", "CN(C)CCc1c[nH]c2ccc(Cn3nncc3)cc12"),
    ("levetiracetam", "CCC(C(N)=O)N1CCCC1=O"),

    # Immunosuppressants / Anti-cancer (fragments)
    ("methotrexate", "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1"),
    ("imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccn2)n1"),
    ("tamoxifen", "CCC(=C(c1ccccc1)c1ccccc1)c1ccc(OCCN(C)C)cc1"),
    ("hydroxychloroquine", "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12"),
    ("lenalidomide", "Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O"),

    # Thyroid
    ("levothyroxine_frag", "Oc1cc(I)c(Oc2ccc(O)c(I)c2)c(I)c1"),  # simplified

    # Antifungals
    ("fluconazole", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"),
    ("ketoconazole_frag", "Clc1ccc(C2(c3ccccc3)OCCO2)cc1"),  # simplified
    ("terbinafine", "CN(C/C=C/C#CC(C)(C)C)Cc1cccc2ccccc12"),

    # Muscle relaxants
    ("cyclobenzaprine", "CN(C)CCC=C1c2ccccc2C=Cc2ccccc21"),
    ("methocarbamol", "COc1ccc(OCC(O)COC(N)=O)cc1"),

    # Anticoagulants (additional)
    ("dabigatran_frag", "Nc1nc(C(=O)OCC)c2cc(C(=O)c3ccc(NCC)cc3)ccc2n1"),  # simplified

    # Dermatology
    ("tretinoin", "CC1=C(/C=C/C(C)=C/C=C/C(=O)O)C(C)(C)CCC1"),
    ("adapalene_frag", "C1CCC(c2ccc(-c3cccc4cc(C(=O)O)ccc34)cc2)CC1"),  # simplified

    # Anti-gout
    ("allopurinol", "O=c1[nH]cnc2[nH]ncc12"),
    ("colchicine_frag", "COc1ccc2c(c1OC)C(NC(C)=O)Cc1cc(=O)c(OC)cc1-2"),

    # Antiepileptics
    ("valproic_acid", "CCCC(CCC)C(=O)O"),
    ("topiramate_frag", "CC1(C)OC2COC3(COS(N)(=O)=O)OC(C)(C)OC3C2O1"),

    # Prokinetics / Antiemetics
    ("ondansetron", "Cc1ncc2c3ccccc3n(CC3CCNCC3)c(=O)c2c1C"),
    ("domperidone", "O=C1NC2(CCN(CCCN3C(=O)c4cc(Cl)ccc4NC3=O)CC2)c2ccccc21"),

    # Urology
    ("sildenafil", "CCCc1nn(C)c2c1nc(nc2OCC)-c1cc(S(=O)(=O)N1CCN(C)CC1)ccc1OCC"),
    ("tadalafil", "CN1CC(=O)N2C(Cc3c2ccc2c3CCN2C(=O)OCc2ccccc2)C1=O"),
    ("tamsulosin", "CCOc1ccc(CC(C)NCCC(OC)c2ccc(OC)c(S(N)(=O)=O)c2)cc1"),

    # Ophthalmology
    ("latanoprost_frag", "CCCCC(O)/C=C/C1CC(O)CC1/C=C/c1ccccc1"),  # simplified

    # Miscellaneous
    ("metoclopramide", "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC"),
    ("prednisone_frag", "O=C1C=C2CCC3C(CCC4(O)C(=O)CCC34C)C2(C)CC1"),  # simplified steroid core
]


# ---------------------------------------------------------------------------
# BRICS compatibility rules (from the original BRICS paper / RDKit)
# Label pairs that can form bonds.
# ---------------------------------------------------------------------------
BRICS_COMPATIBLE_PAIRS: List[Tuple[int, int]] = [
    (1, 3), (1, 5), (1, 10),
    (2, 12), (2, 14), (2, 16),
    (3, 4), (3, 13), (3, 14), (3, 15), (3, 16),
    (4, 5), (4, 11),
    (5, 6), (5, 12), (5, 13), (5, 15),
    (6, 7), (6, 13), (6, 14), (6, 15), (6, 16),
    (7, 7),
    (8, 9), (8, 10), (8, 13), (8, 14), (8, 15), (8, 16),
    (9, 13), (9, 14), (9, 15), (9, 16),
    (10, 13), (10, 14), (10, 15), (10, 16),
    (11, 12), (11, 13), (11, 14), (11, 15), (11, 16),
    (12, 13), (12, 14), (12, 15), (12, 16),
    (13, 14), (13, 15), (13, 16),
    (14, 14), (14, 15), (14, 16),
    (15, 16),
]


def build_compatibility_matrix() -> Dict[str, bool]:
    """Build the BRICS label compatibility matrix as a dict of 'X-Y' -> true."""
    matrix: Dict[str, bool] = {}
    for a, b in BRICS_COMPATIBLE_PAIRS:
        matrix[f"{a}-{b}"] = True
        if a != b:
            matrix[f"{b}-{a}"] = True
    return matrix


def extract_brics_labels(smiles: str) -> List[int]:
    """Extract BRICS dummy atom labels from a fragment SMILES string.

    BRICS fragments use [X*] notation where X is the BRICS label number.
    For example, [3*]OCC has BRICS label 3.
    """
    labels = sorted(set(int(x) for x in re.findall(r'\[(\d+)\*\]', smiles)))
    return labels


def count_attachment_points(smiles: str) -> int:
    """Count the number of attachment points ([*] or [N*]) in a SMILES string."""
    return len(re.findall(r'\[\d*\*\]', smiles))


def get_heavy_atom_count(mol: Chem.Mol) -> int:
    """Count heavy atoms excluding dummy atoms ([*])."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0)


def categorize_fragment(smiles: str, n_attachments: int, heavy_atoms: int,
                        mol: Chem.Mol) -> str:
    """Categorize a fragment based on its structure.

    Categories:
        - ring: contains at least one ring
        - functional_group: single attachment, no rings, small
        - linker: two or more attachments, connects fragments
        - starter: two or more attachments with ring systems (good seed)
        - chain: aliphatic chain fragment
    """
    ring_info = mol.GetRingInfo()
    has_ring = ring_info.NumRings() > 0

    if n_attachments >= 2:
        if has_ring:
            return "starter"
        else:
            return "linker"
    elif has_ring:
        return "ring"
    elif heavy_atoms <= 5:
        return "functional_group"
    else:
        return "chain"


def generate_fragment_name(smiles: str, category: str, idx: int) -> str:
    """Generate a human-readable name for a fragment."""
    # Try to identify common substructures
    name_patterns = [
        ("[*]O", "hydroxyl"),
        ("[*]N", "amine"),
        ("[*]F", "fluoride"),
        ("[*]Cl", "chloride"),
        ("[*]C#N", "nitrile"),
        ("c1ccccc1", "phenyl"),
        ("c1ccncc1", "pyridyl"),
        ("c1ccoc1", "furyl"),
        ("c1ccsc1", "thienyl"),
        ("c1cc[nH]c1", "pyrrolyl"),
        ("C1CCCCC1", "cyclohexyl"),
        ("C1CCNCC1", "piperidyl"),
        ("C1CCOCC1", "morpholino"),
        ("S(=O)(=O)", "sulfonyl"),
        ("C(=O)N", "amide"),
        ("C(=O)O", "ester_acid"),
        ("C(F)(F)F", "trifluoromethyl"),
        ("OC", "methoxy"),
    ]

    for pattern, name in name_patterns:
        if pattern in smiles:
            return f"{name}_{idx}"

    return f"{category}_{idx}"


def compute_fingerprint(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    """Compute Morgan fingerprint for a molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def remove_near_duplicates(
    fragments: List[Dict[str, Any]],
    threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """Remove near-duplicate fragments using Tanimoto similarity on Morgan FPs.

    When two fragments have Tanimoto >= threshold, keep the one with higher
    frequency (or more attachment points as tiebreaker).
    """
    if not fragments:
        return fragments

    # Compute fingerprints
    fps = []
    valid_fragments = []
    for frag in fragments:
        mol = Chem.MolFromSmiles(frag["smiles"])
        if mol is not None:
            fp = compute_fingerprint(mol)
            fps.append(fp)
            valid_fragments.append(frag)

    # Mark duplicates
    keep = [True] * len(valid_fragments)
    for i in range(len(valid_fragments)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(valid_fragments)):
            if not keep[j]:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim >= threshold:
                # Keep the one with higher frequency, or more attachments
                freq_i = valid_fragments[i].get("frequency", 0)
                freq_j = valid_fragments[j].get("frequency", 0)
                att_i = valid_fragments[i].get("n_attachments", 0)
                att_j = valid_fragments[j].get("n_attachments", 0)

                if freq_j > freq_i or (freq_j == freq_i and att_j > att_i):
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return [f for f, k in zip(valid_fragments, keep) if k]


def decompose_drugs(
    min_heavy: int = 3,
    max_heavy: int = 25,
    min_fragment_size: int = 3,
) -> List[Dict[str, Any]]:
    """Decompose all drug molecules using BRICS and collect unique fragments.

    Args:
        min_heavy: Minimum number of heavy atoms in a fragment.
        max_heavy: Maximum number of heavy atoms in a fragment.
        min_fragment_size: Passed to BRICS.BRICSDecompose minFragmentSize.

    Returns:
        List of fragment dictionaries.
    """
    fragment_counter: Counter = Counter()
    all_smiles: Set[str] = set()

    print(f"Decomposing {len(DRUG_MOLECULES)} drug molecules...")
    failed = 0

    for drug_name, smiles in DRUG_MOLECULES:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  WARNING: Could not parse SMILES for {drug_name}: {smiles}")
            failed += 1
            continue

        try:
            frags = BRICS.BRICSDecompose(mol, minFragmentSize=min_fragment_size)
        except Exception as e:
            print(f"  WARNING: BRICS decomposition failed for {drug_name}: {e}")
            failed += 1
            continue

        for frag_smiles in frags:
            # Canonicalize
            frag_mol = Chem.MolFromSmiles(frag_smiles)
            if frag_mol is None:
                continue
            canonical = Chem.MolToSmiles(frag_mol)
            fragment_counter[canonical] += 1

    print(f"  Parsed {len(DRUG_MOLECULES) - failed}/{len(DRUG_MOLECULES)} molecules")
    print(f"  Found {len(fragment_counter)} unique raw fragments")

    # Filter by heavy atom count and build fragment list
    fragments: List[Dict[str, Any]] = []
    for smiles, freq in fragment_counter.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        heavy = get_heavy_atom_count(mol)
        if heavy < min_heavy or heavy > max_heavy:
            continue

        n_attach = count_attachment_points(smiles)
        brics_labels = extract_brics_labels(smiles)
        category = categorize_fragment(smiles, n_attach, heavy, mol)

        fragments.append({
            "smiles": smiles,
            "canonical": smiles,  # already canonical from above
            "brics_labels": brics_labels,
            "n_attachments": n_attach,
            "heavy_atoms": heavy,
            "frequency": freq,
            "category": category,
            "is_starter": n_attach >= 2,
        })

    print(f"  After heavy-atom filter ({min_heavy}-{max_heavy}): {len(fragments)} fragments")
    return fragments


def main():
    parser = argparse.ArgumentParser(
        description="Generate a curated BRICS fragment library from drug molecules."
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "fragment_libraries", "brics_drug_fragments.json"
        ),
        help="Output JSON file path."
    )
    parser.add_argument(
        "--min-heavy", type=int, default=3,
        help="Minimum heavy atoms per fragment (default: 3)."
    )
    parser.add_argument(
        "--max-heavy", type=int, default=25,
        help="Maximum heavy atoms per fragment (default: 25)."
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.9,
        help="Tanimoto threshold for near-duplicate removal (default: 0.9)."
    )
    args = parser.parse_args()

    # Step 1: BRICS decomposition
    fragments = decompose_drugs(
        min_heavy=args.min_heavy,
        max_heavy=args.max_heavy,
    )

    # Step 2: Remove near-duplicates
    print(f"Removing near-duplicates (Tanimoto >= {args.dedup_threshold})...")
    fragments = remove_near_duplicates(fragments, threshold=args.dedup_threshold)
    print(f"  After deduplication: {len(fragments)} fragments")

    # Step 3: Sort by category then frequency (descending)
    category_order = {"ring": 0, "functional_group": 1, "chain": 2, "linker": 3, "starter": 4}
    fragments.sort(key=lambda f: (category_order.get(f["category"], 99), -f["frequency"]))

    # Step 4: Assign IDs and names
    for idx, frag in enumerate(fragments, 1):
        frag["id"] = idx
        frag["name"] = generate_fragment_name(frag["smiles"], frag["category"], idx)

    # Step 5: Build compatibility matrix
    compatibility_matrix = build_compatibility_matrix()

    # Step 6: Build output
    output = {
        "version": "1.0",
        "source": "drug_decomposition",
        "n_fragments": len(fragments),
        "fragments": [
            {
                "id": f["id"],
                "smiles": f["smiles"],
                "canonical": f["canonical"],
                "name": f["name"],
                "brics_labels": f["brics_labels"],
                "n_attachments": f["n_attachments"],
                "heavy_atoms": f["heavy_atoms"],
                "frequency": f["frequency"],
                "category": f["category"],
                "is_starter": f["is_starter"],
            }
            for f in fragments
        ],
        "compatibility_matrix": compatibility_matrix,
    }

    # Step 7: Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(fragments)} fragments to {args.output}")

    # Summary statistics
    cat_counts = Counter(f["category"] for f in fragments)
    print("\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    starter_count = sum(1 for f in fragments if f["is_starter"])
    print(f"\nStarters (2+ attachment points): {starter_count}")

    label_counts = Counter()
    for f in fragments:
        for label in f["brics_labels"]:
            label_counts[label] += 1
    print("\nBRICS label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count} fragments")


if __name__ == "__main__":
    main()
