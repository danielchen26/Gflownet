// Mock data for demo mode — realistic drug-like molecules with properties
// Used when the backend is unavailable so all pages are functional

import type {
  Molecule,
  MolecularProperties,
  ChemicalSpacePoint,
  AtomAttribution,
  RewardDecompositionData,
} from './api'

// --- Drug-like molecules with real SMILES ---

const DRUG_MOLECULES: Array<{
  smiles: string
  name: string
  props: MolecularProperties
}> = [
  {
    smiles: 'CC(=O)Oc1ccccc1C(=O)O',
    name: 'Aspirin',
    props: { molecular_weight: 180.16, logp: 1.2, qed: 0.55, synthetic_accessibility: 1.6, tpsa: 63.6, rotatable_bonds: 3, hbd: 1, hba: 4, num_rings: 1, num_aromatic_rings: 1, formula: 'C9H8O4' },
  },
  {
    smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    name: 'Ibuprofen',
    props: { molecular_weight: 206.28, logp: 3.5, qed: 0.71, synthetic_accessibility: 1.4, tpsa: 37.3, rotatable_bonds: 4, hbd: 1, hba: 2, num_rings: 1, num_aromatic_rings: 1, formula: 'C13H18O2' },
  },
  {
    smiles: 'O=C(O)c1ccccc1O',
    name: 'Salicylic acid',
    props: { molecular_weight: 138.12, logp: 1.1, qed: 0.54, synthetic_accessibility: 1.2, tpsa: 57.5, rotatable_bonds: 1, hbd: 2, hba: 3, num_rings: 1, num_aromatic_rings: 1, formula: 'C7H6O3' },
  },
  {
    smiles: 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
    name: 'Caffeine',
    props: { molecular_weight: 194.19, logp: -0.07, qed: 0.53, synthetic_accessibility: 2.1, tpsa: 58.4, rotatable_bonds: 0, hbd: 0, hba: 6, num_rings: 2, num_aromatic_rings: 2, formula: 'C8H10N4O2' },
  },
  {
    smiles: 'CC(=O)Nc1ccc(O)cc1',
    name: 'Acetaminophen',
    props: { molecular_weight: 151.16, logp: 0.46, qed: 0.74, synthetic_accessibility: 1.0, tpsa: 49.3, rotatable_bonds: 1, hbd: 2, hba: 3, num_rings: 1, num_aromatic_rings: 1, formula: 'C8H9NO2' },
  },
  {
    smiles: 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
    name: 'Pyrene',
    props: { molecular_weight: 202.25, logp: 4.88, qed: 0.38, synthetic_accessibility: 1.5, tpsa: 0.0, rotatable_bonds: 0, hbd: 0, hba: 0, num_rings: 4, num_aromatic_rings: 4, formula: 'C16H10' },
  },
  {
    smiles: 'OC(=O)c1cc(O)c(O)c(O)c1',
    name: 'Gallic acid',
    props: { molecular_weight: 170.12, logp: 0.7, qed: 0.63, synthetic_accessibility: 1.3, tpsa: 97.99, rotatable_bonds: 1, hbd: 4, hba: 5, num_rings: 1, num_aromatic_rings: 1, formula: 'C7H6O5' },
  },
  {
    smiles: 'c1ccc(cc1)-c1ccc(cc1)N(c1ccccc1)c1ccccc1',
    name: 'TPA Derivative',
    props: { molecular_weight: 321.41, logp: 5.94, qed: 0.31, synthetic_accessibility: 2.3, tpsa: 3.24, rotatable_bonds: 4, hbd: 0, hba: 1, num_rings: 4, num_aromatic_rings: 4, formula: 'C24H19N' },
  },
  {
    smiles: 'CC(C)NCC(O)c1ccc(O)c(O)c1',
    name: 'Isoproterenol',
    props: { molecular_weight: 211.26, logp: 0.08, qed: 0.69, synthetic_accessibility: 2.0, tpsa: 72.72, rotatable_bonds: 4, hbd: 4, hba: 4, num_rings: 1, num_aromatic_rings: 1, formula: 'C11H17NO3' },
  },
  {
    smiles: 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
    name: 'Glucose',
    props: { molecular_weight: 180.16, logp: -2.6, qed: 0.25, synthetic_accessibility: 4.8, tpsa: 110.38, rotatable_bonds: 1, hbd: 5, hba: 6, num_rings: 1, num_aromatic_rings: 0, formula: 'C6H12O6' },
  },
  {
    smiles: 'O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12',
    name: 'Apigenin',
    props: { molecular_weight: 270.24, logp: 2.11, qed: 0.67, synthetic_accessibility: 2.2, tpsa: 90.9, rotatable_bonds: 1, hbd: 3, hba: 5, num_rings: 3, num_aromatic_rings: 3, formula: 'C15H10O5' },
  },
  {
    smiles: 'COc1cc2c(cc1OC)-c1cc3ccc(OC)c(OC)c3c[n+]1CC2',
    name: 'Berberine',
    props: { molecular_weight: 336.36, logp: 0.2, qed: 0.52, synthetic_accessibility: 3.5, tpsa: 40.8, rotatable_bonds: 2, hbd: 0, hba: 5, num_rings: 4, num_aromatic_rings: 4, formula: 'C20H18NO4+' },
  },
  {
    smiles: 'CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O',
    name: 'Testosterone',
    props: { molecular_weight: 288.42, logp: 3.32, qed: 0.59, synthetic_accessibility: 5.2, tpsa: 37.3, rotatable_bonds: 0, hbd: 1, hba: 2, num_rings: 4, num_aromatic_rings: 0, formula: 'C19H28O2' },
  },
  {
    smiles: 'CC(C)Cc1ccc(-c2csc(NC(=O)c3ccc(F)cc3)n2)cc1',
    name: 'GFlowNet Mol 14',
    props: { molecular_weight: 370.47, logp: 4.2, qed: 0.81, synthetic_accessibility: 2.8, tpsa: 72.4, rotatable_bonds: 5, hbd: 1, hba: 4, num_rings: 3, num_aromatic_rings: 3, formula: 'C21H21FN2OS' },
  },
  {
    smiles: 'Cc1cc(-c2ccc(NC(=O)CN3CCCC3)cc2)n(-c2ccccc2)n1',
    name: 'GFlowNet Mol 15',
    props: { molecular_weight: 348.44, logp: 2.7, qed: 0.78, synthetic_accessibility: 2.5, tpsa: 56.1, rotatable_bonds: 5, hbd: 1, hba: 4, num_rings: 3, num_aromatic_rings: 3, formula: 'C21H24N4O' },
  },
  {
    smiles: 'O=C(Nc1cccc(-c2ccncc2)c1)c1ccc(F)c(Cl)c1',
    name: 'GFlowNet Mol 16',
    props: { molecular_weight: 330.75, logp: 3.9, qed: 0.76, synthetic_accessibility: 2.1, tpsa: 41.1, rotatable_bonds: 3, hbd: 1, hba: 3, num_rings: 3, num_aromatic_rings: 3, formula: 'C18H12ClFN2O' },
  },
  {
    smiles: 'COc1ccc(-c2nc3ccccc3o2)cc1NC(=O)c1ccco1',
    name: 'GFlowNet Mol 17',
    props: { molecular_weight: 334.35, logp: 3.1, qed: 0.83, synthetic_accessibility: 2.4, tpsa: 72.8, rotatable_bonds: 4, hbd: 1, hba: 5, num_rings: 4, num_aromatic_rings: 4, formula: 'C19H14N2O4' },
  },
  {
    smiles: 'CC(C)(C)c1ccc(-c2nn(-c3ccc(C(=O)O)cc3)c(=O)c3ccccc23)cc1',
    name: 'GFlowNet Mol 18',
    props: { molecular_weight: 414.49, logp: 4.5, qed: 0.72, synthetic_accessibility: 3.1, tpsa: 67.7, rotatable_bonds: 4, hbd: 1, hba: 4, num_rings: 4, num_aromatic_rings: 4, formula: 'C26H22N2O3' },
  },
  {
    smiles: 'Nc1nc2ccc(-c3ccc(S(N)(=O)=O)cc3)cc2s1',
    name: 'GFlowNet Mol 19',
    props: { molecular_weight: 305.37, logp: 1.8, qed: 0.77, synthetic_accessibility: 2.6, tpsa: 101.4, rotatable_bonds: 2, hbd: 3, hba: 5, num_rings: 3, num_aromatic_rings: 3, formula: 'C13H11N3O2S2' },
  },
  {
    smiles: 'O=C(c1ccc(-c2ccccn2)cc1)N1CCN(Cc2ccco2)CC1',
    name: 'GFlowNet Mol 20',
    props: { molecular_weight: 347.41, logp: 2.3, qed: 0.85, synthetic_accessibility: 2.3, tpsa: 47.5, rotatable_bonds: 4, hbd: 0, hba: 5, num_rings: 4, num_aromatic_rings: 3, formula: 'C21H21N3O2' },
  },
]

// --- Generate mock molecule list ---

function computeReward(props: MolecularProperties): number {
  const qedScore = props.qed * 3.0
  const saScore = Math.max(0, (6 - props.synthetic_accessibility) / 6) * 3.0
  const lipinski =
    (props.molecular_weight <= 500 ? 1 : 0) +
    (props.logp <= 5 ? 1 : 0) +
    (props.hbd <= 5 ? 1 : 0) +
    (props.hba <= 10 ? 1 : 0)
  const lipScore = (lipinski / 4) * 2.5
  const diversity = 0.5 + Math.random() * 1.0
  return Math.min(10, Math.max(0, qedScore + saScore + lipScore + diversity))
}

let _mockMolecules: Molecule[] | null = null

export function getMockMolecules(): Molecule[] {
  if (_mockMolecules) return _mockMolecules

  _mockMolecules = DRUG_MOLECULES.map((drug, i) => {
    const reward = computeReward(drug.props)
    return {
      id: `mock-mol-${i.toString().padStart(3, '0')}`,
      smiles: drug.smiles,
      properties: drug.props,
      reward: parseFloat(reward.toFixed(2)),
      generation_step: Math.floor(Math.random() * 2000),
      method: ['de_novo', 'scaffold_hopping', 'fragment_linking', 'optimization'][i % 4],
    }
  }).sort((a, b) => b.reward - a.reward)

  return _mockMolecules
}

// --- Chemical space points with 2D embeddings ---

let _mockSpace: ChemicalSpacePoint[] | null = null

export function getMockChemicalSpace(method: string = 'umap'): ChemicalSpacePoint[] {
  if (_mockSpace) return _mockSpace

  const mols = getMockMolecules()
  const clusters = 4
  const clusterCenters = [
    { x: -3, y: 2 },
    { x: 2, y: 3 },
    { x: -1, y: -3 },
    { x: 4, y: -1 },
  ]

  _mockSpace = mols.map((mol, i) => {
    const cluster = i % clusters
    const center = clusterCenters[cluster]
    // Add Gaussian noise around cluster center
    const jitterX = (Math.random() - 0.5) * 3
    const jitterY = (Math.random() - 0.5) * 3

    return {
      id: mol.id,
      x: center.x + jitterX,
      y: center.y + jitterY,
      smiles: mol.smiles,
      reward: mol.reward,
      properties: mol.properties,
      cluster_id: cluster,
      generation_epoch: mol.generation_step,
    }
  })

  return _mockSpace
}

// --- Atom attribution (gradient saliency) ---

export function getMockAttribution(moleculeId: string): AtomAttribution | null {
  const mols = getMockMolecules()
  const mol = mols.find((m) => m.id === moleculeId)
  if (!mol) return null

  // Generate pseudo-random attribution scores based on SMILES length
  const numAtoms = mol.smiles.replace(/[^A-Z]/gi, '').length
  const scores = Array.from({ length: numAtoms }, (_, i) => {
    // Heteroatoms (N, O, S) get higher attribution
    const char = mol.smiles.charAt(i % mol.smiles.length)
    const base = 'NOS'.includes(char.toUpperCase()) ? 0.3 : 0.0
    return base + (Math.random() - 0.4) * 0.8
  })

  return {
    molecule_id: moleculeId,
    smiles: mol.smiles,
    atom_scores: scores,
    attribution_type: 'reward',
  }
}

// --- Reward decomposition ---

export function getMockRewardDecomposition(moleculeId: string): RewardDecompositionData | null {
  const mols = getMockMolecules()
  const mol = mols.find((m) => m.id === moleculeId)
  if (!mol) return null

  const p = mol.properties
  const components = [
    { name: 'QED Score', value: p.qed, weight: 1.0, contribution: p.qed * 1.0 },
    { name: 'Synthetic Access.', value: Math.max(0, (6 - p.synthetic_accessibility) / 6), weight: 0.8, contribution: Math.max(0, (6 - p.synthetic_accessibility) / 6) * 0.8 },
    { name: 'LogP Penalty', value: p.logp > 5 ? -(p.logp - 5) * 0.3 : 0, weight: 0.5, contribution: p.logp > 5 ? -(p.logp - 5) * 0.3 * 0.5 : 0 },
    { name: 'MW Penalty', value: p.molecular_weight > 500 ? -(p.molecular_weight - 500) / 500 : 0, weight: 0.3, contribution: p.molecular_weight > 500 ? -(p.molecular_weight - 500) / 500 * 0.3 : 0 },
    { name: 'Diversity Bonus', value: 0.15 + Math.random() * 0.15, weight: 0.4, contribution: (0.15 + Math.random() * 0.15) * 0.4 },
    { name: 'Novelty Bonus', value: 0.1 + Math.random() * 0.1, weight: 0.3, contribution: (0.1 + Math.random() * 0.1) * 0.3 },
  ]

  return {
    molecule_id: moleculeId,
    components,
    total_reward: mol.reward,
  }
}

// --- Generation DAG data ---

export interface DAGNode {
  id: string
  label: string
  smiles: string
  depth: number
  flow: number
  x?: number
  y?: number
}

export interface DAGEdge {
  source: string
  target: string
  action: string
  probability: number
}

export function getMockGenerationDAG(moleculeId: string): { nodes: DAGNode[]; edges: DAGEdge[] } | null {
  const mols = getMockMolecules()
  const mol = mols.find((m) => m.id === moleculeId)
  if (!mol) return null

  // Build a simplified generation DAG from SMILES
  const atoms = mol.smiles.replace(/[^A-Za-z]/g, '').split('')
  const steps = Math.min(atoms.length, 8)

  const nodes: DAGNode[] = [
    { id: 'root', label: 'Start', smiles: '', depth: 0, flow: 1.0 },
  ]
  const edges: DAGEdge[] = []

  let currentSmiles = ''
  for (let i = 0; i < steps; i++) {
    const atom = atoms[i] || 'C'
    currentSmiles += atom
    const nodeId = `step-${i}`
    const flow = 1.0 - (i / steps) * 0.7 + Math.random() * 0.2

    nodes.push({
      id: nodeId,
      label: `+${atom}`,
      smiles: currentSmiles,
      depth: i + 1,
      flow: parseFloat(flow.toFixed(3)),
    })

    edges.push({
      source: i === 0 ? 'root' : `step-${i - 1}`,
      target: nodeId,
      action: `Add ${atom}`,
      probability: 0.3 + Math.random() * 0.6,
    })

    // Add a branch at some steps
    if (i % 3 === 1 && i < steps - 1) {
      const altAtom = 'NOF'[i % 3]
      const altId = `alt-${i}`
      nodes.push({
        id: altId,
        label: `+${altAtom} (alt)`,
        smiles: currentSmiles.slice(0, -1) + altAtom,
        depth: i + 1,
        flow: parseFloat((flow * 0.3).toFixed(3)),
      })
      edges.push({
        source: i === 0 ? 'root' : `step-${i - 1}`,
        target: altId,
        action: `Add ${altAtom}`,
        probability: 0.1 + Math.random() * 0.2,
      })
    }
  }

  // Terminal node
  nodes.push({
    id: 'terminal',
    label: 'Complete',
    smiles: mol.smiles,
    depth: steps + 1,
    flow: parseFloat((mol.reward / 10).toFixed(3)),
  })
  edges.push({
    source: `step-${steps - 1}`,
    target: 'terminal',
    action: 'Stop',
    probability: 0.85,
  })

  return { nodes, edges }
}

// --- ADMET predictions ---

export interface ADMETData {
  absorption: { oral_bioavailability: number; caco2_permeability: number; pgp_substrate: boolean }
  distribution: { vd: number; plasma_protein_binding: number; bbb_penetration: boolean }
  metabolism: { cyp2d6_inhibitor: boolean; cyp3a4_inhibitor: boolean; half_life_hours: number }
  excretion: { clearance: number; renal_excretion: boolean }
  toxicity: { herg_inhibition: boolean; ames_mutagenicity: boolean; hepatotoxicity_risk: 'low' | 'medium' | 'high' }
}

export function getMockADMET(moleculeId: string): ADMETData {
  const mols = getMockMolecules()
  const mol = mols.find((m) => m.id === moleculeId)
  const p = mol?.properties

  return {
    absorption: {
      oral_bioavailability: p ? Math.min(1, 0.3 + p.qed * 0.5 + Math.random() * 0.2) : 0.6,
      caco2_permeability: -5.5 + Math.random() * 2,
      pgp_substrate: Math.random() > 0.5,
    },
    distribution: {
      vd: 0.5 + Math.random() * 3,
      plasma_protein_binding: 70 + Math.random() * 25,
      bbb_penetration: p ? p.tpsa < 90 && p.molecular_weight < 450 : Math.random() > 0.5,
    },
    metabolism: {
      cyp2d6_inhibitor: Math.random() > 0.7,
      cyp3a4_inhibitor: Math.random() > 0.6,
      half_life_hours: 2 + Math.random() * 12,
    },
    excretion: {
      clearance: 5 + Math.random() * 30,
      renal_excretion: Math.random() > 0.5,
    },
    toxicity: {
      herg_inhibition: Math.random() > 0.8,
      ames_mutagenicity: Math.random() > 0.85,
      hepatotoxicity_risk: (['low', 'low', 'medium', 'high'] as const)[Math.floor(Math.random() * 4)],
    },
  }
}
