import { useState, useCallback, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  FlaskConical, Atom, Link2, Scissors, RefreshCcw, Grid3X3,
  Beaker, Shield, CheckCircle, Target, Layers, Puzzle, Crosshair,
} from 'lucide-react'
import { SMILESInput } from './SMILESInput'
import { PropertySliderRange } from './PropertySliderRange'
import { PropertyRadarChart } from './PropertyRadarChart'
import { MoleculeViewer2D } from './MoleculeViewer2D'
import FragmentBrowser from './FragmentBrowser'
import DockingPanel from './DockingPanel'
import type { DomainOption } from './DomainSelector'
import type { MolecularConstraints } from '../services/api'

type GenerationMethod = 'de_novo' | 'scaffold_hopping' | 'fragment_linking' | 'r_group' | 'optimization' | 'grid_world'

const METHODS: Array<{
  id: GenerationMethod
  label: string
  description: string
  icon: typeof FlaskConical
}> = [
  { id: 'de_novo', label: 'De Novo', description: 'Build molecules atom-by-atom from scratch', icon: FlaskConical },
  { id: 'scaffold_hopping', label: 'Scaffold Hop', description: 'Replace core scaffold preserving activity', icon: RefreshCcw },
  { id: 'fragment_linking', label: 'Fragment Link', description: 'Connect molecular fragments via linker', icon: Link2 },
  { id: 'r_group', label: 'R-Group', description: 'Enumerate substituent variations', icon: Scissors },
  { id: 'optimization', label: 'Optimize', description: 'Refine existing lead compound', icon: Atom },
  { id: 'grid_world', label: 'Grid World', description: 'Abstract GFlowNet training domain', icon: Grid3X3 },
]

const SCORING_FUNCTIONS = [
  { id: 'qed', label: 'QED', description: 'Quantitative Estimate of Drug-likeness' },
  { id: 'logp', label: 'LogP', description: 'Partition coefficient optimization' },
  { id: 'binding_affinity', label: 'Binding Affinity', description: 'Docking score against target protein' },
  { id: 'multi_objective', label: 'Multi-Objective', description: 'Weighted combination of properties' },
  { id: 'custom', label: 'Custom', description: 'User-defined reward function' },
]

const ATOM_TYPES = [
  { symbol: 'C', name: 'Carbon', color: '#909090' },
  { symbol: 'N', name: 'Nitrogen', color: '#3050F8' },
  { symbol: 'O', name: 'Oxygen', color: '#FF0D0D' },
  { symbol: 'S', name: 'Sulfur', color: '#FFFF30' },
  { symbol: 'F', name: 'Fluorine', color: '#90E050' },
  { symbol: 'Cl', name: 'Chlorine', color: '#1FF01F' },
  { symbol: 'Br', name: 'Bromine', color: '#A62929' },
  { symbol: 'P', name: 'Phosphorus', color: '#FF8000' },
  { symbol: 'I', name: 'Iodine', color: '#940094' },
]

const PRESETS = [
  { label: 'Drug-like', description: 'Lipinski Ro5', constraints: { mw_range: [150, 500] as [number, number], logp_range: [-0.4, 5] as [number, number], qed_range: [0.5, 1] as [number, number], hbd_range: [0, 5] as [number, number], hba_range: [0, 10] as [number, number], lipinski: true } },
  { label: 'Fragment', description: 'Rule of 3', constraints: { mw_range: [100, 300] as [number, number], logp_range: [-1, 3] as [number, number], qed_range: [0, 1] as [number, number], hbd_range: [0, 3] as [number, number], hba_range: [0, 3] as [number, number], lipinski: false } },
  { label: 'PPI', description: 'Protein-protein', constraints: { mw_range: [400, 1000] as [number, number], logp_range: [1, 8] as [number, number], qed_range: [0, 1] as [number, number], hbd_range: [0, 10] as [number, number], hba_range: [0, 15] as [number, number], lipinski: false } },
  { label: 'Kinase', description: 'Kinase inhibitor', constraints: { mw_range: [300, 600] as [number, number], logp_range: [1, 5] as [number, number], qed_range: [0.4, 1] as [number, number], hbd_range: [0, 5] as [number, number], hba_range: [0, 10] as [number, number], lipinski: true } },
  { label: 'CNS', description: 'BBB-penetrant', constraints: { mw_range: [150, 450] as [number, number], logp_range: [1, 4] as [number, number], qed_range: [0.3, 1] as [number, number], tpsa_range: [0, 90] as [number, number], hbd_range: [0, 3] as [number, number], hba_range: [0, 7] as [number, number], lipinski: true } },
]

interface MolecularConfigPanelProps {
  domain: DomainOption
  config: Record<string, unknown>
  onConfigChange: (config: Record<string, unknown>) => void
  onValidationChange?: (isValid: boolean) => void
}

export function MolecularConfigPanel({
  domain,
  config,
  onConfigChange,
  onValidationChange,
}: MolecularConfigPanelProps) {
  const [method, setMethod] = useState<GenerationMethod>(
    (config.method as GenerationMethod) || 'de_novo'
  )
  const [seedSmiles, setSeedSmiles] = useState((config.seed_smiles as string) || '')
  const [scoringFunction, setScoringFunction] = useState('qed')
  const [allowedAtoms, setAllowedAtoms] = useState<string[]>(
    (config.allowed_atoms as string[]) || ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
  )
  const [maxAtoms, setMaxAtoms] = useState((config.max_atoms as number) || 38)
  const [constraints, setConstraints] = useState<MolecularConstraints>({
    mw_range: [150, 800],
    logp_range: [-2, 7],
    qed_range: [0, 1],
    sa_range: [1, 10],
    tpsa_range: [0, 200],
    rotatable_bonds_range: [0, 15],
    hbd_range: [0, 10],
    hba_range: [0, 15],
    lipinski: true,
    veber: false,
    pains_filter: true,
    brenk_filter: false,
    ...(config.constraints as Partial<MolecularConstraints> || {}),
  })

  const syncConfig = useCallback(
    (m: GenerationMethod, smiles: string, c: MolecularConstraints, atoms: string[], maxA: number, scoring: string) => {
      onConfigChange({
        method: m,
        seed_smiles: smiles || undefined,
        constraints: c,
        allowed_atoms: atoms,
        max_atoms: maxA,
        scoring_function: scoring,
      })
      onValidationChange?.(true)
    },
    [onConfigChange, onValidationChange]
  )

  const handleMethodChange = useCallback((m: GenerationMethod) => {
    setMethod(m)
    syncConfig(m, seedSmiles, constraints, allowedAtoms, maxAtoms, scoringFunction)
  }, [seedSmiles, constraints, allowedAtoms, maxAtoms, scoringFunction, syncConfig])

  const handleSmilesChange = useCallback((s: string) => {
    setSeedSmiles(s)
    syncConfig(method, s, constraints, allowedAtoms, maxAtoms, scoringFunction)
  }, [method, constraints, allowedAtoms, maxAtoms, scoringFunction, syncConfig])

  const handleConstraintsChange = useCallback((partial: Partial<MolecularConstraints>) => {
    setConstraints((prev) => {
      const next = { ...prev, ...partial }
      syncConfig(method, seedSmiles, next, allowedAtoms, maxAtoms, scoringFunction)
      return next
    })
  }, [method, seedSmiles, allowedAtoms, maxAtoms, scoringFunction, syncConfig])

  const handleAtomToggle = useCallback((symbol: string) => {
    setAllowedAtoms((prev) => {
      const next = prev.includes(symbol) ? prev.filter((a) => a !== symbol) : [...prev, symbol]
      syncConfig(method, seedSmiles, constraints, next, maxAtoms, scoringFunction)
      return next
    })
  }, [method, seedSmiles, constraints, maxAtoms, scoringFunction, syncConfig])

  const handleMaxAtomsChange = useCallback((val: number) => {
    setMaxAtoms(val)
    syncConfig(method, seedSmiles, constraints, allowedAtoms, val, scoringFunction)
  }, [method, seedSmiles, constraints, allowedAtoms, scoringFunction, syncConfig])

  const handleScoringChange = useCallback((s: string) => {
    setScoringFunction(s)
    syncConfig(method, seedSmiles, constraints, allowedAtoms, maxAtoms, s)
  }, [method, seedSmiles, constraints, allowedAtoms, maxAtoms, syncConfig])

  const applyPreset = useCallback((preset: (typeof PRESETS)[number]) => {
    setConstraints((prev) => {
      const next = { ...prev, ...preset.constraints }
      syncConfig(method, seedSmiles, next, allowedAtoms, maxAtoms, scoringFunction)
      return next
    })
  }, [method, seedSmiles, allowedAtoms, maxAtoms, scoringFunction, syncConfig])

  const previewProperties = useMemo(() => ({
    molecular_weight: (constraints.mw_range![0] + constraints.mw_range![1]) / 2,
    logp: (constraints.logp_range![0] + constraints.logp_range![1]) / 2,
    qed: (constraints.qed_range![0] + constraints.qed_range![1]) / 2,
    synthetic_accessibility: (constraints.sa_range![0] + constraints.sa_range![1]) / 2,
    tpsa: (constraints.tpsa_range![0] + constraints.tpsa_range![1]) / 2,
    rotatable_bonds: (constraints.rotatable_bonds_range![0] + constraints.rotatable_bonds_range![1]) / 2,
    hbd: (constraints.hbd_range![0] + constraints.hbd_range![1]) / 2,
    hba: (constraints.hba_range![0] + constraints.hba_range![1]) / 2,
    num_rings: 2,
    num_aromatic_rings: 1,
  }), [constraints])

  const Icon = domain.icon

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-neon-green/80 to-neon-blue">
            <Icon className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="font-semibold">{domain.name} Configuration</h3>
            <p className="text-xs text-muted-foreground">
              Configure generation strategy, molecular constraints, and scoring
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 text-neon-green text-xs">
          <CheckCircle className="w-4 h-4" />
          Valid
        </div>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* LEFT: Configuration (2/3) */}
        <div className="lg:col-span-2 space-y-5">

          {/* 1. Generation Method */}
          <Section title="Generation Strategy" icon={<FlaskConical className="w-4 h-4" />} step={1}>
            <div className="grid grid-cols-3 gap-2">
              {METHODS.map((m) => {
                const MIcon = m.icon
                return (
                  <button
                    key={m.id}
                    onClick={() => handleMethodChange(m.id)}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      method === m.id
                        ? 'border-neon-purple bg-neon-purple/10'
                        : 'border-dark-border hover:border-neon-purple/30 bg-white/5'
                    }`}
                  >
                    <MIcon className={`w-4 h-4 mb-1.5 ${method === m.id ? 'text-neon-purple' : 'text-muted-foreground'}`} />
                    <div className="text-xs font-semibold">{m.label}</div>
                    <div className="text-[9px] text-muted-foreground mt-0.5 line-clamp-1">{m.description}</div>
                  </button>
                )
              })}
            </div>

            {/* Seed SMILES for non-de-novo methods */}
            {method !== 'de_novo' && method !== 'grid_world' && (
              <div className="mt-3">
                <SMILESInput
                  label="Seed / Scaffold SMILES"
                  value={seedSmiles}
                  onChange={handleSmilesChange}
                  showPreview={true}
                />
              </div>
            )}

            {/* Presets */}
            <div className="flex items-center gap-2 mt-3">
              <span className="text-[10px] text-muted-foreground">Presets:</span>
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  onClick={() => applyPreset(p)}
                  className="px-2 py-0.5 rounded bg-white/5 hover:bg-neon-purple/10 text-[10px] text-muted-foreground hover:text-neon-purple transition-colors"
                  title={p.description}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </Section>

          {/* 2. Molecular Space */}
          <Section title="Molecular Space" icon={<Atom className="w-4 h-4" />} step={2}>
            {/* Allowed Atoms */}
            <div className="mb-4">
              <label className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2 block">Allowed Atom Types</label>
              <div className="flex gap-1.5 flex-wrap">
                {ATOM_TYPES.map((atom) => (
                  <button
                    key={atom.symbol}
                    onClick={() => handleAtomToggle(atom.symbol)}
                    className={`w-9 h-9 rounded-lg border text-xs font-bold transition-all flex items-center justify-center ${
                      allowedAtoms.includes(atom.symbol)
                        ? 'border-neon-purple/50 bg-neon-purple/10 text-white'
                        : 'border-dark-border bg-white/5 text-muted-foreground opacity-40'
                    }`}
                    title={atom.name}
                    style={allowedAtoms.includes(atom.symbol) ? { color: atom.color } : undefined}
                  >
                    {atom.symbol}
                  </button>
                ))}
              </div>
            </div>

            {/* Max Atoms */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-1">
                <label className="text-[10px] text-muted-foreground">Max Heavy Atoms</label>
                <span className="text-xs font-mono text-neon-blue">{maxAtoms}</span>
              </div>
              <input
                type="range" min={5} max={100} value={maxAtoms}
                onChange={(e) => handleMaxAtomsChange(Number(e.target.value))}
                className="w-full accent-neon-purple"
              />
              <div className="flex justify-between text-[9px] text-muted-foreground">
                <span>5 (fragment)</span><span>50 (drug-like)</span><span>100 (macrocycle)</span>
              </div>
            </div>

            {/* Scoring Function */}
            <div>
              <label className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2 block">
                <Target className="w-3 h-3 inline mr-1" />
                Scoring / Reward Function
              </label>
              <div className="grid grid-cols-3 gap-1.5">
                {SCORING_FUNCTIONS.map((sf) => (
                  <button
                    key={sf.id}
                    onClick={() => handleScoringChange(sf.id)}
                    className={`px-2.5 py-2 rounded-lg border text-left transition-all ${
                      scoringFunction === sf.id
                        ? 'border-neon-green/50 bg-neon-green/10'
                        : 'border-dark-border hover:border-neon-green/20 bg-white/5'
                    }`}
                  >
                    <div className={`text-[10px] font-semibold ${scoringFunction === sf.id ? 'text-neon-green' : ''}`}>{sf.label}</div>
                    <div className="text-[8px] text-muted-foreground mt-0.5 line-clamp-1">{sf.description}</div>
                  </button>
                ))}
              </div>
            </div>
          </Section>

          {/* 3. Property Constraints */}
          <Section title="Property Constraints" icon={<Beaker className="w-4 h-4" />} step={3}>
            <div className="space-y-3">
              <PropertySliderRange label="Molecular Weight" min={50} max={1000} value={constraints.mw_range!} onChange={(v) => handleConstraintsChange({ mw_range: v })} step={10} unit=" Da" />
              <PropertySliderRange label="LogP" min={-5} max={10} value={constraints.logp_range!} onChange={(v) => handleConstraintsChange({ logp_range: v })} step={0.1} format={(v) => v.toFixed(1)} />
              <PropertySliderRange label="QED (Drug-likeness)" min={0} max={1} value={constraints.qed_range!} onChange={(v) => handleConstraintsChange({ qed_range: v })} step={0.01} format={(v) => v.toFixed(2)} />
              <PropertySliderRange label="Synthetic Accessibility" min={1} max={10} value={constraints.sa_range!} onChange={(v) => handleConstraintsChange({ sa_range: v })} step={0.1} format={(v) => v.toFixed(1)} />
              <PropertySliderRange label="TPSA" min={0} max={300} value={constraints.tpsa_range!} onChange={(v) => handleConstraintsChange({ tpsa_range: v })} step={5} />
              <PropertySliderRange label="Rotatable Bonds" min={0} max={20} value={constraints.rotatable_bonds_range!} onChange={(v) => handleConstraintsChange({ rotatable_bonds_range: v })} />
              <PropertySliderRange label="H-Bond Donors" min={0} max={15} value={constraints.hbd_range!} onChange={(v) => handleConstraintsChange({ hbd_range: v })} />
              <PropertySliderRange label="H-Bond Acceptors" min={0} max={15} value={constraints.hba_range!} onChange={(v) => handleConstraintsChange({ hba_range: v })} />
            </div>
          </Section>

          {/* 4. ADMET / Medicinal Chemistry Filters */}
          <Section title="ADMET & Med-Chem Filters" icon={<Shield className="w-4 h-4" />} step={4}>
            <div className="grid grid-cols-2 gap-3">
              {([
                { key: 'lipinski', label: 'Lipinski Rule of 5', desc: 'MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10' },
                { key: 'veber', label: 'Veber Rules', desc: 'Rotatable bonds ≤ 10, TPSA ≤ 140' },
                { key: 'pains_filter', label: 'PAINS Filter', desc: 'Pan-assay interference compounds removal' },
                { key: 'brenk_filter', label: 'Brenk Filter', desc: 'Structural alerts for unstable groups' },
              ] as const).map((filter) => (
                <label
                  key={filter.key}
                  className={`flex items-start gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-all ${
                    (constraints as any)[filter.key] ? 'border-neon-green/30 bg-neon-green/5' : 'border-dark-border bg-white/5'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={(constraints as any)[filter.key] ?? false}
                    onChange={(e) => handleConstraintsChange({ [filter.key]: e.target.checked })}
                    className="w-3.5 h-3.5 mt-0.5 rounded border-dark-border accent-neon-green"
                  />
                  <div>
                    <span className="text-[10px] font-semibold block">{filter.label}</span>
                    <span className="text-[9px] text-muted-foreground">{filter.desc}</span>
                  </div>
                </label>
              ))}
            </div>
          </Section>

          {/* 5. Fragment Library (Gap 3) */}
          <Section title="Building Blocks" icon={<Puzzle className="w-4 h-4" />} step={5}>
            <p className="text-[10px] text-muted-foreground mb-3">
              Browse the BRICS fragment library used as molecular building blocks during generation.
            </p>
            <div style={{ maxHeight: '360px', overflow: 'auto' }} className="rounded-lg border border-dark-border">
              <FragmentBrowser compact />
            </div>
          </Section>

          {/* 6. Docking Target (Gap 2) — shown when binding affinity scoring */}
          {scoringFunction === 'binding_affinity' && (
            <Section title="Docking Target" icon={<Crosshair className="w-4 h-4" />} step={6}>
              <p className="text-[10px] text-muted-foreground mb-3">
                Select a protein target for binding affinity scoring during training.
              </p>
              <DockingPanel compact />
            </Section>
          )}
        </div>

        {/* RIGHT: Sticky Live Preview (1/3) */}
        <div className="lg:col-span-1">
          <div className="lg:sticky lg:top-4 space-y-4">
            <h4 className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
              <Layers className="w-3 h-3" />
              Live Preview
            </h4>

            {/* 2D Molecule Preview */}
            {seedSmiles && (
              <div className="glass-dark rounded-lg p-3">
                <h4 className="text-[10px] font-medium mb-2">Seed Molecule</h4>
                <div className="flex justify-center bg-white/5 rounded-lg p-2">
                  <MoleculeViewer2D smiles={seedSmiles} width={180} height={180} />
                </div>
              </div>
            )}

            {/* Property Radar Chart */}
            <div className="glass-dark rounded-lg p-3">
              <h4 className="text-[10px] font-medium mb-2">Target Property Profile</h4>
              <PropertyRadarChart properties={previewProperties} height={200} />
            </div>

            {/* Configuration Summary */}
            <div className="glass-dark rounded-lg p-3">
              <h4 className="text-[10px] font-medium mb-2">Configuration Summary</h4>
              <div className="space-y-1.5">
                <SummaryRow label="Method" value={METHODS.find((m) => m.id === method)?.label ?? method} />
                <SummaryRow label="Scoring" value={SCORING_FUNCTIONS.find((s) => s.id === scoringFunction)?.label ?? scoringFunction} />
                <SummaryRow label="Atoms" value={`${allowedAtoms.join(', ')} (max ${maxAtoms})`} />
                <div className="border-t border-dark-border/50 pt-1.5 mt-1.5" />
                <SummaryRow label="MW" value={`${constraints.mw_range![0]} – ${constraints.mw_range![1]} Da`} />
                <SummaryRow label="LogP" value={`${constraints.logp_range![0].toFixed(1)} – ${constraints.logp_range![1].toFixed(1)}`} />
                <SummaryRow label="QED" value={`${constraints.qed_range![0].toFixed(2)} – ${constraints.qed_range![1].toFixed(2)}`} />
                <SummaryRow label="TPSA" value={`${constraints.tpsa_range![0]} – ${constraints.tpsa_range![1]}`} />
                <div className="border-t border-dark-border/50 pt-1.5 mt-1.5" />
                <SummaryRow label="Lipinski" value={constraints.lipinski ? 'On' : 'Off'} highlight={constraints.lipinski} />
                <SummaryRow label="PAINS" value={constraints.pains_filter ? 'On' : 'Off'} highlight={constraints.pains_filter} />
                <SummaryRow label="Veber" value={constraints.veber ? 'On' : 'Off'} highlight={constraints.veber} />
                <SummaryRow label="Brenk" value={constraints.brenk_filter ? 'On' : 'Off'} highlight={constraints.brenk_filter} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function Section({
  title, icon, step, children,
}: {
  title: string; icon: React.ReactNode; step: number; children: React.ReactNode
}) {
  return (
    <div className="glass-dark rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="w-5 h-5 rounded-full bg-neon-purple/20 text-neon-purple flex items-center justify-center text-[10px] font-bold">{step}</span>
        <span className="text-neon-purple">{icon}</span>
        <span className="text-sm font-semibold">{title}</span>
      </div>
      {children}
    </div>
  )
}

function SummaryRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[10px] text-muted-foreground">{label}</span>
      <span className={`text-[10px] font-mono ${highlight ? 'text-neon-green' : 'text-white'}`}>{value}</span>
    </div>
  )
}
