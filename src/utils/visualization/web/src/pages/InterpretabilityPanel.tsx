import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Brain, Flame, GitBranch, BarChart3, ArrowLeftRight,
  Info, Circle,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell,
} from 'recharts'
import { MoleculeViewer2D } from '../components/MoleculeViewer2D'
import { useChartColors } from '../contexts/ThemeContext'
import { api, type Molecule, type AtomAttribution, type RewardDecompositionData } from '../services/api'
import type { DAGNode, DAGEdge } from '../services/mockData'

type InterpretView = 'attribution' | 'actions' | 'dag' | 'counterfactual' | 'reward'

interface InterpretabilityPanelProps {
  problemConfig?: any
}

export function InterpretabilityPanel({ problemConfig }: InterpretabilityPanelProps) {
  const [activeView, setActiveView] = useState<InterpretView>('attribution')
  const [selectedMolecule, setSelectedMolecule] = useState<Molecule | null>(null)
  const [attributionType, setAttributionType] = useState<'reward' | 'flow' | 'loss'>('reward')
  const colors = useChartColors()

  const { data: molecules } = useQuery({
    queryKey: ['molecules-interpret'],
    queryFn: () => api.molecular.getMolecules({ limit: 20, sort_by: 'reward' }),
    retry: 1,
  })

  const { data: attribution } = useQuery<AtomAttribution>({
    queryKey: ['attribution', selectedMolecule?.id, attributionType],
    queryFn: () => api.molecular.getAttribution(selectedMolecule!.id),
    enabled: !!selectedMolecule && activeView === 'attribution',
    retry: 1,
  })

  const { data: rewardDecomp } = useQuery<RewardDecompositionData>({
    queryKey: ['reward-decomp', selectedMolecule?.id],
    queryFn: () => api.molecular.getRewardDecomposition(selectedMolecule!.id),
    enabled: !!selectedMolecule && activeView === 'reward',
    retry: 1,
  })

  const { data: dagData } = useQuery<{ nodes: DAGNode[]; edges: DAGEdge[] }>({
    queryKey: ['generation-dag', selectedMolecule?.id],
    queryFn: () => api.molecular.getGenerationDAG(selectedMolecule!.id),
    enabled: !!selectedMolecule && activeView === 'dag',
    retry: 1,
  })

  const VIEWS: Array<{ id: InterpretView; label: string; icon: typeof Brain; description: string }> = [
    { id: 'attribution', label: 'Atom Attribution', icon: Flame, description: 'Gradient saliency heatmap on molecular structure' },
    { id: 'actions', label: 'Action Probabilities', icon: BarChart3, description: 'Probability distribution over actions at each step' },
    { id: 'dag', label: 'Generation DAG', icon: GitBranch, description: 'Molecular construction process visualization' },
    { id: 'counterfactual', label: 'Counterfactuals', icon: ArrowLeftRight, description: 'What if atom X was different?' },
    { id: 'reward', label: 'Reward Decomposition', icon: BarChart3, description: 'Contribution of each scoring component' },
  ]

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Context Banner */}
      <div className="mb-3 px-3 py-2 rounded-lg bg-neon-orange/5 border border-neon-orange/20 flex-shrink-0">
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          <span className="text-neon-orange font-medium">Model Interpretability</span> — Understand <em>why</em> your GFlowNet
          {problemConfig?.training_objective && <> ({problemConfig.training_objective.replace(/_/g, ' ')})</>}
          {' '}generates specific molecules. Select a molecule from the left panel to explore atom-level attribution maps,
          generation step sequences, and reward component breakdowns. Attribution and reward decomposition use real model gradients;
          action probabilities and counterfactuals are approximated.
        </p>
      </div>
      <h1 className="text-xl font-bold gradient-text mb-4">GFlowNet Interpretability</h1>

      {/* View Tabs */}
      <div className="flex items-center gap-1 mb-4 overflow-x-auto">
        {VIEWS.map((view) => {
          const Icon = view.icon
          return (
            <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              className={`
                flex items-center gap-2 px-3 py-2 rounded-lg text-xs whitespace-nowrap transition-all
                ${activeView === view.id
                  ? 'bg-neon-purple/15 text-white border border-neon-purple/30'
                  : 'text-muted-foreground hover:text-white hover:bg-white/5 border border-transparent'
                }
              `}
              title={view.description}
            >
              <Icon className="w-3.5 h-3.5" />
              {view.label}
            </button>
          )
        })}
      </div>

      {/* Content */}
      <div className="flex-1 flex gap-4 min-h-0">
        {/* Molecule Selector */}
        <div className="w-56 glass-dark rounded-xl p-3 overflow-y-auto scrollbar-thin flex-shrink-0">
          <h3 className="text-xs font-semibold mb-2">Select Molecule</h3>
          {molecules?.molecules?.length > 0 ? (
            <div className="space-y-1">
              {molecules.molecules.map((mol: Molecule) => (
                <button
                  key={mol.id}
                  onClick={() => setSelectedMolecule(mol)}
                  className={`
                    w-full text-left p-2 rounded-md transition-all
                    ${selectedMolecule?.id === mol.id
                      ? 'bg-neon-purple/15 border border-neon-purple/30'
                      : 'hover:bg-white/5 border border-transparent'
                    }
                  `}
                >
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-white/5 flex-shrink-0 overflow-hidden">
                      <MoleculeViewer2D smiles={mol.smiles} width={32} height={32} />
                    </div>
                    <div className="min-w-0">
                      <p className="text-[9px] font-mono truncate">{mol.smiles}</p>
                      <span className="text-[8px] text-neon-green">R: {mol.reward?.toFixed(1)}</span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-[10px] text-muted-foreground text-center py-4">No molecules available</p>
          )}
        </div>

        {/* Main Interpretation View */}
        <div className="flex-1 glass-dark rounded-xl p-4 overflow-y-auto scrollbar-thin">
          {!selectedMolecule ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Brain className="w-16 h-16 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">Select a molecule to interpret</p>
              <p className="text-[10px] text-muted-foreground/60 mt-1">
                Choose from the list on the left
              </p>
            </div>
          ) : (
            <>
              {activeView === 'attribution' && (
                <AttributionView
                  molecule={selectedMolecule}
                  attribution={attribution ?? null}
                  attributionType={attributionType}
                  onTypeChange={setAttributionType}
                  colors={colors}
                />
              )}
              {activeView === 'actions' && (
                <ActionProbabilityView molecule={selectedMolecule} colors={colors} />
              )}
              {activeView === 'dag' && (
                <GenerationDAGView molecule={selectedMolecule} dagData={dagData ?? null} colors={colors} />
              )}
              {activeView === 'counterfactual' && (
                <CounterfactualView molecule={selectedMolecule} />
              )}
              {activeView === 'reward' && (
                <RewardDecompositionView molecule={selectedMolecule} decomposition={rewardDecomp ?? null} colors={colors} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// --- Attribution Heatmap View ---
function AttributionView({
  molecule, attribution, attributionType, onTypeChange, colors,
}: {
  molecule: Molecule; attribution: AtomAttribution | null
  attributionType: 'reward' | 'flow' | 'loss'; onTypeChange: (t: 'reward' | 'flow' | 'loss') => void
  colors: any
}) {
  const atomColors = useMemo(() => {
    if (!attribution?.atom_scores) return undefined
    const max = Math.max(...attribution.atom_scores.map(Math.abs))
    return attribution.atom_scores.map((score) => {
      const n = max > 0 ? score / max : 0
      if (n > 0) return `rgb(${Math.round(n * 255)}, ${Math.round(n * 76)}, 0)`
      return `rgb(0, ${Math.round(-n * 76)}, ${Math.round(-n * 255)})`
    })
  }, [attribution])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Atom Attribution Heatmap</h3>
        <div className="flex items-center gap-2">
          {(['reward', 'flow', 'loss'] as const).map((t) => (
            <button key={t} onClick={() => onTypeChange(t)}
              className={`px-2 py-1 rounded text-[10px] capitalize ${attributionType === t ? 'bg-neon-purple/20 text-neon-purple' : 'text-muted-foreground hover:text-white'}`}
            >{t}</button>
          ))}
        </div>
      </div>
      <div className="flex justify-center bg-white/5 rounded-lg p-4">
        <MoleculeViewer2D smiles={molecule.smiles} width={400} height={400}
          highlightAtoms={attribution?.atom_scores?.map((_, i) => i)} highlightColors={atomColors} />
      </div>
      {attribution?.atom_scores && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">Attribution Scores by Atom</h4>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={attribution.atom_scores.map((score, i) => ({ atom: i, score }))}>
              <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
              <XAxis dataKey="atom" tick={{ fontSize: 9 }} stroke={colors.axis} />
              <YAxis tick={{ fontSize: 9 }} stroke={colors.axis} />
              <Tooltip contentStyle={{ backgroundColor: 'rgb(var(--dark-panel))', border: '1px solid rgb(var(--dark-border))', fontSize: 10, borderRadius: 8 }} />
              <Bar dataKey="score">
                {attribution.atom_scores.map((score, i) => (
                  <Cell key={i} fill={score >= 0 ? '#EF4444' : '#3B82F6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="flex items-center justify-center gap-6 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-red-500" /> Positive</span>
        <span className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-blue-500" /> Negative</span>
      </div>
    </div>
  )
}

// --- Action Probability View ---
function ActionProbabilityView({ molecule, colors }: { molecule: Molecule; colors: any }) {
  const mockSteps = useMemo(() =>
    Array.from({ length: 8 }, (_, step) => {
      const actions = ['Add C', 'Add N', 'Add O', 'Add S', 'Bond Single', 'Bond Double', 'Bond Aromatic', 'Stop']
      const probs = actions.map(() => Math.random())
      const sum = probs.reduce((a, b) => a + b, 0)
      const chosenIdx = Math.floor(Math.random() * actions.length)
      return { step, actions: actions.map((action, i) => ({ action, probability: probs[i] / sum, chosen: i === chosenIdx })) }
    }), [molecule.id])

  const [selectedStep, setSelectedStep] = useState(0)
  const currentActions = mockSteps[selectedStep]?.actions ?? []

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Action Probability Distribution</h3>
      <div className="flex items-center gap-2 text-xs text-amber-400/80 bg-amber-400/10 border border-amber-400/20 rounded-md px-3 py-2">
        <Info className="w-4 h-4 flex-shrink-0" />
        Simulated data — fragment-level action probabilities require per-step trajectory logging (not yet implemented).
      </div>
      <div className="flex items-center gap-1 overflow-x-auto pb-2">
        {mockSteps.map((s) => (
          <button key={s.step} onClick={() => setSelectedStep(s.step)}
            className={`px-3 py-1.5 rounded-md text-[10px] whitespace-nowrap ${selectedStep === s.step ? 'bg-neon-purple/20 text-neon-purple' : 'text-muted-foreground hover:text-white bg-white/5'}`}
          >Step {s.step + 1}</button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={[...currentActions].sort((a, b) => b.probability - a.probability)} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
          <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 9 }} stroke={colors.axis} />
          <YAxis type="category" dataKey="action" tick={{ fontSize: 10 }} stroke={colors.axis} width={100} />
          <Tooltip contentStyle={{ backgroundColor: 'rgb(var(--dark-panel))', border: '1px solid rgb(var(--dark-border))', fontSize: 10, borderRadius: 8 }}
            formatter={(value: number) => (value * 100).toFixed(1) + '%'} />
          <Bar dataKey="probability">
            {[...currentActions].sort((a, b) => b.probability - a.probability).map((entry, i) => (
              <Cell key={i} fill={entry.chosen ? colors.primary : colors.grid} fillOpacity={entry.chosen ? 1 : 0.4} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 text-[10px] text-muted-foreground">
        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colors.primary }} /> Chosen action</div>
        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm opacity-40" style={{ backgroundColor: colors.grid }} /> Alternative</div>
      </div>
    </div>
  )
}

// --- Generation DAG View (interactive SVG tree) ---
function GenerationDAGView({ molecule, dagData, colors }: {
  molecule: Molecule; dagData: { nodes: DAGNode[]; edges: DAGEdge[] } | null; colors: any
}) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  const layout = useMemo(() => {
    if (!dagData) return null
    const { nodes } = dagData
    const maxDepth = Math.max(...nodes.map((n) => n.depth))
    const depthGroups = new Map<number, DAGNode[]>()
    nodes.forEach((n) => {
      const g = depthGroups.get(n.depth) ?? []
      g.push(n)
      depthGroups.set(n.depth, g)
    })
    const width = 700
    const height = Math.max(400, (maxDepth + 1) * 70)
    const positions = new Map<string, { x: number; y: number }>()
    depthGroups.forEach((group, depth) => {
      const spacing = width / (group.length + 1)
      group.forEach((node, i) => positions.set(node.id, { x: spacing * (i + 1), y: 30 + depth * (height / (maxDepth + 1.5)) }))
    })
    return { positions, width, height }
  }, [dagData])

  if (!dagData || !layout) {
    return (
      <div className="space-y-4">
        <h3 className="text-sm font-semibold">Generation DAG</h3>
        <div className="flex flex-col items-center justify-center h-[400px] bg-white/5 rounded-lg">
          <GitBranch className="w-12 h-12 text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">Loading generation DAG...</p>
        </div>
      </div>
    )
  }

  const { nodes, edges } = dagData
  const { positions, width, height } = layout
  const flowColor = (f: number) => {
    const t = Math.min(1, Math.max(0, f))
    return `rgb(${Math.round(59 + t * (16 - 59))}, ${Math.round(130 + t * (185 - 130))}, ${Math.round(246 + t * (129 - 246))})`
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Generation DAG</h3>
        <span className="text-[10px] text-muted-foreground">{nodes.length} states, {edges.length} transitions</span>
      </div>
      <div className="bg-white/5 rounded-lg overflow-auto" style={{ maxHeight: 500 }}>
        <svg width={width} height={height}>
          <defs>
            <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
              <polygon points="0 0, 6 2, 0 4" fill="rgba(255,255,255,0.3)" />
            </marker>
          </defs>
          {edges.map((edge, i) => {
            const from = positions.get(edge.source), to = positions.get(edge.target)
            if (!from || !to) return null
            const hi = hoveredNode === edge.source || hoveredNode === edge.target
            return (
              <g key={i}>
                <line x1={from.x} y1={from.y + 12} x2={to.x} y2={to.y - 12}
                  stroke={hi ? colors.primary : 'rgba(255,255,255,0.15)'} strokeWidth={hi ? 2 : 1} markerEnd="url(#arrowhead)" />
                <text x={(from.x + to.x) / 2 + 8} y={(from.y + to.y) / 2} fontSize={8} fill="rgba(255,255,255,0.4)" textAnchor="start">
                  {edge.action} ({(edge.probability * 100).toFixed(0)}%)
                </text>
              </g>
            )
          })}
          {nodes.map((node) => {
            const pos = positions.get(node.id)
            if (!pos) return null
            const hi = hoveredNode === node.id
            const r = node.id === 'terminal' || node.id === 'root' ? 16 : 12
            return (
              <g key={node.id} onMouseEnter={() => setHoveredNode(node.id)} onMouseLeave={() => setHoveredNode(null)} className="cursor-pointer">
                <circle cx={pos.x} cy={pos.y} r={r} fill={flowColor(node.flow)}
                  stroke={hi ? '#fff' : 'rgba(255,255,255,0.2)'} strokeWidth={hi ? 2 : 1} opacity={hi ? 1 : 0.8} />
                <text x={pos.x} y={pos.y + 3} textAnchor="middle" fontSize={9} fontWeight="bold" fill="white">{node.label}</text>
                <text x={pos.x} y={pos.y + r + 12} textAnchor="middle" fontSize={8} fill="rgba(255,255,255,0.5)">f={node.flow}</text>
              </g>
            )
          })}
        </svg>
      </div>
      <div className="flex items-center gap-6 text-[10px] text-muted-foreground">
        <div className="flex items-center gap-1.5"><Circle className="w-3 h-3 text-blue-400" fill="currentColor" /> Low flow</div>
        <div className="flex items-center gap-1.5"><Circle className="w-3 h-3 text-emerald-400" fill="currentColor" /> High flow</div>
        <span className="ml-auto">Node = intermediate state, Edge = action</span>
      </div>
      {hoveredNode && (() => {
        const node = nodes.find((n) => n.id === hoveredNode)
        if (!node) return null
        return (
          <div className="bg-white/5 rounded-md p-2 flex items-center gap-3">
            {node.smiles && <MoleculeViewer2D smiles={node.smiles} width={60} height={60} />}
            <div className="text-[10px]">
              <p className="font-mono">{node.smiles || '(empty)'}</p>
              <p className="text-muted-foreground">Depth: {node.depth} | Flow: {node.flow}</p>
            </div>
          </div>
        )
      })()}
    </div>
  )
}

// --- Counterfactual View ---
function CounterfactualView({ molecule }: { molecule: Molecule }) {
  const counterfactual = useMemo(() => {
    const s = molecule.smiles
    let alt = s
    let change = 'C → N'
    if (s.includes('N')) { alt = s.replace('N', 'O'); change = 'N → O' }
    else if (s.includes('O')) { alt = s.replace('O', 'N'); change = 'O → N' }
    else { alt = s.replace('C', 'N') }
    const delta = (Math.random() - 0.5) * 2
    return { smiles: alt, reward: Math.max(0, (molecule.reward ?? 5) + delta), change }
  }, [molecule])

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Counterfactual Analysis</h3>
      <div className="flex items-center gap-2 text-xs text-amber-400/80 bg-amber-400/10 border border-amber-400/20 rounded-md px-3 py-2">
        <Info className="w-4 h-4 flex-shrink-0" />
        Simulated — performs simple atom substitution with random reward delta. Real counterfactuals require backward policy evaluation.
      </div>
      <p className="text-xs text-muted-foreground">Explore "what if" scenarios — substituting atoms and observing property changes.</p>
      <div className="grid grid-cols-2 gap-4">
        <div className="glass rounded-lg p-3">
          <h4 className="text-[10px] font-semibold text-muted-foreground mb-2">Original</h4>
          <div className="bg-white/5 rounded-md p-2 flex justify-center">
            <MoleculeViewer2D smiles={molecule.smiles} width={180} height={180} />
          </div>
          <div className="mt-2 space-y-1">
            <RowPair label="Reward" value={molecule.reward?.toFixed(2)} color="#00FF88" />
            <RowPair label="QED" value={molecule.properties?.qed?.toFixed(3)} />
            <RowPair label="MW" value={molecule.properties?.molecular_weight?.toFixed(0)} />
          </div>
        </div>
        <div className="glass rounded-lg p-3 border-dashed border-neon-purple/30">
          <h4 className="text-[10px] font-semibold text-neon-purple mb-2">Counterfactual ({counterfactual.change})</h4>
          <div className="bg-white/5 rounded-md p-2 flex justify-center">
            <MoleculeViewer2D smiles={counterfactual.smiles} width={180} height={180} />
          </div>
          <div className="mt-2 space-y-1">
            <RowPair label="Reward"
              value={`${counterfactual.reward.toFixed(2)} (${counterfactual.reward > (molecule.reward ?? 0) ? '+' : ''}${(counterfactual.reward - (molecule.reward ?? 0)).toFixed(2)})`}
              color={counterfactual.reward > (molecule.reward ?? 0) ? '#00FF88' : '#EF4444'} />
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground bg-white/5 rounded-md p-3">
        <Info className="w-4 h-4 flex-shrink-0" />
        Performs single-atom substitutions and re-evaluates the reward function. In production, uses the trained GFlowNet backward policy.
      </div>
    </div>
  )
}

// --- Reward Decomposition View ---
function RewardDecompositionView({ molecule, decomposition, colors }: {
  molecule: Molecule; decomposition: RewardDecompositionData | null; colors: any
}) {
  const COLORS = ['#3B82F6', '#10B981', '#EF4444', '#F59E0B', '#8B5CF6', '#EC4899']
  const data = useMemo(() => {
    if (!decomposition?.components) return []
    return decomposition.components.map((c, i) => ({ ...c, fill: COLORS[i % COLORS.length] }))
  }, [decomposition])

  if (!data.length) {
    return <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">Loading reward decomposition...</div>
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Reward Decomposition</h3>
      <p className="text-xs text-muted-foreground">
        Breakdown for: <span className="font-mono">{molecule.smiles.slice(0, 40)}{molecule.smiles.length > 40 ? '...' : ''}</span>
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
          <XAxis dataKey="name" tick={{ fontSize: 9 }} stroke={colors.axis} angle={-20} textAnchor="end" height={60} />
          <YAxis tick={{ fontSize: 9 }} stroke={colors.axis} />
          <Tooltip contentStyle={{ backgroundColor: 'rgb(var(--dark-panel))', border: '1px solid rgb(var(--dark-border))', fontSize: 10, borderRadius: 8 }} />
          <Bar dataKey="contribution" radius={[4, 4, 0, 0]}>
            {data.map((e, i) => <Cell key={i} fill={e.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="glass rounded-lg overflow-hidden">
        <table className="w-full text-[10px]">
          <thead><tr className="border-b border-dark-border/30">
            <th className="text-left p-2 text-muted-foreground">Component</th>
            <th className="text-right p-2 text-muted-foreground">Raw</th>
            <th className="text-right p-2 text-muted-foreground">Weight</th>
            <th className="text-right p-2 text-muted-foreground">Contribution</th>
          </tr></thead>
          <tbody>
            {data.map((d) => (
              <tr key={d.name} className="border-b border-dark-border/20">
                <td className="p-2 flex items-center gap-1.5"><div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.fill }} />{d.name}</td>
                <td className="p-2 text-right font-mono">{d.value.toFixed(3)}</td>
                <td className="p-2 text-right font-mono text-muted-foreground">{d.weight.toFixed(1)}</td>
                <td className={`p-2 text-right font-mono font-medium ${d.contribution >= 0 ? 'text-neon-green' : 'text-red-400'}`}>
                  {d.contribution >= 0 ? '+' : ''}{d.contribution.toFixed(3)}
                </td>
              </tr>
            ))}
            <tr className="font-medium">
              <td className="p-2">Total</td><td /><td />
              <td className="p-2 text-right font-mono text-neon-green">{data.reduce((s, d) => s + d.contribution, 0).toFixed(3)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

function RowPair({ label, value, color }: { label: string; value?: string; color?: string }) {
  return (
    <div className="flex justify-between text-[10px]">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono" style={{ color }}>{value ?? '—'}</span>
    </div>
  )
}
