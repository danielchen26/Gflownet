import { useState, useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart3, Grid3X3, List, Download, Filter, X,
  ChevronRight, Copy, Check, Eye, Trash2, SlidersHorizontal,
} from 'lucide-react'
import { MoleculeViewer2D } from '../components/MoleculeViewer2D'
import { MoleculeGallery } from '../components/MoleculeCard'
import { PropertySliderRange } from '../components/PropertySliderRange'
import { PropertyRadarChart } from '../components/PropertyRadarChart'
import { api, type Molecule } from '../services/api'
import type { ViewId } from '../components/Sidebar'

type ViewMode = 'grid' | 'table'
type SortBy = 'reward' | 'qed' | 'mw' | 'logp' | 'sa'

interface Filters {
  mw_range: [number, number]
  logp_range: [number, number]
  qed_range: [number, number]
  sa_range: [number, number]
  reward_min: number
  lipinski: boolean
  no_pains: boolean
}

const DEFAULT_FILTERS: Filters = {
  mw_range: [0, 1000],
  logp_range: [-5, 10],
  qed_range: [0, 1],
  sa_range: [1, 10],
  reward_min: 0,
  lipinski: false,
  no_pains: false,
}

interface ResultsHubProps {
  onNavigate?: (view: ViewId) => void
  problemConfig?: any
}

export function ResultsHub({ onNavigate, problemConfig }: ResultsHubProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('grid')
  const [sortBy, setSortBy] = useState<SortBy>('reward')
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [showFilters, setShowFilters] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [detailMolecule, setDetailMolecule] = useState<Molecule | null>(null)
  const [copied, setCopied] = useState(false)

  const { data, isLoading } = useQuery({
    queryKey: ['molecules-results', sortBy],
    queryFn: () => api.molecular.getMolecules({ limit: 100, sort_by: sortBy }),
    retry: 1,
    refetchInterval: 10000,
  })

  const molecules: Molecule[] = data?.molecules ?? []

  // Apply client-side sorting + filters
  const filteredMolecules = useMemo(() => {
    // Sort (backend sorts too, but this ensures consistent behavior)
    const sortFns: Record<SortBy, (a: Molecule, b: Molecule) => number> = {
      reward: (a, b) => (b.reward ?? 0) - (a.reward ?? 0),
      qed: (a, b) => (b.properties?.qed ?? 0) - (a.properties?.qed ?? 0),
      mw: (a, b) => (b.properties?.molecular_weight ?? 0) - (a.properties?.molecular_weight ?? 0),
      logp: (a, b) => (b.properties?.logp ?? 0) - (a.properties?.logp ?? 0),
      sa: (a, b) => (a.properties?.synthetic_accessibility ?? 10) - (b.properties?.synthetic_accessibility ?? 10),
    }
    const sorted = [...molecules].sort(sortFns[sortBy] ?? sortFns.reward)

    return sorted.filter((mol) => {
      const p = mol.properties
      if (!p) return true
      if (p.molecular_weight < filters.mw_range[0] || p.molecular_weight > filters.mw_range[1]) return false
      if (p.logp < filters.logp_range[0] || p.logp > filters.logp_range[1]) return false
      if (p.qed < filters.qed_range[0] || p.qed > filters.qed_range[1]) return false
      if (p.synthetic_accessibility < filters.sa_range[0] || p.synthetic_accessibility > filters.sa_range[1]) return false
      if (mol.reward < filters.reward_min) return false
      if (filters.lipinski) {
        if (p.molecular_weight > 500 || p.logp > 5 || p.hbd > 5 || p.hba > 10) return false
      }
      return true
    })
  }, [molecules, filters, sortBy])

  const handleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const handleExport = useCallback(async (format: 'smiles' | 'sdf' | 'csv') => {
    const ids = selectedIds.size > 0
      ? Array.from(selectedIds)
      : filteredMolecules.map((m) => m.id)
    try {
      const result = await api.molecular.exportMolecules({ ids, format })
      const blob = new Blob([result], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `molecules.${format}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }, [selectedIds, filteredMolecules])

  const handleCopy = useCallback(() => {
    if (detailMolecule) {
      navigator.clipboard.writeText(detailMolecule.smiles)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }
  }, [detailMolecule])

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Context Banner */}
      <div className="mb-3 px-3 py-2 rounded-lg bg-neon-blue/5 border border-neon-blue/20">
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          <span className="text-neon-blue font-medium">Generated Molecules</span> — These are candidates discovered by your GFlowNet
          {problemConfig?.training_objective && (
            <> using <span className="text-neon-purple font-mono">{problemConfig.training_objective.replace(/_/g, ' ')}</span> objective</>
          )}. Each molecule is scored by a reward function combining drug-likeness (QED), synthetic accessibility, and molecular property targets.
          {data?.total != null && <> Showing top {filteredMolecules.length} of <span className="font-mono text-white">{data.total}</span> total molecules from training + database.</>}
        </p>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-bold gradient-text">Results & Analysis</h1>
          <p className="text-[10px] text-muted-foreground mt-0.5">
            {filteredMolecules.length} molecules {molecules.length !== filteredMolecules.length && `(filtered from ${molecules.length})`}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortBy)}
            className="px-2 py-1.5 rounded-md bg-dark-bg border border-dark-border text-xs"
          >
            <option value="reward">Sort: Reward</option>
            <option value="qed">Sort: QED</option>
            <option value="mw">Sort: MW</option>
            <option value="logp">Sort: LogP</option>
            <option value="sa">Sort: SA</option>
          </select>

          {/* View Mode */}
          <div className="flex rounded-md border border-dark-border overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-1.5 ${viewMode === 'grid' ? 'bg-neon-purple/20 text-neon-purple' : 'text-muted-foreground hover:text-white'}`}
            >
              <Grid3X3 className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`p-1.5 ${viewMode === 'table' ? 'bg-neon-purple/20 text-neon-purple' : 'text-muted-foreground hover:text-white'}`}
            >
              <List className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Filter Toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`p-1.5 rounded-md border ${showFilters ? 'border-neon-purple bg-neon-purple/20 text-neon-purple' : 'border-dark-border text-muted-foreground hover:text-white'}`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
          </button>

          {/* Export */}
          <div className="relative group">
            <button className="flex items-center gap-1 px-2 py-1.5 rounded-md bg-white/5 text-xs text-muted-foreground hover:text-white">
              <Download className="w-3.5 h-3.5" />
              Export
            </button>
            <div className="hidden group-hover:block absolute right-0 top-full mt-1 bg-dark-panel border border-dark-border rounded-md shadow-lg z-10">
              <button onClick={() => handleExport('smiles')} className="block w-full px-3 py-1.5 text-xs text-left hover:bg-white/5">SMILES</button>
              <button onClick={() => handleExport('sdf')} className="block w-full px-3 py-1.5 text-xs text-left hover:bg-white/5">SDF</button>
              <button onClick={() => handleExport('csv')} className="block w-full px-3 py-1.5 text-xs text-left hover:bg-white/5">CSV</button>
            </div>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden mb-4"
          >
            <div className="glass-dark rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold">Filters</span>
                <button
                  onClick={() => setFilters(DEFAULT_FILTERS)}
                  className="text-[10px] text-muted-foreground hover:text-white"
                >
                  Reset
                </button>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                <PropertySliderRange
                  label="Molecular Weight" min={0} max={1000} step={10}
                  value={filters.mw_range}
                  onChange={(v) => setFilters({ ...filters, mw_range: v })}
                  unit=" Da"
                />
                <PropertySliderRange
                  label="LogP" min={-5} max={10} step={0.1}
                  value={filters.logp_range}
                  onChange={(v) => setFilters({ ...filters, logp_range: v })}
                  format={(v) => v.toFixed(1)}
                />
                <PropertySliderRange
                  label="QED" min={0} max={1} step={0.01}
                  value={filters.qed_range}
                  onChange={(v) => setFilters({ ...filters, qed_range: v })}
                  format={(v) => v.toFixed(2)}
                />
                <PropertySliderRange
                  label="SA Score" min={1} max={10} step={0.1}
                  value={filters.sa_range}
                  onChange={(v) => setFilters({ ...filters, sa_range: v })}
                  format={(v) => v.toFixed(1)}
                />
                <div className="flex items-center gap-4">
                  <label className="flex items-center gap-2 text-[10px] cursor-pointer">
                    <input type="checkbox" checked={filters.lipinski} onChange={(e) => setFilters({ ...filters, lipinski: e.target.checked })} className="w-3 h-3 accent-neon-purple" />
                    <span className="text-muted-foreground">Lipinski</span>
                  </label>
                  <label className="flex items-center gap-2 text-[10px] cursor-pointer">
                    <input type="checkbox" checked={filters.no_pains} onChange={(e) => setFilters({ ...filters, no_pains: e.target.checked })} className="w-3 h-3 accent-neon-purple" />
                    <span className="text-muted-foreground">No PAINS</span>
                  </label>
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground uppercase tracking-wider block mb-1">Min Reward</label>
                  <input
                    type="number" value={filters.reward_min} step={0.5}
                    onChange={(e) => setFilters({ ...filters, reward_min: parseFloat(e.target.value) || 0 })}
                    className="w-full px-2 py-1 rounded-md bg-dark-bg border border-dark-border text-xs font-mono"
                  />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Content */}
      <div className="flex-1 flex gap-4 min-h-0">
        {/* Main Area */}
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <p className="text-sm text-muted-foreground">Loading molecules...</p>
            </div>
          ) : filteredMolecules.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <BarChart3 className="w-16 h-16 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">No molecules match your filters</p>
            </div>
          ) : viewMode === 'grid' ? (
            <MoleculeGallery
              molecules={filteredMolecules}
              selectedIds={selectedIds}
              onSelect={handleSelect}
              onClick={(id) => setDetailMolecule(filteredMolecules.find((m) => m.id === id) || null)}
              columns={4}
              size="md"
            />
          ) : (
            <div className="glass-dark rounded-lg overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-dark-border">
                    <th className="text-left p-2 text-muted-foreground font-medium">#</th>
                    <th className="text-left p-2 text-muted-foreground font-medium">Structure</th>
                    <th className="text-left p-2 text-muted-foreground font-medium">SMILES</th>
                    <th className="text-right p-2 text-muted-foreground font-medium">Reward</th>
                    <th className="text-right p-2 text-muted-foreground font-medium">MW</th>
                    <th className="text-right p-2 text-muted-foreground font-medium">QED</th>
                    <th className="text-right p-2 text-muted-foreground font-medium">LogP</th>
                    <th className="text-right p-2 text-muted-foreground font-medium">SA</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredMolecules.map((mol, i) => (
                    <tr
                      key={mol.id}
                      onClick={() => setDetailMolecule(mol)}
                      className="border-b border-dark-border/30 hover:bg-white/5 cursor-pointer transition-colors"
                    >
                      <td className="p-2 text-muted-foreground">{i + 1}</td>
                      <td className="p-1">
                        <MoleculeViewer2D smiles={mol.smiles} width={40} height={40} />
                      </td>
                      <td className="p-2 font-mono truncate max-w-[200px]">{mol.smiles}</td>
                      <td className="p-2 text-right font-mono text-neon-green">{mol.reward?.toFixed(2)}</td>
                      <td className="p-2 text-right font-mono">{mol.properties?.molecular_weight?.toFixed(0)}</td>
                      <td className="p-2 text-right font-mono">{mol.properties?.qed?.toFixed(3)}</td>
                      <td className="p-2 text-right font-mono">{mol.properties?.logp?.toFixed(2)}</td>
                      <td className="p-2 text-right font-mono">{mol.properties?.synthetic_accessibility?.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Detail Drawer */}
        <AnimatePresence>
          {detailMolecule && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 300, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="glass-dark rounded-xl p-4 overflow-y-auto scrollbar-thin flex-shrink-0"
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold">Molecule Detail</h3>
                <button onClick={() => setDetailMolecule(null)} className="text-muted-foreground hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* 2D Preview */}
              <div className="bg-white/5 rounded-lg p-2 flex justify-center mb-3">
                <MoleculeViewer2D smiles={detailMolecule.smiles} width={240} height={240} />
              </div>

              {/* SMILES */}
              <div className="flex items-center gap-2 bg-white/5 rounded-md p-2 mb-3">
                <span className="text-[10px] font-mono break-all flex-1">{detailMolecule.smiles}</span>
                <button onClick={handleCopy} className="flex-shrink-0 text-muted-foreground hover:text-white">
                  {copied ? <Check className="w-3 h-3 text-neon-green" /> : <Copy className="w-3 h-3" />}
                </button>
              </div>

              {/* Properties */}
              <div className="space-y-1.5 mb-3">
                <DetailRow label="Reward" value={detailMolecule.reward?.toFixed(3)} color="#00FF88" />
                <DetailRow label="Molecular Weight" value={`${detailMolecule.properties?.molecular_weight?.toFixed(1)} Da`} />
                <DetailRow label="LogP" value={detailMolecule.properties?.logp?.toFixed(2)} />
                <DetailRow label="QED" value={detailMolecule.properties?.qed?.toFixed(3)} />
                <DetailRow label="SA Score" value={detailMolecule.properties?.synthetic_accessibility?.toFixed(2)} />
                <DetailRow label="TPSA" value={`${detailMolecule.properties?.tpsa?.toFixed(1)} A²`} />
                <DetailRow label="Rotatable Bonds" value={detailMolecule.properties?.rotatable_bonds?.toString()} />
                <DetailRow label="HBD" value={detailMolecule.properties?.hbd?.toString()} />
                <DetailRow label="HBA" value={detailMolecule.properties?.hba?.toString()} />
                <DetailRow label="Rings" value={detailMolecule.properties?.num_rings?.toString()} />
                <DetailRow label="Aromatic Rings" value={detailMolecule.properties?.num_aromatic_rings?.toString()} />
                {detailMolecule.properties?.formula && (
                  <DetailRow label="Formula" value={detailMolecule.properties.formula} />
                )}
              </div>

              {/* Radar Chart */}
              {detailMolecule.properties && (
                <div className="mb-3">
                  <h4 className="text-[10px] font-semibold text-muted-foreground mb-1">Property Profile</h4>
                  <PropertyRadarChart properties={detailMolecule.properties} height={200} />
                </div>
              )}

              {/* Actions */}
              <div className="space-y-2">
                <button
                  onClick={() => onNavigate?.('structure')}
                  className="w-full px-3 py-2 rounded-md bg-neon-purple/20 text-neon-purple text-xs hover:bg-neon-purple/30 transition-colors flex items-center justify-center gap-2"
                >
                  <Eye className="w-3.5 h-3.5" />
                  View in 3D
                </button>
                <div className="grid grid-cols-2 gap-2">
                  <button onClick={() => handleExport('smiles')} className="px-2 py-1.5 rounded-md bg-white/5 text-muted-foreground text-[10px] hover:bg-white/10">
                    Export SMILES
                  </button>
                  <button onClick={() => handleExport('sdf')} className="px-2 py-1.5 rounded-md bg-white/5 text-muted-foreground text-[10px] hover:bg-white/10">
                    Export SDF
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function DetailRow({ label, value, color }: { label: string; value?: string; color?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[10px] text-muted-foreground">{label}</span>
      <span className="text-[10px] font-mono font-medium" style={{ color }}>{value ?? '—'}</span>
    </div>
  )
}
