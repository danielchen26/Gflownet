import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Fingerprint, Crosshair, Puzzle, FlaskConical, Target, Zap,
} from 'lucide-react'

// Gap components
import DiversityStats from '../components/DiversityStats'
import DiversityMatrix from '../components/DiversityMatrix'
import DockingPanel from '../components/DockingPanel'
import FragmentBrowser from '../components/FragmentBrowser'
import SynthesisRoute from '../components/SynthesisRoute'
import ParetoFrontExplorer from '../components/ParetoFrontExplorer'
import PreferenceSliders from '../components/PreferenceSliders'
import OraclePanel from '../components/OraclePanel'

const TABS = [
  {
    id: 'diversity',
    label: 'Diversity Analysis',
    shortLabel: 'Diversity',
    icon: Fingerprint,
    color: '#10B981',
    gap: 1,
    description: 'Tanimoto similarity metrics, scaffold diversity, and molecular fingerprint analysis',
  },
  {
    id: 'docking',
    label: 'Docking Rewards',
    shortLabel: 'Docking',
    icon: Crosshair,
    color: '#EF4444',
    gap: 2,
    description: 'Target-specific drug design with AutoDock Vina and proxy MLP scoring',
  },
  {
    id: 'fragments',
    label: 'Fragment Library',
    shortLabel: 'Fragments',
    icon: Puzzle,
    color: '#F59E0B',
    gap: 3,
    description: 'BRICS-compatible fragment library with compatibility checking and categories',
  },
  {
    id: 'synthesis',
    label: 'Reaction Constraints',
    shortLabel: 'Synthesis',
    icon: FlaskConical,
    color: '#8B5CF6',
    gap: 4,
    description: 'Reaction-based molecular generation with synthesis route guarantees',
  },
  {
    id: 'pareto',
    label: 'Multi-Objective Pareto',
    shortLabel: 'Pareto',
    icon: Target,
    color: '#3B82F6',
    gap: 5,
    description: 'MOGFN-PC preference-conditioned optimization across competing objectives',
  },
  {
    id: 'oracles',
    label: 'TDC Oracles',
    shortLabel: 'Oracles',
    icon: Zap,
    color: '#06B6D4',
    gap: 'PMO',
    description: 'Target-specific activity oracles for DRD2, GSK3B, JNK3, and PMO benchmark evaluation',
  },
] as const

type TabId = typeof TABS[number]['id']

export function MolecularToolkit() {
  const [activeTab, setActiveTab] = useState<TabId>('diversity')
  const currentTab = TABS.find(t => t.id === activeTab)!

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold gradient-text">Molecular Toolkit</h1>
        <p className="text-xs text-muted-foreground mt-1">
          Six integrated capabilities for production-grade molecular design
        </p>
      </div>

      {/* Tab bar */}
      <div style={{
        display: 'flex', gap: '4px', padding: '4px',
        background: '#111827', borderRadius: '10px', border: '1px solid #1F2937',
      }}>
        {TABS.map(tab => {
          const Icon = tab.icon
          const active = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                padding: '10px 12px', borderRadius: '8px', cursor: 'pointer',
                fontSize: '12px', fontWeight: active ? 600 : 500,
                background: active ? `${tab.color}15` : 'transparent',
                color: active ? tab.color : '#6B7280',
                border: active ? `1px solid ${tab.color}30` : '1px solid transparent',
                transition: 'all 0.2s',
              }}
            >
              <Icon style={{ width: 14, height: 14 }} />
              <span className="hidden lg:inline">{tab.shortLabel}</span>
              <span style={{
                fontSize: '9px', fontWeight: 700, padding: '1px 5px',
                borderRadius: '3px', background: active ? `${tab.color}20` : '#1F2937',
                color: active ? tab.color : '#4B5563',
              }}>
                {tab.gap}
              </span>
            </button>
          )
        })}
      </div>

      {/* Tab description */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.15 }}
        style={{
          padding: '12px 16px', borderRadius: '8px',
          background: `${currentTab.color}08`,
          border: `1px solid ${currentTab.color}20`,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            fontSize: '10px', fontWeight: 700, padding: '2px 6px',
            borderRadius: '4px', background: `${currentTab.color}20`, color: currentTab.color,
          }}>
            Gap {currentTab.gap}
          </span>
          <span style={{ fontSize: '13px', fontWeight: 600, color: '#E5E7EB' }}>
            {currentTab.label}
          </span>
        </div>
        <p style={{ fontSize: '11px', color: '#9CA3AF', marginTop: '4px' }}>
          {currentTab.description}
        </p>
      </motion.div>

      {/* Tab content */}
      <motion.div
        key={activeTab + '-content'}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.2 }}
        style={{
          background: '#111827', borderRadius: '12px',
          border: '1px solid #1F2937', overflow: 'hidden',
        }}
      >
        {activeTab === 'diversity' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: '400px' }}>
              <div style={{ borderRight: '1px solid #1F2937' }}>
                <DiversityStats autoRefresh refreshInterval={30000} />
              </div>
              <div>
                <DiversityMatrix maxMolecules={50} />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'docking' && (
          <div style={{ maxWidth: '600px' }}>
            <DockingPanel />
          </div>
        )}

        {activeTab === 'fragments' && (
          <FragmentBrowser />
        )}

        {activeTab === 'synthesis' && (
          <div style={{ padding: '16px' }}>
            <div style={{ marginBottom: '16px' }}>
              <p style={{ fontSize: '12px', color: '#9CA3AF', marginBottom: '12px' }}>
                Select a molecule from the <strong style={{ color: '#E5E7EB' }}>Candidates</strong> page
                to view its synthesis route, or enter a molecule ID below.
              </p>
              <SynthesisLookup />
            </div>
          </div>
        )}

        {activeTab === 'pareto' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: '400px' }}>
            <div style={{ borderRight: '1px solid #1F2937' }}>
              <PreferenceSliders />
            </div>
            <div>
              <ParetoFrontExplorer autoRefresh refreshInterval={15000} />
            </div>
          </div>
        )}

        {activeTab === 'oracles' && (
          <div style={{ maxWidth: '600px' }}>
            <OraclePanel />
          </div>
        )}
      </motion.div>
    </div>
  )
}

/** Small helper: enter a molecule ID to look up its synthesis route */
function SynthesisLookup() {
  const [molId, setMolId] = useState('')
  const [activeMolId, setActiveMolId] = useState<string | null>(null)

  return (
    <div>
      <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
        <input
          type="text"
          value={molId}
          onChange={e => setMolId(e.target.value)}
          placeholder="Enter molecule ID..."
          style={{
            flex: 1, padding: '6px 10px', fontSize: '12px', fontFamily: 'monospace',
            background: '#1F2937', border: '1px solid #374151', borderRadius: '6px',
            color: '#E5E7EB',
          }}
        />
        <button
          onClick={() => setActiveMolId(molId.trim() || null)}
          disabled={!molId.trim()}
          style={{
            padding: '6px 14px', fontSize: '12px', fontWeight: 600,
            background: '#8B5CF620', color: '#8B5CF6',
            border: '1px solid #8B5CF650', borderRadius: '6px',
            cursor: molId.trim() ? 'pointer' : 'not-allowed',
          }}
        >
          Look Up
        </button>
      </div>
      <SynthesisRoute moleculeId={activeMolId} />
    </div>
  )
}
