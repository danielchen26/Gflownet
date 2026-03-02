import { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Network, Maximize2, BarChart3, Settings, ChevronDown, ChevronUp, RotateCcw, Play, FastForward, Shield, Sparkles, Zap, Gauge, Database, Target, FlaskConical, TrendingUp, Fingerprint, Layers } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { GFlowNet2DTrajectory } from '../visualizations/GFlowNet2DTrajectory'
import { GFlowNetTrainingDashboard } from './GFlowNetTrainingDashboard'
import { RealtimeMetrics } from './RealtimeMetrics'
import { ErrorPanel } from './ErrorPanel'
import { TrainingControls } from './TrainingControls'
import { MoleculeViewer2D } from './MoleculeViewer2D'
import DiversityStats from './DiversityStats'
import ParetoFrontExplorer from './ParetoFrontExplorer'
import PreferenceSliders from './PreferenceSliders'
import DockingPanel from './DockingPanel'
import SynthesisRoute from './SynthesisRoute'
import { api } from '../services/api'
import type { Molecule } from '../services/api'
import { useThemeLayout, useChartColors } from '../contexts/ThemeContext'

interface MonitoringDashboardProps {
  problemConfig: any
  onRestart?: (config: any) => void
}

export function MonitoringDashboard({ problemConfig, onRestart }: MonitoringDashboardProps) {
  const [expandedView, setExpandedView] = useState<'none' | 'trajectory' | 'training'>('none')
  const [clearedErrorCount, setClearedErrorCount] = useState(0)
  const [showConfig, setShowConfig] = useState(false)
  const layout = useThemeLayout()

  // Editable config state - initialized from problemConfig
  const [editConfig, setEditConfig] = useState<any>({ ...problemConfig })
  const [hasChanges, setHasChanges] = useState(false)

  // Track changes vs original
  useEffect(() => {
    if (!problemConfig) return
    const changed = Object.keys(editConfig).some(key => {
      if (key === 'reward_peaks') return false // don't compare arrays
      return editConfig[key] !== problemConfig[key]
    })
    setHasChanges(changed)
  }, [editConfig, problemConfig])

  // Reset edit config and cleared errors when problemConfig changes (after restart)
  useEffect(() => {
    if (problemConfig) setEditConfig({ ...problemConfig })
    setClearedErrorCount(0)
  }, [problemConfig])

  const handleRerun = () => {
    if (onRestart) {
      onRestart(editConfig)
      setHasChanges(false)
    }
  }

  const [extendAmount, setExtendAmount] = useState(300)
  const [isExtending, setIsExtending] = useState(false)

  // Fetch training state for error info and status
  // NOTE: All hooks must be called before any conditional returns (React Rules of Hooks)
  const { data: trainingState } = useQuery({
    queryKey: ['training-state'],
    queryFn: () => api.training.getState(),
    refetchInterval: 1000,
  })

  const rawErrorCount = trainingState?.error_count || 0
  const errorCount = Math.max(0, rawErrorCount - clearedErrorCount)
  const lastError = errorCount > 0 ? (trainingState?.last_error || null) : null
  const isTraining = trainingState?.is_training || false
  const isPaused = trainingState?.is_paused || false

  // Detect molecular domain
  const isMolecularDomain = problemConfig?.domain_type === 'molecule' || problemConfig?.domain === 'molecule'
  // Detect MOGFN multi-objective mode
  const isMOGFN = problemConfig?.training_objective === 'MULTI_OBJECTIVE_TB'
  const chartColors = useChartColors()

  // Fetch latest generated molecules for the live feed (molecular domain only)
  const { data: moleculeData } = useQuery({
    queryKey: ['molecule-feed'],
    queryFn: () => api.molecular.getMolecules({ limit: 8, sort_by: 'generation_step' }),
    refetchInterval: isMolecularDomain ? 2000 : false,
    enabled: isMolecularDomain,
  })

  const feedMolecules: Molecule[] = moleculeData?.molecules ?? []

  // Compute diversity metrics from the feed molecules
  const diversityMetrics = useMemo(() => {
    if (!feedMolecules.length) return { unique: 0, avgQED: 0, avgReward: 0, topReward: 0, scaffoldCount: 0 }
    const uniqueSmiles = new Set(feedMolecules.map((m) => m.smiles))
    const avgQED = feedMolecules.reduce((s, m) => s + (m.properties?.qed ?? 0), 0) / feedMolecules.length
    const avgReward = feedMolecules.reduce((s, m) => s + (m.reward ?? 0), 0) / feedMolecules.length
    const topReward = Math.max(...feedMolecules.map((m) => m.reward ?? 0))
    // Approximate scaffold diversity by counting unique ring counts
    const scaffolds = new Set(feedMolecules.map((m) => `${m.properties?.num_rings ?? 0}-${m.properties?.num_aromatic_rings ?? 0}`))
    return { unique: uniqueSmiles.size, avgQED, avgReward, topReward, scaffoldCount: scaffolds.size }
  }, [feedMolecules])

  // Guard: show placeholder when no training has been configured yet
  if (!problemConfig) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-dark-bg text-center py-20">
        <Activity className="w-12 h-12 text-muted-foreground mb-4 opacity-30" />
        <h2 className="text-lg font-semibold text-muted-foreground mb-2">No Training Active</h2>
        <p className="text-sm text-muted-foreground/60 max-w-md">
          Go to the <span className="text-neon-purple font-medium">Setup</span> tab to configure your domain and start training.
        </p>
      </div>
    )
  }

  const handleExtend = async () => {
    setIsExtending(true)
    try {
      await api.training.extend(extendAmount)
    } catch (err) {
      console.error('Failed to extend training:', err)
    }
    setIsExtending(false)
  }
  
  return (
    <div className="h-full flex flex-col bg-dark-bg">
      {/* Header */}
      <div className={`border-b border-dark-border/50 ${layout.compact ? 'px-2 py-1' : layout.spacious ? 'px-6 py-3' : 'px-4 py-2'} flex items-center justify-between`}>
        <div className={`flex items-center ${layout.compact ? 'gap-2' : 'gap-4'}`}>
          <h2 className={`${layout.compact ? 'text-sm' : layout.spacious ? 'text-xl' : 'text-lg'} font-semibold`}>Training Monitor</h2>
          <div className={`flex items-center gap-2 ${layout.labelSize} text-muted-foreground`}>
            <Activity className={`${layout.compact ? 'w-3 h-3' : 'w-4 h-4'} ${isTraining && !isPaused ? 'text-neon-green animate-pulse' : 'text-gray-500'}`} />
            <span>{isTraining ? (isPaused ? 'Paused' : 'Live Training') : 'Idle'}</span>
          </div>
          {/* Training Controls */}
          <TrainingControls
            isTraining={isTraining}
            isPaused={isPaused}
          />
        </div>

        {/* Config Summary + Toggle */}
        <div className={`flex items-center ${layout.compact ? 'gap-2' : 'gap-4'} ${layout.tinySize}`}>
          {isMolecularDomain ? (
            <>
              <span className="text-muted-foreground flex items-center gap-1">
                <FlaskConical className="w-3 h-3 text-neon-green" />
                Molecular
              </span>
              {!layout.compact && feedMolecules.length > 0 && (
                <span className="text-muted-foreground">
                  Generated: <span className="text-neon-green font-mono">{moleculeData?.total ?? feedMolecules.length}</span>
                </span>
              )}
            </>
          ) : (
            <span className="text-muted-foreground">
              Grid: {problemConfig.grid_size}×{problemConfig.grid_size}
            </span>
          )}
          {!layout.compact && (
            <span className="text-muted-foreground">
              Objective: <span className="text-neon-purple">{problemConfig.training_objective}</span>
            </span>
          )}
          <span className="text-muted-foreground">
            Ep: <span className="text-neon-blue">{problemConfig.n_episodes}</span>
          </span>
          <button
            onClick={() => setShowConfig(!showConfig)}
            className={`flex items-center gap-1 ${layout.compact ? 'px-1.5 py-0.5' : 'px-2 py-1'} ${layout.tinySize} bg-dark-panel border border-dark-border rounded hover:border-neon-purple/50 transition-colors`}
          >
            <Settings className="w-3 h-3 text-neon-purple" />
            <span>Config</span>
            {showConfig ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* Hyperparameter Config Panel (collapsible, editable) */}
      <AnimatePresence>
        {showConfig && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-b border-dark-border/50 overflow-hidden"
          >
            <div className={`${layout.compact ? 'px-2 py-1.5' : layout.spacious ? 'px-6 py-4' : 'px-4 py-3'} bg-dark-panel/30`}>
              {/* Objective selector */}
              <div className={`flex items-center gap-2 ${layout.compact ? 'mb-1.5' : 'mb-3'}`}>
                <span className={`${layout.tinySize} text-muted-foreground font-medium`}>Objective:</span>
                <select
                  value={editConfig.training_objective}
                  onChange={(e) => setEditConfig({ ...editConfig, training_objective: e.target.value })}
                  className="text-xs bg-dark-bg border border-dark-border rounded px-2 py-1 text-white"
                >
                  <option value="TRAJECTORY_BALANCE">Trajectory Balance (TB)</option>
                  <option value="SUB_TRAJECTORY_BALANCE">Sub-Trajectory Balance (STB)</option>
                  <option value="DETAILED_BALANCE">Detailed Balance (DB)</option>
                  <option value="FLOW_MATCHING">Flow Matching (FM)</option>
                  <option value="TRAJECTORY_LIKELIHOOD_MAXIMIZATION">TLM (ICLR 2025 - Backward Policy Training)</option>
                  <option value="MULTI_OBJECTIVE_TB">MOGFN-PC (Multi-Objective Pareto)</option>
                </select>
                {editConfig.training_objective === 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION' && (
                  <span className="px-1.5 py-0.5 text-[9px] bg-neon-cyan/20 text-neon-cyan rounded">
                    Trains backward policy P_B
                  </span>
                )}
              </div>

              {/* ── Core Training ── */}
              <div className={layout.compact ? 'mb-1.5' : 'mb-3'}>
                <div className={`flex items-center gap-2 ${layout.compact ? 'mb-1' : 'mb-2'}`}>
                  <Settings className={`${layout.compact ? 'w-3 h-3' : 'w-3.5 h-3.5'} text-neon-blue`} />
                  <span className={`${layout.headingSize} font-medium text-neon-blue`}>Core Training</span>
                </div>
                <div className={`grid ${layout.configColumns} ${layout.gap}`}>
                  <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border border-neon-blue/20`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className={`${layout.labelSize} text-muted-foreground`}>Episodes</span>
                      <span className={`${layout.tinySize} font-mono text-neon-blue`}>{editConfig.n_episodes}</span>
                    </div>
                    <input type="range" min={100} max={3000} step={100}
                      value={editConfig.n_episodes ?? 1000}
                      onChange={(e) => setEditConfig({ ...editConfig, n_episodes: Number(e.target.value) })}
                      className="w-full h-1"
                    />
                    <div className={`flex justify-between ${layout.tinySize} text-muted-foreground mt-0.5`}>
                      <span>100</span><span>3000</span>
                    </div>
                  </div>
                  <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border border-neon-blue/20`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className={`${layout.labelSize} text-muted-foreground`}>Batch Size</span>
                      <span className={`${layout.tinySize} font-mono text-neon-blue`}>{editConfig.batch_size ?? 32}</span>
                    </div>
                    <input type="range" min={8} max={128} step={8}
                      value={editConfig.batch_size ?? 32}
                      onChange={(e) => setEditConfig({ ...editConfig, batch_size: Number(e.target.value) })}
                      className="w-full h-1"
                    />
                    <div className={`flex justify-between ${layout.tinySize} text-muted-foreground mt-0.5`}>
                      <span>8</span><span>128</span>
                    </div>
                  </div>
                  <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border border-neon-blue/20`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className={`${layout.labelSize} text-muted-foreground`}>Learning Rate</span>
                      <span className={`${layout.tinySize} font-mono text-neon-blue`}>{(editConfig.learning_rate ?? 0.005).toFixed(4)}</span>
                    </div>
                    <input type="range" min={0.0005} max={0.02} step={0.0005}
                      value={editConfig.learning_rate ?? 0.005}
                      onChange={(e) => setEditConfig({ ...editConfig, learning_rate: Number(e.target.value) })}
                      className="w-full h-1"
                    />
                    <div className={`flex justify-between ${layout.tinySize} text-muted-foreground mt-0.5`}>
                      <span>0.0005</span><span>0.02</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* ── Mode Collapse Prevention ── */}
              <div className={layout.compact ? 'mb-1.5' : 'mb-3'}>
                <div className={`flex items-center gap-2 ${layout.compact ? 'mb-1' : 'mb-2'}`}>
                  <Shield className={`${layout.compact ? 'w-3 h-3' : 'w-3.5 h-3.5'} text-neon-purple`} />
                  <span className={`${layout.headingSize} font-medium text-neon-purple`}>Mode Collapse Prevention</span>
                  {!layout.compact && (
                    <span className={`px-1.5 py-0.5 ${layout.tinySize} font-medium bg-neon-green/15 text-neon-green rounded-full`}>Recommended</span>
                  )}
                </div>
                <div className={`grid ${layout.configColumns} ${layout.gap}`}>
                  {/* Epsilon */}
                  <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border border-neon-purple/20`}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1">
                        <Sparkles className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-purple`} />
                        <span className={`${layout.labelSize} text-muted-foreground`}>Epsilon (ε)</span>
                      </div>
                      <span className={`${layout.tinySize} font-mono text-neon-purple`}>
                        {((editConfig.epsilon ?? 0.15) * 100).toFixed(0)}%
                        {editConfig.epsilon_decay && <span className="text-neon-green ml-0.5">↓</span>}
                      </span>
                    </div>
                    <input type="range" min={0} max={0.4} step={0.01}
                      value={editConfig.epsilon ?? 0.15}
                      onChange={(e) => setEditConfig({ ...editConfig, epsilon: Number(e.target.value) })}
                      className="w-full h-1"
                    />
                    <div className="flex items-center gap-1 mt-1">
                      <input type="checkbox" checked={editConfig.epsilon_decay ?? true}
                        onChange={(e) => setEditConfig({ ...editConfig, epsilon_decay: e.target.checked })}
                        className="w-2.5 h-2.5"
                      />
                      <span className={`${layout.tinySize} text-muted-foreground`}>Decay</span>
                    </div>
                  </div>

                  {/* Entropy Weight */}
                  <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border border-neon-purple/20`}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1">
                        <Zap className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-blue`} />
                        <span className={`${layout.labelSize} text-muted-foreground`}>Entropy</span>
                      </div>
                      <span className={`${layout.tinySize} font-mono text-neon-blue`}>{(editConfig.entropy_weight ?? 0.02).toFixed(3)}</span>
                    </div>
                    <input type="range" min={0} max={0.1} step={0.002}
                      value={editConfig.entropy_weight ?? 0.02}
                      onChange={(e) => setEditConfig({ ...editConfig, entropy_weight: Number(e.target.value) })}
                      className="w-full h-1"
                    />
                  </div>

                  {/* Z Learning Rate Multiplier */}
                  <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border border-neon-orange/20`}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1">
                        <Gauge className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-orange`} />
                        <span className={`${layout.labelSize} text-muted-foreground`}>Z Rate</span>
                      </div>
                      <span className={`${layout.tinySize} font-mono text-neon-orange`}>{(editConfig.z_learning_rate_multiplier ?? 10).toFixed(0)}×</span>
                    </div>
                    <input type="range" min={1} max={20} step={1}
                      value={editConfig.z_learning_rate_multiplier ?? 10}
                      onChange={(e) => setEditConfig({ ...editConfig, z_learning_rate_multiplier: Number(e.target.value) })}
                      className="w-full h-1"
                    />
                  </div>

                  {/* Replay Buffer Toggle */}
                  <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border ${editConfig.use_replay_buffer ? 'border-cyan-500/40' : 'border-dark-border/30'}`}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1">
                        <Database className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-cyan-400`} />
                        <span className={`${layout.labelSize} text-muted-foreground`}>Replay</span>
                      </div>
                      <input type="checkbox" checked={editConfig.use_replay_buffer ?? false}
                        onChange={(e) => setEditConfig({ ...editConfig, use_replay_buffer: e.target.checked })}
                        className="w-3 h-3"
                      />
                    </div>
                    <span className={`${layout.tinySize} font-mono text-cyan-400`}>
                      {editConfig.use_replay_buffer ? `${((editConfig.replay_ratio ?? 0.5) * 100).toFixed(0)}% mix` : 'OFF'}
                    </span>
                  </div>

                  {/* Reward Shaping Toggle */}
                  <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border ${editConfig.reward_shaping !== false ? 'border-neon-green/40' : 'border-dark-border/30'}`}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1">
                        <Target className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-green`} />
                        <span className={`${layout.labelSize} text-muted-foreground`}>Shaping</span>
                      </div>
                      <input type="checkbox" checked={editConfig.reward_shaping !== false}
                        onChange={(e) => setEditConfig({ ...editConfig, reward_shaping: e.target.checked })}
                        className="w-3 h-3"
                      />
                    </div>
                    <span className={`${layout.tinySize} font-mono text-neon-green`}>
                      {editConfig.reward_shaping !== false ? 'ON' : 'OFF'}
                    </span>
                  </div>
                </div>

                {/* TLM parameters (show when TLM selected) */}
                {editConfig.training_objective === 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION' && (
                  <div className={`grid grid-cols-2 ${layout.gap} ${layout.compact ? 'mt-1' : 'mt-2'}`}>
                    <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border border-neon-cyan/20`}>
                      <div className="flex items-center justify-between mb-1">
                        <span className={`${layout.labelSize} text-muted-foreground`}>TLM Backward λ</span>
                        <span className={`${layout.tinySize} font-mono text-neon-cyan`}>{(editConfig.tlm_backward_weight ?? 1.0).toFixed(1)}</span>
                      </div>
                      <input type="range" min={0.1} max={5.0} step={0.1}
                        value={editConfig.tlm_backward_weight ?? 1.0}
                        onChange={(e) => setEditConfig({ ...editConfig, tlm_backward_weight: Number(e.target.value) })}
                        className="w-full h-1"
                      />
                    </div>
                    <div className={`${layout.innerPad} bg-dark-bg/50 rounded-lg border border-neon-cyan/20`}>
                      <div className="flex items-center justify-between mb-1">
                        <span className={`${layout.labelSize} text-muted-foreground`}>TLM Entropy</span>
                        <span className={`${layout.tinySize} font-mono text-neon-cyan`}>{(editConfig.tlm_entropy_coeff ?? 0.01).toFixed(3)}</span>
                      </div>
                      <input type="range" min={0} max={0.1} step={0.002}
                        value={editConfig.tlm_entropy_coeff ?? 0.01}
                        onChange={(e) => setEditConfig({ ...editConfig, tlm_entropy_coeff: Number(e.target.value) })}
                        className="w-full h-1"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Bottom row: live info + two actions */}
              <div className={`${layout.compact ? 'mt-1.5' : 'mt-3'} flex items-center justify-between`}>
                {/* Live training info */}
                <div className={`flex items-center ${layout.compact ? 'gap-2' : 'gap-4'} ${layout.tinySize} text-muted-foreground`}>
                  {trainingState?.current_epsilon !== undefined && (
                    <>
                      <span>Live ε: <span className="font-mono text-neon-purple">{(trainingState.current_epsilon * 100).toFixed(1)}%</span></span>
                      <span>Progress: <span className="font-mono text-neon-blue">{((trainingState.progress ?? 0) * 100).toFixed(1)}%</span></span>
                    </>
                  )}
                  <span>Temp: <span className="font-mono">{(editConfig.temperature ?? 1.0).toFixed(1)}</span></span>
                  {!layout.compact && (
                    <>
                      <span>Replay: <span className={`font-mono ${editConfig.use_replay_buffer ? 'text-cyan-400' : 'text-gray-500'}`}>{editConfig.use_replay_buffer ? 'ON' : 'OFF'}</span></span>
                      <span>Shaping: <span className={`font-mono ${editConfig.reward_shaping !== false ? 'text-neon-green' : 'text-gray-500'}`}>{editConfig.reward_shaping !== false ? 'ON' : 'OFF'}</span></span>
                    </>
                  )}
                </div>

                {/* Two actions: Extend or Retrain */}
                <div className={`flex items-center ${layout.compact ? 'gap-1.5' : 'gap-3'}`}>
                  {/* Extend Training — continue from current weights */}
                  <div className="flex items-center gap-1.5">
                    <span className={`${layout.tinySize} text-muted-foreground`}>+</span>
                    <input
                      type="number"
                      min={100}
                      max={5000}
                      step={100}
                      value={extendAmount}
                      onChange={(e) => setExtendAmount(Number(e.target.value))}
                      className={`${layout.compact ? 'w-12 px-1 py-0.5' : 'w-16 px-1.5 py-1'} ${layout.tinySize} font-mono bg-dark-bg border border-dark-border rounded text-white text-center`}
                    />
                    <button
                      onClick={handleExtend}
                      disabled={isExtending}
                      className={`flex items-center gap-1.5 ${layout.compact ? 'px-2 py-1 text-[10px]' : 'px-3 py-1.5 text-xs'} font-medium bg-neon-orange/20 text-neon-orange border border-neon-orange/30 rounded hover:bg-neon-orange/30 transition-colors`}
                    >
                      <FastForward className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'}`} />
                      {isExtending ? 'Extending...' : layout.compact ? 'Extend' : 'Extend Training'}
                    </button>
                  </div>

                  <span className={`${layout.tinySize} text-muted-foreground`}>or</span>

                  {/* Retrain — start fresh with current config */}
                  <button
                    onClick={handleRerun}
                    disabled={!onRestart}
                    className={`flex items-center gap-1.5 ${layout.compact ? 'px-2 py-1 text-[10px]' : 'px-3 py-1.5 text-xs'} font-medium bg-red-500/20 text-red-400 border border-red-500/30 rounded hover:bg-red-500/30 transition-colors`}
                  >
                    <RotateCcw className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'}`} />
                    {layout.compact ? 'Retrain' : 'Retrain from Scratch'}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Panel */}
      {(errorCount > 0 || lastError) && (
        <div className="px-4 py-2">
          <ErrorPanel
            errorCount={errorCount}
            lastError={lastError}
            onClear={() => setClearedErrorCount(rawErrorCount)}
          />
        </div>
      )}
      
      {/* Molecular Diversity Metrics Row (shown when in molecular domain) */}
      {isMolecularDomain && expandedView === 'none' && feedMolecules.length > 0 && (
        <div className={`${layout.compact ? 'px-1 pt-1' : layout.spacious ? 'px-4 pt-4' : 'px-2 pt-2'}`}>
          <div className={`grid grid-cols-5 ${layout.gap}`}>
            <DiversityMetricCard
              icon={<FlaskConical className="w-3.5 h-3.5" />}
              label="Total Generated"
              value={moleculeData?.total ?? feedMolecules.length}
              color={chartColors.primary}
              layout={layout}
            />
            <DiversityMetricCard
              icon={<Fingerprint className="w-3.5 h-3.5" />}
              label="Unique Molecules"
              value={`${diversityMetrics.unique}/${feedMolecules.length}`}
              sub={`${((diversityMetrics.unique / feedMolecules.length) * 100).toFixed(0)}%`}
              color={chartColors.secondary}
              layout={layout}
            />
            <DiversityMetricCard
              icon={<TrendingUp className="w-3.5 h-3.5" />}
              label="Top Reward"
              value={diversityMetrics.topReward.toFixed(2)}
              sub={`avg ${diversityMetrics.avgReward.toFixed(2)}`}
              color={chartColors.green}
              layout={layout}
            />
            <DiversityMetricCard
              icon={<Sparkles className="w-3.5 h-3.5" />}
              label="Avg QED"
              value={diversityMetrics.avgQED.toFixed(3)}
              color={chartColors.tertiary}
              layout={layout}
            />
            <DiversityMetricCard
              icon={<Layers className="w-3.5 h-3.5" />}
              label="Scaffold Diversity"
              value={diversityMetrics.scaffoldCount}
              sub="unique scaffolds"
              color={chartColors.primary}
              layout={layout}
            />
          </div>
        </div>
      )}

      {/* Main Content - Two Rows Layout (with optional molecule feed) */}
      <div className={`flex-1 flex ${layout.compact ? 'p-1 gap-1' : layout.spacious ? 'p-4 gap-4' : 'p-2 gap-2'} overflow-hidden`}>
        {/* Left: Training views */}
        <div className={`flex-1 flex flex-col ${layout.compact ? 'gap-1' : layout.spacious ? 'gap-4' : 'gap-2'} min-w-0`}>
        {expandedView === 'none' ? (
          <>
            {/* Row 1: Trajectory Sampling and Metrics */}
            <div className={`flex-1 grid grid-cols-1 ${layout.compact ? 'lg:grid-cols-4' : 'lg:grid-cols-3'} ${layout.gap} min-h-0`}>
              {/* Left: 2D Trajectory Visualization */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`${layout.compact ? 'lg:col-span-3' : 'lg:col-span-2'} glass-dark rounded-lg ${layout.cardPad} relative group`}
              >
                <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => setExpandedView('trajectory')}
                    className="p-1 bg-dark-panel/80 rounded hover:bg-dark-panel"
                  >
                    <Maximize2 className="w-3 h-3" />
                  </button>
                </div>

                <h3 className={`${layout.headingSize} font-medium mb-2 flex items-center gap-2`}>
                  <Network className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-purple`} />
                  {isMolecularDomain ? 'Generated Molecules' : 'Trajectory Sampling'}
                </h3>

                <div className="h-[calc(100%-1.5rem)] min-h-0">
                  {isMolecularDomain ? (
                    <div className="h-full overflow-y-auto scrollbar-thin">
                      {feedMolecules.length > 0 ? (
                        <div className="grid grid-cols-2 gap-2">
                          {feedMolecules.map((mol) => (
                            <div key={mol.id} className="glass rounded-lg p-2 flex flex-col items-center">
                              <div className="bg-white/5 rounded overflow-hidden">
                                <MoleculeViewer2D smiles={mol.smiles} width={120} height={120} />
                              </div>
                              <div className="w-full mt-1.5 space-y-0.5">
                                <div className="flex items-center justify-between text-[9px]">
                                  <span className="text-neon-green font-mono font-bold">{(mol.reward ?? 0).toFixed(2)}</span>
                                  <span className="text-muted-foreground">QED {(mol.properties?.qed ?? 0).toFixed(2)}</span>
                                </div>
                                <p className="text-[8px] font-mono text-muted-foreground truncate">{mol.smiles}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                          <FlaskConical className="w-10 h-10 text-muted-foreground/20 mb-2" />
                          <p className={`${layout.labelSize} text-muted-foreground`}>Waiting for molecules...</p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <GFlowNet2DTrajectory />
                  )}
                </div>
              </motion.div>

              {/* Right: Metrics */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="h-full overflow-auto"
              >
                <RealtimeMetrics trainingState={trainingState} />
              </motion.div>
            </div>

            {/* Row 2: Training Progress */}
            <div className="flex-1 min-h-0">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`glass-dark rounded-lg ${layout.cardPad} relative group h-full`}
              >
                <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => setExpandedView('training')}
                    className="p-1 bg-dark-panel/80 rounded hover:bg-dark-panel"
                  >
                    <Maximize2 className="w-3 h-3" />
                  </button>
                </div>

                <h3 className={`${layout.headingSize} font-medium mb-2 flex items-center gap-2`}>
                  <BarChart3 className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-blue`} />
                  Training Progress
                </h3>

                <div className="h-[calc(100%-1.5rem)] min-h-0">
                  <GFlowNetTrainingDashboard problemConfig={problemConfig} trainingState={trainingState} />
                </div>
              </motion.div>
            </div>
          </>
        ) : expandedView === 'trajectory' ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="h-full glass-dark rounded-xl p-4 relative"
          >
            <button
              onClick={() => setExpandedView('none')}
              className="absolute top-4 right-4 z-10 p-2 bg-dark-panel/80 rounded-lg hover:bg-dark-panel"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
            
            <div className="h-full">
              <GFlowNet2DTrajectory />
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="h-full glass-dark rounded-xl p-4 relative"
          >
            <button
              onClick={() => setExpandedView('none')}
              className="absolute top-4 right-4 z-10 p-2 bg-dark-panel/80 rounded-lg hover:bg-dark-panel"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
            
            <div className="h-full">
              <GFlowNetTrainingDashboard problemConfig={problemConfig} trainingState={trainingState} />
            </div>
          </motion.div>
        )}
        </div>

        {/* Right: Molecule Feed Side Panel (molecular domain only) */}
        {isMolecularDomain && expandedView === 'none' && (
          <MoleculeFeedPanel molecules={feedMolecules} layout={layout} chartColors={chartColors} isMOGFN={isMOGFN} />
        )}
      </div>
    </div>
  )
}

// --- Helper Components ---

function DiversityMetricCard({ icon, label, value, sub, color, layout }: {
  icon: React.ReactNode
  label: string
  value: number | string
  sub?: string
  color: string
  layout: ReturnType<typeof useThemeLayout>
}) {
  return (
    <div className={`glass-dark rounded-lg ${layout.cardPad} flex items-center gap-2`}>
      <div className="w-7 h-7 rounded-md bg-white/5 flex items-center justify-center flex-shrink-0" style={{ color }}>
        {icon}
      </div>
      <div className="min-w-0">
        <div className={`${layout.tinySize} text-muted-foreground truncate`}>{label}</div>
        <div className={`${layout.labelSize} font-bold font-mono`} style={{ color }}>
          {typeof value === 'number' ? value.toLocaleString() : value}
        </div>
        {sub && <div className={`${layout.tinySize} text-muted-foreground/60`}>{sub}</div>}
      </div>
    </div>
  )
}

function MoleculeFeedPanel({ molecules, layout, chartColors, isMOGFN }: {
  molecules: Molecule[]
  layout: ReturnType<typeof useThemeLayout>
  chartColors: ReturnType<typeof useChartColors>
  isMOGFN?: boolean
}) {
  const [selectedIdx, setSelectedIdx] = useState(0)
  const selected = molecules[selectedIdx] ?? null

  if (!molecules.length) {
    return (
      <div className={`w-64 flex-shrink-0 glass-dark rounded-lg ${layout.cardPad} flex flex-col items-center justify-center text-center`}>
        <FlaskConical className="w-8 h-8 text-muted-foreground/30 mb-2" />
        <span className={`${layout.labelSize} text-muted-foreground`}>Waiting for molecules...</span>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`w-72 flex-shrink-0 glass-dark rounded-lg ${layout.cardPad} flex flex-col overflow-hidden`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <h3 className={`${layout.headingSize} font-medium flex items-center gap-1.5`}>
          <FlaskConical className="w-3 h-3 text-neon-green" />
          Live Molecule Feed
        </h3>
        <span className={`${layout.tinySize} px-1.5 py-0.5 rounded bg-neon-green/15 text-neon-green font-mono`}>
          {molecules.length}
        </span>
      </div>

      {/* Selected molecule detail */}
      {selected && (
        <div className="mb-2 p-2 rounded-lg bg-dark-bg/50 border border-dark-border/30">
          <div className="flex gap-2">
            <div className="w-24 h-24 flex-shrink-0 rounded bg-white/5 overflow-hidden">
              <MoleculeViewer2D smiles={selected.smiles} width={96} height={96} />
            </div>
            <div className="flex-1 min-w-0">
              <div className={`${layout.tinySize} text-muted-foreground truncate font-mono`} title={selected.smiles}>
                {selected.smiles.length > 20 ? selected.smiles.slice(0, 20) + '...' : selected.smiles}
              </div>
              <div className={`${layout.labelSize} font-bold mt-1`} style={{ color: chartColors.green }}>
                Reward: {(selected.reward ?? 0).toFixed(2)}
              </div>
              <div className={`grid grid-cols-2 gap-x-2 gap-y-0.5 mt-1 ${layout.tinySize}`}>
                <span className="text-muted-foreground">MW: <span className="font-mono text-foreground">{(selected.properties?.molecular_weight ?? 0).toFixed(0)}</span></span>
                <span className="text-muted-foreground">QED: <span className="font-mono text-foreground">{(selected.properties?.qed ?? 0).toFixed(2)}</span></span>
                <span className="text-muted-foreground">LogP: <span className="font-mono text-foreground">{(selected.properties?.logp ?? 0).toFixed(1)}</span></span>
                <span className="text-muted-foreground">SA: <span className="font-mono text-foreground">{(selected.properties?.synthetic_accessibility ?? 0).toFixed(1)}</span></span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Synthesis Route (Gap 4) */}
      {selected && (
        <div className="mb-2 border border-dark-border/20 rounded-md overflow-hidden bg-dark-bg/30">
          <SynthesisRoute moleculeId={selected.id} compact />
        </div>
      )}

      {/* Molecule list */}
      <div className="flex-1 overflow-y-auto space-y-1 scrollbar-thin">
        {molecules.map((mol, idx) => (
          <button
            key={mol.id}
            onClick={() => setSelectedIdx(idx)}
            className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md transition-all ${
              idx === selectedIdx
                ? 'bg-neon-green/10 border border-neon-green/30'
                : 'hover:bg-white/5 border border-transparent'
            }`}
          >
            <div className="w-10 h-10 flex-shrink-0 rounded bg-white/5 overflow-hidden">
              <MoleculeViewer2D smiles={mol.smiles} width={40} height={40} />
            </div>
            <div className="flex-1 min-w-0 text-left">
              <div className={`${layout.tinySize} font-mono truncate`}>
                {mol.smiles.length > 16 ? mol.smiles.slice(0, 16) + '...' : mol.smiles}
              </div>
              <div className={`${layout.tinySize} flex items-center gap-2`}>
                <span className="text-neon-green font-mono">{(mol.reward ?? 0).toFixed(1)}</span>
                <span className="text-muted-foreground">QED {(mol.properties?.qed ?? 0).toFixed(2)}</span>
              </div>
            </div>
            <div className={`w-1.5 h-6 rounded-full flex-shrink-0`}
              style={{
                backgroundColor: chartColors.green,
                opacity: 0.2 + (mol.reward / 10) * 0.8,
              }}
            />
          </button>
        ))}
      </div>

      {/* Diversity Analysis (Gap 1) */}
      <div className="border-t border-dark-border/30 mt-2">
        <DiversityStats autoRefresh refreshInterval={30000} />
      </div>

      {/* Docking Panel (Gap 2) */}
      <div className="border-t border-dark-border/30 mt-2">
        <DockingPanel compact />
      </div>

      {/* MOGFN Pareto Optimization (Gap 5) */}
      {isMOGFN && (
        <>
          <div className="border-t border-dark-border/30 mt-2">
            <PreferenceSliders compact />
          </div>
          <div className="border-t border-dark-border/30 mt-2">
            <ParetoFrontExplorer autoRefresh refreshInterval={15000} />
          </div>
        </>
      )}
    </motion.div>
  )
}