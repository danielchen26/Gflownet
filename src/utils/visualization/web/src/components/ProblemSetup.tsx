import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Settings, Grid3x3, Target, Zap, Play, RefreshCw, Info, ChevronLeft, ChevronRight, Sparkles, BookOpen, Lightbulb, Shield, Gauge, Database, ChevronDown, Wrench, AlertTriangle } from 'lucide-react'
import { DomainSelector, DomainOption, BUILT_IN_DOMAINS } from './DomainSelector'
import { DomainConfigPanel } from './DomainConfigPanel'
import { InfoTooltip, TOOLTIPS } from './InfoTooltip'
import { useThemeLayout } from '../contexts/ThemeContext'

interface ProblemConfig {
  domain_type: string
  grid_size: number
  reward_peaks: Array<{
    position: [number, number]
    intensity: number
  }>
  training_objective: 'TRAJECTORY_BALANCE' | 'SUB_TRAJECTORY_BALANCE' | 'DETAILED_BALANCE' | 'FLOW_MATCHING' | 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION'
  n_episodes: number
  batch_size?: number
  learning_rate?: number
  hidden_dim?: number
  temperature?: number
  epsilon?: number
  epsilon_decay?: boolean
  entropy_weight?: number
  z_learning_rate_multiplier?: number
  use_replay_buffer?: boolean
  replay_buffer_size?: number
  replay_ratio?: number
  replay_priority_alpha?: number
  tlm_backward_weight?: number
  tlm_entropy_coeff?: number
  reward_shaping?: boolean
  domain_config?: Record<string, unknown>
}

type SetupStep = 'domain' | 'configure' | 'train'

interface ProblemSetupProps {
  onStart: (config: ProblemConfig) => void
}

export function ProblemSetup({ onStart }: ProblemSetupProps) {
  const layout = useThemeLayout()
  const [currentStep, setCurrentStep] = useState<SetupStep>('domain')
  const [selectedDomain, setSelectedDomain] = useState<DomainOption | null>(
    BUILT_IN_DOMAINS.find(d => d.id === 'grid_world') || null
  )
  const [domainConfig, setDomainConfig] = useState<Record<string, unknown>>({})
  const [isDomainConfigValid, setIsDomainConfigValid] = useState(true)

  const [config, setConfig] = useState<ProblemConfig>({
    domain_type: 'grid_world',
    grid_size: 8,
    reward_peaks: [
      { position: [7, 7], intensity: 10 },
      { position: [0, 7], intensity: 8 },
    ],
    training_objective: 'TRAJECTORY_BALANCE',
    n_episodes: 1000,
    batch_size: 32,
    learning_rate: 0.005,
    hidden_dim: 64,
    temperature: 1.0,
    epsilon: 0.15,
    epsilon_decay: true,
    entropy_weight: 0.02,
    z_learning_rate_multiplier: 10.0,
    use_replay_buffer: true,
    replay_buffer_size: 10000,
    replay_ratio: 0.5,
    replay_priority_alpha: 0.6,
    tlm_backward_weight: 1.0,
    tlm_entropy_coeff: 0.01,
    reward_shaping: true,
  })

  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [showCoreTraining, setShowCoreTraining] = useState(true)
  const [showModeCollapse, setShowModeCollapse] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleDomainSelect = (domain: DomainOption) => {
    setSelectedDomain(domain)
    setConfig(prev => ({ ...prev, domain_type: domain.id }))
    if (domain.id === 'grid_world') {
      setCurrentStep('train')
    } else {
      setCurrentStep('configure')
    }
  }

  const handleDomainConfigChange = (newConfig: Record<string, unknown>) => {
    setDomainConfig(newConfig)
    setConfig(prev => ({ ...prev, domain_config: newConfig }))
  }

  const handleCellClick = (x: number, y: number) => {
    setSelectedCell([x, y])
  }

  const addRewardPeak = () => {
    if (selectedCell) {
      const existing = config.reward_peaks.findIndex(
        p => p.position[0] === selectedCell[0] && p.position[1] === selectedCell[1]
      )
      if (existing === -1) {
        setConfig({
          ...config,
          reward_peaks: [...config.reward_peaks, { position: selectedCell, intensity: 5 }]
        })
      }
    }
  }

  const removeRewardPeak = (index: number) => {
    setConfig({
      ...config,
      reward_peaks: config.reward_peaks.filter((_, i) => i !== index)
    })
  }

  const updatePeakIntensity = (index: number, intensity: number) => {
    const newPeaks = [...config.reward_peaks]
    newPeaks[index].intensity = intensity
    setConfig({ ...config, reward_peaks: newPeaks })
  }

  const startTraining = () => {
    console.log('ProblemSetup: Starting training with config:', config)
    setIsRunning(true)
    onStart(config)
  }

  const steps: { id: SetupStep; label: string; description: string }[] = [
    { id: 'domain', label: 'Select Domain', description: 'Choose a domain type' },
    { id: 'configure', label: 'Configure', description: 'Set domain parameters' },
    { id: 'train', label: 'Training Setup', description: 'Configure training' },
  ]

  const goToStep = (step: SetupStep) => {
    setCurrentStep(step)
  }

  const canProceedToTrain = selectedDomain && (
    selectedDomain.id === 'grid_world' || isDomainConfigValid
  )

  return (
    <div className={`max-w-6xl mx-auto ${layout.sectionPad} ${layout.sectionGap.replace('gap', 'space-y')}`}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`text-center ${layout.compact ? 'mb-4' : layout.spacious ? 'mb-10' : 'mb-8'}`}
      >
        <h1 className={`${layout.compact ? 'text-2xl' : layout.spacious ? 'text-5xl' : 'text-4xl'} font-bold ${layout.compact ? 'mb-2' : 'mb-4'} gradient-text`}>
          GFlowNet Interactive Setup
        </h1>
        <p className={`${layout.compact ? 'text-sm' : 'text-lg'} text-muted-foreground`}>
          Configure your problem domain and start training
        </p>
      </motion.div>

      {/* Step Indicator */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`flex items-center justify-center gap-2 ${layout.compact ? 'mb-4' : layout.spacious ? 'mb-10' : 'mb-8'}`}
      >
        {steps.map((step, index) => {
          const isActive = currentStep === step.id
          const isCompleted = steps.findIndex(s => s.id === currentStep) > index
          const isClickable = step.id === 'domain' ||
            (step.id === 'configure' && selectedDomain && selectedDomain.id !== 'grid_world') ||
            (step.id === 'train' && selectedDomain)

          return (
            <div key={step.id} className="flex items-center">
              <button
                onClick={() => isClickable && goToStep(step.id)}
                disabled={!isClickable}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-neon-purple/20 to-neon-blue/20 border border-neon-purple text-white'
                    : isCompleted
                      ? 'bg-neon-green/10 border border-neon-green/30 text-neon-green'
                      : 'bg-dark-card border border-dark-border text-muted-foreground'
                } ${isClickable ? 'cursor-pointer hover:border-neon-purple/50' : 'cursor-not-allowed'}`}
              >
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                  isActive
                    ? 'bg-neon-purple text-white'
                    : isCompleted
                      ? 'bg-neon-green text-white'
                      : 'bg-dark-border text-muted-foreground'
                }`}>
                  {isCompleted ? '✓' : index + 1}
                </span>
                <div className="text-left">
                  <div className="text-sm font-medium">{step.label}</div>
                  <div className="text-xs text-muted-foreground">{step.description}</div>
                </div>
              </button>
              {index < steps.length - 1 && (
                <ChevronRight className={`w-5 h-5 mx-2 ${
                  isCompleted ? 'text-neon-green' : 'text-dark-border'
                }`} />
              )}
            </div>
          )
        })}
      </motion.div>

      {/* Domain Selection Step */}
      <AnimatePresence mode="wait">
        {currentStep === 'domain' && (
          <motion.div
            key="domain-step"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="glass-dark rounded-xl p-6"
          >
            <DomainSelector
              selectedDomain={selectedDomain?.id}
              onDomainSelect={handleDomainSelect}
            />
          </motion.div>
        )}

        {/* Domain Configuration Step */}
        {currentStep === 'configure' && selectedDomain && selectedDomain.id !== 'grid_world' && (
          <motion.div
            key="configure-step"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="glass-dark rounded-xl p-6"
          >
            <DomainConfigPanel
              domain={selectedDomain}
              config={domainConfig}
              onConfigChange={handleDomainConfigChange}
              onValidationChange={setIsDomainConfigValid}
            />

            <div className="flex justify-between mt-6">
              <button
                onClick={() => setCurrentStep('domain')}
                className="flex items-center gap-2 px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:border-neon-purple/50 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back to Domains
              </button>
              <button
                onClick={() => setCurrentStep('train')}
                disabled={!isDomainConfigValid}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  isDomainConfigValid
                    ? 'bg-gradient-to-r from-neon-purple to-neon-blue text-white hover:shadow-lg hover:shadow-neon-purple/30'
                    : 'bg-dark-card text-muted-foreground cursor-not-allowed'
                }`}
              >
                Continue to Training
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        )}

        {/* Training Setup Step */}
        {currentStep === 'train' && (
          <motion.div
            key="train-step"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            {/* Back to Domain button */}
            <div className="mb-4">
              <button
                onClick={() => setCurrentStep('domain')}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-dark-card border border-dark-border rounded-lg hover:border-neon-purple/50 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Change Domain
                {selectedDomain && (
                  <span className="ml-2 px-2 py-0.5 bg-neon-purple/20 text-neon-purple text-xs rounded">
                    {selectedDomain.name}
                  </span>
                )}
              </button>
            </div>

            {/* ═══════════════════════════════════════════════════════════════
                ROW 1: Two-column — Grid Config (left) + Objective & Peaks (right)
               ═══════════════════════════════════════════════════════════════ */}
            <div className={`grid grid-cols-1 lg:grid-cols-2 ${layout.sectionGap} ${layout.compact ? 'mb-3' : layout.spacious ? 'mb-8' : 'mb-6'}`}>

              {/* Left Column: Grid Configuration */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`glass-dark rounded-xl ${layout.sectionPad}`}
              >
                <h2 className={`${layout.compact ? 'text-base' : layout.spacious ? 'text-2xl' : 'text-xl'} font-semibold ${layout.compact ? 'mb-2' : 'mb-4'} flex items-center gap-2`}>
                  <Grid3x3 className={`${layout.compact ? 'w-4 h-4' : 'w-5 h-5'} text-neon-purple`} />
                  Grid World Configuration
                </h2>

                {/* Interactive Grid */}
                <div className="mb-4">
                  <div
                    className="inline-block border border-dark-border rounded-lg p-2 bg-dark-panel"
                    style={{
                      display: 'grid',
                      gridTemplateColumns: `repeat(${config.grid_size}, 1fr)`,
                      gap: '2px',
                      width: 'fit-content'
                    }}
                  >
                    {Array.from({ length: config.grid_size * config.grid_size }).map((_, i) => {
                      const x = i % config.grid_size
                      const y = Math.floor(i / config.grid_size)
                      const peak = config.reward_peaks.find(
                        p => p.position[0] === x && p.position[1] === y
                      )
                      const isSelected = selectedCell?.[0] === x && selectedCell?.[1] === y

                      return (
                        <motion.div
                          key={i}
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => handleCellClick(x, y)}
                          className={`
                            w-8 h-8 rounded cursor-pointer transition-all
                            ${peak
                              ? 'bg-gradient-to-br from-neon-green to-neon-green/50'
                              : 'bg-dark-bg hover:bg-dark-border'
                            }
                            ${isSelected ? 'ring-2 ring-neon-purple' : ''}
                          `}
                          style={{
                            opacity: peak ? 0.3 + (peak.intensity / 10) * 0.7 : 1
                          }}
                        >
                          {peak && (
                            <div className="w-full h-full flex items-center justify-center text-xs font-bold">
                              {peak.intensity}
                            </div>
                          )}
                        </motion.div>
                      )
                    })}
                  </div>

                  {selectedCell && (() => {
                    const existingPeakIndex = config.reward_peaks.findIndex(
                      p => p.position[0] === selectedCell[0] && p.position[1] === selectedCell[1]
                    )
                    return (
                      <div className="mt-4 flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          Selected: ({selectedCell[0]}, {selectedCell[1]})
                        </span>
                        {existingPeakIndex === -1 ? (
                          <button
                            onClick={addRewardPeak}
                            className="px-3 py-1 text-sm bg-neon-green/20 text-neon-green rounded-md hover:bg-neon-green/30"
                          >
                            Add Reward Peak
                          </button>
                        ) : (
                          <>
                            <span className="text-xs text-neon-green font-mono">
                              R={config.reward_peaks[existingPeakIndex].intensity}
                            </span>
                            <button
                              onClick={() => removeRewardPeak(existingPeakIndex)}
                              className="px-3 py-1 text-sm bg-red-500/20 text-red-400 rounded-md hover:bg-red-500/30"
                            >
                              Remove Peak
                            </button>
                          </>
                        )}
                      </div>
                    )
                  })()}
                </div>

                {/* Grid Size */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Grid Size</label>
                  <input
                    type="range"
                    min={4}
                    max={16}
                    value={config.grid_size}
                    onChange={(e) => setConfig({ ...config, grid_size: Number(e.target.value), reward_peaks: [] })}
                    className="w-full"
                  />
                  <span className="text-xs text-muted-foreground">{config.grid_size} × {config.grid_size}</span>
                </div>
              </motion.div>

              {/* Right Column: Training Objective + Reward Peaks */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`glass-dark rounded-xl ${layout.sectionPad} ${layout.compact ? 'space-y-2' : 'space-y-4'}`}
              >
                {/* Training Objective */}
                <div>
                  <h2 className={`${layout.compact ? 'text-base' : layout.spacious ? 'text-2xl' : 'text-xl'} font-semibold ${layout.compact ? 'mb-2' : 'mb-3'} flex items-center gap-2`}>
                    <BookOpen className={`${layout.compact ? 'w-4 h-4' : 'w-5 h-5'} text-neon-blue`} />
                    Training Objective
                  </h2>

                  <div className="grid grid-cols-2 gap-2 mb-3">
                    {[
                      { value: 'TRAJECTORY_BALANCE', label: 'Trajectory Balance', short: 'TB', desc: 'Standard with learnable Z', color: 'neon-purple', tooltip: TOOLTIPS.TRAJECTORY_BALANCE },
                      { value: 'SUB_TRAJECTORY_BALANCE', label: 'Sub-Trajectory', short: 'STB', desc: 'O(T²) credit assignment', color: 'neon-blue', tooltip: TOOLTIPS.SUB_TRAJECTORY_BALANCE },
                      { value: 'DETAILED_BALANCE', label: 'Detailed Balance', short: 'DB', desc: 'Local balance + backward policy', color: 'neon-green', tooltip: TOOLTIPS.DETAILED_BALANCE },
                      { value: 'FLOW_MATCHING', label: 'Flow Matching', short: 'FM', desc: 'Direct flow estimation', color: 'neon-orange', tooltip: TOOLTIPS.FLOW_MATCHING },
                    ].map((obj) => (
                      <button
                        key={obj.value}
                        onClick={() => setConfig({ ...config, training_objective: obj.value as any })}
                        className={`p-2.5 rounded-lg text-left transition-all ${
                          config.training_objective === obj.value
                            ? `border-2 border-${obj.color} bg-${obj.color}/10`
                            : 'border border-dark-border hover:border-dark-border/70 bg-dark-panel/50'
                        }`}
                      >
                        <div className="flex items-center gap-1.5 mb-0.5">
                          <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${
                            config.training_objective === obj.value ? `bg-${obj.color}/30 text-${obj.color}` : 'bg-dark-border text-muted-foreground'
                          }`}>
                            {obj.short}
                          </span>
                          <span className="text-xs font-medium">{obj.label}</span>
                          <InfoTooltip {...obj.tooltip} size="sm" />
                        </div>
                        <p className="text-[10px] text-muted-foreground">{obj.desc}</p>
                      </button>
                    ))}
                  </div>

                  {/* TLM — full-width highlight */}
                  <button
                    onClick={() => setConfig({ ...config, training_objective: 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION' })}
                    className={`w-full p-2.5 rounded-lg text-left transition-all ${
                      config.training_objective === 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION'
                        ? 'border-2 border-neon-cyan bg-neon-cyan/10'
                        : 'border border-neon-cyan/30 hover:border-neon-cyan/50 bg-dark-panel/50'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${
                        config.training_objective === 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION' ? 'bg-neon-cyan/30 text-neon-cyan' : 'bg-dark-border text-muted-foreground'
                      }`}>
                        TLM
                      </span>
                      <span className="text-xs font-medium">Trajectory Likelihood Maximization</span>
                      <InfoTooltip {...TOOLTIPS.TRAJECTORY_LIKELIHOOD_MAXIMIZATION} size="sm" />
                      <span className="px-1.5 py-0.5 text-[9px] bg-neon-cyan/20 text-neon-cyan rounded ml-auto">
                        ICLR 2025
                      </span>
                    </div>
                    <p className="text-[10px] text-muted-foreground">
                      Trains backward policy P_B to encode path counts. Solves extreme mode collapse.
                    </p>
                  </button>

                  {/* TLM Parameters — inline when TLM selected */}
                  {config.training_objective === 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION' && (
                    <div className="mt-2 p-3 bg-neon-cyan/5 border border-neon-cyan/20 rounded-lg space-y-3">
                      <div className="flex items-center gap-2 text-xs text-neon-cyan font-medium">
                        <Lightbulb className="w-3 h-3" />
                        TLM Parameters
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-1">
                            <span className="text-[10px] text-muted-foreground">Backward Weight (λ)</span>
                            <InfoTooltip {...TOOLTIPS.TLM_BACKWARD_WEIGHT} size="sm" />
                          </div>
                          <span className="text-[10px] text-neon-cyan font-mono">{(config.tlm_backward_weight ?? 1.0).toFixed(1)}</span>
                        </div>
                        <input
                          type="range"
                          min={0.1}
                          max={5.0}
                          step={0.1}
                          value={config.tlm_backward_weight ?? 1.0}
                          onChange={(e) => setConfig({ ...config, tlm_backward_weight: Number(e.target.value) })}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-1">
                            <span className="text-[10px] text-muted-foreground">Backward Entropy Coeff</span>
                            <InfoTooltip {...TOOLTIPS.TLM_ENTROPY_COEFF} size="sm" />
                          </div>
                          <span className="text-[10px] text-neon-cyan font-mono">{(config.tlm_entropy_coeff ?? 0.01).toFixed(3)}</span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={0.1}
                          step={0.001}
                          value={config.tlm_entropy_coeff ?? 0.01}
                          onChange={(e) => setConfig({ ...config, tlm_entropy_coeff: Number(e.target.value) })}
                          className="w-full"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Reward Peaks */}
                <div className="p-3 border border-dark-border rounded-lg bg-dark-panel/30">
                  <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Target className="w-4 h-4 text-neon-green" />
                    Reward Peaks
                    <span className="text-xs text-muted-foreground ml-auto">{config.reward_peaks.length} peaks</span>
                  </h3>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {config.reward_peaks.map((peak, i) => (
                      <div key={i} className="flex items-center gap-2 p-2 bg-dark-panel rounded-md">
                        <span className="text-sm font-mono text-neon-green">
                          ({peak.position[0]}, {peak.position[1]})
                        </span>
                        <input
                          type="range"
                          min={1}
                          max={10}
                          step={0.5}
                          value={peak.intensity}
                          onChange={(e) => updatePeakIntensity(i, Number(e.target.value))}
                          className="flex-1"
                        />
                        <span className="text-sm w-12 text-right font-mono">{peak.intensity}</span>
                        <button
                          onClick={() => removeRewardPeak(i)}
                          className="p-1 text-red-500 hover:bg-red-500/20 rounded"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    {config.reward_peaks.length === 0 && (
                      <p className="text-xs text-muted-foreground text-center py-2">
                        Click cells on the grid to add reward peaks
                      </p>
                    )}
                  </div>
                </div>

                {/* Reward Peak Presets */}
                <div className="flex gap-2">
                  <button
                    onClick={() => setConfig({
                      ...config,
                      reward_peaks: [
                        { position: [7, 7], intensity: 10 },
                        { position: [1, 7], intensity: 8 },
                        { position: [4, 4], intensity: 6 },
                        { position: [7, 1], intensity: 7 },
                      ]
                    })}
                    className="flex-1 px-2 py-1.5 text-[10px] bg-dark-panel text-muted-foreground border border-dark-border rounded hover:bg-dark-border transition-colors"
                  >
                    Multi-Modal
                  </button>
                  <button
                    onClick={() => setConfig({
                      ...config,
                      reward_peaks: [
                        { position: [config.grid_size - 1, config.grid_size - 1], intensity: 10 },
                      ]
                    })}
                    className="flex-1 px-2 py-1.5 text-[10px] bg-dark-panel text-muted-foreground border border-dark-border rounded hover:bg-dark-border transition-colors"
                  >
                    Single Goal
                  </button>
                  <button
                    onClick={() => setConfig({
                      ...config,
                      reward_peaks: [
                        { position: [1, 1], intensity: 5 },
                        { position: [config.grid_size - 2, config.grid_size - 2], intensity: 5 },
                        { position: [1, config.grid_size - 2], intensity: 5 },
                        { position: [config.grid_size - 2, 1], intensity: 5 },
                      ]
                    })}
                    className="flex-1 px-2 py-1.5 text-[10px] bg-dark-panel text-muted-foreground border border-dark-border rounded hover:bg-dark-border transition-colors"
                  >
                    Four Corners
                  </button>
                </div>
              </motion.div>
            </div>

            {/* ═══════════════════════════════════════════════════════════════
                SECTION 1: Core Training Parameters
               ═══════════════════════════════════════════════════════════════ */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className={`glass-dark rounded-xl ${layout.sectionPad} ${layout.compact ? 'mb-2' : 'mb-4'}`}
            >
              {/* Collapsible Header */}
              <button
                onClick={() => setShowCoreTraining(!showCoreTraining)}
                className="w-full flex items-center justify-between"
              >
                <div>
                  <h2 className={`${layout.compact ? 'text-sm' : layout.spacious ? 'text-xl' : 'text-lg'} font-semibold mb-0.5 flex items-center gap-2`}>
                    <Settings className={`${layout.compact ? 'w-4 h-4' : 'w-5 h-5'} text-neon-blue`} />
                    Core Training
                  </h2>
                  {!showCoreTraining && (
                    <div className="flex items-center gap-3 text-[10px] text-muted-foreground ml-7">
                      <span>Ep: <span className="text-neon-blue font-mono">{config.n_episodes}</span></span>
                      <span>Batch: <span className="text-neon-blue font-mono">{config.batch_size}</span></span>
                      <span>LR: <span className="text-neon-blue font-mono">{config.learning_rate?.toFixed(4)}</span></span>
                    </div>
                  )}
                </div>
                <ChevronDown className={`w-5 h-5 text-muted-foreground transition-transform ${showCoreTraining ? 'rotate-180' : ''}`} />
              </button>

              <AnimatePresence>
                {showCoreTraining && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <p className="text-xs text-muted-foreground mt-1 mb-4">
                      Standard training hyperparameters for gradient-based optimization
                    </p>

                    <div className={`grid grid-cols-1 md:grid-cols-3 ${layout.gap}`}>
                      {/* Episodes */}
                      <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border border-dark-border`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium">Episodes</span>
                          <span className="text-sm font-mono text-neon-blue">{config.n_episodes}</span>
                        </div>
                        <input
                          type="range"
                          min={50}
                          max={2000}
                          step={50}
                          value={config.n_episodes}
                          onChange={(e) => setConfig({ ...config, n_episodes: Number(e.target.value) })}
                          className="w-full mb-1"
                        />
                        <div className="flex justify-between text-[9px] text-muted-foreground">
                          <span>50</span>
                          <span>2000</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground mt-2">
                          Training iterations. More episodes = better convergence but longer training.
                        </p>
                      </div>

                      {/* Batch Size */}
                      <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border border-dark-border`}>
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-1">
                            <span className="text-sm font-medium">Batch Size</span>
                            <InfoTooltip {...TOOLTIPS.BATCH_SIZE} size="sm" />
                          </div>
                          <span className="text-sm font-mono text-neon-blue">{config.batch_size}</span>
                        </div>
                        <input
                          type="range"
                          min={4}
                          max={64}
                          step={4}
                          value={config.batch_size ?? 32}
                          onChange={(e) => setConfig({ ...config, batch_size: Number(e.target.value) })}
                          className="w-full mb-1"
                        />
                        <div className="flex justify-between text-[9px] text-muted-foreground">
                          <span>4 (faster)</span>
                          <span>64 (stable)</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground mt-2">
                          Trajectories per gradient update. More = stable gradients, slower iteration.
                        </p>
                      </div>

                      {/* Learning Rate */}
                      <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border border-dark-border`}>
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-1">
                            <span className="text-sm font-medium">Learning Rate</span>
                            <InfoTooltip {...TOOLTIPS.LEARNING_RATE} size="sm" />
                          </div>
                          <span className="text-sm font-mono text-neon-blue">{config.learning_rate?.toFixed(4)}</span>
                        </div>
                        <input
                          type="range"
                          min={0.0001}
                          max={0.01}
                          step={0.0001}
                          value={config.learning_rate ?? 0.005}
                          onChange={(e) => setConfig({ ...config, learning_rate: Number(e.target.value) })}
                          className="w-full mb-1"
                        />
                        <div className="flex justify-between text-[9px] text-muted-foreground">
                          <span>0.0001 (cautious)</span>
                          <span>0.01 (aggressive)</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground mt-2">
                          Step size for weight updates. Too high = unstable, too low = slow.
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* ═══════════════════════════════════════════════════════════════
                SECTION 2: Mode Collapse Prevention
               ═══════════════════════════════════════════════════════════════ */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className={`glass-dark rounded-xl ${layout.sectionPad} ${layout.compact ? 'mb-2' : 'mb-4'}`}
            >
              {/* Collapsible Header */}
              <button
                onClick={() => setShowModeCollapse(!showModeCollapse)}
                className="w-full flex items-center justify-between"
              >
                <div>
                  <div className="flex items-center gap-3 mb-0.5">
                    <h2 className={`${layout.compact ? 'text-sm' : layout.spacious ? 'text-xl' : 'text-lg'} font-semibold flex items-center gap-2`}>
                      <Shield className={`${layout.compact ? 'w-4 h-4' : 'w-5 h-5'} text-neon-green`} />
                      Mode Collapse Prevention
                    </h2>
                    <span className="px-2 py-0.5 text-[10px] font-medium bg-neon-green/15 text-neon-green rounded-full">
                      Recommended
                    </span>
                  </div>
                  {!showModeCollapse && (
                    <div className="flex items-center gap-2 text-[10px] text-muted-foreground ml-7 flex-wrap">
                      {(config.epsilon ?? 0) > 0 && (
                        <span className="px-1.5 py-0.5 bg-neon-purple/15 text-neon-purple rounded">ε={((config.epsilon ?? 0) * 100).toFixed(0)}%</span>
                      )}
                      {(config.entropy_weight ?? 0) > 0 && (
                        <span className="px-1.5 py-0.5 bg-neon-blue/15 text-neon-blue rounded">H={(config.entropy_weight ?? 0).toFixed(3)}</span>
                      )}
                      {(config.z_learning_rate_multiplier ?? 1) > 1 && (
                        <span className="px-1.5 py-0.5 bg-neon-orange/15 text-neon-orange rounded">Z={(config.z_learning_rate_multiplier ?? 1)}×</span>
                      )}
                      {config.use_replay_buffer && (
                        <span className="px-1.5 py-0.5 bg-cyan-500/15 text-cyan-400 rounded">Replay</span>
                      )}
                      {config.reward_shaping && (
                        <span className="px-1.5 py-0.5 bg-neon-green/15 text-neon-green rounded">Shaping</span>
                      )}
                    </div>
                  )}
                </div>
                <ChevronDown className={`w-5 h-5 text-muted-foreground transition-transform ${showModeCollapse ? 'rotate-180' : ''}`} />
              </button>

              <AnimatePresence>
                {showModeCollapse && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <p className="text-xs text-muted-foreground mt-1 mb-5">
                      These techniques prevent the GFlowNet from collapsing to a single mode. Based on Malkin et al. (2022) and AISTATS 2024.
                    </p>

              <div className={`grid grid-cols-1 md:grid-cols-2 ${layout.gap} ${layout.compact ? 'mb-2' : 'mb-4'}`}>
                {/* ε-Uniform Exploration */}
                <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border border-neon-purple/30`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <Sparkles className="w-4 h-4 text-neon-purple" />
                      <span className="text-sm font-medium">ε-Uniform Exploration</span>
                      <InfoTooltip {...TOOLTIPS.EPSILON_EXPLORATION} size="sm" />
                    </div>
                    <span className="text-sm font-mono text-neon-purple">{(config.epsilon ?? 0.15).toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={0.3}
                    step={0.01}
                    value={config.epsilon ?? 0.15}
                    onChange={(e) => setConfig({ ...config, epsilon: Number(e.target.value) })}
                    className="w-full mb-1"
                  />
                  <div className="flex justify-between text-[9px] text-muted-foreground">
                    <span>0% (greedy)</span>
                    <span>10% (default)</span>
                    <span>30% (max explore)</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-2">
                    Probability of taking random actions. Formula: P(a|s) = (1-ε)P_F + ε Uniform
                  </p>
                  <div className="flex items-center gap-1.5 mt-2">
                    <input
                      type="checkbox"
                      id="epsilon_decay"
                      checked={config.epsilon_decay ?? true}
                      onChange={(e) => setConfig({ ...config, epsilon_decay: e.target.checked })}
                      className="w-3.5 h-3.5 rounded bg-dark-panel border-dark-border accent-neon-purple"
                    />
                    <label htmlFor="epsilon_decay" className="text-[10px] text-muted-foreground">
                      Anneal ε → 0 during training (recommended for convergence)
                    </label>
                  </div>
                </div>

                {/* Entropy Regularization */}
                <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border border-neon-purple/30`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <Zap className="w-4 h-4 text-neon-blue" />
                      <span className="text-sm font-medium">Entropy Regularization</span>
                      <InfoTooltip {...TOOLTIPS.ENTROPY_REGULARIZATION} size="sm" />
                    </div>
                    <span className="text-sm font-mono text-neon-blue">{(config.entropy_weight ?? 0.02).toFixed(3)}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={0.1}
                    step={0.001}
                    value={config.entropy_weight ?? 0.02}
                    onChange={(e) => setConfig({ ...config, entropy_weight: Number(e.target.value) })}
                    className="w-full mb-1"
                  />
                  <div className="flex justify-between text-[9px] text-muted-foreground">
                    <span>0 (off)</span>
                    <span>0.1 (strong)</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-2">
                    Bonus for policy diversity. Prevents premature convergence to deterministic policy.
                  </p>
                </div>

                {/* Z Learning Rate Multiplier */}
                <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border border-neon-orange/30`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <Gauge className="w-4 h-4 text-neon-orange" />
                      <span className="text-sm font-medium">Z Learning Rate Multiplier</span>
                      <InfoTooltip {...TOOLTIPS.Z_LEARNING_RATE} size="sm" />
                    </div>
                    <span className="text-sm font-mono text-neon-orange">{(config.z_learning_rate_multiplier ?? 10.0).toFixed(0)}×</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={20}
                    step={1}
                    value={config.z_learning_rate_multiplier ?? 10.0}
                    onChange={(e) => setConfig({ ...config, z_learning_rate_multiplier: Number(e.target.value) })}
                    className="w-full mb-1"
                  />
                  <div className="flex justify-between text-[9px] text-muted-foreground">
                    <span>1× (same as policy)</span>
                    <span>20× (fastest)</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-2">
                    Faster partition function Z learning. 10× recommended (peptide paper).
                  </p>
                </div>

                {/* Experience Replay Buffer */}
                <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border ${config.use_replay_buffer ? 'border-neon-cyan/40' : 'border-dark-border'}`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <Database className="w-4 h-4 text-neon-cyan" />
                      <span className="text-sm font-medium">Experience Replay Buffer</span>
                      <InfoTooltip {...TOOLTIPS.EXPERIENCE_REPLAY} size="sm" />
                      <span className="px-1.5 py-0.5 text-[9px] bg-neon-cyan/15 text-neon-cyan rounded ml-1">
                        JMLR 2023
                      </span>
                    </div>
                    <input
                      type="checkbox"
                      checked={config.use_replay_buffer ?? false}
                      onChange={(e) => setConfig({ ...config, use_replay_buffer: e.target.checked })}
                      className="w-4 h-4 rounded bg-dark-panel border-dark-border accent-neon-cyan"
                    />
                  </div>
                  {config.use_replay_buffer && (
                    <div className="mt-2 pt-2 border-t border-dark-border/50">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-muted-foreground">Replay Ratio</span>
                        <span className="text-xs font-mono text-neon-cyan">{((config.replay_ratio ?? 0.5) * 100).toFixed(0)}%</span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.1}
                        value={config.replay_ratio ?? 0.5}
                        onChange={(e) => setConfig({ ...config, replay_ratio: Number(e.target.value) })}
                        className="w-full mb-1"
                      />
                      <div className="flex justify-between text-[9px] text-muted-foreground">
                        <span>0% (fresh only)</span>
                        <span>50%</span>
                        <span>100% (replay only)</span>
                      </div>
                    </div>
                  )}
                  <p className="text-[10px] text-muted-foreground mt-2">
                    Off-policy learning from past trajectories. Helps discover minority modes.
                  </p>
                </div>
              </div>

              {/* Reward Shaping — full-width highlight card */}
              <div className={`${layout.innerPad} rounded-lg border ${config.reward_shaping ? 'border-neon-green/40 bg-neon-green/5' : 'border-dark-border bg-dark-panel/50'}`}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    <Target className="w-4 h-4 text-neon-green" />
                    <span className="text-sm font-medium">Reward Shaping</span>
                    <span className="px-1.5 py-0.5 text-[9px] font-medium bg-neon-green/15 text-neon-green rounded">
                      Essential
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Enable</span>
                    <input
                      type="checkbox"
                      checked={config.reward_shaping ?? true}
                      onChange={(e) => setConfig({ ...config, reward_shaping: e.target.checked })}
                      className="w-4 h-4 rounded bg-dark-panel border-dark-border accent-neon-green"
                    />
                  </div>
                </div>
                <p className="text-[10px] text-muted-foreground mt-1">
                  Auto-compensates path asymmetry by scaling rewards inversely to path counts. Without this, peaks with more paths dominate
                  (e.g., 70:1 on 5×5, 3432:1 on 8×8).
                </p>
              </div>

              {/* Presets */}
              <div className="flex items-center gap-2 mt-4">
                <span className="text-xs text-muted-foreground">Presets:</span>
                <button
                  onClick={() => setConfig({
                    ...config,
                    epsilon: 0.15,
                    epsilon_decay: true,
                    entropy_weight: 0.02,
                    z_learning_rate_multiplier: 10.0,
                    use_replay_buffer: true,
                    reward_shaping: true,
                  })}
                  className="px-3 py-1.5 text-xs bg-neon-green/10 text-neon-green border border-neon-green/30 rounded-md hover:bg-neon-green/20 transition-colors"
                >
                  Balanced (Default)
                </button>
                <button
                  onClick={() => setConfig({
                    ...config,
                    epsilon: 0.25,
                    epsilon_decay: true,
                    entropy_weight: 0.05,
                    z_learning_rate_multiplier: 15.0,
                    use_replay_buffer: true,
                    reward_shaping: true,
                  })}
                  className="px-3 py-1.5 text-xs bg-neon-purple/10 text-neon-purple border border-neon-purple/30 rounded-md hover:bg-neon-purple/20 transition-colors"
                >
                  High Exploration
                </button>
                <button
                  onClick={() => setConfig({
                    ...config,
                    epsilon: 0,
                    epsilon_decay: false,
                    entropy_weight: 0,
                    z_learning_rate_multiplier: 1.0,
                    use_replay_buffer: false,
                    reward_shaping: false,
                  })}
                  className="px-3 py-1.5 text-xs bg-dark-panel text-muted-foreground border border-dark-border rounded-md hover:bg-dark-border transition-colors"
                >
                  Minimal
                </button>
              </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* ═══════════════════════════════════════════════════════════════
                SECTION 3: Advanced Parameters (collapsible)
               ═══════════════════════════════════════════════════════════════ */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className={`glass-dark rounded-xl ${layout.sectionPad} ${layout.compact ? 'mb-3' : 'mb-6'}`}
            >
              {/* Collapsible Header */}
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full flex items-center justify-between"
              >
                <div className="flex items-center gap-2">
                  <Wrench className={`${layout.compact ? 'w-4 h-4' : 'w-5 h-5'} text-muted-foreground`} />
                  <h2 className={`${layout.compact ? 'text-sm' : layout.spacious ? 'text-xl' : 'text-lg'} font-semibold`}>Advanced Parameters</h2>
                  <span className="px-2 py-0.5 text-[10px] text-muted-foreground bg-dark-panel border border-dark-border rounded-full">
                    Optional
                  </span>
                </div>
                <ChevronDown className={`w-5 h-5 text-muted-foreground transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
              </button>

              <AnimatePresence>
                {showAdvanced && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <p className="text-xs text-muted-foreground mt-2 mb-4">
                      These parameters rarely need adjustment. Defaults work well for most experiments.
                    </p>

                    <div className={`grid grid-cols-1 md:grid-cols-2 ${layout.gap}`}>
                      {/* Temperature */}
                      <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border border-dark-border`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium">Temperature</span>
                          <span className="text-sm font-mono text-neon-blue">{(config.temperature ?? 1.0).toFixed(1)}</span>
                        </div>
                        <input
                          type="range"
                          min={0.5}
                          max={3.0}
                          step={0.1}
                          value={config.temperature ?? 1.0}
                          onChange={(e) => setConfig({ ...config, temperature: Number(e.target.value) })}
                          className="w-full mb-1"
                        />
                        <div className="flex justify-between text-[9px] text-muted-foreground">
                          <span>0.5 (focused)</span>
                          <span>1.0 (default)</span>
                          <span>3.0 (random)</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground mt-2">
                          Controls sampling randomness. Higher = more random exploration.
                        </p>
                        {(config.temperature ?? 1.0) > 2.0 && (
                          <div className="flex items-center gap-1 mt-2 text-[10px] text-amber-400">
                            <AlertTriangle className="w-3 h-3" />
                            <span>Values above 2.0 make sampling nearly random — training may fail</span>
                          </div>
                        )}
                      </div>

                      {/* Hidden Dimension */}
                      <div className={`${layout.innerPad} bg-dark-panel/50 rounded-lg border border-dark-border`}>
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-1">
                            <span className="text-sm font-medium">Hidden Dimension</span>
                            <InfoTooltip {...TOOLTIPS.HIDDEN_DIM} size="sm" />
                          </div>
                          <span className="text-sm font-mono text-neon-blue">{config.hidden_dim}</span>
                        </div>
                        <input
                          type="range"
                          min={32}
                          max={256}
                          step={32}
                          value={config.hidden_dim ?? 64}
                          onChange={(e) => setConfig({ ...config, hidden_dim: Number(e.target.value) })}
                          className="w-full mb-1"
                        />
                        <div className="flex justify-between text-[9px] text-muted-foreground">
                          <span>32 (fast)</span>
                          <span>256 (expressive)</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground mt-2">
                          Neural network width. Larger = more capacity but slower. 64 sufficient for grids.
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* ═══════════════════════════════════════════════════════════════
                START TRAINING BUTTON
               ═══════════════════════════════════════════════════════════════ */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
              className={layout.compact ? 'mb-3' : 'mb-6'}
            >
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={startTraining}
                disabled={isRunning || config.reward_peaks.length === 0}
                className={`
                  w-full ${layout.compact ? 'py-2.5 text-sm' : layout.spacious ? 'py-4 text-xl' : 'py-3.5 text-lg'} rounded-xl font-medium flex items-center justify-center gap-2
                  ${isRunning || config.reward_peaks.length === 0
                    ? 'bg-dark-panel text-muted-foreground cursor-not-allowed'
                    : 'bg-gradient-to-r from-neon-purple to-neon-blue text-white hover:shadow-lg hover:shadow-neon-purple/50'
                  }
                `}
              >
                {isRunning ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Starting Training...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start Training
                  </>
                )}
              </motion.button>

              {config.reward_peaks.length === 0 && (
                <p className="text-xs text-red-500 text-center mt-2">
                  Please add at least one reward peak by clicking on the grid
                </p>
              )}
            </motion.div>

            {/* ═══════════════════════════════════════════════════════════════
                ROW 3: Configuration Summary
               ═══════════════════════════════════════════════════════════════ */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className={`glass-dark rounded-xl ${layout.cardPad}`}
            >
              <h3 className={`${layout.compact ? 'text-xs' : 'text-sm'} font-medium ${layout.compact ? 'mb-2' : 'mb-3'} flex items-center gap-2`}>
                <Info className={`${layout.compact ? 'w-3 h-3' : 'w-4 h-4'} text-neon-blue`} />
                Configuration Summary
              </h3>

              <div className={`grid grid-cols-2 ${layout.compact ? 'lg:grid-cols-6' : 'lg:grid-cols-6'} ${layout.gap} ${layout.compact ? 'mb-2' : 'mb-4'}`}>
                <div className="p-2 bg-dark-panel/50 rounded-lg border-l-2 border-neon-purple">
                  <span className="text-[10px] text-muted-foreground block">Domain</span>
                  <span className="text-sm font-medium">{selectedDomain?.name || 'Grid World'}</span>
                </div>
                <div className="p-2 bg-dark-panel/50 rounded-lg border-l-2 border-neon-purple">
                  <span className="text-[10px] text-muted-foreground block">Grid</span>
                  <span className="text-sm font-medium">{config.grid_size}×{config.grid_size}</span>
                </div>
                <div className="p-2 bg-dark-panel/50 rounded-lg border-l-2 border-neon-green">
                  <span className="text-[10px] text-muted-foreground block">Objective</span>
                  <span className="text-sm font-medium">{config.training_objective.split('_').map(w => w[0]).join('')}</span>
                </div>
                <div className="p-2 bg-dark-panel/50 rounded-lg border-l-2 border-neon-blue">
                  <span className="text-[10px] text-muted-foreground block">Episodes</span>
                  <span className="text-sm font-medium">{config.n_episodes}</span>
                </div>
                <div className="p-2 bg-dark-panel/50 rounded-lg border-l-2 border-neon-orange">
                  <span className="text-[10px] text-muted-foreground block">Exploration ε</span>
                  <span className="text-sm font-medium">{((config.epsilon ?? 0.05) * 100).toFixed(0)}%{config.epsilon_decay ? ' ↓' : ''}</span>
                </div>
                <div className="p-2 bg-dark-panel/50 rounded-lg border-l-2 border-neon-orange">
                  <span className="text-[10px] text-muted-foreground block">Z Rate</span>
                  <span className="text-sm font-medium">{(config.z_learning_rate_multiplier ?? 10).toFixed(0)}×</span>
                </div>
              </div>

              <div className={`grid grid-cols-2 ${layout.gap}`}>
                <div className="p-3 bg-dark-panel/30 rounded-lg border border-dark-border/50">
                  <h4 className="text-xs font-medium mb-2 flex items-center gap-2">
                    <Shield className="w-3 h-3 text-neon-green" />
                    Mode Collapse Prevention
                  </h4>
                  <div className="flex flex-wrap gap-1">
                    {(config.epsilon ?? 0) > 0 && (
                      <span className="px-2 py-0.5 text-[10px] bg-neon-purple/20 text-neon-purple rounded-full">
                        ε-Exploration: {((config.epsilon ?? 0) * 100).toFixed(0)}%
                      </span>
                    )}
                    {config.epsilon_decay && (
                      <span className="px-2 py-0.5 text-[10px] bg-neon-blue/20 text-neon-blue rounded-full">
                        ε Decay
                      </span>
                    )}
                    {(config.entropy_weight ?? 0) > 0 && (
                      <span className="px-2 py-0.5 text-[10px] bg-neon-green/20 text-neon-green rounded-full">
                        Entropy: {(config.entropy_weight ?? 0).toFixed(3)}
                      </span>
                    )}
                    {(config.z_learning_rate_multiplier ?? 1) > 1 && (
                      <span className="px-2 py-0.5 text-[10px] bg-neon-orange/20 text-neon-orange rounded-full">
                        Z: {(config.z_learning_rate_multiplier ?? 1)}×
                      </span>
                    )}
                    {config.use_replay_buffer && (
                      <span className="px-2 py-0.5 text-[10px] bg-cyan-500/20 text-cyan-400 rounded-full">
                        Replay: {((config.replay_ratio ?? 0.5) * 100).toFixed(0)}%
                      </span>
                    )}
                    {config.reward_shaping && (
                      <span className="px-2 py-0.5 text-[10px] bg-neon-green/20 text-neon-green rounded-full">
                        Reward Shaping
                      </span>
                    )}
                    {config.training_objective === 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION' && (
                      <span className="px-2 py-0.5 text-[10px] bg-neon-cyan/20 text-neon-cyan rounded-full font-medium">
                        TLM λ={config.tlm_backward_weight ?? 1.0}
                      </span>
                    )}
                    {(config.epsilon ?? 0) === 0 && !(config.entropy_weight ?? 0) && (config.z_learning_rate_multiplier ?? 1) === 1 && !config.use_replay_buffer && config.training_objective !== 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION' && (
                      <span className="px-2 py-0.5 text-[10px] bg-red-500/20 text-red-400 rounded-full">
                        ⚠ None Active
                      </span>
                    )}
                  </div>
                </div>

                <div className="p-3 bg-dark-panel/30 rounded-lg border border-dark-border/50">
                  <h4 className="text-xs font-medium mb-2 flex items-center gap-2">
                    <BookOpen className="w-3 h-3 text-neon-purple" />
                    {config.training_objective.replace(/_/g, ' ')}
                  </h4>
                  <p className="text-[10px] text-muted-foreground">
                    {config.training_objective === 'TRAJECTORY_BALANCE' && (
                      "Standard GFlowNet objective using full trajectory log-probability with learnable partition function Z."
                    )}
                    {config.training_objective === 'SUB_TRAJECTORY_BALANCE' && (
                      "O(T²) learning signals from sub-trajectories. Better credit assignment for long trajectories."
                    )}
                    {config.training_objective === 'DETAILED_BALANCE' && (
                      "Local balance conditions using backward policy. Requires joint state-pair features."
                    )}
                    {config.training_objective === 'TRAJECTORY_LIKELIHOOD_MAXIMIZATION' && (
                      `TLM (ICLR 2025): Trains backward policy P_B to encode path counts. λ=${(config.tlm_backward_weight ?? 1.0).toFixed(1)}, entropy=${(config.tlm_entropy_coeff ?? 0.01).toFixed(3)}. Solves extreme mode collapse.`
                    )}
                    {config.training_objective === 'FLOW_MATCHING' && (
                      "Direct flow estimation minimizing (Z(s) - F(s))². Trades accuracy for efficiency."
                    )}
                  </p>
                </div>
              </div>
            </motion.div>

          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
