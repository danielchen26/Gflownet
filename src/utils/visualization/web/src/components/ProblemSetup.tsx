import { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings, Grid3x3, Target, Zap, Play, RefreshCw } from 'lucide-react'
import axios from '../lib/axios'

interface ProblemConfig {
  grid_size: number
  reward_peaks: Array<{
    position: [number, number]
    intensity: number
  }>
  training_objective: 'TB' | 'SubTB' | 'DB' | 'FM'
  n_episodes: number
}

interface ProblemSetupProps {
  onStart: (config: ProblemConfig) => void
}

export function ProblemSetup({ onStart }: ProblemSetupProps) {
  const [config, setConfig] = useState<ProblemConfig>({
    grid_size: 10,
    reward_peaks: [
      { position: [8, 8], intensity: 10 },
      { position: [2, 8], intensity: 8 },
      { position: [5, 5], intensity: 6 },
    ],
    training_objective: 'TB',
    n_episodes: 1000,
  })
  
  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  
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
  
  const startTraining = async () => {
    setIsRunning(true)
    try {
      // Server expects 1-based grid coordinates; convert before sending
      const payload = {
        ...config,
        reward_peaks: config.reward_peaks.map((peak) => ({
          ...peak,
          position: [peak.position[0] + 1, peak.position[1] + 1] as [number, number],
        })),
      }
      await axios.post('/api/training/start', payload)
      onStart(config)
    } catch (error) {
      console.error('Failed to start training:', error)
      setIsRunning(false)
    }
  }
  
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h1 className="text-4xl font-bold mb-4 gradient-text">
          GFlowNet Interactive Setup
        </h1>
        <p className="text-lg text-muted-foreground">
          Configure your problem domain and start training
        </p>
      </motion.div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Grid Configuration */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass-dark rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Grid3x3 className="w-5 h-5 text-neon-purple" />
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
                        ? `bg-gradient-to-br from-neon-green to-neon-green/50` 
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
            
            {selectedCell && (
              <div className="mt-4 flex items-center gap-2">
                <span className="text-sm text-muted-foreground">
                  Selected: ({selectedCell[0]}, {selectedCell[1]})
                </span>
                <button
                  onClick={addRewardPeak}
                  className="px-3 py-1 text-sm bg-neon-green/20 text-neon-green rounded-md hover:bg-neon-green/30"
                >
                  Add Reward Peak
                </button>
              </div>
            )}
          </div>
          
          {/* Grid Size */}
          <div className="space-y-2 mb-4">
            <label className="text-sm font-medium">Grid Size</label>
            <input
              type="range"
              min={5}
              max={20}
              value={config.grid_size}
              onChange={(e) => setConfig({ ...config, grid_size: Number(e.target.value) })}
              className="w-full"
            />
            <span className="text-xs text-muted-foreground">{config.grid_size} × {config.grid_size}</span>
          </div>
        </motion.div>
        
        {/* Training Configuration */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass-dark rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-neon-blue" />
            Training Configuration
          </h2>
          
          {/* Reward Peaks List */}
          <div className="mb-4">
            <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Target className="w-4 h-4 text-neon-green" />
              Reward Peaks
            </h3>
            <div className="space-y-2">
              {config.reward_peaks.map((peak, i) => (
                <div key={i} className="flex items-center gap-2 p-2 bg-dark-panel rounded-md">
                  <span className="text-sm font-mono">
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
                  <span className="text-sm w-12 text-right">{peak.intensity}</span>
                  <button
                    onClick={() => removeRewardPeak(i)}
                    className="p-1 text-red-500 hover:bg-red-500/20 rounded"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
          
          {/* Training Objective */}
          <div className="mb-4">
            <label className="text-sm font-medium block mb-2">Training Objective</label>
            <select
              value={config.training_objective}
              onChange={(e) => setConfig({ ...config, training_objective: e.target.value as any })}
              className="w-full px-3 py-2 bg-dark-panel border border-dark-border rounded-md"
            >
              <option value="TB">Trajectory Balance (TB)</option>
              <option value="SubTB">Sub-Trajectory Balance</option>
              <option value="DB">Detailed Balance</option>
              <option value="FM">Flow Matching</option>
            </select>
          </div>
          
          {/* Episodes */}
          <div className="mb-4">
            <label className="text-sm font-medium block mb-2">Number of Episodes</label>
            <input
              type="number"
              value={config.n_episodes}
              onChange={(e) => setConfig({ ...config, n_episodes: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-dark-panel border border-dark-border rounded-md"
            />
          </div>
          
          
          {/* Start Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={startTraining}
            disabled={isRunning || config.reward_peaks.length === 0}
            className={`
              w-full py-3 rounded-lg font-medium flex items-center justify-center gap-2
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
            <p className="text-xs text-red-500 mt-2">
              Please add at least one reward peak
            </p>
          )}
        </motion.div>
      </div>
      
      {/* Quick Presets */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-dark rounded-xl p-4"
      >
        <h3 className="text-sm font-medium mb-3">Quick Presets</h3>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setConfig({
              ...config,
              reward_peaks: [
                { position: [8, 8], intensity: 10 },
                { position: [2, 8], intensity: 8 },
                { position: [5, 5], intensity: 6 },
                { position: [8, 2], intensity: 7 },
              ]
            })}
            className="px-3 py-1 text-sm bg-dark-panel hover:bg-dark-border rounded-md"
          >
            Multi-Modal
          </button>
          <button
            onClick={() => setConfig({
              ...config,
              reward_peaks: [
                { position: [9, 9], intensity: 10 },
              ]
            })}
            className="px-3 py-1 text-sm bg-dark-panel hover:bg-dark-border rounded-md"
          >
            Single Goal
          </button>
          <button
            onClick={() => setConfig({
              ...config,
              reward_peaks: [
                { position: [2, 2], intensity: 5 },
                { position: [7, 7], intensity: 5 },
                { position: [2, 7], intensity: 5 },
                { position: [7, 2], intensity: 5 },
              ]
            })}
            className="px-3 py-1 text-sm bg-dark-panel hover:bg-dark-border rounded-md"
          >
            Four Corners
          </button>
        </div>
      </motion.div>
    </div>
  )
}
