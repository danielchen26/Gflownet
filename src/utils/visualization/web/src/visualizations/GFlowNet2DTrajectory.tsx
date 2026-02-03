import { useState, useMemo, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Play, Pause, SkipBack, SkipForward, Info, Target, Flag } from 'lucide-react'
import { api } from '../services/api'
import { COLORS, interpolateRewardColor } from '../utils/colors'

interface Trajectory {
  id: string
  states: Array<[number, number]>
  actions: string[]
  rewards: number[]
  total_reward: number
  length: number
}

interface TrajectoryData {
  trajectories: Trajectory[]
  grid_size: [number, number]
  reward_peaks: Array<{
    position: [number, number]
    intensity: number
    name: string
  }>
}

export function GFlowNet2DTrajectory() {
  const [selectedTrajectory, setSelectedTrajectory] = useState<number>(0)
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState(false)
  
  // Check if training is active
  const { data: metrics } = useQuery({
    queryKey: ['training-state'],
    queryFn: async () => {
      return await api.training.getState()
    },
    refetchInterval: 1000,
  })

  const isTraining = metrics?.is_training || false

  const { data, error, isError, dataUpdatedAt } = useQuery({
    queryKey: ['trajectories'],
    queryFn: async () => {
      try {
        console.log('🔄 Fetching trajectories from /api/v2/trajectories...')
        const result = await api.trajectories.getRecent(20)
        console.log('✅ GFlowNet2DTrajectory API SUCCESS:', {
          hasData: !!result,
          hasDomain: !!result?.domain,
          domainType: typeof result?.domain,
          hasTrajectories: !!result?.trajectories,
          trajCount: result?.trajectories?.length || 0,
          domainKeys: result?.domain ? Object.keys(result.domain) : 'NO DOMAIN',
          gridSize: result?.domain?.grid_size,
          rewardPeaksCount: result?.domain?.reward_peaks?.length
        })
        return result
      } catch (err) {
        console.error('❌ GFlowNet2DTrajectory API ERROR:', err)
        throw err
      }
    },
    refetchInterval: isTraining ? 500 : 3000, // Poll every 3s when not training
    retry: 3,
    gcTime: 0, // React Query v5: use gcTime instead of cacheTime
    staleTime: 0,
    refetchOnMount: true,
  })
  
  const [sortMode, setSortMode] = useState<'reward' | 'recent' | 'length'>('reward')
  
  const trajectories = useMemo(() => {
    if (!data || !data.trajectories || !Array.isArray(data.trajectories)) return []
    const arr = [...data.trajectories]
    switch (sortMode) {
      case 'length':
        return arr.sort((a, b) => b.length - a.length)
      case 'recent':
        return arr // preserve arrival order (newest last)
      case 'reward':
      default:
        return arr.sort((a, b) => b.total_reward - a.total_reward)
    }
  }, [data, sortMode])
  
  // Reset selection when sort changes or list size changes
  useEffect(() => {
    setSelectedTrajectory(0)
    setCurrentStep(0)
  }, [sortMode, trajectories.length])
  
  // Auto-play animation when training or when user hits play
  useEffect(() => {
    if (!trajectories.length) return
    const trajectory = trajectories[selectedTrajectory]
    if (!trajectory) return
    
    // Auto-start during live training
    if (isTraining && !isPlaying) {
      setIsPlaying(true)
    }
    
    if (isPlaying) {
      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= trajectory.length - 1) {
            // Loop for both training and manual playback
            return 0
          }
          return prev + 1
        })
      }, 800) // Slower animation for better visibility
      return () => clearInterval(interval)
    }
  }, [isPlaying, isTraining, trajectories, selectedTrajectory])
  
  // Auto-advance to new trajectories when training
  useEffect(() => {
    if (isTraining && trajectories.length > 0) {
      const interval = setInterval(() => {
        setSelectedTrajectory(prev => {
          const next = (prev + 1) % Math.min(trajectories.length, 5) // Cycle through first 5
          setCurrentStep(0)
          return next
        })
      }, 3000) // Switch trajectories every 3 seconds during training
      return () => clearInterval(interval)
    }
  }, [isTraining, trajectories.length])
  
  // Ensure current step stays in range when trajectories update
  useEffect(() => {
    if (!trajectories.length) return
    const trajectory = trajectories[selectedTrajectory]
    if (!trajectory) return
    if (currentStep >= trajectory.length) {
      setCurrentStep(Math.max(trajectory.length - 1, 0))
    }
  }, [trajectories, selectedTrajectory, currentStep])
  
  // Debug logging
  console.log('🔍 GFlowNet2DTrajectory render check:', {
    hasData: !!data,
    hasDomain: !!data?.domain,
    trajectoriesLength: trajectories.length,
    condition: !data || !data.domain || trajectories.length === 0,
    isError,
    errorMessage: error?.toString()
  })

  if (isError) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-500">Error loading trajectory data</p>
          <p className="text-xs text-muted-foreground mt-2">{error?.toString()}</p>
          <p className="text-xs text-muted-foreground mt-2">Check console for details</p>
        </div>
      </div>
    )
  }

  if (!data || !data.domain || trajectories.length === 0) {
    // Return a simple placeholder for now
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <p className="text-muted-foreground">Waiting for trajectory data...</p>
          <p className="text-sm text-muted-foreground mt-2">Start training to see visualizations</p>
          <p className="text-xs text-yellow-500 mt-4">
            Debug: data={String(!!data)} | domain={String(!!data?.domain)} | trajectories={trajectories.length}
          </p>
          <p className="text-xs text-muted-foreground">Check browser console (F12) for API response details</p>
        </div>
      </div>
    )
  }

  const trajectory = trajectories[selectedTrajectory]
  const cellSize = 40
  const gridWidth = data.domain.grid_size[0] * cellSize
  const gridHeight = data.domain.grid_size[1] * cellSize
  const toCanvasCoord = (coord: number) => (coord - 1) * cellSize + cellSize / 2 + 1
  
  return (
    <div className="h-full flex gap-4">
      {/* Main Grid Visualization */}
      <div className="flex-1 flex flex-col">
        <div className="glass-dark rounded-lg p-4 mb-4">
          <h3 className="text-sm font-medium mb-2">Grid World Navigation</h3>
          <div className="relative" style={{ width: gridWidth + 2, height: gridHeight + 2 }}>
            {/* Grid Background */}
            <svg width={gridWidth + 2} height={gridHeight + 2} className="absolute top-0 left-0">
              {/* Grid lines */}
              {Array.from({ length: data.domain.grid_size[0] + 1 }).map((_, i) => (
                <line
                  key={`v${i}`}
                  x1={i * cellSize + 1}
                  y1={1}
                  x2={i * cellSize + 1}
                  y2={gridHeight + 1}
                  stroke="#2A2A2D"
                  strokeWidth="1"
                />
              ))}
              {Array.from({ length: data.domain.grid_size[1] + 1 }).map((_, i) => (
                <line
                  key={`h${i}`}
                  x1={1}
                  y1={i * cellSize + 1}
                  x2={gridWidth + 1}
                  y2={i * cellSize + 1}
                  stroke="#2A2A2D"
                  strokeWidth="1"
                />
              ))}
              
              {/* Reward peaks as gradients */}
              <defs>
                {data.domain.reward_peaks.map((peak: any, i: number) => (
                  <radialGradient key={i} id={`peak${i}`}>
                    <stop offset="0%" stopColor={COLORS.reward.high} stopOpacity="0.6" />
                    <stop offset="100%" stopColor={COLORS.reward.high} stopOpacity="0" />
                  </radialGradient>
                ))}
              </defs>

              {/* Render reward peaks */}
              {data.domain.reward_peaks.map((peak: any, i: number) => (
                <g key={i}>
                  <circle
                    cx={toCanvasCoord(peak.position[0])}
                    cy={toCanvasCoord(peak.position[1])}
                    r={cellSize * 2}
                    fill={`url(#peak${i})`}
                  />
                  <text
                    x={toCanvasCoord(peak.position[0])}
                    y={toCanvasCoord(peak.position[1])}
                    textAnchor="middle"
                    className="fill-white text-xs font-medium"
                  >
                    R={peak.intensity}
                  </text>
                </g>
              ))}
              
              {/* Trajectory path up to current step */}
              {trajectory.states.slice(0, currentStep + 1).map((state: any, i: number) => {
                if (i === 0) return null
                const prevState = trajectory.states[i - 1]
                return (
                  <line
                    key={i}
                    x1={toCanvasCoord(prevState[0])}
                    y1={toCanvasCoord(prevState[1])}
                    x2={toCanvasCoord(state[0])}
                    y2={toCanvasCoord(state[1])}
                    stroke={COLORS.primary.purple}
                    strokeWidth="3"
                    strokeDasharray={i === currentStep ? "5,5" : ""}
                    opacity={0.8 - (currentStep - i) * 0.1}
                  />
                )
              })}
              
              {/* Start marker */}
              <g>
                <circle
                  cx={toCanvasCoord(trajectory.states[0][0])}
                  cy={toCanvasCoord(trajectory.states[0][1])}
                  r="12"
                  fill={COLORS.primary.green}
                  stroke={COLORS.primary.green}
                  strokeWidth="2"
                  fillOpacity="0.3"
                />
                <Flag 
                  className="w-4 h-4"
                  x={toCanvasCoord(trajectory.states[0][0]) - 8}
                  y={toCanvasCoord(trajectory.states[0][1]) - 8}
                  fill={COLORS.primary.green}
                />
              </g>
              
              {/* Current position */}
              {currentStep < trajectory.states.length && (
                <motion.g
                  initial={{ 
                    x: toCanvasCoord(trajectory.states[0][0]),
                    y: toCanvasCoord(trajectory.states[0][1])
                  }}
                  animate={{ 
                    x: toCanvasCoord(trajectory.states[currentStep][0]),
                    y: toCanvasCoord(trajectory.states[currentStep][1])
                  }}
                  transition={{ duration: 0.6, ease: "easeInOut" }}
                >
                  <motion.circle
                    r="8"
                    fill={COLORS.primary.purple}
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.5, repeat: Infinity }}
                  />
                  {/* Trail effect */}
                  <motion.circle
                    r="12"
                    fill={COLORS.primary.purple}
                    fillOpacity="0.3"
                    animate={{ scale: [0.8, 1.5, 0.8] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                </motion.g>
              )}
              
              {/* Goal marker if at terminal state */}
              {currentStep === trajectory.states.length - 1 && (
                <g>
                  <circle
                    cx={toCanvasCoord(trajectory.states[currentStep][0])}
                    cy={toCanvasCoord(trajectory.states[currentStep][1])}
                    r="12"
                    fill={COLORS.primary.pink}
                    stroke={COLORS.primary.pink}
                    strokeWidth="2"
                    fillOpacity="0.3"
                  />
                  <Target 
                    className="w-4 h-4"
                    x={toCanvasCoord(trajectory.states[currentStep][0]) - 8}
                    y={toCanvasCoord(trajectory.states[currentStep][1]) - 8}
                    fill={COLORS.primary.pink}
                  />
                </g>
              )}
            </svg>
          </div>
        </div>
        
        {/* Playback Controls */}
        <div className="glass-dark rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">
              Step {currentStep + 1} / {trajectory.length}
            </span>
            <span className="text-sm font-medium text-neon-green">
              Reward: {trajectory.rewards[currentStep]?.toFixed(2) || '0.00'}
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentStep(0)}
              className="p-2 glass-dark rounded-lg hover:bg-dark-panel transition-colors"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 glass-dark rounded-lg hover:bg-dark-panel transition-colors"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            
            <button
              onClick={() => setCurrentStep(Math.min(currentStep + 1, trajectory.length - 1))}
              className="p-2 glass-dark rounded-lg hover:bg-dark-panel transition-colors"
            >
              <SkipForward className="w-4 h-4" />
            </button>
            
            <input
              type="range"
              min={0}
              max={trajectory.length - 1}
              value={currentStep}
              onChange={(e) => setCurrentStep(Number(e.target.value))}
              className="flex-1 ml-4"
            />
          </div>
          
          {currentStep < trajectory.actions.length && (
            <div className="mt-2 text-sm">
              <span className="text-muted-foreground">Action: </span>
              <span className="font-mono text-neon-purple">{trajectory.actions[currentStep]}</span>
            </div>
          )}
        </div>
      </div>
      
      {/* Trajectory List */}
        <div className="w-80 glass-dark rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-4 h-4 text-neon-purple" />
            <h3 className="text-sm font-medium">Sampled Trajectories</h3>
            {isTraining && (
              <motion.span 
                className="ml-auto text-xs text-neon-green"
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                Live Training...
              </motion.span>
            )}
          </div>
          
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground">Sort:</span>
            <div className="flex gap-1">
              <button
                onClick={() => setSortMode('reward')}
                className={`px-2 py-1 text-xs rounded ${sortMode === 'reward' ? 'bg-neon-purple/30 text-white' : 'glass-dark hover:bg-dark-panel'}`}
              >
                Reward
              </button>
              <button
                onClick={() => setSortMode('recent')}
                className={`px-2 py-1 text-xs rounded ${sortMode === 'recent' ? 'bg-neon-purple/30 text-white' : 'glass-dark hover:bg-dark-panel'}`}
              >
                Recent
              </button>
              <button
                onClick={() => setSortMode('length')}
                className={`px-2 py-1 text-xs rounded ${sortMode === 'length' ? 'bg-neon-purple/30 text-white' : 'glass-dark hover:bg-dark-panel'}`}
              >
                Length
              </button>
            </div>
          </div>
        
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {trajectories.slice(0, 10).map((traj, idx) => (
            <motion.div
              key={traj.id}
              whileHover={{ scale: 1.02 }}
              onClick={() => {
                setSelectedTrajectory(idx)
                setCurrentStep(0)
                setIsPlaying(true) // keep animation visible after selection
              }}
              className={`p-3 rounded-lg cursor-pointer transition-colors ${
                selectedTrajectory === idx
                  ? 'bg-neon-purple/20 border border-neon-purple/50'
                  : 'glass-dark hover:bg-dark-panel'
              }`}
              animate={selectedTrajectory === idx ? { 
                borderColor: '#BD00FF',
                boxShadow: '0 0 20px rgba(189, 0, 255, 0.3)'
              } : {}}
              transition={{ duration: 0.3 }}
            >
              <div className="flex justify-between items-start mb-1">
                <span className="text-sm font-medium">Trajectory {idx + 1}</span>
                <span className="text-xs text-neon-green">
                  R: {traj.total_reward.toFixed(2)}
                </span>
              </div>
              <div className="text-xs text-muted-foreground">
                Length: {traj.length} steps
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Start: ({traj.states[0][0]}, {traj.states[0][1]}) → 
                End: ({traj.states[traj.states.length - 1][0]}, {traj.states[traj.states.length - 1][1]})
              </div>
            </motion.div>
          ))}
        </div>
        
        {/* Legend */}
        <div className="mt-4 pt-4 border-t border-dark-border space-y-2">
          <h4 className="text-xs font-medium mb-2">Legend</h4>
          <div className="flex items-center gap-2 text-xs">
            <Flag className="w-3 h-3 text-neon-green" />
            <span>Start Position</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <Target className="w-3 h-3 text-neon-pink" />
            <span>Goal/Terminal State</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-neon-purple"></div>
            <span>Current Position</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-12 h-0.5 bg-gradient-to-r from-green-500 to-transparent"></div>
            <span>Reward Field</span>
          </div>
        </div>
      </div>
    </div>
  )
}
