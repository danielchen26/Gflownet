import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Network, Maximize2, BarChart3 } from 'lucide-react'
import { GFlowNet2DTrajectory } from '../visualizations/GFlowNet2DTrajectory'
import { GFlowNetTrainingDashboard } from './GFlowNetTrainingDashboard'
import { RealtimeMetrics } from './RealtimeMetrics'

interface MonitoringDashboardProps {
  problemConfig: any
}

export function MonitoringDashboard({ problemConfig }: MonitoringDashboardProps) {
  const [expandedView, setExpandedView] = useState<'none' | 'trajectory' | 'training'>('none')
  
  return (
    <div className="h-full flex flex-col bg-dark-bg">
      {/* Header */}
      <div className="border-b border-dark-border/50 px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold">Training Monitor</h2>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Activity className="w-4 h-4 text-neon-green animate-pulse" />
            <span>Live Training</span>
          </div>
        </div>
        
        {/* Grid Info */}
        <div className="flex items-center gap-4 text-xs">
          <span className="text-muted-foreground">
            Grid: {problemConfig.grid_size}×{problemConfig.grid_size}
          </span>
          <span className="text-muted-foreground">
            Objective: <span className="text-neon-purple">{problemConfig.training_objective}</span>
          </span>
          <span className="text-muted-foreground">
            Episodes: <span className="text-neon-blue">{problemConfig.n_episodes}</span>
          </span>
        </div>
      </div>
      
      {/* Main Content - Two Rows Layout */}
      <div className="flex-1 flex flex-col p-2 gap-2 overflow-hidden">
        {expandedView === 'none' ? (
          <>
            {/* Row 1: Trajectory Sampling and Metrics */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-2 min-h-0">
              {/* Left: 2D Trajectory Visualization */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="lg:col-span-2 glass-dark rounded-lg p-3 relative group"
              >
                <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => setExpandedView('trajectory')}
                    className="p-1 bg-dark-panel/80 rounded hover:bg-dark-panel"
                  >
                    <Maximize2 className="w-3 h-3" />
                  </button>
                </div>
                
                <h3 className="text-xs font-medium mb-2 flex items-center gap-2">
                  <Network className="w-3 h-3 text-neon-purple" />
                  Trajectory Sampling
                </h3>
                
                <div className="h-[calc(100%-1.5rem)] min-h-0">
                  <GFlowNet2DTrajectory />
                </div>
              </motion.div>
              
              {/* Right: Metrics */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="h-full overflow-auto"
              >
                <RealtimeMetrics />
              </motion.div>
            </div>
            
            {/* Row 2: Training Progress */}
            <div className="flex-1 min-h-0">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-dark rounded-lg p-3 relative group h-full"
              >
                <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => setExpandedView('training')}
                    className="p-1 bg-dark-panel/80 rounded hover:bg-dark-panel"
                  >
                    <Maximize2 className="w-3 h-3" />
                  </button>
                </div>
                
                <h3 className="text-xs font-medium mb-2 flex items-center gap-2">
                  <BarChart3 className="w-3 h-3 text-neon-blue" />
                  Training Progress
                </h3>
                
                <div className="h-[calc(100%-1.5rem)] min-h-0">
                  <GFlowNetTrainingDashboard />
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
              <GFlowNetTrainingDashboard />
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}