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
    <div className="h-screen flex flex-col bg-dark-bg">
      {/* Header */}
      <div className="border-b border-dark-border/50 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold">Training Monitor</h2>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Activity className="w-4 h-4 text-neon-green animate-pulse" />
            <span>Live Training</span>
          </div>
        </div>
        
        {/* Grid Info */}
        <div className="flex items-center gap-4 text-sm">
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
      
      {/* Main Content */}
      <div className="flex-1 p-4 overflow-auto">
        {expandedView === 'none' ? (
          <div className="h-full grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
            {/* Left: 2D Trajectory Visualization */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="lg:col-span-2 glass-dark rounded-xl p-4 relative group"
            >
              <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => setExpandedView('trajectory')}
                  className="p-2 bg-dark-panel/80 rounded-lg hover:bg-dark-panel"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>
              
              <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
                <Network className="w-4 h-4 text-neon-purple" />
                Trajectory Sampling
              </h3>
              
              <div className="h-[calc(100%-2rem)] min-h-0">
                <GFlowNet2DTrajectory />
              </div>
            </motion.div>
            
            {/* Right: Metrics */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-4"
            >
              <RealtimeMetrics />
            </motion.div>
          </div>
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
      
      {/* Bottom: Training Metrics */}
      {expandedView === 'none' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="border-t border-dark-border/50 p-4"
        >
          <div className="glass-dark rounded-xl p-4 relative group">
            <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={() => setExpandedView('training')}
                className="p-2 bg-dark-panel/80 rounded-lg hover:bg-dark-panel"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </div>
            
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-neon-blue" />
              Training Progress
            </h3>
            
            <div className="h-80 min-h-80">
              <GFlowNetTrainingDashboard />
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}