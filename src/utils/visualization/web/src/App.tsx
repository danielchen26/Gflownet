import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Sparkles, Zap, Network, Grid3X3, Box, Monitor, Home } from 'lucide-react'
import { GFlowNetDistribution3D } from './visualizations/GFlowNetDistribution3D'
import { GFlowNetFlowField } from './visualizations/GFlowNetFlowField'
import { ProblemSetup } from './components/ProblemSetup'
import { MonitoringDashboard } from './components/MonitoringDashboard'
import { useWebSocket } from './hooks/useWebSocket'
import { ErrorBoundary } from './components/ErrorBoundary'
import { api } from './services/api'

function App() {
  const [activeView, setActiveView] = useState<'setup' | 'monitor' | 'distribution' | 'flow'>('setup')
  const [problemConfig, setProblemConfig] = useState<any>(null)
  const { isConnected } = useWebSocket()

  const handleStartTraining = async (config: any) => {
    console.log('App: Starting training with config:', config)

    // Convert 0-based JavaScript coordinates to 1-based Julia coordinates
    const reward_peaks = config.reward_peaks.map((peak: any) => ({
      position: [peak.position[0] + 1, peak.position[1] + 1],
      intensity: peak.intensity
    }))

    // Start real training via v2 API
    try {
      const result = await api.training.start({
        domain_type: 'grid_world',
        grid_size: config.grid_size,
        n_episodes: config.n_episodes,
        batch_size: config.batch_size || 8,
        learning_rate: config.learning_rate || 0.01,
        objective: config.training_objective,
        hidden_dim: config.hidden_dim || 64,
        reward_peaks: reward_peaks,
        // Exploration parameters for mode discovery (Phase 7: Mode Collapse Fix)
        epsilon: config.epsilon ?? 0.05,           // ε-uniform exploration
        epsilon_decay: config.epsilon_decay ?? true, // Anneal to 0
        entropy_weight: config.entropy_weight ?? 0.01, // Policy entropy
        z_learning_rate_multiplier: config.z_learning_rate_multiplier ?? 10.0, // Faster Z
      })
      console.log('App: Training started successfully:', result)
      setProblemConfig(config)
      setActiveView('monitor')
    } catch (err: any) {
      console.error('App: Failed to start training:', err)
      const errorMsg = err.response?.data?.error || err.message || 'Unknown error'
      alert(`Failed to start training: ${errorMsg}\n\nMake sure the backend server is running on port 8080.`)
    }
  }

  return (
    <div className="min-h-screen bg-dark-bg text-white overflow-hidden">
      {/* Background gradient */}
      <div className="fixed inset-0 bg-gradient-radial from-neon-purple/10 via-transparent to-transparent pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 border-b border-dark-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-purple to-neon-blue flex items-center justify-center">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">GFlowNet Visualization</h1>
                <p className="text-xs text-muted-foreground">Real-time Training Monitor</p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
                isConnected ? 'bg-neon-green/20 text-neon-green' : 'bg-red-500/20 text-red-500'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-neon-green animate-pulse' : 'bg-red-500'}`} />
                <span className="text-xs font-medium">{isConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="relative z-10 border-b border-dark-border/30 backdrop-blur-sm">
        <div className="container mx-auto px-6">
          <div className="flex space-x-1">
            {[
              { id: 'setup', label: 'Setup', icon: Grid3X3 },
              { id: 'monitor', label: 'Monitor', icon: Monitor },
              { id: 'distribution', label: '3D Distribution', icon: Box },
              { id: 'flow', label: 'Flow Field', icon: Network },
            ].map((tab) => {
              const Icon = tab.icon
              const isActive = activeView === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveView(tab.id as any)}
                  className={`
                    relative px-6 py-3 font-medium transition-all
                    ${isActive
                      ? 'text-white'
                      : 'text-muted-foreground hover:text-white'
                    }
                  `}
                >
                  <div className="flex items-center space-x-2">
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </div>
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-neon-purple to-neon-blue"
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    />
                  )}
                </button>
              )
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative z-0 container mx-auto px-6 py-8">
        <ErrorBoundary>
          {activeView === 'setup' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <ProblemSetup onStart={handleStartTraining} />
            </motion.div>
          )}

          {activeView === 'monitor' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <MonitoringDashboard problemConfig={problemConfig} />
            </motion.div>
          )}

          {activeView === 'distribution' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="h-[800px]"
            >
              <GFlowNetDistribution3D />
            </motion.div>
          )}

          {activeView === 'flow' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="h-[800px]"
            >
              <GFlowNetFlowField />
            </motion.div>
          )}
        </ErrorBoundary>
      </main>
    </div>
  )
}

export default App
