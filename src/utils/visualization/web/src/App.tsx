import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Sparkles, Zap, Network, Grid3X3, Box, Monitor, Home } from 'lucide-react'
import { GFlowNetDistribution3D } from './visualizations/GFlowNetDistribution3D'
import { GFlowNetFlowField } from './visualizations/GFlowNetFlowField'
import { ProblemSetup } from './components/ProblemSetup'
import { MonitoringDashboard } from './components/MonitoringDashboard'
import { ThemeSelector } from './components/ThemeSelector'
import { ThemeProvider, useThemeLayout } from './contexts/ThemeContext'
import { useWebSocket } from './hooks/useWebSocket'
import { ErrorBoundary } from './components/ErrorBoundary'
import { api } from './services/api'

function AppContent() {
  const [activeView, setActiveView] = useState<'setup' | 'monitor' | 'distribution' | 'flow'>('setup')
  const [problemConfig, setProblemConfig] = useState<any>(null)
  const { isConnected } = useWebSocket()
  const layout = useThemeLayout()

  const handleStartTraining = async (config: any) => {
    console.log('App: Starting training with config:', config)

    // Convert 0-based JavaScript coordinates to 1-based Julia coordinates
    const reward_peaks = config.reward_peaks.map((peak: any) => ({
      position: [peak.position[0] + 1, peak.position[1] + 1],
      intensity: peak.intensity
    }))

    // Navigate to monitor immediately — don't block on API response
    // Julia JIT compilation can take 10-30s on first call; the Monitor page
    // handles the "waiting for data" state via its polling loop.
    setProblemConfig(config)
    setActiveView('monitor')

    // Fire API call in background
    try {
      const result = await api.training.start({
        domain_type: 'grid_world',
        grid_size: config.grid_size,
        n_episodes: config.n_episodes ?? 1000,
        batch_size: config.batch_size ?? 32,
        learning_rate: config.learning_rate ?? 0.005,
        objective: config.training_objective,
        hidden_dim: config.hidden_dim ?? 64,
        temperature: config.temperature ?? 1.0,
        reward_peaks: reward_peaks,
        // Exploration parameters for mode discovery (Phase 7: Mode Collapse Fix)
        epsilon: config.epsilon ?? 0.15,
        epsilon_decay: config.epsilon_decay ?? true,
        entropy_weight: config.entropy_weight ?? 0.02,
        z_learning_rate_multiplier: config.z_learning_rate_multiplier ?? 10.0,
        // Experience Replay Buffer (JMLR 2023)
        use_replay_buffer: config.use_replay_buffer ?? false,
        replay_buffer_size: config.replay_buffer_size ?? 10000,
        replay_ratio: config.replay_ratio ?? 0.5,
        replay_priority_alpha: config.replay_priority_alpha ?? 0.6,
        // TLM parameters (ICLR 2025)
        tlm_backward_weight: config.tlm_backward_weight ?? 1.0,
        tlm_entropy_coeff: config.tlm_entropy_coeff ?? 0.01,
        // Reward Shaping (auto-compensate path asymmetry)
        reward_shaping: config.reward_shaping ?? true,
      })
      console.log('App: Training started successfully:', result)
    } catch (err: any) {
      console.error('App: Failed to start training:', err)
      const errorMsg = err.response?.data?.error || err.message || 'Unknown error'
      alert(`Failed to start training: ${errorMsg}\n\nMake sure the backend server is running on port 8080.`)
      setActiveView('setup')
    }
  }

  return (
    <div className="min-h-screen bg-dark-bg text-white overflow-hidden">
      {/* Background gradient */}
      <div className="fixed inset-0 bg-gradient-radial from-neon-purple/10 via-transparent to-transparent pointer-events-none" />

      {/* Header — z-20 so the theme dropdown (child) renders above the nav (z-10) */}
      <header className="relative z-20 border-b border-dark-border/50 backdrop-blur-sm">
        <div className={`container mx-auto ${layout.compact ? 'px-3 py-1.5' : layout.spacious ? 'px-8 py-5' : 'px-6 py-4'}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`${layout.compact ? 'w-7 h-7' : layout.spacious ? 'w-12 h-12' : 'w-10 h-10'} rounded-lg bg-gradient-to-br from-neon-purple to-neon-blue flex items-center justify-center`}>
                <Sparkles className={`${layout.compact ? 'w-4 h-4' : layout.spacious ? 'w-7 h-7' : 'w-6 h-6'}`} />
              </div>
              <div>
                <h1 className={`${layout.titleSize} font-bold gradient-text`}>GFlowNet Visualization</h1>
                {!layout.compact && (
                  <p className={`${layout.tinySize} text-muted-foreground`}>Real-time Training Monitor</p>
                )}
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <ThemeSelector />
              <div className={`flex items-center space-x-2 ${layout.compact ? 'px-2 py-0.5' : 'px-3 py-1'} rounded-full ${
                isConnected ? 'bg-neon-green/20 text-neon-green' : 'bg-red-500/20 text-red-500'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-neon-green animate-pulse' : 'bg-red-500'}`} />
                <span className={`${layout.tinySize} font-medium`}>{isConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="relative z-10 border-b border-dark-border/30 backdrop-blur-sm">
        <div className={`container mx-auto ${layout.compact ? 'px-3' : layout.spacious ? 'px-8' : 'px-6'}`}>
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
                    relative ${layout.compact ? 'px-3 py-1.5 text-xs' : layout.spacious ? 'px-8 py-4 text-sm' : 'px-6 py-3'} font-medium transition-all
                    ${isActive
                      ? 'text-white'
                      : 'text-muted-foreground hover:text-white'
                    }
                  `}
                >
                  <div className="flex items-center space-x-2">
                    <Icon className={`${layout.compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
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
      <main className={`relative z-0 container mx-auto ${layout.compact ? 'px-3 py-3' : layout.spacious ? 'px-8 py-10' : 'px-6 py-8'}`}>
        <ErrorBoundary key={activeView}>
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
              <MonitoringDashboard problemConfig={problemConfig} onRestart={handleStartTraining} />
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

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  )
}

export default App
