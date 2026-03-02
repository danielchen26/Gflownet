import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { GFlowNetDistribution3D } from './visualizations/GFlowNetDistribution3D'
import { GFlowNetFlowField } from './visualizations/GFlowNetFlowField'
import { ProblemSetup } from './components/ProblemSetup'
import { MonitoringDashboard } from './components/MonitoringDashboard'
import { Sidebar, type ViewId } from './components/Sidebar'
import { StatusBar } from './components/StatusBar'
import { CommandPalette } from './components/CommandPalette'
import { ThemeProvider, useThemeLayout } from './contexts/ThemeContext'
import { useWebSocket } from './hooks/useWebSocket'
import { ErrorBoundary } from './components/ErrorBoundary'
import { AIAssistant } from './components/AIAssistant'
import { api } from './services/api'

// Pages
import { Home } from './pages/Home'
import { FormulatePage } from './pages/FormulatePage'
import { MolecularViewer3D } from './pages/MolecularViewer3D'
import { ChemicalSpaceExplorer } from './pages/ChemicalSpaceExplorer'
import { ResultsHub } from './pages/ResultsHub'
import { InterpretabilityPanel } from './pages/InterpretabilityPanel'

function AppContent() {
  const [activeView, setActiveView] = useState<ViewId>('home')
  const [problemConfig, setProblemConfig] = useState<any>(null)
  const [pendingDomainId, setPendingDomainId] = useState<string | null>(null)
  const { isConnected } = useWebSocket()
  const layout = useThemeLayout()

  const handleNavigate = useCallback((view: ViewId) => {
    setActiveView(view)
  }, [])

  // Home/Formulate → Configure: user selected a domain, jump to configure for that domain
  const handleDomainSelect = useCallback((domainId: string) => {
    setPendingDomainId(domainId)
    setActiveView('configure')
  }, [])

  const handleStartTraining = useCallback(async (config: any) => {
    console.log('App: Starting training with config:', config)

    // Convert 0-based JavaScript coordinates to 1-based Julia coordinates
    const reward_peaks = config.reward_peaks?.map((peak: any) => ({
      position: [peak.position[0] + 1, peak.position[1] + 1],
      intensity: peak.intensity
    })) ?? []

    // Navigate to train immediately — don't block on API response
    setProblemConfig(config)
    setActiveView('train')

    // Fire API call in background
    try {
      const result = await api.training.start({
        domain_type: config.domain_type ?? 'grid_world',
        grid_size: config.grid_size,
        n_episodes: config.n_episodes ?? 1000,
        batch_size: config.batch_size ?? 32,
        learning_rate: config.learning_rate ?? 0.005,
        objective: config.training_objective ?? config.objective,
        hidden_dim: config.hidden_dim ?? 64,
        temperature: config.temperature ?? 1.0,
        reward_peaks: reward_peaks,
        epsilon: config.epsilon ?? 0.15,
        epsilon_decay: config.epsilon_decay ?? true,
        entropy_weight: config.entropy_weight ?? 0.02,
        z_learning_rate_multiplier: config.z_learning_rate_multiplier ?? 10.0,
        use_replay_buffer: config.use_replay_buffer ?? false,
        replay_buffer_size: config.replay_buffer_size ?? 10000,
        replay_ratio: config.replay_ratio ?? 0.5,
        replay_priority_alpha: config.replay_priority_alpha ?? 0.6,
        tlm_backward_weight: config.tlm_backward_weight ?? 1.0,
        tlm_entropy_coeff: config.tlm_entropy_coeff ?? 0.01,
        reward_shaping: config.reward_shaping ?? true,
        // Domain-specific config
        ...(config.domain_config ? { domain_config: config.domain_config } : {}),
      })
      console.log('App: Training started successfully:', result)
    } catch (err: any) {
      console.error('App: Failed to start training:', err)
      const errorMsg = err.response?.data?.error || err.message || 'Unknown error'
      alert(`Failed to start training: ${errorMsg}\n\nMake sure the backend server is running on port 8080.`)
      setActiveView('configure')
    }
  }, [])

  return (
    <div className="h-screen flex flex-col bg-dark-bg text-white overflow-hidden">
      {/* Background gradient */}
      <div className="fixed inset-0 bg-gradient-radial from-neon-purple/10 via-transparent to-transparent pointer-events-none" />

      {/* Main layout: Sidebar + Content */}
      <div className="flex-1 flex min-h-0 relative z-0">
        {/* Sidebar Navigation */}
        <Sidebar
          activeView={activeView}
          onViewChange={handleNavigate}
          isConnected={isConnected}
          domainType={problemConfig?.domain_type ?? problemConfig?.domain}
        />

        {/* Main Content */}
        <main className={`flex-1 overflow-y-auto scrollbar-thin ${layout.compact ? 'p-3' : layout.spacious ? 'p-8' : 'p-6'}`}>
          <ErrorBoundary key={activeView}>
            <motion.div
              key={activeView}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
            >
              {activeView === 'home' && (
                <Home onNavigate={handleNavigate} onDomainSelect={handleDomainSelect} />
              )}

              {activeView === 'formulate' && (
                <FormulatePage onNavigate={handleNavigate} onDomainSelect={handleDomainSelect} />
              )}

              {activeView === 'configure' && (
                <ProblemSetup
                  onStart={handleStartTraining}
                  initialDomainId={pendingDomainId}
                  onDomainConsumed={() => setPendingDomainId(null)}
                />
              )}

              {activeView === 'train' && (
                <MonitoringDashboard problemConfig={problemConfig} onRestart={handleStartTraining} />
              )}

              {activeView === 'candidates' && (
                <ResultsHub onNavigate={handleNavigate} problemConfig={problemConfig} />
              )}

              {activeView === 'structure' && (
                <MolecularViewer3D problemConfig={problemConfig} />
              )}

              {activeView === 'landscape' && (
                problemConfig?.domain_type === 'grid_world' ? (
                  <div className="h-[800px]">
                    <GFlowNetDistribution3D />
                  </div>
                ) : (
                  <ChemicalSpaceExplorer onNavigate={handleNavigate} problemConfig={problemConfig} />
                )
              )}

              {activeView === 'interpret' && (
                <InterpretabilityPanel problemConfig={problemConfig} />
              )}

              {activeView === 'flow' && (
                <div className="h-[800px]">
                  <GFlowNetFlowField />
                </div>
              )}
            </motion.div>
          </ErrorBoundary>
        </main>
      </div>

      {/* Status Bar */}
      <StatusBar isConnected={isConnected} />

      {/* Command Palette (Cmd+K) */}
      <CommandPalette onNavigate={handleNavigate} />

      {/* AI Assistant — contextual help overlay */}
      <AIAssistant activeView={activeView} problemConfig={problemConfig} />
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
