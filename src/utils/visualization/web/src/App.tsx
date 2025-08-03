import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Sparkles, Zap, Network, Grid3X3, Box, Monitor, Home } from 'lucide-react'
import { GFlowNetDistribution3D } from './visualizations/GFlowNetDistribution3D'
import { GFlowNetFlowField } from './visualizations/GFlowNetFlowField'
import { ProblemSetup } from './components/ProblemSetup'
import { MonitoringDashboard } from './components/MonitoringDashboard'
import { useWebSocket } from './hooks/useWebSocket'
import { ErrorBoundary } from './components/ErrorBoundary'

function App() {
  const [activeView, setActiveView] = useState<'setup' | 'monitor' | 'distribution' | 'flow'>('setup')
  const [problemConfig, setProblemConfig] = useState<any>(null)
  const { isConnected } = useWebSocket()
  
  const handleStartTraining = async (config: any) => {
    console.log('Starting training with config:', config)
    
    // Reset training on server
    try {
      await fetch('http://localhost:8080/api/training/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
    } catch (err) {
      console.error('Failed to reset training:', err)
    }
    
    setProblemConfig(config)
    console.log('Setting active view to monitor')
    setActiveView('monitor')
  }

  return (
    <div className="min-h-screen bg-dark-bg text-white overflow-hidden">
      {/* Background gradient */}
      <div className="fixed inset-0 bg-gradient-radial from-neon-purple/10 via-transparent to-transparent pointer-events-none" />
      
      {/* Header */}
      <header className="relative z-10 border-b border-dark-border/50 backdrop-blur-lg">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <motion.div
                initial={{ rotate: 0 }}
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-purple to-neon-blue flex items-center justify-center"
              >
                <Sparkles className="w-6 h-6 text-white" />
              </motion.div>
              <h1 className="text-2xl font-bold gradient-text">GFlowNet Visualization</h1>
            </div>
            
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-neon-green' : 'bg-red-500'} animate-pulse`} />
                <span className="text-sm text-muted-foreground">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              
              <nav className="flex space-x-1 bg-dark-panel/50 p-1 rounded-lg">
                <button
                  onClick={() => setActiveView('setup')}
                  className={`px-4 py-2 rounded-md transition-all duration-200 flex items-center space-x-2 ${
                    activeView === 'setup'
                      ? 'bg-neon-purple/20 text-neon-purple'
                      : 'text-muted-foreground hover:text-white hover:bg-dark-panel'
                  }`}
                >
                  <Home className="w-4 h-4" />
                  <span>Setup</span>
                </button>
                {problemConfig && (
                  <>
                    <button
                      onClick={() => setActiveView('monitor')}
                      className={`px-4 py-2 rounded-md transition-all duration-200 flex items-center space-x-2 ${
                        activeView === 'monitor'
                          ? 'bg-neon-purple/20 text-neon-purple'
                          : 'text-muted-foreground hover:text-white hover:bg-dark-panel'
                      }`}
                    >
                      <Monitor className="w-4 h-4" />
                      <span>Monitor</span>
                    </button>
                    <button
                      onClick={() => setActiveView('distribution')}
                      className={`px-4 py-2 rounded-md transition-all duration-200 flex items-center space-x-2 ${
                        activeView === 'distribution'
                          ? 'bg-neon-purple/20 text-neon-purple'
                          : 'text-muted-foreground hover:text-white hover:bg-dark-panel'
                      }`}
                    >
                      <Box className="w-4 h-4" />
                      <span>3D Distribution</span>
                    </button>
                    <button
                      onClick={() => setActiveView('flow')}
                      className={`px-4 py-2 rounded-md transition-all duration-200 flex items-center space-x-2 ${
                        activeView === 'flow'
                          ? 'bg-neon-purple/20 text-neon-purple'
                          : 'text-muted-foreground hover:text-white hover:bg-dark-panel'
                      }`}
                    >
                      <Zap className="w-4 h-4" />
                      <span>Policy Flow</span>
                    </button>
                  </>
                )}
              </nav>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="relative z-10 flex-1 overflow-hidden h-[calc(100vh-73px)]">
        {activeView === 'setup' && (
          <ProblemSetup onStart={handleStartTraining} />
        )}
        
        {activeView === 'monitor' && problemConfig && (
          <ErrorBoundary>
            {console.log('Rendering MonitoringDashboard with config:', problemConfig)}
            <MonitoringDashboard problemConfig={problemConfig} />
          </ErrorBoundary>
        )}
        
        {activeView === 'distribution' && (
          <div className="h-full p-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="h-full glass-dark rounded-2xl p-6"
            >
              <GFlowNetDistribution3D />
            </motion.div>
          </div>
        )}
        
        {activeView === 'flow' && (
          <div className="h-full p-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="h-full glass-dark rounded-2xl p-6"
            >
              <GFlowNetFlowField />
            </motion.div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App