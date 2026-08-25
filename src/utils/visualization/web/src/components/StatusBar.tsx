import { useQuery } from '@tanstack/react-query'
import { Activity, Wifi, WifiOff, Clock, Zap, TrendingDown, Gauge } from 'lucide-react'
import { api, type TrainingState } from '../services/api'

interface StatusBarProps {
  isConnected: boolean
}

export function StatusBar({ isConnected }: StatusBarProps) {
  const { data: state } = useQuery<TrainingState>({
    queryKey: ['training-state-status'],
    queryFn: api.training.getState,
    refetchInterval: 1000,
    retry: 1,
  })

  const isTraining = state?.is_training && !state?.is_paused
  const progress = state?.total_iterations
    ? Math.round((state.current_iteration / state.total_iterations) * 100)
    : 0
  const eta = state?.total_iterations && state?.current_iteration && isTraining
    ? formatETA(state.total_iterations - state.current_iteration)
    : null

  return (
    <div className="h-7 border-t border-dark-border/50 bg-dark-panel/90 backdrop-blur-sm flex items-center px-4 text-[10px] text-muted-foreground select-none">
      {/* Training Status */}
      <div className="flex items-center space-x-2 mr-4">
        <div className={`w-1.5 h-1.5 rounded-full ${
          isTraining ? 'bg-neon-green animate-pulse' :
          state?.is_paused ? 'bg-neon-orange' :
          'bg-muted-foreground'
        }`} />
        <span className="font-medium">
          {isTraining ? 'Training' : state?.is_paused ? 'Paused' : 'Idle'}
        </span>
      </div>

      {/* Progress */}
      {state?.is_training && (
        <>
          <div className="flex items-center space-x-2 mr-4">
            <Activity className="w-3 h-3" />
            <span>Ep {state.current_iteration}/{state.total_iterations}</span>
            <div className="w-20 h-1.5 rounded-full bg-dark-border overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-neon-purple to-neon-blue transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span>{progress}%</span>
          </div>

          <div className="flex items-center space-x-2 mr-4">
            <TrendingDown className="w-3 h-3" />
            <span>Loss: {state.latest_loss?.toFixed(4) ?? '—'}</span>
          </div>

          <div className="flex items-center space-x-2 mr-4">
            <Zap className="w-3 h-3" />
            <span>Reward: {state.metrics?.mean_reward?.toFixed(2) ?? '—'}</span>
          </div>

          {eta && (
            <div className="flex items-center space-x-2 mr-4">
              <Clock className="w-3 h-3" />
              <span>ETA: {eta}</span>
            </div>
          )}
        </>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Connection */}
      <div className="flex items-center space-x-1.5">
        {isConnected ? (
          <Wifi className="w-3 h-3 text-neon-green" />
        ) : (
          <WifiOff className="w-3 h-3 text-red-500" />
        )}
        <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {/* Time */}
      <div className="ml-4 font-mono">
        {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </div>
    </div>
  )
}

function formatETA(remainingIterations: number): string {
  // Rough estimate: ~50ms per iteration on average
  const seconds = Math.round(remainingIterations * 0.05)
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`
}
