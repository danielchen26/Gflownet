import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, Square, Loader2 } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../services/api'

type TrainingStatus = 'idle' | 'running' | 'paused' | 'stopped'

interface TrainingControlsProps {
  isTraining: boolean
  isPaused: boolean
  onStatusChange?: (status: TrainingStatus) => void
}

export function TrainingControls({
  isTraining,
  isPaused,
  onStatusChange
}: TrainingControlsProps) {
  const queryClient = useQueryClient()
  const [isLoading, setIsLoading] = useState<'pause' | 'stop' | null>(null)

  const pauseMutation = useMutation({
    mutationFn: () => api.training.pause(),
    onMutate: () => setIsLoading('pause'),
    onSettled: () => {
      setIsLoading(null)
      queryClient.invalidateQueries({ queryKey: ['training-state'] })
    },
    onSuccess: (data) => {
      onStatusChange?.(data.paused ? 'paused' : 'running')
    }
  })

  const stopMutation = useMutation({
    mutationFn: () => api.training.stop(),
    onMutate: () => setIsLoading('stop'),
    onSettled: () => {
      setIsLoading(null)
      queryClient.invalidateQueries({ queryKey: ['training-state'] })
    },
    onSuccess: () => {
      onStatusChange?.('stopped')
    }
  })

  // Determine current status for display
  const getStatus = (): TrainingStatus => {
    if (!isTraining) return 'idle'
    if (isPaused) return 'paused'
    return 'running'
  }

  const status = getStatus()

  const getStatusColor = () => {
    switch (status) {
      case 'running':
        return 'bg-neon-green'
      case 'paused':
        return 'bg-yellow-400'
      case 'stopped':
      case 'idle':
        return 'bg-gray-500'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'running':
        return 'Running'
      case 'paused':
        return 'Paused'
      case 'stopped':
        return 'Stopped'
      case 'idle':
        return 'Idle'
    }
  }

  return (
    <div className="flex items-center gap-3">
      {/* Status Indicator */}
      <div className="flex items-center gap-2">
        <motion.div
          animate={status === 'running' ? { scale: [1, 1.2, 1] } : {}}
          transition={{ repeat: Infinity, duration: 1 }}
          className={`w-2 h-2 rounded-full ${getStatusColor()}`}
        />
        <span className="text-xs text-muted-foreground">{getStatusText()}</span>
      </div>

      {/* Control Buttons */}
      {isTraining && (
        <div className="flex items-center gap-1">
          {/* Pause/Resume Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => pauseMutation.mutate()}
            disabled={isLoading !== null}
            className={`p-1.5 rounded-md transition-colors ${
              isPaused
                ? 'bg-neon-green/20 hover:bg-neon-green/30 text-neon-green'
                : 'bg-yellow-400/20 hover:bg-yellow-400/30 text-yellow-400'
            } disabled:opacity-50`}
            title={isPaused ? 'Resume Training' : 'Pause Training'}
          >
            {isLoading === 'pause' ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : isPaused ? (
              <Play className="w-4 h-4" />
            ) : (
              <Pause className="w-4 h-4" />
            )}
          </motion.button>

          {/* Stop Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => stopMutation.mutate()}
            disabled={isLoading !== null}
            className="p-1.5 rounded-md bg-red-500/20 hover:bg-red-500/30 text-red-400 disabled:opacity-50"
            title="Stop Training"
          >
            {isLoading === 'stop' ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Square className="w-4 h-4" />
            )}
          </motion.button>
        </div>
      )}
    </div>
  )
}
