import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Zap, TrendingUp, Clock, Layers, Award, Shuffle } from 'lucide-react'
import { api } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import { useEffect, useState } from 'react'

interface Metric {
  label: string
  value: string | number
  icon: React.ElementType
  color: string
  trend?: 'up' | 'down' | 'stable'
}

// Animated number display
function AnimatedNumber({ value }: { value: number }) {
  const [displayValue, setDisplayValue] = useState(0)
  
  useEffect(() => {
    const duration = 1000
    const steps = 60
    const stepValue = (value - displayValue) / steps
    let current = displayValue
    let step = 0
    
    const interval = setInterval(() => {
      step++
      current += stepValue
      setDisplayValue(current)
      
      if (step >= steps) {
        setDisplayValue(value)
        clearInterval(interval)
      }
    }, duration / steps)
    
    return () => clearInterval(interval)
  }, [value])
  
  return <>{displayValue.toFixed(2)}</>
}

// Single metric display
function MetricItem({ metric }: { metric: Metric }) {
  const Icon = metric.icon
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      whileHover={{ scale: 1.02 }}
      className="glass-dark rounded-lg p-3 border border-dark-border/50"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg bg-gradient-to-br ${metric.color}`}>
            <Icon className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-xs text-muted-foreground">{metric.label}</p>
            <p className="text-lg font-semibold">
              {typeof metric.value === 'number' ? (
                <AnimatedNumber value={metric.value} />
              ) : (
                metric.value
              )}
            </p>
          </div>
        </div>
        {metric.trend && (
          <div className={`text-xs ${
            metric.trend === 'up' ? 'text-neon-green' : 
            metric.trend === 'down' ? 'text-neon-pink' : 
            'text-muted-foreground'
          }`}>
            {metric.trend === 'up' ? '↑' : metric.trend === 'down' ? '↓' : '→'}
          </div>
        )}
      </div>
    </motion.div>
  )
}

// Progress ring
function ProgressRing({ progress, size = 60 }: { progress: number, size?: number }) {
  const radius = (size - 8) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (progress / 100) * circumference
  
  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        stroke="#2A2A2D"
        strokeWidth="4"
        fill="none"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        stroke="url(#progressGradient)"
        strokeWidth="4"
        fill="none"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        className="transition-all duration-1000 ease-out"
      />
      <defs>
        <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#BD00FF" />
          <stop offset="100%" stopColor="#00D9FF" />
        </linearGradient>
      </defs>
    </svg>
  )
}

export function RealtimeMetrics() {
  const [realtimeMetrics, setRealtimeMetrics] = useState({
    currentLoss: 0,
    currentReward: 0,
    episodeCount: 0,
    convergenceProgress: 0,
  })
  
  const { lastMessage } = useWebSocket()
  
  const { data: metrics } = useQuery({
    queryKey: ['training-state'],
    queryFn: async () => {
      const state = await api.training.getState()
      return state
    },
    refetchInterval: 250, // Update every 250ms for real-time feel
  })

  const { data: trajectories } = useQuery({
    queryKey: ['trajectories'],
    queryFn: async () => {
      const data = await api.trajectories.getRecent(10)
      return { count: data.trajectories?.length || 0 }
    },
    refetchInterval: metrics?.is_training ? 1000 : 5000, // Update more frequently during training
  })
  
  // Update from WebSocket
  useEffect(() => {
    if (lastMessage?.type === 'training.update') {
      const data = lastMessage.data
      setRealtimeMetrics(prev => ({
        currentLoss: data.loss,
        currentReward: data.reward,
        episodeCount: data.episode,
        convergenceProgress: Math.min(95, prev.convergenceProgress + Math.random() * 2),
      }))
    }
  }, [lastMessage])
  
  const displayMetrics: Metric[] = [
    {
      label: 'Current Loss',
      value: metrics?.latest_loss?.toFixed(2) || '0.00',
      icon: Activity,
      color: 'from-neon-purple to-neon-purple/50',
      trend: (metrics?.latest_loss || 0) < 10 ? 'down' : 'stable',
    },
    {
      label: 'Mean Reward',
      value: metrics?.metrics?.mean_reward?.toFixed(2) || '0.00',
      icon: Award,
      color: 'from-neon-green to-neon-green/50',
      trend: (metrics?.metrics?.mean_reward || 0) > 1 ? 'up' : 'stable',
    },
    {
      label: 'Iteration',
      value: `${metrics?.current_iteration || 0}/${metrics?.total_iterations || 0}`,
      icon: Layers,
      color: 'from-neon-blue to-neon-blue/50',
    },
    {
      label: 'Gradient Norm',
      value: metrics?.latest_gradient_norm?.toFixed(2) || '0.00',
      icon: Zap,
      color: 'from-gradient-orange-from to-gradient-orange-to',
    },
    // Exploration metric (Phase 7: Mode Collapse Fix)
    {
      label: 'Current Epsilon',
      value: metrics?.current_epsilon?.toFixed(3) || '0.000',
      icon: Shuffle,
      color: 'from-yellow-400 to-yellow-600',
      trend: metrics?.epsilon_decay ? 'down' : 'stable',  // Decreasing if annealing
    },
  ]
  
  return (
    <div className="space-y-2">
      {/* Convergence Progress */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-dark rounded-lg p-3 text-center"
      >
        <h3 className="text-xs font-medium text-muted-foreground mb-2">Training Progress</h3>
        <div className="relative inline-block">
          <ProgressRing progress={((metrics?.current_iteration || 0) / (metrics?.total_iterations || 1)) * 100} size={80} />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-lg font-bold gradient-text transform rotate-90">
              {(((metrics?.current_iteration || 0) / (metrics?.total_iterations || 1)) * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          {metrics?.current_iteration || 0} / {metrics?.total_iterations || 0} iterations
        </p>
      </motion.div>
      
      {/* Metrics List */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-muted-foreground px-1">Live Metrics</h3>
        <AnimatePresence mode="wait">
          {displayMetrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <MetricItem metric={metric} />
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
      
      {/* Status Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="glass-dark rounded-lg p-4"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Status</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${metrics?.is_training ? 'bg-neon-green animate-pulse' : 'bg-gray-500'}`} />
            <span className="text-sm">{metrics?.is_training ? 'Training Active' : 'Idle'}</span>
          </div>
        </div>
      </motion.div>
    </div>
  )
}