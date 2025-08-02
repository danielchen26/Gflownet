import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Activity, Zap } from 'lucide-react'
import axios from 'axios'
import { useWebSocket } from '../hooks/useWebSocket'
import { useEffect, useState } from 'react'

interface TrainingData {
  episode: number
  loss: number
  reward: number
  timestamp: string
}

interface Metrics {
  mean_loss: number
  mean_reward: number
  total_episodes: number
}

// Custom tooltip with glassmorphism
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-dark rounded-lg px-4 py-2 border border-neon-purple/20">
        <p className="text-sm text-muted-foreground">Episode {label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm font-medium" style={{ color: entry.color }}>
            {entry.name}: {entry.value.toFixed(4)}
          </p>
        ))}
      </div>
    )
  }
  return null
}

// Animated metric card
function MetricCard({ 
  title, 
  value, 
  change, 
  icon: Icon, 
  color 
}: { 
  title: string
  value: number
  change?: number
  icon: React.ElementType
  color: string
}) {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="glass-dark rounded-xl p-4 border border-dark-border/50"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">{title}</span>
        <Icon className={`w-4 h-4 ${color}`} />
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-bold">{value.toFixed(4)}</span>
        {change !== undefined && (
          <div className={`flex items-center space-x-1 text-sm ${change >= 0 ? 'text-neon-green' : 'text-neon-pink'}`}>
            {change >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            <span>{Math.abs(change).toFixed(2)}%</span>
          </div>
        )}
      </div>
    </motion.div>
  )
}

export function TrainingDashboard() {
  const [realtimeData, setRealtimeData] = useState<TrainingData[]>([])
  const { lastMessage } = useWebSocket()
  
  const { data: history } = useQuery({
    queryKey: ['training-history'],
    queryFn: async () => {
      const response = await axios.get('/api/training/history')
      return response.data
    },
  })
  
  // Update realtime data from WebSocket
  useEffect(() => {
    if (lastMessage?.type === 'training.update') {
      setRealtimeData(prev => [...prev.slice(-99), lastMessage.data as TrainingData])
    }
  }, [lastMessage])
  
  // Combine historical and realtime data
  const chartData = history ? history.episodes.map((episode: number, i: number) => ({
    episode,
    loss: history.losses[i],
    reward: history.rewards[i],
  })) : []
  
  const combinedData = [...chartData, ...realtimeData]
  const displayData = combinedData.slice(-100) // Show last 100 points
  
  const metrics: Metrics = history?.metrics || {
    mean_loss: 0,
    mean_reward: 0,
    total_episodes: 0,
  }
  
  // Calculate changes
  const recentLoss = displayData.slice(-10).reduce((acc, d) => acc + d.loss, 0) / 10
  const previousLoss = displayData.slice(-20, -10).reduce((acc, d) => acc + d.loss, 0) / 10
  const lossChange = previousLoss ? ((recentLoss - previousLoss) / previousLoss) * 100 : 0
  
  const recentReward = displayData.slice(-10).reduce((acc, d) => acc + d.reward, 0) / 10
  const previousReward = displayData.slice(-20, -10).reduce((acc, d) => acc + d.reward, 0) / 10
  const rewardChange = previousReward ? ((recentReward - previousReward) / previousReward) * 100 : 0
  
  return (
    <div className="h-full flex flex-col space-y-6">
      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          title="Loss"
          value={metrics.mean_loss}
          change={-lossChange}
          icon={Activity}
          color="text-neon-purple"
        />
        <MetricCard
          title="Reward"
          value={metrics.mean_reward}
          change={rewardChange}
          icon={TrendingUp}
          color="text-neon-green"
        />
        <MetricCard
          title="Episodes"
          value={metrics.total_episodes}
          icon={Zap}
          color="text-neon-blue"
        />
      </div>
      
      {/* Charts */}
      <div className="flex-1 grid grid-cols-2 gap-6">
        {/* Loss Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-dark rounded-xl p-4"
        >
          <h3 className="text-sm font-medium text-muted-foreground mb-4">Training Loss</h3>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={displayData}>
              <defs>
                <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#BD00FF" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#BD00FF" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A2A2D" />
              <XAxis 
                dataKey="episode" 
                stroke="#666"
                tick={{ fill: '#666', fontSize: 12 }}
              />
              <YAxis 
                stroke="#666"
                tick={{ fill: '#666', fontSize: 12 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="loss"
                stroke="#BD00FF"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#lossGradient)"
              />
              <ReferenceLine 
                y={metrics.mean_loss} 
                stroke="#BD00FF" 
                strokeDasharray="5 5"
                strokeOpacity={0.5}
              />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>
        
        {/* Reward Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-dark rounded-xl p-4"
        >
          <h3 className="text-sm font-medium text-muted-foreground mb-4">Mean Reward</h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={displayData}>
              <defs>
                <linearGradient id="rewardGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#00FF88" stopOpacity={1}/>
                  <stop offset="100%" stopColor="#00D9FF" stopOpacity={1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A2A2D" />
              <XAxis 
                dataKey="episode" 
                stroke="#666"
                tick={{ fill: '#666', fontSize: 12 }}
              />
              <YAxis 
                stroke="#666"
                tick={{ fill: '#666', fontSize: 12 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="reward"
                stroke="url(#rewardGradient)"
                strokeWidth={3}
                dot={false}
                filter="drop-shadow(0px 0px 8px rgba(0, 255, 136, 0.5))"
              />
              <ReferenceLine 
                y={metrics.mean_reward} 
                stroke="#00FF88" 
                strokeDasharray="5 5"
                strokeOpacity={0.5}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      </div>
      
      {/* Live indicator */}
      {realtimeData.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center justify-center space-x-2"
        >
          <div className="w-2 h-2 rounded-full bg-neon-green animate-pulse" />
          <span className="text-xs text-muted-foreground">Live training data</span>
        </motion.div>
      )}
    </div>
  )
}