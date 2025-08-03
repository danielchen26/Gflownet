import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Brush } from 'recharts'
import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Activity, Zap, Target, Gauge } from 'lucide-react'
import axios from '../lib/axios'
import { useMemo, useState } from 'react'

interface TrainingData {
  episodes: number[]
  losses: number[]
  rewards: number[]
  tb_losses: number[]
  flow_losses: number[]
  metrics: {
    mean_loss: number
    mean_reward: number
    mean_tb_loss: number
    mean_flow_loss: number
    total_episodes: number
    convergence_estimate: number
  }
}

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-dark rounded-lg px-3 py-2 border border-neon-purple/20">
        <p className="text-xs text-muted-foreground mb-1">Episode {label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-xs font-medium" style={{ color: entry.color }}>
            {entry.name}: {entry.value.toFixed(4)}
          </p>
        ))}
      </div>
    )
  }
  return null
}

// Metric card
function MetricCard({ 
  title, 
  value, 
  subtitle,
  icon: Icon, 
  color,
  trend,
  percentage
}: { 
  title: string
  value: number | string
  subtitle?: string
  icon: React.ElementType
  color: string
  trend?: 'up' | 'down'
  percentage?: number
}) {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="glass-dark rounded-lg p-2 border border-dark-border/50"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">{title}</span>
        <Icon className={`w-4 h-4 ${color}`} />
      </div>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-lg font-bold">
            {typeof value === 'number' ? value.toFixed(4) : value}
          </div>
          {subtitle && (
            <div className="text-[10px] text-muted-foreground">{subtitle}</div>
          )}
        </div>
        {percentage !== undefined && (
          <div className={`flex items-center space-x-1 text-sm ${
            trend === 'up' ? 'text-neon-green' : 'text-neon-pink'
          }`}>
            {trend === 'up' ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            <span>{Math.abs(percentage).toFixed(1)}%</span>
          </div>
        )}
      </div>
    </motion.div>
  )
}

export function GFlowNetTrainingDashboard() {
  const [brushStartIndex, setBrushStartIndex] = useState<number | null>(null)
  const [brushEndIndex, setBrushEndIndex] = useState<number | null>(null)
  
  // Poll for current training state
  const { data: trainingState } = useQuery({
    queryKey: ['training-state'],
    queryFn: async () => {
      const response = await axios.get('/api/training/state')
      return response.data
    },
    refetchInterval: 250, // Fast updates for smooth animation
  })
  
  const isTraining = trainingState?.is_training || false
  
  // Get full training history
  const { data: history } = useQuery({
    queryKey: ['training-history'],
    queryFn: async () => {
      const response = await axios.get('/api/training/history')
      return response.data as TrainingData
    },
    refetchInterval: isTraining ? 500 : false, // Update during training
  })
  
  // Prepare chart data - show ALL episodes for full history
  const chartData = useMemo(() => {
    if (!history) return []
    
    return history.episodes.map((episode, i) => ({
      episode: episode,
      loss: history.losses[i],
      reward: history.rewards[i],
      tb_loss: history.tb_losses[i],
      flow_loss: history.flow_losses[i],
    }))
  }, [history])
  
  if (!history || !chartData.length) return <div className="flex items-center justify-center h-full text-muted-foreground">Waiting for training data...</div>
  
  const metrics = history.metrics
  
  return (
    <div className="h-full flex flex-col space-y-1 p-2">
      {/* Metrics Grid with live updates */}
      <div className="grid grid-cols-3 gap-1">
        <MetricCard
          title="Current Loss"
          value={trainingState?.current_loss || history.losses[history.losses.length - 1]}
          subtitle="Latest Episode"
          icon={Activity}
          color="text-neon-purple"
          trend={trainingState?.current_loss < metrics.mean_loss ? "down" : "up"}
          percentage={Math.abs(((trainingState?.current_loss || history.losses[history.losses.length - 1]) - metrics.mean_loss) / metrics.mean_loss * 100)}
        />
        <MetricCard
          title="Current Reward"
          value={trainingState?.current_reward || history.rewards[history.rewards.length - 1]}
          subtitle={`Episode ${trainingState?.current_episode || metrics.total_episodes}`}
          icon={Target}
          color="text-neon-green"
          trend={trainingState?.current_reward > metrics.mean_reward ? "up" : "down"}
          percentage={Math.abs(((trainingState?.current_reward || history.rewards[history.rewards.length - 1]) - metrics.mean_reward) / metrics.mean_reward * 100)}
        />
        <MetricCard
          title="Convergence"
          value={`${(metrics.convergence_estimate * 100).toFixed(1)}%`}
          subtitle={isTraining ? "Training..." : "Complete"}
          icon={Gauge}
          color="text-neon-blue"
        />
      </div>
      
      {/* Charts Grid - 2x2 layout */}
      <div className="flex-1 grid grid-cols-2 gap-1">
        {/* Loss Curves - Top Left */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-dark rounded-lg p-2"
        >
          <h3 className="text-xs font-medium text-muted-foreground mb-2">
            Training Losses
          </h3>
          <div style={{ width: '100%', height: 'calc(100% - 32px)' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A2D" />
                <XAxis 
                  dataKey="episode" 
                  stroke="#666"
                  tick={{ fill: '#666', fontSize: 10 }}
                />
                <YAxis 
                  stroke="#666"
                  tick={{ fill: '#666', fontSize: 10 }}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<CustomTooltip />} />
                
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#BD00FF"
                  strokeWidth={2}
                  dot={false}
                  name="Total"
                />
                <Line
                  type="monotone"
                  dataKey="tb_loss"
                  stroke="#00D9FF"
                  strokeWidth={2}
                  dot={false}
                  name="TB"
                  strokeDasharray="5 5"
                />
                <Line
                  type="monotone"
                  dataKey="flow_loss"
                  stroke="#FF006E"
                  strokeWidth={2}
                  dot={false}
                  name="Flow"
                  strokeDasharray="3 3"
                />
                <Legend 
                  verticalAlign="top"
                  align="right"
                  height={20}
                  iconType="line"
                  wrapperStyle={{ fontSize: '10px' }}
                />
                <Brush
                  dataKey="episode"
                  height={30}
                  stroke="#666"
                  fill="#1a1a1d"
                  onChange={(e: any) => {
                    setBrushStartIndex(e.startIndex)
                    setBrushEndIndex(e.endIndex)
                  }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
        
        {/* Reward Chart - Top Right */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-dark rounded-lg p-2"
        >
          <h3 className="text-xs font-medium text-muted-foreground mb-2">
            Mean Reward Progress
          </h3>
          <div style={{ width: '100%', height: 'calc(100% - 32px)' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 2, right: 30, left: 2, bottom: 2 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A2D" />
                <XAxis 
                  dataKey="episode" 
                  stroke="#666"
                  tick={{ fill: '#666', fontSize: 10 }}
                  domain={brushStartIndex !== null && brushEndIndex !== null ? [chartData[brushStartIndex].episode, chartData[brushEndIndex].episode] : ['dataMin', 'dataMax']}
                />
                <YAxis 
                  stroke="#00FF88"
                  tick={{ fill: '#00FF88', fontSize: 10 }}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<CustomTooltip />} />
                
                <defs>
                  <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00FF88" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#00FF88" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                
                <Area
                  type="monotone"
                  dataKey="reward"
                  stroke="#00FF88"
                  strokeWidth={2}
                  fill="url(#rewardGradient)"
                  fillOpacity={0.3}
                  name="Mean Reward"
                />
                
                <Legend 
                  verticalAlign="top"
                  align="right"
                  height={20}
                  iconType="line"
                  wrapperStyle={{ fontSize: '10px' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
        
        {/* Loss Components - Bottom Left */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-dark rounded-lg p-2"
        >
          <h3 className="text-xs font-medium text-muted-foreground mb-2">
            Loss Components Comparison
          </h3>
          <div style={{ width: '100%', height: 'calc(100% - 32px)' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A2D" />
                <XAxis 
                  dataKey="episode" 
                  stroke="#666"
                  tick={{ fill: '#666', fontSize: 10 }}
                  domain={brushStartIndex !== null && brushEndIndex !== null ? [chartData[brushStartIndex].episode, chartData[brushEndIndex].episode] : ['dataMin', 'dataMax']}
                />
                <YAxis 
                  stroke="#666"
                  tick={{ fill: '#666', fontSize: 10 }}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<CustomTooltip />} />
                
                <defs>
                  <linearGradient id="tbGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00D9FF" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#00D9FF" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="flowGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#FF006E" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#FF006E" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                
                <Area
                  type="monotone"
                  dataKey="tb_loss"
                  stackId="1"
                  stroke="#00D9FF"
                  fill="url(#tbGradient)"
                  name="TB Loss"
                />
                <Area
                  type="monotone"
                  dataKey="flow_loss"
                  stackId="1"
                  stroke="#FF006E"
                  fill="url(#flowGradient)"
                  name="Flow Loss"
                />
                
                <Legend 
                  verticalAlign="top"
                  align="right"
                  height={20}
                  iconType="line"
                  wrapperStyle={{ fontSize: '10px' }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
        
        {/* Metrics Summary - Bottom Right */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-dark rounded-lg p-2"
        >
          <h3 className="text-xs font-medium text-muted-foreground mb-2">
            Training Statistics
          </h3>
          <div className="grid grid-cols-2 gap-2 h-[calc(100%-32px)]">
            <div className="space-y-2">
              <div className="glass-dark rounded-lg px-2 py-1">
                <div className="text-[10px] text-muted-foreground">Mean Loss (20ep)</div>
                <div className="text-sm font-medium text-neon-blue">
                  {metrics.mean_loss.toFixed(4)}
                </div>
              </div>
              <div className="glass-dark rounded-lg px-2 py-1">
                <div className="text-[10px] text-muted-foreground">Mean Reward (20ep)</div>
                <div className="text-sm font-medium text-neon-pink">
                  {metrics.mean_reward.toFixed(3)}
                </div>
              </div>
            </div>
            <div className="space-y-2">
              <div className="glass-dark rounded-lg px-2 py-1">
                <div className="text-[10px] text-muted-foreground">TB Loss (20ep)</div>
                <div className="text-sm font-medium text-neon-purple">
                  {metrics.mean_tb_loss.toFixed(4)}
                </div>
              </div>
              <div className="glass-dark rounded-lg px-2 py-1">
                <div className="text-[10px] text-muted-foreground">Total Episodes</div>
                <div className="text-sm font-medium">
                  {trainingState?.current_episode || metrics.total_episodes}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}