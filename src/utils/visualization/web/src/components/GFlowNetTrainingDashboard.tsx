import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Activity, Zap, Target, Gauge } from 'lucide-react'
import axios from '../lib/axios'
import { useMemo } from 'react'

interface TrainingData {
  episodes: number[]
  losses: number[]
  rewards: number[]
  tb_losses: number[]
  flow_losses: number[]
  exploration_rates: number[]
  metrics: {
    mean_loss: number
    mean_reward: number
    mean_tb_loss: number
    mean_flow_loss: number
    current_exploration: number
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
      className="glass-dark rounded-xl p-4 border border-dark-border/50"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">{title}</span>
        <Icon className={`w-4 h-4 ${color}`} />
      </div>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-2xl font-bold">
            {typeof value === 'number' ? value.toFixed(4) : value}
          </div>
          {subtitle && (
            <div className="text-xs text-muted-foreground mt-1">{subtitle}</div>
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
  
  // Prepare chart data - only show last 50 episodes for better visualization
  const chartData = useMemo(() => {
    if (!history) return []
    
    const startIdx = Math.max(0, history.episodes.length - 50)
    return history.episodes.slice(startIdx).map((episode, i) => ({
      episode: episode,
      loss: history.losses[startIdx + i],
      reward: history.rewards[startIdx + i],
      tb_loss: history.tb_losses[startIdx + i],
      flow_loss: history.flow_losses[startIdx + i],
      exploration_rate: history.exploration_rates[startIdx + i],
    }))
  }, [history])
  
  if (!history || !chartData.length) return <div className="flex items-center justify-center h-full text-muted-foreground">Waiting for training data...</div>
  
  const metrics = history.metrics
  
  return (
    <div className="h-full flex flex-col space-y-4 overflow-hidden">
      {/* Metrics Grid with live updates */}
      <div className="grid grid-cols-3 gap-3">
        <MetricCard
          title="Current Loss"
          value={trainingState?.current_loss || metrics.mean_loss}
          subtitle="Trajectory Balance"
          icon={Activity}
          color="text-neon-purple"
          trend={trainingState?.current_loss < metrics.mean_loss ? "down" : "up"}
          percentage={Math.abs(((trainingState?.current_loss || metrics.mean_loss) - metrics.mean_loss) / metrics.mean_loss * 100)}
        />
        <MetricCard
          title="Current Reward"
          value={trainingState?.current_reward || metrics.mean_reward}
          subtitle={`Episode ${trainingState?.current_episode || metrics.total_episodes}`}
          icon={Target}
          color="text-neon-green"
          trend={trainingState?.current_reward > metrics.mean_reward ? "up" : "down"}
          percentage={Math.abs(((trainingState?.current_reward || metrics.mean_reward) - metrics.mean_reward) / metrics.mean_reward * 100)}
        />
        <MetricCard
          title="Convergence"
          value={`${(metrics.convergence_estimate * 100).toFixed(1)}%`}
          subtitle={isTraining ? "Training..." : "Complete"}
          icon={Gauge}
          color="text-neon-blue"
        />
      </div>
      
      {/* Charts with better layout */}
      <div className="flex-1 space-y-4 min-h-0">
        {/* Loss Curves - Full width */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-dark rounded-xl p-4 h-[45%]"
        >
          <h3 className="text-sm font-medium text-muted-foreground mb-3 flex items-center justify-between">
            <span>Training Losses</span>
            <div className="flex gap-4 text-xs">
              <span className="flex items-center gap-1"><div className="w-3 h-0.5 bg-[#BD00FF]"></div>Total Loss</span>
              <span className="flex items-center gap-1"><div className="w-3 h-0.5 bg-[#00D9FF]"></div>TB Loss</span>
              <span className="flex items-center gap-1"><div className="w-3 h-0.5 bg-[#FF006E]"></div>Flow Loss</span>
            </div>
          </h3>
          <div style={{ width: '100%', height: 'calc(100% - 48px)' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
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
                  name="Total Loss"
                />
                <Line
                  type="monotone"
                  dataKey="tb_loss"
                  stroke="#00D9FF"
                  strokeWidth={2}
                  dot={false}
                  name="TB Loss"
                  strokeDasharray="5 5"
                />
                <Line
                  type="monotone"
                  dataKey="flow_loss"
                  stroke="#FF006E"
                  strokeWidth={2}
                  dot={false}
                  name="Flow Loss"
                  strokeDasharray="3 3"
                />
                <Legend 
                  verticalAlign="bottom"
                  height={20}
                  iconType="line"
                  wrapperStyle={{ paddingTop: '10px', fontSize: '11px' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
        
        {/* Reward Chart - Full width */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-dark rounded-xl p-4 h-[45%]"
        >
          <h3 className="text-sm font-medium text-muted-foreground mb-3 flex items-center justify-between">
            <span>Reward & Exploration</span>
            <div className="flex gap-4 text-xs">
              <span className="flex items-center gap-1"><div className="w-3 h-0.5 bg-[#00FF88]"></div>Mean Reward (left)</span>
              <span className="flex items-center gap-1"><div className="w-3 h-0.5 bg-[#FFA07A] stroke-dasharray-[5,5]"></div>Exploration Rate (right)</span>
            </div>
          </h3>
          <div style={{ width: '100%', height: 'calc(100% - 48px)' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A2D" />
                <XAxis 
                  dataKey="episode" 
                  stroke="#666"
                  tick={{ fill: '#666', fontSize: 10 }}
                />
                <YAxis 
                  yAxisId="left"
                  stroke="#00FF88"
                  tick={{ fill: '#00FF88', fontSize: 10 }}
                  domain={['auto', 'auto']}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  stroke="#FFA07A"
                  tick={{ fill: '#FFA07A', fontSize: 10 }}
                  domain={[0, 1]}
                />
                <Tooltip content={<CustomTooltip />} />
                
                <defs>
                  <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00FF88" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#00FF88" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="reward"
                  stroke="#00FF88"
                  strokeWidth={2}
                  fill="url(#rewardGradient)"
                  fillOpacity={0.3}
                  name="Mean Reward"
                />
                
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="exploration_rate"
                  stroke="#FFA07A"
                  strokeWidth={2}
                  dot={false}
                  name="Exploration Rate"
                  strokeDasharray="5 5"
                />
                
                <Legend 
                  verticalAlign="bottom"
                  height={20}
                  iconType="line"
                  wrapperStyle={{ paddingTop: '10px', fontSize: '11px' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>
      
      {/* Additional Metrics */}
      <div className="grid grid-cols-4 gap-3">
        <div className="glass-dark rounded-lg px-3 py-2">
          <div className="text-xs text-muted-foreground">TB Loss</div>
          <div className="text-sm font-medium text-neon-blue">
            {metrics.mean_tb_loss.toFixed(4)}
          </div>
        </div>
        <div className="glass-dark rounded-lg px-3 py-2">
          <div className="text-xs text-muted-foreground">Flow Loss</div>
          <div className="text-sm font-medium text-neon-pink">
            {metrics.mean_flow_loss.toFixed(4)}
          </div>
        </div>
        <div className="glass-dark rounded-lg px-3 py-2">
          <div className="text-xs text-muted-foreground">Exploration</div>
          <div className="text-sm font-medium text-neon-orange">
            {(metrics.current_exploration * 100).toFixed(1)}%
          </div>
        </div>
        <div className="glass-dark rounded-lg px-3 py-2">
          <div className="text-xs text-muted-foreground">Episodes</div>
          <div className="text-sm font-medium">
            {metrics.total_episodes}
          </div>
        </div>
      </div>
    </div>
  )
}