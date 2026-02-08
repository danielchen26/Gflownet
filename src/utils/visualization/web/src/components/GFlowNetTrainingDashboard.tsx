import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Brush } from 'recharts'
import { motion } from 'framer-motion'
import { Zap, Target, Gauge, Sparkles, Database } from 'lucide-react'
import { api } from '../services/api'
import { useChartColors, useThemeLayout } from '../contexts/ThemeContext'
import { useEffect, useMemo, useState } from 'react'

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

// Metric card — currently unused but kept for potential future use

interface GFlowNetTrainingDashboardProps {
  problemConfig?: any
}

export function GFlowNetTrainingDashboard({ problemConfig }: GFlowNetTrainingDashboardProps) {
  const [brushStartIndex, setBrushStartIndex] = useState<number | null>(null)
  const [brushEndIndex, setBrushEndIndex] = useState<number | null>(null)
  const cc = useChartColors()
  const layout = useThemeLayout()
  
  // Poll for current training state
  const { data: trainingState } = useQuery({
    queryKey: ['training-state'],
    queryFn: async () => {
      return await api.training.getState()
    },
    refetchInterval: 250, // Fast updates for smooth animation
  })

  const isTraining = trainingState?.is_training || false

  // Get full training history from v2 API
  const { data: historyRaw } = useQuery({
    queryKey: ['training-history'],
    queryFn: async () => {
      return await api.training.getHistory()
    },
    refetchInterval: isTraining ? 500 : false, // Update during training
  })

  // Transform v2 API response to match expected format
  // If backend doesn't have history, fall back to showing current state
  const history: TrainingData | undefined = useMemo(() => {
    // Backend returns {losses: [], rewards: [], gradient_norms: [], iteration_times: []}
    // We need to transform this to our expected format

    console.log('📊 Training history raw data:', {
      hasHistoryRaw: !!historyRaw,
      hasLosses: historyRaw?.losses?.length > 0,
      hasRewards: historyRaw?.rewards?.length > 0,
      lossesLength: historyRaw?.losses?.length || 0,
      rewardsLength: historyRaw?.rewards?.length || 0,
    })

    // Check if backend returned historical arrays
    if (historyRaw && historyRaw.losses && historyRaw.losses.length > 0) {
      const length = historyRaw.losses.length
      return {
        episodes: Array.from({ length }, (_, i) => i + 1), // 1, 2, 3, ..., n
        losses: historyRaw.losses,
        rewards: historyRaw.rewards || Array(length).fill(0),
        tb_losses: historyRaw.losses, // V2 doesn't separate loss components
        flow_losses: Array(length).fill(0),
        metrics: {
          mean_loss: historyRaw.losses.reduce((a:number, b:number) => a + b, 0) / length,
          mean_reward: historyRaw.rewards ? historyRaw.rewards.reduce((a:number, b:number) => a + b, 0) / length : 0,
          mean_tb_loss: 0,
          mean_flow_loss: 0,
          total_episodes: length,
          convergence_estimate: (trainingState?.current_iteration || 0) / (trainingState?.total_iterations || 1),
        }
      }
    }

    // Fallback: If no history, create a single-point dataset from current training state
    if (trainingState && trainingState.current_iteration > 0) {
      console.log('📊 Using fallback: creating single-point history from current state')
      return {
        episodes: [trainingState.current_iteration],
        losses: [trainingState.latest_loss || 0],
        rewards: [trainingState.latest_reward || 0],
        tb_losses: [trainingState.latest_loss || 0],
        flow_losses: [0],
        metrics: {
          mean_loss: trainingState.latest_loss || 0,
          mean_reward: trainingState.metrics?.mean_reward || 0,
          mean_tb_loss: 0,
          mean_flow_loss: 0,
          total_episodes: trainingState.current_iteration,
          convergence_estimate: trainingState.current_iteration / (trainingState.total_iterations || 1),
        }
      }
    }

    console.log('📊 No history data available')
    return undefined
  }, [historyRaw, trainingState])
  
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

  // Reset brush indices when chartData shrinks (e.g., on training restart)
  // to prevent out-of-bounds access on stale indices
  useEffect(() => {
    if (brushEndIndex !== null && brushEndIndex >= chartData.length) {
      setBrushStartIndex(null)
      setBrushEndIndex(null)
    }
  }, [chartData.length, brushEndIndex])

  // Safe accessors for brush-synced domain — guards against stale indices
  const brushDomain = brushStartIndex !== null && brushEndIndex !== null
    && brushStartIndex < chartData.length && brushEndIndex < chartData.length
    ? [chartData[brushStartIndex].episode, chartData[brushEndIndex].episode]
    : undefined

  if (!history || !chartData.length) return <div className="flex items-center justify-center h-full text-muted-foreground">Waiting for training data...</div>
  
  const metrics = history.metrics
  
  return (
    <div className={`h-full flex flex-col ${layout.compact ? 'space-y-0.5 p-1' : layout.spacious ? 'space-y-3 p-4' : 'space-y-1 p-2'}`}>
      {/* Charts Grid - 2x2 layout */}
      <div className={`flex-1 grid grid-cols-2 ${layout.compact ? 'gap-0.5' : layout.spacious ? 'gap-3' : 'gap-1'}`}>
        {/* Loss Curves - Top Left */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`glass-dark rounded-lg ${layout.compact ? 'p-1' : layout.spacious ? 'p-4' : 'p-2'}`}
        >
          <h3 className={`${layout.headingSize} font-medium text-muted-foreground ${layout.compact ? 'mb-1' : 'mb-2'}`}>
            Training Losses
          </h3>
          <div style={{ width: '100%', height: `calc(100% - ${layout.compact ? '20px' : layout.spacious ? '40px' : '32px'})` }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
                <XAxis
                  dataKey="episode"
                  stroke={cc.axis}
                  tick={{ fill: cc.axis, fontSize: 10 }}
                />
                <YAxis
                  stroke={cc.axis}
                  tick={{ fill: cc.axis, fontSize: 10 }}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<CustomTooltip />} />

                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke={cc.primary}
                  strokeWidth={2}
                  dot={false}
                  name="Total"
                />
                <Line
                  type="monotone"
                  dataKey="tb_loss"
                  stroke={cc.secondary}
                  strokeWidth={2}
                  dot={false}
                  name="TB"
                  strokeDasharray="5 5"
                />
                <Line
                  type="monotone"
                  dataKey="flow_loss"
                  stroke={cc.tertiary}
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
                  stroke={cc.axis}
                  fill={cc.brushFill}
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
          className={`glass-dark rounded-lg ${layout.compact ? 'p-1' : layout.spacious ? 'p-4' : 'p-2'}`}
        >
          <h3 className={`${layout.headingSize} font-medium text-muted-foreground ${layout.compact ? 'mb-1' : 'mb-2'}`}>
            Mean Reward Progress
          </h3>
          <div style={{ width: '100%', height: `calc(100% - ${layout.compact ? '20px' : layout.spacious ? '40px' : '32px'})` }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 2, right: 30, left: 2, bottom: 2 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
                <XAxis
                  dataKey="episode"
                  stroke={cc.axis}
                  tick={{ fill: cc.axis, fontSize: 10 }}
                  domain={brushDomain ?? ['dataMin', 'dataMax']}
                />
                <YAxis
                  stroke={cc.green}
                  tick={{ fill: cc.green, fontSize: 10 }}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<CustomTooltip />} />

                <defs>
                  <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={cc.green} stopOpacity={0.8}/>
                    <stop offset="95%" stopColor={cc.green} stopOpacity={0}/>
                  </linearGradient>
                </defs>

                <Area
                  type="monotone"
                  dataKey="reward"
                  stroke={cc.green}
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
          className={`glass-dark rounded-lg ${layout.compact ? 'p-1' : layout.spacious ? 'p-4' : 'p-2'}`}
        >
          <h3 className={`${layout.headingSize} font-medium text-muted-foreground ${layout.compact ? 'mb-1' : 'mb-2'}`}>
            Loss Components Comparison
          </h3>
          <div style={{ width: '100%', height: `calc(100% - ${layout.compact ? '20px' : layout.spacious ? '40px' : '32px'})` }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
                <XAxis
                  dataKey="episode"
                  stroke={cc.axis}
                  tick={{ fill: cc.axis, fontSize: 10 }}
                  domain={brushDomain ?? ['dataMin', 'dataMax']}
                />
                <YAxis
                  stroke={cc.axis}
                  tick={{ fill: cc.axis, fontSize: 10 }}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<CustomTooltip />} />

                <defs>
                  <linearGradient id="tbGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={cc.secondary} stopOpacity={0.8}/>
                    <stop offset="95%" stopColor={cc.secondary} stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="flowGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={cc.tertiary} stopOpacity={0.8}/>
                    <stop offset="95%" stopColor={cc.tertiary} stopOpacity={0.1}/>
                  </linearGradient>
                </defs>

                <Area
                  type="monotone"
                  dataKey="tb_loss"
                  stackId="1"
                  stroke={cc.secondary}
                  fill="url(#tbGradient)"
                  name="TB Loss"
                />
                <Area
                  type="monotone"
                  dataKey="flow_loss"
                  stackId="1"
                  stroke={cc.tertiary}
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
        
        {/* Training Dynamics - Bottom Right */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className={`glass-dark rounded-lg ${layout.compact ? 'p-1' : layout.spacious ? 'p-4' : 'p-2'} flex flex-col`}
        >
          <h3 className={`${layout.headingSize} font-medium text-muted-foreground ${layout.compact ? 'mb-1' : 'mb-2'}`}>
            Training Dynamics
          </h3>
          <div className={`flex-1 flex flex-col ${layout.compact ? 'gap-1' : layout.spacious ? 'gap-3' : 'gap-2'} overflow-auto`}>
            {/* Objective Badge */}
            <div className="flex items-center gap-2">
              <span className={`px-2 py-0.5 ${layout.tinySize} font-semibold bg-neon-purple/20 text-neon-purple rounded`}>
                {problemConfig?.training_objective?.replace(/_/g, ' ') || 'TB'}
              </span>
              <span className={`${layout.tinySize} text-muted-foreground`}>
                Ep {trainingState?.current_iteration || metrics.total_episodes} / {trainingState?.total_iterations || metrics.total_episodes}
              </span>
            </div>

            {/* Rolling Averages */}
            <div className={`grid grid-cols-2 ${layout.compact ? 'gap-0.5' : 'gap-1.5'}`}>
              <div className={`glass-dark rounded ${layout.compact ? 'px-1 py-0.5' : 'px-2 py-1'}`}>
                <div className={`${layout.tinySize} text-muted-foreground`}>Mean Loss</div>
                <div className={`${layout.labelSize} font-mono font-medium text-neon-blue`}>{metrics.mean_loss.toFixed(2)}</div>
              </div>
              <div className={`glass-dark rounded ${layout.compact ? 'px-1 py-0.5' : 'px-2 py-1'}`}>
                <div className={`${layout.tinySize} text-muted-foreground`}>Mean Reward</div>
                <div className={`${layout.labelSize} font-mono font-medium text-neon-green`}>{metrics.mean_reward.toFixed(2)}</div>
              </div>
            </div>

            {/* Active Features */}
            <div className={`border-t border-dark-border/30 ${layout.compact ? 'pt-0.5' : 'pt-1.5'}`}>
              <div className={`${layout.tinySize} text-muted-foreground ${layout.compact ? 'mb-0.5' : 'mb-1.5'} font-medium`}>Active Features</div>
              <div className={`${layout.compact ? 'space-y-0.5' : 'space-y-1'}`}>
                {/* Epsilon */}
                {(problemConfig?.epsilon ?? 0) > 0 && (
                  <div className={`flex items-center gap-1.5 ${layout.tinySize}`}>
                    <Sparkles className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-purple`} />
                    <span className="text-muted-foreground">ε-Exploration:</span>
                    <span className="font-mono text-neon-purple">
                      {trainingState?.current_epsilon !== undefined
                        ? `${(trainingState.current_epsilon * 100).toFixed(1)}%`
                        : `${((problemConfig?.epsilon ?? 0) * 100).toFixed(0)}%`
                      }
                    </span>
                    {problemConfig?.epsilon_decay && (
                      <span className={`${layout.compact ? 'text-[7px]' : 'text-[8px]'} text-neon-green`}>(decaying)</span>
                    )}
                  </div>
                )}

                {/* Entropy */}
                {(problemConfig?.entropy_weight ?? 0) > 0 && (
                  <div className={`flex items-center gap-1.5 ${layout.tinySize}`}>
                    <Zap className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-blue`} />
                    <span className="text-muted-foreground">Entropy:</span>
                    <span className="font-mono text-neon-blue">{(problemConfig?.entropy_weight ?? 0).toFixed(3)}</span>
                  </div>
                )}

                {/* Replay Buffer */}
                <div className={`flex items-center gap-1.5 ${layout.tinySize}`}>
                  <Database className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-cyan-400`} />
                  <span className="text-muted-foreground">Replay:</span>
                  <span className={`font-mono ${problemConfig?.use_replay_buffer ? 'text-cyan-400' : 'text-gray-500'}`}>
                    {problemConfig?.use_replay_buffer ? `ON (${((problemConfig?.replay_ratio ?? 0.5) * 100).toFixed(0)}%)` : 'OFF'}
                  </span>
                </div>

                {/* Reward Shaping */}
                <div className={`flex items-center gap-1.5 ${layout.tinySize}`}>
                  <Target className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-green`} />
                  <span className="text-muted-foreground">Reward Shaping:</span>
                  <span className={`font-mono ${problemConfig?.reward_shaping !== false ? 'text-neon-green' : 'text-gray-500'}`}>
                    {problemConfig?.reward_shaping !== false ? 'ON' : 'OFF'}
                  </span>
                </div>

                {/* Z Rate */}
                {(problemConfig?.z_learning_rate_multiplier ?? 1) > 1 && (
                  <div className={`flex items-center gap-1.5 ${layout.tinySize}`}>
                    <Gauge className={`${layout.compact ? 'w-2.5 h-2.5' : 'w-3 h-3'} text-neon-orange`} />
                    <span className="text-muted-foreground">Z Rate:</span>
                    <span className="font-mono text-neon-orange">{(problemConfig?.z_learning_rate_multiplier ?? 1).toFixed(0)}×</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}