import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Grid3x3, Target, Compass, Info } from 'lucide-react'
import axios from '../lib/axios'

interface DomainInfo {
  name: string
  description: string
  state_space: {
    type: string
    size: number[]
    total_states: number
  }
  action_space: {
    type: string
    actions: string[]
    size: number
  }
  reward_info: {
    type: string
    range: number[]
    peaks: Array<{
      position: number[]
      intensity: number
      name: string
    }>
  }
  objectives: string[]
}

export function DomainInfo() {
  const { data: domainInfo } = useQuery({
    queryKey: ['domain-info'],
    queryFn: async () => {
      const response = await axios.get('/api/domain/info')
      return response.data as DomainInfo
    },
  })

  if (!domainInfo) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      <div className="glass-dark rounded-xl p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Grid3x3 className="w-5 h-5 text-neon-purple" />
          <h3 className="text-lg font-semibold">{domainInfo.name}</h3>
        </div>
        <p className="text-sm text-muted-foreground">{domainInfo.description}</p>
      </div>

      <div className="glass-dark rounded-xl p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Compass className="w-5 h-5 text-neon-blue" />
          <h4 className="font-medium">State & Action Space</h4>
        </div>
        <div className="space-y-2 text-sm">
          <div>
            <span className="text-muted-foreground">States:</span>{' '}
            <span className="font-mono">
              {domainInfo.state_space.size.join('×')} = {domainInfo.state_space.total_states}
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">Actions:</span>{' '}
            <span className="font-mono">{domainInfo.action_space.actions.join(', ')}</span>
          </div>
        </div>
      </div>

      <div className="glass-dark rounded-xl p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Target className="w-5 h-5 text-neon-green" />
          <h4 className="font-medium">Reward Peaks</h4>
        </div>
        <div className="space-y-2">
          {domainInfo.reward_info.peaks.map((peak, i) => (
            <div key={i} className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">{peak.name}</span>
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xs">({peak.position.join(',')})</span>
                <div className="w-12 bg-gradient-to-r from-transparent to-neon-green rounded-full h-2">
                  <div 
                    className="h-full bg-neon-green rounded-full"
                    style={{ width: `${(peak.intensity / 10) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-dark rounded-xl p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Info className="w-5 h-5 text-neon-purple" />
          <h4 className="font-medium">Training Objectives</h4>
        </div>
        <div className="flex flex-wrap gap-2">
          {domainInfo.objectives.map((obj) => (
            <span
              key={obj}
              className="px-2 py-1 text-xs rounded-md bg-neon-purple/20 text-neon-purple"
            >
              {obj}
            </span>
          ))}
        </div>
      </div>
    </motion.div>
  )
}