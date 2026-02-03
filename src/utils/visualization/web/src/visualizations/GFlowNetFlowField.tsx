import { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Html, Line } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'
import { motion } from 'framer-motion'
import { Info, Map, BarChart3, Activity, Target, Zap, TrendingUp } from 'lucide-react'
import { COLORS, interpolateRewardColor, interpolateFlowColor } from '../utils/colors'

interface FlowFieldData {
  resolution: [number, number, number]
  bounds: {
    x: [number, number]
    y: [number, number]
  }
  data: Array<{
    position: [number, number, number]
    velocity: [number, number, number]
    magnitude: number
    reward: number
    flow_value: number
  }>
  reward_peaks: Array<{
    position: [number, number]
    intensity: number
    name: string
  }>
}

interface StateStats {
  visitation_counts: Record<string, number>
  value_estimates: Record<string, number>
  total_states_visited: number
  max_visits: number
  coverage: number
  flow_statistics: {
    mean_flow: number
    max_flow: number
    convergence_ratio: number
    policy_entropy: number
  }
}

// Enhanced reward heatmap with smoother visualization
function RewardHeatmap({ flowData }: { flowData: FlowFieldData }) {
  const texture = useMemo(() => {
    const size = 512 // High resolution
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!
    
    // Create smooth gradient background
    ctx.fillStyle = '#050510'
    ctx.fillRect(0, 0, size, size)
    
    // Draw each reward peak with smooth gradients
    flowData.reward_peaks.forEach(peak => {
      const x = (peak.position[0] / 10) * size
      const y = (1 - peak.position[1] / 10) * size
      const radius = peak.intensity * 30
      
      for (let r = radius; r > 0; r -= 2) {
        const alpha = (r / radius) * 0.3
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, r)
        
        if (r === radius) {
          gradient.addColorStop(0, `rgba(0, 255, 136, ${alpha})`)
          gradient.addColorStop(0.5, `rgba(0, 217, 255, ${alpha * 0.7})`)
          gradient.addColorStop(1, `rgba(0, 0, 0, 0)`)
        } else {
          const hue = 120 + (1 - r/radius) * 60 // Green to cyan
          gradient.addColorStop(0, `hsla(${hue}, 100%, 60%, ${alpha})`)
          gradient.addColorStop(1, `hsla(${hue}, 100%, 30%, 0)`)
        }
        
        ctx.fillStyle = gradient
        ctx.fillRect(0, 0, size, size)
      }
    })
    
    return new THREE.CanvasTexture(canvas)
  }, [flowData])
  
  return (
    <mesh rotation={[0, 0, 0]} position={[5, 5, -0.01]}>
      <planeGeometry args={[10, 10]} />
      <meshBasicMaterial 
        map={texture} 
        transparent 
        opacity={0.6}
      />
    </mesh>
  )
}

// Policy flow vectors visualization
function FlowVectors({ flowData }: { flowData: FlowFieldData }) {
  const arrowsRef = useRef<THREE.Group>(null!)
  
  useFrame((state) => {
    if (arrowsRef.current) {
      // Subtle animation
      arrowsRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.1) * 0.02
    }
  })
  
  return (
    <group ref={arrowsRef}>
      {flowData.data.map((point, i) => {
        if (point.magnitude < 0.01 && point.flow_value < 0.05) return null // Skip very low flow
        
        const dir = new THREE.Vector3(...point.velocity).normalize()
        const flowNorm = Math.max(0, Math.min(1, point.flow_value))
        const length = Math.min(0.6, Math.max(0.15, point.magnitude * 1.5 + flowNorm * 0.2))
        const rewardNorm = Math.max(0, Math.min(1, point.reward / 10))
        
        // Color gradient based on flow value
        const color = new THREE.Color()
        if (flowNorm < 0.33) {
          color.setRGB(0.5, 0.5, 0.5) // Gray for low flow
        } else if (flowNorm < 0.66) {
          color.setRGB(0.5, 0.5 + flowNorm, 1) // Blue for medium
        } else {
          color.setRGB(0.5 + flowNorm * 0.5, 0, 1) // Purple for high
        }
        
        const angle = Math.atan2(dir.y, dir.x)
        
        return (
          <group key={i} position={[point.position[0], point.position[1], 0.05]}>
            {/* Arrow visualization */}
            <group rotation={[0, 0, angle]}>
              {/* Shaft */}
              <mesh position={[length * 0.35, 0, 0]}>
                <boxGeometry args={[length * 0.7, 0.06, 0.02]} />
                <meshBasicMaterial 
                  color={color}
                  transparent
                  opacity={0.7 + flowNorm * 0.3}
                />
              </mesh>
              {/* Head */}
              <mesh position={[length * 0.7, 0, 0]}>
                <coneGeometry args={[0.12, 0.2, 4]} />
                <meshBasicMaterial 
                  color={color}
                  transparent
                  opacity={0.8 + flowNorm * 0.2}
                />
              </mesh>
            </group>
            
            {/* Value indicator at base */}
            {rewardNorm > 0.3 && (
              <mesh position={[0, 0, -0.05]} rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[0.1, 0.1 + rewardNorm * 0.1, 16]} />
                <meshBasicMaterial 
                  color={interpolateRewardColor(rewardNorm)}
                  transparent
                  opacity={0.5}
                />
              </mesh>
            )}
          </group>
        )
      })}
    </group>
  )
}

// Enhanced state visitation with value estimates
// Trajectory paths visualization
function TrajectoryPaths() {
  const { data: trajectories } = useQuery({
    queryKey: ['recent-trajectories'],
    queryFn: async () => {
      const response = await api.trajectories.getRecent(20)
      return response.data.trajectories
    },
  })
  
  if (!trajectories) return null
  
  return (
    <group>
      {trajectories.slice(0, 20).map((traj: any, i: number) => {
        const points = traj.states.map((state: [number, number]) => 
          new THREE.Vector3(state[0], state[1], 0.02)
        )
        
        // Color based on recency - newer trajectories are brighter
        const recency = 1 - (i / 20)
        const color = new THREE.Color().lerpColors(
          new THREE.Color(0x666666), // Old - gray
          new THREE.Color(0x00ffff), // New - cyan
          recency
        )
        
        return (
          <Line
            key={traj.id}
            points={points}
            color={color}
            lineWidth={2}
            opacity={0.3 + recency * 0.4}
            transparent
          />
        )
      })}
    </group>
  )
}

// State visitation heatmap
function StateVisitation({ stats }: { stats: StateStats }) {
  const texture = useMemo(() => {
    const size = 256
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!
    const imageData = ctx.createImageData(size, size)
    const data = imageData.data
    
    // Create visitation heatmap
    Object.entries(stats.visitation_counts).forEach(([key, count]) => {
      const [x, y] = key.split(',').map(Number)
      const tx = Math.floor((x / 10) * size)
      const ty = Math.floor((1 - y / 10) * size)
      const normalized = count / stats.max_visits
      
      // Apply gaussian smoothing
      const radius = 10
      for (let dx = -radius; dx <= radius; dx++) {
        for (let dy = -radius; dy <= radius; dy++) {
          const px = tx + dx
          const py = ty + dy
          if (px >= 0 && px < size && py >= 0 && py < size) {
            const dist2 = dx * dx + dy * dy
            const weight = Math.exp(-dist2 / (2 * 4 * 4)) * normalized
            const idx = (py * size + px) * 4
            
            // Blue channel for visitation
            data[idx + 2] = Math.min(255, data[idx + 2] + weight * 255)
            data[idx + 3] = Math.min(255, data[idx + 3] + weight * 128)
          }
        }
      }
    })
    
    ctx.putImageData(imageData, 0, 0)
    return new THREE.CanvasTexture(canvas)
  }, [stats])
  
  return (
    <mesh position={[5, 5, 0.01]} rotation={[0, 0, 0]}>
      <planeGeometry args={[10, 10]} />
      <meshBasicMaterial 
        map={texture}
        transparent
        opacity={0.5}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  )
}

// Create arrow from components
function createArrow(position: [number, number, number], direction: THREE.Vector3, length: number, color: number) {
  const arrowGroup = new THREE.Group()
  
  // Shaft
  const shaftGeometry = new THREE.CylinderGeometry(0.05, 0.05, length * 0.7, 8)
  const shaftMaterial = new THREE.MeshBasicMaterial({ color })
  const shaft = new THREE.Mesh(shaftGeometry, shaftMaterial)
  shaft.position.set(0, length * 0.35, 0)
  shaft.rotation.z = Math.PI / 2
  
  // Head
  const headGeometry = new THREE.ConeGeometry(0.15, length * 0.3, 8)
  const headMaterial = new THREE.MeshBasicMaterial({ color })
  const head = new THREE.Mesh(headGeometry, headMaterial)
  head.position.set(length * 0.85, 0, 0)
  head.rotation.z = -Math.PI / 2
  
  arrowGroup.add(shaft)
  arrowGroup.add(head)
  arrowGroup.position.set(...position)
  arrowGroup.rotation.z = Math.atan2(direction.y, direction.x)
  
  return arrowGroup
}

// Main scene with enhanced visualization
function Scene({ viewMode }: { viewMode: 'flow' | 'visits' | 'combined' }) {
  const { data: flowField } = useQuery({
    queryKey: ['flow-field'],
    queryFn: async () => {
      const response = await api.analysis.getFlowField()
      return response.data as FlowFieldData
    },
  })
  
  const { data: stateStats } = useQuery({
    queryKey: ['state-statistics'],
    queryFn: async () => {
      const response = await api.training.getState()
      return response.data as StateStats
    },
  })
  
  if (!flowField || !stateStats) return null
  
  return (
    <group position={[0, 0, 0]}>
      {/* Grid on XY plane */}
      <gridHelper args={[10, 10, '#2A2A2D', '#1A1A1D']} position={[5, 5, 0]} rotation={[0, 0, 0]} />
      
      {/* Always show reward heatmap */}
      <RewardHeatmap flowData={flowField} />
      
      {/* Conditional rendering based on view mode */}
      {(viewMode === 'flow' || viewMode === 'combined') && (
        <>
          <FlowVectors flowData={flowField} />
          <TrajectoryPaths />
        </>
      )}
      
      {(viewMode === 'visits' || viewMode === 'combined') && (
        <StateVisitation stats={stateStats} />
      )}
      
      {/* Reward peak indicators */}
      {flowField.reward_peaks.map((peak, i) => (
        <group key={i} position={[peak.position[0], peak.position[1], 0]}>
          {/* Glowing cylinder at peak */}
          <mesh position={[0, 0, 0.3]}>
            <cylinderGeometry args={[0.3, 0.3, 0.6, 16]} />
            <meshStandardMaterial 
              color={COLORS.reward.high}
              transparent 
              opacity={0.7}
              emissive={COLORS.reward.high}
              emissiveIntensity={0.3}
            />
          </mesh>
          {/* Light effect */}
          <pointLight 
            position={[0, 0, 0.5]}
            color={COLORS.reward.high}
            intensity={peak.intensity / 20}
            distance={2}
          />
          {/* Label */}
          <Html position={[0, 0, 0.8]} distanceFactor={10}>
            <div className="text-xs bg-dark-panel/90 px-2 py-1 rounded-md pointer-events-none">
              <div className="text-white font-medium">{peak.name}</div>
              <div className="text-[10px] text-neon-green">R={peak.intensity}</div>
            </div>
          </Html>
        </group>
      ))}
    </group>
  )
}

export function GFlowNetFlowField() {
  const [viewMode, setViewMode] = useState<'flow' | 'visits' | 'combined'>('combined')
  
  const { data: stateStats } = useQuery({
    queryKey: ['state-statistics'],
    queryFn: async () => {
      const response = await api.training.getState()
      return response.data as StateStats
    },
  })
  
  return (
    <div className="relative w-full h-full flex">
      {/* Control Panel */}
      <div className="w-80 p-4 space-y-4 overflow-y-auto bg-dark-bg/50">
        {/* View Mode Controls */}
        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Map className="w-4 h-4 text-neon-purple" />
            Policy Analysis
          </h3>
          <div className="space-y-2">
            {[
              { key: 'flow', label: 'Flow Directions', icon: Zap },
              { key: 'visits', label: 'State Visits', icon: Activity },
              { key: 'combined', label: 'Combined View', icon: Target }
            ].map(({ key, label, icon: Icon }) => (
              <motion.button
                key={key}
                onClick={() => setViewMode(key as any)}
                className={`w-full p-3 rounded-lg flex items-center gap-3 transition-colors ${
                  viewMode === key
                    ? 'bg-neon-purple/20 border border-neon-purple/50 text-neon-purple'
                    : 'glass-dark hover:bg-dark-panel'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Icon className="w-4 h-4" />
                <span className="text-sm">{label}</span>
              </motion.button>
            ))}
          </div>
        </div>
        
        {/* Explanation Panel */}
        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-2 flex items-center space-x-2">
            <Info className="w-4 h-4 text-neon-purple" />
            <span>Understanding Flow Factors</span>
          </h3>
          <div className="text-xs text-muted-foreground space-y-2">
            <div className="p-2 bg-dark-panel/50 rounded-md">
              <p className="font-medium text-white mb-2">Understanding GFlowNet Policy Quality</p>
              
              <div className="space-y-2">
                <div>
                  <p className="text-neon-blue font-medium">What makes a good policy?</p>
                  <ul className="mt-1 space-y-1 text-[11px] ml-2">
                    <li>• <strong>Efficient paths:</strong> Arrows point directly toward rewards</li>
                    <li>• <strong>Multi-modal:</strong> Can reach multiple reward peaks</li>
                    <li>• <strong>Smooth flow:</strong> No chaotic or circular patterns</li>
                    <li>• <strong>Good coverage:</strong> Explores but focuses on rewards</li>
                  </ul>
                </div>
                
                <div>
                  <p className="text-neon-green font-medium">Visual indicators:</p>
                  <ul className="mt-1 space-y-1 text-[11px] ml-2">
                    <li>• <strong>Arrow direction:</strong> Where the policy moves from each state</li>
                    <li>• <strong>Arrow size:</strong> Confidence in that action (longer = more confident)</li>
                    <li>• <strong>Arrow color:</strong> Expected value (purple = high, gray = low)</li>
                    <li>• <strong>Background heat:</strong> Actual reward landscape being learned</li>
                  </ul>
                </div>
                
                <div>
                  <p className="text-neon-orange font-medium">Policy problems to spot:</p>
                  <ul className="mt-1 space-y-1 text-[11px] ml-2">
                    <li>• <strong>Stuck in local optima:</strong> All arrows point to one peak</li>
                    <li>• <strong>Poor exploration:</strong> Large unexplored areas</li>
                    <li>• <strong>Circular flows:</strong> Arrows form loops</li>
                    <li>• <strong>Weak confidence:</strong> Very short arrows everywhere</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Statistics Panel */}
        {stateStats && (
          <div className="glass-dark rounded-lg p-4">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-neon-blue" />
              Policy Statistics
            </h3>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="text-center p-2 bg-dark-panel/50 rounded-lg">
                  <div className="text-lg font-bold text-neon-green">
                    {(stateStats.coverage * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">Coverage</div>
                </div>
                <div className="text-center p-2 bg-dark-panel/50 rounded-lg">
                  <div className="text-lg font-bold text-neon-blue">
                    {stateStats.total_states_visited}
                  </div>
                  <div className="text-xs text-muted-foreground">States Visited</div>
                </div>
              </div>
              
              {stateStats.flow_statistics && (
                <>
                  <div className="pt-2 border-t border-dark-border space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Mean Flow:</span>
                      <span className="text-neon-purple">{stateStats.flow_statistics.mean_flow.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Policy Entropy:</span>
                      <span className="text-neon-orange">{stateStats.flow_statistics.policy_entropy.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Convergence:</span>
                      <span className="flex items-center gap-1">
                        <span className="text-neon-green">{(stateStats.flow_statistics.convergence_ratio * 100).toFixed(1)}%</span>
                        <TrendingUp className="w-3 h-3 text-neon-green" />
                      </span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
        
        {/* Legend */}
        <div className="glass-dark rounded-lg p-3">
          <h4 className="text-xs font-medium mb-2 flex items-center space-x-2">
            <BarChart3 className="w-4 h-4" />
            <span>Visual Guide</span>
          </h4>
          <div className="space-y-1 text-xs">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-gradient-to-r from-green-500 to-red-500 rounded"></div>
              <span>Reward Landscape</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-1 bg-gradient-to-r from-gray-500 to-purple-500"></div>
              <span>Flow Magnitude</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-6 bg-gradient-to-t from-transparent to-blue-500 rounded"></div>
              <span>Visit Frequency</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full border-2 border-yellow-500"></div>
              <span>State Values</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* 3D Canvas */}
      <div className="flex-1">
        <Canvas camera={{ position: [5, -8, 15], fov: 45 }}>
          <color attach="background" args={['#0A0A0B']} />
          <fog attach="fog" args={['#0A0A0B', 15, 50]} />
          
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 10, 10]} intensity={1.5} color={COLORS.primary.purple} />
          <pointLight position={[-5, -5, 8]} intensity={1.0} color={COLORS.reward.high} />
          <directionalLight position={[0, 15, 5]} intensity={0.4} />
          
          <Scene viewMode={viewMode} />
          
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            target={[5, 5, 0]}
            minDistance={10}
            maxDistance={40}
            minPolarAngle={0.2}
            maxPolarAngle={Math.PI / 2.5}
            autoRotate={false}
          />
          
          <EffectComposer>
            <Bloom
              intensity={1.2}
              luminanceThreshold={0.3}
              luminanceSmoothing={0.9}
            />
          </EffectComposer>
        </Canvas>
      </div>
    </div>
  )
}
