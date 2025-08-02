import { useRef, useMemo, useState, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Html, Line } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useQuery } from '@tanstack/react-query'
import axios from '../lib/axios'
import { motion } from 'framer-motion'
import { BarChart3, Info, Layers, Eye, Target, Activity } from 'lucide-react'
import { COLORS, interpolateRewardColor } from '../utils/colors'

interface TrajectoryBundle {
  trajectories: Array<{
    id: string
    states: Array<[number, number]>
    rewards: number[]
    total_reward: number
  }>
  reward_peaks: Array<{
    position: [number, number]
    intensity: number
    name: string
  }>
}

// Smooth density heatmap visualization
function TrajectoryDensity({ trajectories }: { trajectories: TrajectoryBundle['trajectories'] }) {
  const densityTexture = useMemo(() => {
    const size = 128 // Reduced for better performance
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!
    
    // Clear canvas
    ctx.fillStyle = 'rgba(0,0,0,0)'
    ctx.fillRect(0, 0, size, size)
    
    // Count visits per grid cell
    const gridSize = 20 // 20x20 grid
    const visitCounts = new Map<string, number>()
    
    trajectories.forEach(traj => {
      traj.states.forEach(state => {
        // Ensure state is within bounds [1, 10]
        if (state[0] >= 1 && state[0] <= 10 && state[1] >= 1 && state[1] <= 10) {
          const gridX = Math.floor((state[0] - 1) / 9 * (gridSize - 1))
          const gridY = Math.floor((state[1] - 1) / 9 * (gridSize - 1))
          const key = `${gridX},${gridY}`
          visitCounts.set(key, (visitCounts.get(key) || 0) + 1)
        }
      })
    })
    
    // Find max visits for normalization
    const maxVisits = Math.max(...visitCounts.values(), 1)
    
    // Draw density spots
    visitCounts.forEach((count, key) => {
      const [gridX, gridY] = key.split(',').map(Number)
      const x = (gridX / (gridSize - 1)) * size
      const y = size - (gridY / (gridSize - 1)) * size // Flip Y
      
      const intensity = count / maxVisits
      const radius = Math.max(8, Math.min(20, intensity * 30))
      
      // Draw radial gradient
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius)
      
      if (intensity > 0.7) {
        // Hot spots - yellow/white
        gradient.addColorStop(0, `rgba(255, 255, 200, ${intensity})`)
        gradient.addColorStop(0.5, `rgba(255, 200, 0, ${intensity * 0.7})`)
        gradient.addColorStop(1, 'rgba(255, 100, 0, 0)')
      } else if (intensity > 0.3) {
        // Medium - purple/blue
        gradient.addColorStop(0, `rgba(200, 100, 255, ${intensity})`)
        gradient.addColorStop(0.5, `rgba(100, 50, 255, ${intensity * 0.7})`)
        gradient.addColorStop(1, 'rgba(50, 0, 200, 0)')
      } else {
        // Low - blue
        gradient.addColorStop(0, `rgba(0, 150, 255, ${intensity})`)
        gradient.addColorStop(0.5, `rgba(0, 100, 200, ${intensity * 0.5})`)
        gradient.addColorStop(1, 'rgba(0, 50, 150, 0)')
      }
      
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, size, size)
    })
    
    return new THREE.CanvasTexture(canvas)
  }, [trajectories])
  
  return (
    <mesh position={[0, 0, 0.02]} rotation={[0, 0, 0]}>
      <planeGeometry args={[10, 10]} />
      <meshBasicMaterial 
        map={densityTexture} 
        transparent 
        opacity={0.9}
      />
    </mesh>
  )
}

// Trajectory paths with reward-based coloring
function TrajectoryPaths({ trajectories }: { trajectories: TrajectoryBundle['trajectories'] }) {
  const sortedTrajectories = useMemo(() => {
    return [...trajectories].sort((a, b) => b.total_reward - a.total_reward).slice(0, 20) // Show top 20
  }, [trajectories])
  
  return (
    <group>
      {sortedTrajectories.map((traj, i) => {
        // Map coordinates from [1,10] to [-5,5] for centering
        const points = traj.states.map((state) => 
          new THREE.Vector3(state[0] - 5.5, state[1] - 5.5, 0.05)
        )
        
        // Color gradient from high reward (green) to low reward (red)
        const normalizedRank = i / Math.max(sortedTrajectories.length - 1, 1)
        const color = new THREE.Color().lerpColors(
          new THREE.Color(0x00ff88), // Green for high reward
          new THREE.Color(0xff0066), // Red for low reward
          normalizedRank
        )
        
        return (
          <Line
            key={traj.id}
            points={points}
            color={color}
            lineWidth={2}
            opacity={0.7 - normalizedRank * 0.4}
            transparent
          />
        )
      })}
    </group>
  )
}

// Reward landscape as smooth field
function RewardLandscape({ peaks }: { peaks: TrajectoryBundle['reward_peaks'] }) {
  const rewardTexture = useMemo(() => {
    const size = 256
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!
    
    // Clear with dark background
    ctx.fillStyle = 'rgba(0, 0, 0, 0)'
    ctx.fillRect(0, 0, size, size)
    
    // Create gradient for each peak
    peaks.forEach(peak => {
      // Map from [1,10] range to canvas coordinates
      const x = ((peak.position[0] - 1) / 9) * size
      const y = size - ((peak.position[1] - 1) / 9) * size // Flip Y
      const radius = peak.intensity * 20
      
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius)
      gradient.addColorStop(0, `rgba(0, 255, 136, ${peak.intensity / 15})`)
      gradient.addColorStop(0.5, `rgba(0, 200, 100, ${peak.intensity / 30})`)
      gradient.addColorStop(1, 'rgba(0, 100, 50, 0)')
      
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, size, size)
    })
    
    return new THREE.CanvasTexture(canvas)
  }, [peaks])
  
  return (
    <group>
      {/* Reward heatmap on ground */}
      <mesh position={[0, 0, -0.01]} rotation={[0, 0, 0]}>
        <planeGeometry args={[10, 10]} />
        <meshBasicMaterial 
          map={rewardTexture}
          transparent
          opacity={0.6}
        />
      </mesh>
      
      {/* Peak markers - adjust position to be within [1,10] range */}
      {peaks.map((peak, i) => (
        <group key={i} position={[peak.position[0] - 5.5, peak.position[1] - 5.5, 0]}>
          <mesh position={[0, 0, 0.3]}>
            <cylinderGeometry args={[0.2, 0.2, 0.6, 16]} />
            <meshStandardMaterial 
              color={COLORS.reward.high}
              emissive={COLORS.reward.high}
              emissiveIntensity={0.2}
              transparent
              opacity={0.7}
            />
          </mesh>
          <pointLight 
            position={[0, 0, 0.5]}
            color={COLORS.reward.high}
            intensity={peak.intensity / 20}
            distance={2}
          />
          <Html position={[0, 0, 0.8]} distanceFactor={10}>
            <div className="text-xs bg-dark-panel/90 px-2 py-1 rounded-md whitespace-nowrap pointer-events-none">
              <div className="text-white font-medium">{peak.name}</div>
              <div className="text-[10px] text-neon-green">R={peak.intensity}</div>
            </div>
          </Html>
        </group>
      ))}
    </group>
  )
}

// Posterior probability as contour plot
function PosteriorVisualization({ trajectories }: { trajectories: TrajectoryBundle['trajectories'] }) {
  const contourLines = useMemo(() => {
    // Calculate endpoint density
    const gridSize = 20
    const density = new Float32Array(gridSize * gridSize)
    let maxDensity = 0
    
    trajectories.forEach(traj => {
      const endpoint = traj.states[traj.states.length - 1]
      // Map from [1,10] to grid coordinates
      if (endpoint[0] >= 1 && endpoint[0] <= 10 && endpoint[1] >= 1 && endpoint[1] <= 10) {
        const gx = Math.floor((endpoint[0] - 1) / 9 * (gridSize - 1))
        const gy = Math.floor((endpoint[1] - 1) / 9 * (gridSize - 1))
        
        // Gaussian smoothing with smaller kernel
        for (let dx = -2; dx <= 2; dx++) {
          for (let dy = -2; dy <= 2; dy++) {
            const x = gx + dx
            const y = gy + dy
            if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
              const weight = Math.exp(-(dx*dx + dy*dy) / 2)
              const idx = y * gridSize + x
              density[idx] += weight * traj.total_reward
              maxDensity = Math.max(maxDensity, density[idx])
            }
          }
        }
      }
    })
    
    // Generate contour lines at different levels
    const levels = [0.3, 0.5, 0.7, 0.9]
    const lines: JSX.Element[] = []
    
    levels.forEach((level, levelIdx) => {
      const threshold = level * maxDensity
      const points: THREE.Vector3[] = []
      
      // Simple contour following
      for (let y = 0; y < gridSize - 1; y++) {
        for (let x = 0; x < gridSize - 1; x++) {
          const idx = y * gridSize + x
          const v0 = density[idx]
          const v1 = density[idx + 1]
          const v2 = density[idx + gridSize]
          const v3 = density[idx + gridSize + 1]
          
          // Check if contour crosses this cell
          if (v0 > 0 || v1 > 0 || v2 > 0 || v3 > 0) {
            const avg = (v0 + v1 + v2 + v3) / 4
            if (Math.abs(avg - threshold) < threshold * 0.2) {
              // Map back to world coordinates centered at origin
              const wx = (x / (gridSize - 1)) * 10 - 5
              const wy = (y / (gridSize - 1)) * 10 - 5
              points.push(new THREE.Vector3(wx, wy, 0.1 + levelIdx * 0.02))
            }
          }
        }
      }
      
      if (points.length > 5) {
        const geometry = new THREE.BufferGeometry().setFromPoints(points)
        const color = new THREE.Color().setHSL(0.8 - level * 0.3, 1, 0.6)
        
        lines.push(
          <line key={levelIdx} geometry={geometry}>
            <lineBasicMaterial 
              color={color} 
              linewidth={2} 
              transparent 
              opacity={0.6}
            />
          </line>
        )
      }
    })
    
    return lines
  }, [trajectories])
  
  return (
    <group>
      {contourLines}
      <Html position={[3, 3, 1]}>
        <div className="text-xs bg-dark-panel/90 px-3 py-2 rounded-md pointer-events-none">
          <div className="font-medium mb-1">Posterior P(x|R)</div>
          <div className="text-[10px] space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-purple-600"></div>
              <span>Low probability</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-yellow-500"></div>
              <span>High probability</span>
            </div>
          </div>
        </div>
      </Html>
    </group>
  )
}

// Main scene with conditional rendering
function Scene({ viewMode }: { viewMode: 'density' | 'posterior' | 'combined' }) {
  const { data } = useQuery({
    queryKey: ['all-trajectories'],
    queryFn: async () => {
      const response = await axios.get('/api/trajectories/all')
      return response.data as TrajectoryBundle
    },
    refetchInterval: 10000,
  })
  
  if (!data) return null
  
  return (
    <group position={[0, 0, 0]}>
      {/* Grid on XY plane - centered at origin */}
      <gridHelper args={[10, 10, '#3A3A3D', '#2A2A2D']} position={[0, 0, 0]} rotation={[0, 0, 0]} />
      
      {/* Always show reward landscape as base layer */}
      <RewardLandscape peaks={data.reward_peaks} />
      
      {/* Conditional rendering based on view mode */}
      {(viewMode === 'density' || viewMode === 'combined') && (
        <>
          {/* Trajectory paths on XY plane */}
          <TrajectoryPaths trajectories={data.trajectories} />
          {/* Trajectory density heatmap overlaid */}
          <TrajectoryDensity trajectories={data.trajectories} />
        </>
      )}
      
      {(viewMode === 'posterior' || viewMode === 'combined') && (
        <>
          {/* Posterior visualization */}
          <PosteriorVisualization trajectories={data.trajectories} />
        </>
      )}
    </group>
  )
}

export function GFlowNetDistribution3D() {
  const [viewMode, setViewMode] = useState<'density' | 'posterior' | 'combined'>('combined')
  
  const { data: stats } = useQuery({
    queryKey: ['distribution-stats'],
    queryFn: async () => {
      const response = await axios.get('/api/analysis/distribution')
      return response.data
    },
  })
  
  return (
    <div className="relative w-full h-full flex">
      {/* Control Panel */}
      <div className="w-80 p-4 space-y-4 overflow-y-auto bg-dark-bg/50">
        {/* View Mode Controls */}
        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Eye className="w-4 h-4 text-neon-purple" />
            Visualization Mode
          </h3>
          <div className="space-y-2">
            {[
              { key: 'density', label: 'Trajectory Density', icon: Activity },
              { key: 'posterior', label: 'Posterior P(x|R)', icon: Target },
              { key: 'combined', label: 'Combined View', icon: Layers }
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
        
        {/* Info Panel */}
        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-2 flex items-center space-x-2">
            <Info className="w-4 h-4 text-neon-purple" />
            <span>About This View</span>
          </h3>
          <div className="text-xs text-muted-foreground space-y-2">
            {viewMode === 'density' && (
              <>
                <p><strong>Height bars:</strong> State visitation frequency - taller bars indicate states visited more often during sampling.</p>
                <p><strong>Path colors:</strong> Trajectory quality - green paths have higher rewards, red paths have lower rewards.</p>
              </>
            )}
            {viewMode === 'posterior' && (
              <>
                <p><strong>Purple spheres:</strong> Posterior distribution P(x|R) showing where high-reward trajectories tend to terminate.</p>
                <p><strong>Size indicates:</strong> Combined probability and reward - larger spheres represent both frequent and high-reward endpoints.</p>
              </>
            )}
            {viewMode === 'combined' && (
              <>
                <p><strong>Multi-layered view:</strong> Shows both trajectory density and posterior distribution.</p>
                <p><strong>Green cylinders:</strong> Reward peaks that attract trajectories.</p>
                <p><strong>Purple spheres:</strong> High-value endpoints weighted by both frequency and reward.</p>
              </>
            )}
          </div>
          
          {stats && (
            <div className="mt-3 pt-3 border-t border-dark-border space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Trajectories:</span>
                <span>{stats.total_trajectories}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Unique Endpoints:</span>
                <span>{stats.unique_endpoints}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mode Diversity:</span>
                <span className="text-neon-green">{stats.diversity_score?.toFixed(2)}</span>
              </div>
            </div>
          )}
        </div>
        
        {/* Legend */}
        <div className="glass-dark rounded-lg p-3">
          <h4 className="text-xs font-medium mb-2 flex items-center space-x-2">
            <BarChart3 className="w-4 h-4" />
            <span>Visual Elements</span>
          </h4>
          <div className="space-y-1 text-xs">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-6 bg-gradient-to-t from-red-500 to-yellow-500 rounded"></div>
              <span>Visitation Density</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-12 h-0.5 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"></div>
              <span>Reward Quality</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-neon-purple"></div>
              <span>Posterior P(x|R)</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* 3D Canvas - Better camera angle for XY plane viewing */}
      <div className="flex-1">
        <Canvas 
          camera={{ position: [0, -12, 10], fov: 50 }}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        >
          <color attach="background" args={['#0A0A0B']} />
          <fog attach="fog" args={['#0A0A0B', 20, 50]} />
          
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 15]} intensity={1.5} color={COLORS.primary.purple} />
          <pointLight position={[-10, -10, 15]} intensity={0.8} color={COLORS.reward.high} />
          <directionalLight position={[0, 20, 10]} intensity={0.5} />
          
          <Scene viewMode={viewMode} />
          
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            target={[0, 0, 0]}
            minDistance={8}
            maxDistance={25}
            minPolarAngle={0.1}
            maxPolarAngle={Math.PI / 2.5}
            autoRotate={false}
          />
          
          <EffectComposer>
            <Bloom
              intensity={1.5}
              luminanceThreshold={0.2}
              luminanceSmoothing={0.9}
            />
          </EffectComposer>
        </Canvas>
      </div>
    </div>
  )
}