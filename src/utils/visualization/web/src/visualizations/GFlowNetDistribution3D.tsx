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
    const size = 256 // High resolution for smooth appearance
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!
    
    // Create image data for direct pixel manipulation
    const imageData = ctx.createImageData(size, size)
    const data = imageData.data
    
    // Build density map with gaussian smoothing
    const densityGrid = new Float32Array(size * size)
    let maxDensity = 0
    
    trajectories.forEach(traj => {
      traj.states.forEach((state, idx) => {
        // Map state to texture coordinates
        const tx = Math.floor((state[0] / 10) * size)
        const ty = Math.floor((1 - state[1] / 10) * size) // Flip Y
        
        // Apply gaussian kernel for smooth density
        const sigma = 8 // Spread of gaussian
        const kernelSize = 25
        
        for (let dx = -kernelSize; dx <= kernelSize; dx++) {
          for (let dy = -kernelSize; dy <= kernelSize; dy++) {
            const px = tx + dx
            const py = ty + dy
            
            if (px >= 0 && px < size && py >= 0 && py < size) {
              const dist2 = dx * dx + dy * dy
              const weight = Math.exp(-dist2 / (2 * sigma * sigma))
              const index = py * size + px
              densityGrid[index] += weight
              maxDensity = Math.max(maxDensity, densityGrid[index])
            }
          }
        }
      })
    })
    
    // Convert density to color
    for (let i = 0; i < size * size; i++) {
      const normalized = densityGrid[i] / maxDensity
      const intensity = Math.pow(normalized, 0.7) // Gamma correction
      
      // Color gradient: transparent -> blue -> purple -> yellow -> white
      let r, g, b, a
      
      if (intensity < 0.01) {
        r = 0; g = 0; b = 0; a = 0
      } else if (intensity < 0.25) {
        const t = intensity * 4
        r = 0; g = 0; b = t * 255
        a = t * 255
      } else if (intensity < 0.5) {
        const t = (intensity - 0.25) * 4
        r = t * 128; g = 0; b = 255
        a = 255
      } else if (intensity < 0.75) {
        const t = (intensity - 0.5) * 4
        r = 128 + t * 127; g = t * 128; b = 255 - t * 128
        a = 255
      } else {
        const t = (intensity - 0.75) * 4
        r = 255; g = 128 + t * 127; b = 128 + t * 127
        a = 255
      }
      
      data[i * 4] = r
      data[i * 4 + 1] = g
      data[i * 4 + 2] = b
      data[i * 4 + 3] = a
    }
    
    ctx.putImageData(imageData, 0, 0)
    return new THREE.CanvasTexture(canvas)
  }, [trajectories])
  
  return (
    <mesh position={[0, 0, 0.01]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[10, 10]} />
      <meshBasicMaterial 
        map={densityTexture} 
        transparent 
        opacity={0.9}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

// Trajectory paths with reward-based coloring
function TrajectoryPaths({ trajectories }: { trajectories: TrajectoryBundle['trajectories'] }) {
  const sortedTrajectories = useMemo(() => {
    return [...trajectories].sort((a, b) => b.total_reward - a.total_reward)
  }, [trajectories])
  
  return (
    <group>
      {sortedTrajectories.map((traj, i) => {
        const points = traj.states.map((state, j) => 
          new THREE.Vector3(state[0], state[1], j * 0.1)
        )
        
        // Color based on reward rank
        const color = new THREE.Color().setHSL(
          0.3 - (i / trajectories.length) * 0.3, // Green (high) to Red (low)
          1,
          0.5
        )
        
        return (
          <Line
            key={traj.id}
            points={points}
            color={color}
            lineWidth={2}
            opacity={0.3 + (1 - i / trajectories.length) * 0.4}
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
    
    // Create gradient for each peak
    peaks.forEach(peak => {
      const x = (peak.position[0] / 10) * size
      const y = (1 - peak.position[1] / 10) * size
      const radius = peak.intensity * 15
      
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius)
      gradient.addColorStop(0, `rgba(0, 255, 136, ${peak.intensity / 10})`)
      gradient.addColorStop(0.5, `rgba(0, 255, 136, ${peak.intensity / 20})`)
      gradient.addColorStop(1, 'rgba(0, 255, 136, 0)')
      
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, size, size)
    })
    
    return new THREE.CanvasTexture(canvas)
  }, [peaks])
  
  return (
    <group>
      {/* Reward heatmap on ground */}
      <mesh position={[0, 0, -0.01]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[10, 10]} />
        <meshBasicMaterial 
          map={rewardTexture}
          transparent
          opacity={0.6}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Peak markers */}
      {peaks.map((peak, i) => (
        <group key={i} position={[peak.position[0], peak.position[1], 0]}>
          <mesh position={[0, 0, 0.1]}>
            <sphereGeometry args={[0.2, 16, 16]} />
            <meshStandardMaterial 
              color={COLORS.reward.high}
              emissive={COLORS.reward.high}
              emissiveIntensity={0.5}
            />
          </mesh>
          <pointLight 
            position={[0, 0, 1]}
            color={COLORS.reward.high}
            intensity={peak.intensity / 10}
            distance={5}
          />
          <Html position={[0, 0, 1]} distanceFactor={10}>
            <div className="text-xs bg-dark-panel/90 px-2 py-1 rounded-md whitespace-nowrap pointer-events-none">
              <div className="text-white font-medium">{peak.name}</div>
              <div className="text-[10px] text-neon-green">Reward: {peak.intensity}</div>
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
    const gridSize = 30
    const density = new Float32Array(gridSize * gridSize)
    let maxDensity = 0
    
    trajectories.forEach(traj => {
      const endpoint = traj.states[traj.states.length - 1]
      const gx = Math.floor((endpoint[0] / 10) * gridSize)
      const gy = Math.floor((endpoint[1] / 10) * gridSize)
      
      // Gaussian smoothing
      for (let dx = -3; dx <= 3; dx++) {
        for (let dy = -3; dy <= 3; dy++) {
          const x = gx + dx
          const y = gy + dy
          if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
            const weight = Math.exp(-(dx*dx + dy*dy) / 4)
            const idx = y * gridSize + x
            density[idx] += weight * traj.total_reward
            maxDensity = Math.max(maxDensity, density[idx])
          }
        }
      }
    })
    
    // Generate contour lines at different levels
    const levels = [0.2, 0.4, 0.6, 0.8]
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
          const crosses = [
            (v0 < threshold && v1 >= threshold) || (v0 >= threshold && v1 < threshold),
            (v1 < threshold && v3 >= threshold) || (v1 >= threshold && v3 < threshold),
            (v3 < threshold && v2 >= threshold) || (v3 >= threshold && v2 < threshold),
            (v2 < threshold && v0 >= threshold) || (v2 >= threshold && v0 < threshold)
          ]
          
          if (crosses.some(c => c)) {
            const wx = (x + 0.5) / gridSize * 10
            const wy = (y + 0.5) / gridSize * 10
            points.push(new THREE.Vector3(wx, wy, 0.1 + levelIdx * 0.05))
          }
        }
      }
      
      if (points.length > 2) {
        const geometry = new THREE.BufferGeometry().setFromPoints(points)
        const color = new THREE.Color().setHSL(0.8 - level * 0.3, 1, 0.6)
        
        lines.push(
          <line key={levelIdx} geometry={geometry}>
            <lineBasicMaterial 
              color={color} 
              linewidth={2} 
              transparent 
              opacity={0.8}
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
      <Html position={[5, 8, 1]}>
        <div className="text-xs bg-dark-panel/90 px-3 py-2 rounded-md">
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
      {/* Grid helper on the ground plane */}
      <gridHelper args={[10, 10, '#2A2A2D', '#1A1A1D']} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]} />
      
      {/* Always show reward landscape as base layer */}
      <RewardLandscape peaks={data.reward_peaks} />
      
      {/* Conditional rendering based on view mode */}
      {(viewMode === 'density' || viewMode === 'combined') && (
        <>
          {/* Trajectory density heatmap on same plane */}
          <TrajectoryDensity trajectories={data.trajectories} />
          {/* Trajectory paths in 3D space above */}
          <TrajectoryPaths trajectories={data.trajectories} />
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
      
      {/* 3D Canvas - Optimized camera and lighting */}
      <div className="flex-1">
        <Canvas 
          camera={{ position: [15, 15, 15], fov: 50 }}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        >
          <color attach="background" args={['#0A0A0B']} />
          <fog attach="fog" args={['#0A0A0B', 20, 80]} />
          
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 15]} intensity={1.5} color={COLORS.primary.purple} />
          <pointLight position={[-10, -10, 15]} intensity={0.8} color={COLORS.reward.high} />
          <directionalLight position={[0, 20, 10]} intensity={0.5} />
          
          <Scene viewMode={viewMode} />
          
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            target={[0, 0, 3]}
            minDistance={10}
            maxDistance={50}
            minPolarAngle={0.2}
            maxPolarAngle={Math.PI / 2.2}
            autoRotate={true}
            autoRotateSpeed={0.5}
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