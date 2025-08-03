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

// Smooth density surface visualization
function DensitySurface({ trajectories }: { trajectories: TrajectoryBundle['trajectories'] }) {
  const { geometry, colorAttribute } = useMemo(() => {
    const gridSize = 32 // Higher resolution for smoother surface
    const visitCounts = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0))
    
    // Count visits with Gaussian smoothing
    trajectories.forEach(traj => {
      traj.states.forEach(state => {
        // Map from [1,10] to grid coordinates
        const centerX = (state[0] - 1) / 9 * (gridSize - 1)
        const centerY = (state[1] - 1) / 9 * (gridSize - 1)
        
        // Apply Gaussian kernel with reward weighting
        const sigma = 2.0
        const rewardWeight = Math.max(0.1, traj.total_reward / 10)
        
        for (let x = 0; x < gridSize; x++) {
          for (let y = 0; y < gridSize; y++) {
            const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)
            const weight = Math.exp(-(dist ** 2) / (2 * sigma ** 2)) * rewardWeight
            visitCounts[x][y] += weight
          }
        }
      })
    })
    
    // Apply additional smoothing pass
    const smoothedCounts = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0))
    for (let x = 1; x < gridSize - 1; x++) {
      for (let y = 1; y < gridSize - 1; y++) {
        smoothedCounts[x][y] = (
          visitCounts[x][y] * 0.4 +
          visitCounts[x-1][y] * 0.15 +
          visitCounts[x+1][y] * 0.15 +
          visitCounts[x][y-1] * 0.15 +
          visitCounts[x][y+1] * 0.15
        )
      }
    }
    
    // Create plane geometry with proper orientation
    const geometry = new THREE.PlaneGeometry(10, 10, gridSize - 1, gridSize - 1)
    geometry.rotateX(-Math.PI / 2) // Rotate to horizontal
    
    const vertices = geometry.attributes.position.array as Float32Array
    const colors = new Float32Array(vertices.length)
    
    // Find max for normalization
    let maxCount = 0
    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        maxCount = Math.max(maxCount, smoothedCounts[x][y])
      }
    }
    
    // Set vertex heights and colors
    let vertexIndex = 0
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const height = smoothedCounts[x][y] / maxCount
        const smoothHeight = height * height // Square for more dramatic effect
        
        vertices[vertexIndex + 1] = smoothHeight * 2.5 // Y is up
        
        // Color based on height
        const color = new THREE.Color()
        if (height < 0.2) {
          color.setRGB(0, 0.2, 0.8) // Deep blue
        } else if (height < 0.4) {
          color.lerpColors(new THREE.Color(0, 0.2, 0.8), new THREE.Color(0, 0.8, 0.8), (height - 0.2) * 5)
        } else if (height < 0.6) {
          color.lerpColors(new THREE.Color(0, 0.8, 0.8), new THREE.Color(0, 1, 0.4), (height - 0.4) * 5)
        } else if (height < 0.8) {
          color.lerpColors(new THREE.Color(0, 1, 0.4), new THREE.Color(1, 1, 0), (height - 0.6) * 5)
        } else {
          color.lerpColors(new THREE.Color(1, 1, 0), new THREE.Color(1, 0.5, 0), (height - 0.8) * 5)
        }
        
        colors[vertexIndex] = color.r
        colors[vertexIndex + 1] = color.g
        colors[vertexIndex + 2] = color.b
        
        vertexIndex += 3
      }
    }
    
    geometry.computeVertexNormals()
    const colorAttribute = new THREE.BufferAttribute(colors, 3)
    geometry.setAttribute('color', colorAttribute)
    
    return { geometry, colorAttribute }
  }, [trajectories])
  
  return (
    <mesh geometry={geometry} position={[0, 0, 0]}>
      <meshStandardMaterial
        vertexColors
        wireframe={false}
        side={THREE.DoubleSide}
        transparent
        opacity={0.9}
        metalness={0.2}
        roughness={0.6}
        emissive={new THREE.Color(0x001144)}
        emissiveIntensity={0.05}
      />
    </mesh>
  )
}

// 3D Density bars visualization
function DensityBars({ trajectories }: { trajectories: TrajectoryBundle['trajectories'] }) {
  const densityData = useMemo(() => {
    const gridSize = 10 // 10x10 grid to match coordinate system
    const visitCounts = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0))
    const maxCount = { value: 0 }
    
    // Count visits per grid cell
    trajectories.forEach(traj => {
      traj.states.forEach(state => {
        // Map from [1,10] to [0,9] grid indices
        const gridX = Math.floor(Math.max(0, Math.min(9, state[0] - 1)))
        const gridY = Math.floor(Math.max(0, Math.min(9, state[1] - 1)))
        visitCounts[gridX][gridY]++
        maxCount.value = Math.max(maxCount.value, visitCounts[gridX][gridY])
      })
    })
    
    return { visitCounts, maxCount: maxCount.value }
  }, [trajectories])
  
  return (
    <group>
      {densityData.visitCounts.map((row, x) => 
        row.map((count, y) => {
          if (count === 0) return null
          
          const height = (count / densityData.maxCount) * 3 // Max height of 3 units
          const intensity = count / densityData.maxCount
          
          // Color gradient from blue (low) to yellow (high)
          const color = new THREE.Color()
          if (intensity < 0.5) {
            color.lerpColors(new THREE.Color(0x0066ff), new THREE.Color(0x00ff88), intensity * 2)
          } else {
            color.lerpColors(new THREE.Color(0x00ff88), new THREE.Color(0xffff00), (intensity - 0.5) * 2)
          }
          
          return (
            <mesh key={`${x}-${y}`} position={[x - 4.5, height / 2, y - 4.5]}>
              <boxGeometry args={[0.8, height, 0.8]} />
              <meshStandardMaterial 
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
                transparent
                opacity={0.8}
              />
            </mesh>
          )
        })
      )}
    </group>
  )
}

// Smooth density heatmap visualization (for texture overlay)
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
    <mesh position={[0, 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
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
    return [...trajectories].sort((a, b) => b.total_reward - a.total_reward).slice(0, 20) // Show top 20
  }, [trajectories])
  
  return (
    <group>
      {sortedTrajectories.map((traj, i) => {
        // Map coordinates from [1,10] to [-5,5] for centering
        const points = traj.states.map((state) => 
          new THREE.Vector3(state[0] - 5.5, 0.01, state[1] - 5.5) // Y=0 for ground level
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
      <mesh position={[0, -0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[10, 10]} />
        <meshBasicMaterial 
          map={rewardTexture}
          transparent
          opacity={0.6}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Peak markers - adjust position to be within [1,10] range */}
      {peaks.map((peak, i) => (
        <group key={i} position={[peak.position[0] - 5.5, 0, peak.position[1] - 5.5]}>
          <mesh position={[0, 0.3, 0]}>
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
            position={[0, 0.5, 0]}
            color={COLORS.reward.high}
            intensity={peak.intensity / 20}
            distance={2}
          />
          <Html position={[0, 0.8, 0]} distanceFactor={10}>
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

// Posterior probability as 3D spheres
function PosteriorVisualization({ trajectories }: { trajectories: TrajectoryBundle['trajectories'] }) {
  const sphereData = useMemo(() => {
    // Calculate endpoint density and rewards
    const endpoints = new Map<string, { count: number; totalReward: number; position: [number, number] }>()
    
    trajectories.forEach(traj => {
      if (traj.states.length > 0) {
        const endpoint = traj.states[traj.states.length - 1]
        const key = `${endpoint[0]},${endpoint[1]}`
        
        if (!endpoints.has(key)) {
          endpoints.set(key, { count: 0, totalReward: 0, position: endpoint })
        }
        
        const data = endpoints.get(key)!
        data.count++
        data.totalReward += traj.total_reward
      }
    })
    
    // Convert to array and calculate probabilities
    const spheres = Array.from(endpoints.values()).map(ep => ({
      position: ep.position,
      probability: ep.count / trajectories.length,
      avgReward: ep.totalReward / ep.count,
      size: Math.sqrt(ep.count / trajectories.length) * 2, // Size based on probability
    }))
    
    // Sort by size for better rendering (larger spheres first)
    spheres.sort((a, b) => b.size - a.size)
    
    return spheres
  }, [trajectories])
  
  return (
    <group>
      {sphereData.map((sphere, i) => {
        // Color based on average reward
        const color = new THREE.Color()
        const normalizedReward = Math.min(1, sphere.avgReward / 10)
        color.lerpColors(new THREE.Color(0x6600ff), new THREE.Color(0xff00ff), normalizedReward)
        
        return (
          <group key={i} position={[sphere.position[0] - 5.5, 0.5 + sphere.size / 2, sphere.position[1] - 5.5]}>
            <mesh>
              <sphereGeometry args={[sphere.size * 0.3, 16, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
                transparent
                opacity={0.7}
                metalness={0.4}
                roughness={0.2}
              />
            </mesh>
            {/* Add glow effect for larger spheres */}
            {sphere.size > 0.5 && (
              <pointLight
                color={color}
                intensity={sphere.size * 0.5}
                distance={sphere.size * 2}
              />
            )}
            {/* Label for significant endpoints */}
            {sphere.probability > 0.05 && (
              <Html position={[0, sphere.size * 0.4, 0]} distanceFactor={10}>
                <div className="text-xs bg-dark-panel/90 px-2 py-1 rounded-md whitespace-nowrap pointer-events-none">
                  <div className="text-white font-medium">P={sphere.probability.toFixed(2)}</div>
                  <div className="text-[10px] text-purple-400">R̄={sphere.avgReward.toFixed(1)}</div>
                </div>
              </Html>
            )}
          </group>
        )
      })}
    </group>
  )
}


// Main scene with conditional rendering
function Scene({ viewMode, densityMode }: { viewMode: 'density' | 'posterior' | 'combined', densityMode: 'bars' | 'surface' }) {
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
      {/* Grid on horizontal plane */}
      <gridHelper args={[10, 10, '#3A3A3D', '#2A2A2D']} position={[0, 0, 0]} />
      
      {/* Always show reward landscape as base layer */}
      <RewardLandscape peaks={data.reward_peaks} />
      
      {/* Conditional rendering based on view mode */}
      {(viewMode === 'density' || viewMode === 'combined') && (
        <>
          {/* 3D Density visualization - bars or surface based on mode */}
          {densityMode === 'bars' ? (
            <DensityBars trajectories={data.trajectories} />
          ) : (
            <DensitySurface trajectories={data.trajectories} />
          )}
          {/* Trajectory paths on XY plane */}
          <TrajectoryPaths trajectories={data.trajectories} />
          {/* Trajectory density heatmap overlaid - only for bars mode */}
          {densityMode === 'bars' && <TrajectoryDensity trajectories={data.trajectories} />}
        </>
      )}
      
      {(viewMode === 'posterior' || viewMode === 'combined') && (
        <>
          {/* Posterior visualization - 3D spheres */}
          <PosteriorVisualization trajectories={data.trajectories} />
        </>
      )}
    </group>
  )
}

export function GFlowNetDistribution3D() {
  const [viewMode, setViewMode] = useState<'density' | 'posterior' | 'combined'>('combined')
  const [densityMode, setDensityMode] = useState<'bars' | 'surface'>('surface')
  
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
        
        {/* Density Display Mode - Only show when density view is active */}
        {(viewMode === 'density' || viewMode === 'combined') && (
          <div className="glass-dark rounded-lg p-4">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-neon-blue" />
              Density Display
            </h3>
            <div className="space-y-2">
              <motion.button
                onClick={() => setDensityMode('surface')}
                className={`w-full p-3 rounded-lg flex items-center gap-3 transition-colors ${
                  densityMode === 'surface'
                    ? 'bg-neon-blue/20 border border-neon-blue/50 text-neon-blue'
                    : 'glass-dark hover:bg-dark-panel'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="w-4 h-4 rounded bg-gradient-to-br from-blue-500 via-green-500 to-yellow-500" />
                <span className="text-sm">Smooth Surface</span>
              </motion.button>
              <motion.button
                onClick={() => setDensityMode('bars')}
                className={`w-full p-3 rounded-lg flex items-center gap-3 transition-colors ${
                  densityMode === 'bars'
                    ? 'bg-neon-blue/20 border border-neon-blue/50 text-neon-blue'
                    : 'glass-dark hover:bg-dark-panel'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="w-4 h-4 grid grid-cols-2 gap-0.5">
                  <div className="bg-blue-500 rounded-sm" />
                  <div className="bg-green-500 rounded-sm" />
                  <div className="bg-yellow-500 rounded-sm" />
                  <div className="bg-red-500 rounded-sm" />
                </div>
                <span className="text-sm">Discrete Bars</span>
              </motion.button>
            </div>
          </div>
        )}
        
        {/* Info Panel */}
        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-2 flex items-center space-x-2">
            <Info className="w-4 h-4 text-neon-purple" />
            <span>About This View</span>
          </h3>
          <div className="text-xs text-muted-foreground space-y-2">
            {viewMode === 'density' && (
              <>
                {densityMode === 'surface' ? (
                  <>
                    <p><strong>3D surface:</strong> Smooth density landscape showing state visitation frequency.</p>
                    <p><strong>Surface height:</strong> Taller regions indicate states visited more often.</p>
                    <p><strong>Color gradient:</strong> Blue (low) → Cyan → Green → Yellow (high frequency)</p>
                  </>
                ) : (
                  <>
                    <p><strong>3D bars:</strong> Discrete state visitation frequency per grid cell.</p>
                    <p><strong>Bar height:</strong> Taller bars indicate higher visitation counts.</p>
                    <p><strong>Bar colors:</strong> Blue (low) → Green → Yellow (high frequency)</p>
                  </>
                )}
                <p><strong>Path colors:</strong> Trajectory quality - green paths have higher rewards, red paths have lower rewards.</p>
              </>
            )}
            {viewMode === 'posterior' && (
              <>
                <p><strong>3D spheres:</strong> Posterior distribution P(x|R) showing trajectory endpoints.</p>
                <p><strong>Sphere size:</strong> Proportional to endpoint probability.</p>
                <p><strong>Sphere color:</strong> Purple (low reward) → Pink (high reward)</p>
                <p><strong>Labels:</strong> Show probability P and average reward R̄ for significant endpoints.</p>
              </>
            )}
            {viewMode === 'combined' && (
              <>
                <p><strong>Multi-layered view:</strong> Shows density, trajectory paths, and posterior distribution.</p>
                {densityMode === 'surface' ? (
                  <p><strong>3D surface:</strong> Smooth density landscape with color gradient.</p>
                ) : (
                  <p><strong>3D bars:</strong> Discrete state visitation frequency.</p>
                )}
                <p><strong>Green cylinders:</strong> Reward peaks that attract trajectories.</p>
                <p><strong>3D spheres:</strong> Posterior probability distribution at endpoints.</p>
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
              <div className="w-3 h-6 bg-gradient-to-t from-blue-500 via-green-500 to-yellow-500 rounded"></div>
              <span>Density Bars</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-12 h-0.5 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"></div>
              <span>Trajectory Reward</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-neon-green"></div>
              <span>Reward Peaks</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* 3D Canvas - Better camera angle for horizontal plane viewing */}
      <div className="flex-1">
        <Canvas 
          camera={{ position: [8, 8, 8], fov: 50 }}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        >
          <color attach="background" args={['#0A0A0B']} />
          <fog attach="fog" args={['#0A0A0B', 20, 50]} />
          
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 15]} intensity={1.5} color={COLORS.primary.purple} />
          <pointLight position={[-10, -10, 15]} intensity={0.8} color={COLORS.reward.high} />
          <directionalLight position={[0, 20, 10]} intensity={0.5} />
          
          <Scene viewMode={viewMode} densityMode={densityMode} />
          
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