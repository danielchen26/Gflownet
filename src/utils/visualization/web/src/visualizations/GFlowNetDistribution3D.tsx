import { useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html, Line } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'
import { motion } from 'framer-motion'
import { BarChart3, Info, Layers, Eye, Target, Activity, MapPin, CheckCircle2, XCircle } from 'lucide-react'

// =============================================================================
// DYNAMIC COORDINATE SYSTEM
// =============================================================================
// World space: Fixed 10x10 units spanning [-5, +5] on X and Z axes
// State coordinates from API: [1, gridSize] for both dimensions
//
// COORDINATE MAPPING:
// - State [1, 1] → World (-5, 0, -5) [front-left corner]
// - State [gridSize, gridSize] → World (+5, 0, +5) [back-right corner]
// - State X increases → World X increases (left to right)
// - State Y increases → World Z increases (front to back)
//
// TEXTURE MAPPING (after rotateX(-Math.PI/2)):
// - UV (0, 0) → World (-5, 0, +5) [back-left]
// - UV (0, 1) → World (-5, 0, -5) [front-left]
// - UV (1, 0) → World (+5, 0, +5) [back-right]
// - UV (1, 1) → World (+5, 0, -5) [front-right]
// =============================================================================

const WORLD_SIZE = 10
const WORLD_HALF = 5

// Convert state coordinate [1, gridSize] to world coordinate [-5, 5]
function stateToWorld(state: number, gridSize: number): number {
  if (gridSize <= 1) return 0
  return ((state - 1) / (gridSize - 1)) * WORLD_SIZE - WORLD_HALF
}

// Convert state coordinate to texture UV [0, 1]
function stateToTextureUV(state: number, gridSize: number): number {
  if (gridSize <= 1) return 0.5
  return (state - 1) / (gridSize - 1)
}

// Convert cell index [0, gridSize-1] to world coordinate [-5, 5]
function cellToWorld(cellIndex: number, gridSize: number): number {
  if (gridSize <= 1) return 0
  return (cellIndex / (gridSize - 1)) * WORLD_SIZE - WORLD_HALF
}

interface TrajectoryData {
  id: string
  states: Array<[number, number]>
  rewards: number[]
  total_reward: number
}

interface RewardPeak {
  position: [number, number]
  intensity: number
  name: string
}

interface TrajectoryBundle {
  trajectories: TrajectoryData[]
  reward_peaks: RewardPeak[]
  gridSize: number
}

// =============================================================================
// DENSITY SURFACE - Smooth 3D terrain showing visitation frequency
// =============================================================================
function DensitySurface({ trajectories, gridSize }: { trajectories: TrajectoryData[], gridSize: number }) {
  const geometry = useMemo(() => {
    const resolution = 64
    const visitCounts = Array(resolution).fill(null).map(() => Array(resolution).fill(0))

    // Count visits with Gaussian smoothing
    trajectories.forEach(traj => {
      traj.states.forEach(state => {
        // Map state [1, gridSize] to normalized [0, 1]
        const normX = (state[0] - 1) / Math.max(1, gridSize - 1)
        const normY = (state[1] - 1) / Math.max(1, gridSize - 1)

        const centerX = normX * (resolution - 1)
        const centerY = normY * (resolution - 1)

        // Apply Gaussian kernel
        const sigma = resolution / 12
        const rewardWeight = Math.max(0.1, traj.total_reward / 10)

        for (let x = 0; x < resolution; x++) {
          for (let y = 0; y < resolution; y++) {
            const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)
            const weight = Math.exp(-(dist ** 2) / (2 * sigma ** 2)) * rewardWeight
            visitCounts[x][y] += weight
          }
        }
      })
    })

    // Find max for normalization
    let maxCount = 0
    for (let x = 0; x < resolution; x++) {
      for (let y = 0; y < resolution; y++) {
        maxCount = Math.max(maxCount, visitCounts[x][y])
      }
    }
    if (maxCount === 0) maxCount = 1

    // Create plane geometry
    const geo = new THREE.PlaneGeometry(WORLD_SIZE, WORLD_SIZE, resolution - 1, resolution - 1)
    geo.rotateX(-Math.PI / 2)

    const vertices = geo.attributes.position.array as Float32Array
    const colors = new Float32Array(vertices.length)

    // After rotateX(-Math.PI/2), the plane lies on XZ with:
    // - row 0 → highest local Y → world Z = -WORLD_HALF (front)
    // - row max → lowest local Y → world Z = +WORLD_HALF (back)
    // - col 0 → X = -WORLD_HALF (left)
    // - col max → X = +WORLD_HALF (right)
    //
    // For state [1,1] at world (-5, 0, -5) [front-left]:
    // - normX=0, normY=0 → visitCounts[0][0]
    // - We want this at col=0 (X=-5), row=0 (Z=-5)
    // - So dataX = col, dataY = row

    let vertexIndex = 0
    for (let row = 0; row < resolution; row++) {
      for (let col = 0; col < resolution; col++) {
        // FIXED: Direct mapping without Y-flip
        const dataX = col
        const dataY = row

        const height = visitCounts[dataX][dataY] / maxCount
        const smoothHeight = height * height * 2.5

        vertices[vertexIndex + 1] = smoothHeight

        // Color gradient
        const color = new THREE.Color()
        if (height < 0.2) {
          color.setRGB(0.05, 0.1, 0.4)
        } else if (height < 0.4) {
          color.lerpColors(new THREE.Color(0.05, 0.1, 0.4), new THREE.Color(0, 0.5, 0.6), (height - 0.2) * 5)
        } else if (height < 0.6) {
          color.lerpColors(new THREE.Color(0, 0.5, 0.6), new THREE.Color(0, 0.8, 0.3), (height - 0.4) * 5)
        } else if (height < 0.8) {
          color.lerpColors(new THREE.Color(0, 0.8, 0.3), new THREE.Color(0.9, 0.9, 0), (height - 0.6) * 5)
        } else {
          color.lerpColors(new THREE.Color(0.9, 0.9, 0), new THREE.Color(1, 0.5, 0), (height - 0.8) * 5)
        }

        colors[vertexIndex] = color.r
        colors[vertexIndex + 1] = color.g
        colors[vertexIndex + 2] = color.b

        vertexIndex += 3
      }
    }

    geo.computeVertexNormals()
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    return geo
  }, [trajectories, gridSize])

  return (
    <mesh geometry={geometry} position={[0, 0, 0]}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        transparent
        opacity={0.92}
        metalness={0.1}
        roughness={0.7}
      />
    </mesh>
  )
}

// =============================================================================
// DENSITY BARS - Discrete 3D bars per cell
// =============================================================================
function DensityBars({ trajectories, gridSize }: { trajectories: TrajectoryData[], gridSize: number }) {
  const densityData = useMemo(() => {
    const visitCounts = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0))
    let maxCount = 0

    trajectories.forEach(traj => {
      traj.states.forEach(state => {
        // State [1, gridSize] to index [0, gridSize-1]
        const cellX = Math.floor(Math.max(0, Math.min(gridSize - 1, state[0] - 1)))
        const cellY = Math.floor(Math.max(0, Math.min(gridSize - 1, state[1] - 1)))
        visitCounts[cellX][cellY]++
        maxCount = Math.max(maxCount, visitCounts[cellX][cellY])
      })
    })

    return { visitCounts, maxCount: maxCount || 1 }
  }, [trajectories, gridSize])

  return (
    <group>
      {densityData.visitCounts.map((row, cellX) =>
        row.map((count, cellY) => {
          if (count === 0) return null

          const height = (count / densityData.maxCount) * 3
          const intensity = count / densityData.maxCount

          // Use dynamic coordinate conversion
          const worldX = cellToWorld(cellX, gridSize)
          const worldZ = cellToWorld(cellY, gridSize)

          // Bar width based on grid size
          const barWidth = (WORLD_SIZE / gridSize) * 0.85

          const color = new THREE.Color()
          if (intensity < 0.33) {
            color.lerpColors(new THREE.Color(0x0044aa), new THREE.Color(0x00aacc), intensity * 3)
          } else if (intensity < 0.66) {
            color.lerpColors(new THREE.Color(0x00aacc), new THREE.Color(0x00cc44), (intensity - 0.33) * 3)
          } else {
            color.lerpColors(new THREE.Color(0x00cc44), new THREE.Color(0xdddd00), (intensity - 0.66) * 3)
          }

          return (
            <mesh key={`${cellX}-${cellY}`} position={[worldX, height / 2, worldZ]}>
              <boxGeometry args={[barWidth, height, barWidth]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.25}
                transparent
                opacity={0.85}
              />
            </mesh>
          )
        })
      )}
    </group>
  )
}

// =============================================================================
// TRAJECTORY DENSITY HEATMAP - Ground texture overlay
// =============================================================================
function TrajectoryDensity({ trajectories, gridSize }: { trajectories: TrajectoryData[], gridSize: number }) {
  const texture = useMemo(() => {
    const size = 256
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!

    ctx.clearRect(0, 0, size, size)

    const visitCounts = new Map<string, number>()
    trajectories.forEach(traj => {
      traj.states.forEach(state => {
        const cellX = Math.floor(Math.max(0, Math.min(gridSize - 1, state[0] - 1)))
        const cellY = Math.floor(Math.max(0, Math.min(gridSize - 1, state[1] - 1)))
        const key = `${cellX},${cellY}`
        visitCounts.set(key, (visitCounts.get(key) || 0) + 1)
      })
    })

    const maxVisits = Math.max(...visitCounts.values(), 1)

    visitCounts.forEach((count, key) => {
      const [cellX, cellY] = key.split(',').map(Number)

      // Cell to texture UV
      const uvX = cellX / Math.max(1, gridSize - 1)
      const uvY = cellY / Math.max(1, gridSize - 1)

      // FIXED: Canvas Y mapping to match world Z
      // UV (0,0) → world back-left, UV (0,1) → world front-left
      // Canvas top (Y=0) maps to UV Y=1 (front), Canvas bottom (Y=size) maps to UV Y=0 (back)
      // So: canvasY = (1 - uvY) * size... but wait, state Y=1 should be front (UV Y=1)
      // state Y=1 → cellY=0 → uvY=0 → we want canvas bottom (back)
      // Actually the issue is: state Y=1 should map to world Z=-5 (front)
      // After rotation, UV Y=1 maps to world Z=-5 (front)
      // So state Y=1 (cellY=0, uvY=0) should map to canvas position that gives UV Y=1
      // canvasY = 0 gives UV Y = 1 - 0/size = 1
      // So: canvasY = uvY * size (NOT (1-uvY)*size)
      const canvasX = uvX * size
      const canvasY = uvY * size

      const intensity = count / maxVisits
      const radius = Math.max(12, Math.min(40, intensity * 50))

      const gradient = ctx.createRadialGradient(canvasX, canvasY, 0, canvasX, canvasY, radius)

      if (intensity > 0.7) {
        gradient.addColorStop(0, `rgba(255, 255, 100, ${intensity * 0.8})`)
        gradient.addColorStop(0.5, `rgba(255, 180, 0, ${intensity * 0.5})`)
        gradient.addColorStop(1, 'rgba(255, 100, 0, 0)')
      } else if (intensity > 0.3) {
        gradient.addColorStop(0, `rgba(100, 200, 255, ${intensity * 0.8})`)
        gradient.addColorStop(0.5, `rgba(50, 150, 255, ${intensity * 0.5})`)
        gradient.addColorStop(1, 'rgba(0, 100, 200, 0)')
      } else {
        gradient.addColorStop(0, `rgba(100, 100, 255, ${intensity * 0.6})`)
        gradient.addColorStop(0.5, `rgba(50, 50, 200, ${intensity * 0.3})`)
        gradient.addColorStop(1, 'rgba(0, 0, 150, 0)')
      }

      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, size, size)
    })

    return new THREE.CanvasTexture(canvas)
  }, [trajectories, gridSize])

  return (
    <mesh position={[0, 0.015, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[WORLD_SIZE, WORLD_SIZE]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={0.9}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}

// =============================================================================
// TRAJECTORY PATHS - Line rendering of trajectories
// =============================================================================
function TrajectoryPaths({ trajectories, gridSize }: { trajectories: TrajectoryData[], gridSize: number }) {
  const sortedTrajectories = useMemo(() => {
    return [...trajectories].sort((a, b) => b.total_reward - a.total_reward).slice(0, 25)
  }, [trajectories])

  return (
    <group>
      {sortedTrajectories.map((traj, i) => {
        const points = traj.states.map((state) =>
          new THREE.Vector3(
            stateToWorld(state[0], gridSize),
            0.02,
            stateToWorld(state[1], gridSize)
          )
        )

        const normalizedRank = i / Math.max(sortedTrajectories.length - 1, 1)
        const color = new THREE.Color().lerpColors(
          new THREE.Color(0x00ff66),
          new THREE.Color(0x666666),
          normalizedRank
        )

        return (
          <Line
            key={traj.id}
            points={points}
            color={color}
            lineWidth={2.5 - normalizedRank * 1.5}
            opacity={0.8 - normalizedRank * 0.5}
            transparent
          />
        )
      })}
    </group>
  )
}

// =============================================================================
// REWARD PEAK MARKER - Diamond crystal with glow (replaces cylinder)
// =============================================================================
function RewardPeakMarker({ peak, gridSize }: { peak: RewardPeak, gridSize: number }) {
  const worldX = stateToWorld(peak.position[0], gridSize)
  const worldZ = stateToWorld(peak.position[1], gridSize)

  // Create diamond geometry (octahedron)
  const diamondGeometry = useMemo(() => {
    const geo = new THREE.OctahedronGeometry(0.35, 0)
    return geo
  }, [])

  return (
    <group position={[worldX, 0, worldZ]}>
      {/* Ground ring indicator */}
      <mesh position={[0, 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.4, 0.55, 32]} />
        <meshBasicMaterial
          color="#00ff88"
          transparent
          opacity={0.6}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Inner glow circle */}
      <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.4, 32]} />
        <meshBasicMaterial
          color="#00ff88"
          transparent
          opacity={0.15}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Diamond marker */}
      <mesh position={[0, 0.6, 0]} geometry={diamondGeometry}>
        <meshStandardMaterial
          color="#00ff88"
          emissive="#00ff88"
          emissiveIntensity={0.5}
          transparent
          opacity={0.9}
          metalness={0.3}
          roughness={0.2}
        />
      </mesh>

      {/* Point light for glow effect */}
      <pointLight
        position={[0, 0.6, 0]}
        color="#00ff88"
        intensity={peak.intensity / 8}
        distance={4}
        decay={2}
      />

      {/* Label */}
      <Html position={[0, 1.2, 0]} distanceFactor={10}>
        <div className="text-xs bg-dark-bg/95 px-2.5 py-1.5 rounded-md whitespace-nowrap pointer-events-none border border-emerald-500/50 shadow-lg shadow-emerald-500/20">
          <div className="text-emerald-400 font-semibold flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            {peak.name}
          </div>
          <div className="text-[10px] text-emerald-300/80 mt-0.5">
            Reward: {peak.intensity.toFixed(1)}
          </div>
          <div className="text-[9px] text-muted-foreground">
            [{peak.position[0]}, {peak.position[1]}]
          </div>
        </div>
      </Html>
    </group>
  )
}

// =============================================================================
// REWARD LANDSCAPE - Ground heatmap showing reward distribution
// =============================================================================
function RewardLandscape({ peaks, gridSize }: { peaks: RewardPeak[], gridSize: number }) {
  const texture = useMemo(() => {
    const size = 512
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!

    ctx.clearRect(0, 0, size, size)

    peaks.forEach(peak => {
      const uvX = stateToTextureUV(peak.position[0], gridSize)
      const uvY = stateToTextureUV(peak.position[1], gridSize)

      const x = uvX * size
      // FIXED: Same Y mapping fix as TrajectoryDensity
      const y = uvY * size

      const radius = peak.intensity * 25

      for (let layer = 0; layer < 3; layer++) {
        const r = radius * (1 - layer * 0.2)
        const alpha = (0.15 - layer * 0.04) * (peak.intensity / 10)

        const gradient = ctx.createRadialGradient(x, y, 0, x, y, r)
        gradient.addColorStop(0, `rgba(0, 255, 136, ${alpha})`)
        gradient.addColorStop(0.5, `rgba(0, 200, 100, ${alpha * 0.6})`)
        gradient.addColorStop(1, 'rgba(0, 100, 50, 0)')

        ctx.fillStyle = gradient
        ctx.fillRect(0, 0, size, size)
      }
    })

    return new THREE.CanvasTexture(canvas)
  }, [peaks, gridSize])

  return (
    <group>
      {/* Reward heatmap on ground */}
      <mesh position={[0, -0.005, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[WORLD_SIZE, WORLD_SIZE]} />
        <meshBasicMaterial
          map={texture}
          transparent
          opacity={0.7}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Peak markers */}
      {peaks.map((peak, i) => (
        <RewardPeakMarker key={i} peak={peak} gridSize={gridSize} />
      ))}
    </group>
  )
}

// =============================================================================
// POSTERIOR VISUALIZATION - Spheres showing endpoint distribution
// =============================================================================
function PosteriorVisualization({ trajectories, gridSize }: { trajectories: TrajectoryData[], gridSize: number }) {
  const sphereData = useMemo(() => {
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

    const totalTrajectories = trajectories.length || 1
    const spheres = Array.from(endpoints.values()).map(ep => ({
      position: ep.position,
      probability: ep.count / totalTrajectories,
      avgReward: ep.totalReward / ep.count,
      count: ep.count,
    }))

    spheres.sort((a, b) => b.probability - a.probability)

    return spheres
  }, [trajectories])

  return (
    <group>
      {sphereData.map((sphere, i) => {
        const worldX = stateToWorld(sphere.position[0], gridSize)
        const worldZ = stateToWorld(sphere.position[1], gridSize)

        const sphereRadius = Math.sqrt(sphere.probability) * 1.2
        const sphereY = 0.5 + sphereRadius

        const normalizedReward = Math.min(1, sphere.avgReward / 50)
        const color = new THREE.Color().lerpColors(
          new THREE.Color(0x6633ff),
          new THREE.Color(0xff44aa),
          normalizedReward
        )

        return (
          <group key={i} position={[worldX, sphereY, worldZ]}>
            <mesh>
              <sphereGeometry args={[sphereRadius * 0.4, 20, 20]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
                transparent
                opacity={0.85}
              />
            </mesh>

            {sphere.probability > 0.05 && (
              <Html position={[0, sphereRadius * 0.6, 0]} distanceFactor={10}>
                <div className="text-xs bg-dark-bg/95 px-2 py-1.5 rounded-md whitespace-nowrap pointer-events-none border border-purple-500/50 shadow-lg shadow-purple-500/20">
                  <div className="text-purple-300 font-semibold">
                    P = {(sphere.probability * 100).toFixed(0)}%
                  </div>
                  <div className="text-[10px] text-purple-200/70">
                    Avg R: {sphere.avgReward.toFixed(1)}
                  </div>
                  <div className="text-[9px] text-muted-foreground">
                    [{sphere.position[0]}, {sphere.position[1]}]
                  </div>
                </div>
              </Html>
            )}
          </group>
        )
      })}
    </group>
  )
}

// =============================================================================
// FLOATING LEGEND - Clear explanation of visual elements
// =============================================================================
function FloatingLegend({ viewMode }: { viewMode: 'density' | 'posterior' | 'combined' }) {
  return (
    <div className="absolute top-4 right-4 z-10">
      <div className="bg-dark-bg/95 backdrop-blur-sm rounded-lg p-3 border border-dark-border shadow-xl min-w-[180px]">
        <h4 className="text-xs font-semibold text-white mb-2 pb-1.5 border-b border-dark-border">
          Legend
        </h4>
        <div className="space-y-2 text-[11px]">
          {/* Reward Peaks */}
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 flex items-center justify-center">
              <div className="w-2.5 h-2.5 bg-emerald-400 rotate-45 shadow-sm shadow-emerald-400/50" />
            </div>
            <span className="text-emerald-400">Target Reward Peak</span>
          </div>

          {/* Density Surface/Bars */}
          {(viewMode === 'density' || viewMode === 'combined') && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-sm bg-gradient-to-br from-blue-500 via-cyan-400 to-yellow-400" />
              <span className="text-cyan-300">Trajectory Density</span>
            </div>
          )}

          {/* Trajectory Paths */}
          {(viewMode === 'density' || viewMode === 'combined') && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-gradient-to-r from-green-400 to-gray-500 rounded" />
              <span className="text-green-300">Trajectory Paths</span>
            </div>
          )}

          {/* Posterior Spheres */}
          {(viewMode === 'posterior' || viewMode === 'combined') && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 shadow-sm shadow-purple-500/50" />
              <span className="text-purple-300">Endpoint P(x|R)</span>
            </div>
          )}

          {/* Explanatory note */}
          <div className="pt-1.5 mt-1.5 border-t border-dark-border text-[10px] text-muted-foreground">
            {viewMode === 'density' && (
              <p>Shows where trajectories visit most frequently during exploration.</p>
            )}
            {viewMode === 'posterior' && (
              <p>Shows where trajectories terminate. Sphere size = probability.</p>
            )}
            {viewMode === 'combined' && (
              <p>Density surface shows path frequency; spheres show endpoint distribution.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// LOADING PLACEHOLDER
// =============================================================================
function SceneLoadingPlaceholder() {
  return (
    <group position={[0, 0, 0]}>
      <gridHelper args={[WORLD_SIZE, 10, '#3A3A3D', '#2A2A2D']} position={[0, 0, 0]} />
      <Html position={[0, 2, 0]} center>
        <div className="text-sm text-muted-foreground bg-dark-panel/80 px-4 py-2 rounded-lg">
          Loading distribution data...
        </div>
      </Html>
    </group>
  )
}

// =============================================================================
// MAIN 3D SCENE
// =============================================================================
function Scene({ viewMode, densityMode }: {
  viewMode: 'density' | 'posterior' | 'combined'
  densityMode: 'bars' | 'surface'
}) {
  const { data } = useQuery({
    queryKey: ['distribution-data'],
    queryFn: async () => {
      const trajs = await api.trajectories.getRecent(50)

      // Extract grid size from API response
      const gridSizeArr = trajs.domain?.grid_size
      const gridSize = Array.isArray(gridSizeArr) ? gridSizeArr[0] : (gridSizeArr || 10)

      return {
        trajectories: trajs.trajectories || [],
        reward_peaks: trajs.domain?.reward_peaks || [],
        gridSize: gridSize
      } as TrajectoryBundle
    },
    refetchInterval: 5000,
    staleTime: 4000,
  })

  if (!data) return <SceneLoadingPlaceholder />

  const { trajectories, reward_peaks, gridSize } = data

  return (
    <group position={[0, 0, 0]}>
      {/* Base grid */}
      <gridHelper args={[WORLD_SIZE, gridSize, '#3A3A3D', '#2A2A2D']} position={[0, 0, 0]} />

      {/* Reward landscape (always visible as reference) */}
      <RewardLandscape peaks={reward_peaks} gridSize={gridSize} />

      {/* Density visualization */}
      {(viewMode === 'density' || viewMode === 'combined') && (
        <>
          {densityMode === 'bars' ? (
            <>
              <DensityBars trajectories={trajectories} gridSize={gridSize} />
              <TrajectoryDensity trajectories={trajectories} gridSize={gridSize} />
            </>
          ) : (
            <DensitySurface trajectories={trajectories} gridSize={gridSize} />
          )}
          <TrajectoryPaths trajectories={trajectories} gridSize={gridSize} />
        </>
      )}

      {/* Posterior visualization */}
      {(viewMode === 'posterior' || viewMode === 'combined') && (
        <PosteriorVisualization trajectories={trajectories} gridSize={gridSize} />
      )}
    </group>
  )
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================
export function GFlowNetDistribution3D() {
  const [viewMode, setViewMode] = useState<'density' | 'posterior' | 'combined'>('combined')
  const [densityMode, setDensityMode] = useState<'bars' | 'surface'>('surface')

  const { data: stats } = useQuery({
    queryKey: ['distribution-stats'],
    queryFn: async () => {
      return await api.analysis.getDistribution()
    },
    refetchInterval: 5000,
    staleTime: 4000,
  })

  return (
    <div className="relative w-full h-full flex">
      {/* Control Panel */}
      <div className="w-80 p-4 space-y-4 overflow-y-auto bg-dark-bg/50">
        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Eye className="w-4 h-4 text-neon-purple" />
            Visualization Mode
          </h3>
          <div className="space-y-2">
            {[
              { key: 'density', label: 'Trajectory Density', icon: Activity, desc: 'Where paths visit' },
              { key: 'posterior', label: 'Posterior P(x|R)', icon: Target, desc: 'Where paths end' },
              { key: 'combined', label: 'Combined View', icon: Layers, desc: 'Full picture' }
            ].map(({ key, label, icon: Icon, desc }) => (
              <motion.button
                key={key}
                onClick={() => setViewMode(key as typeof viewMode)}
                className={`w-full p-3 rounded-lg flex items-center gap-3 transition-colors ${
                  viewMode === key
                    ? 'bg-neon-purple/20 border border-neon-purple/50 text-neon-purple'
                    : 'glass-dark hover:bg-dark-panel'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Icon className="w-4 h-4 flex-shrink-0" />
                <div className="text-left">
                  <span className="text-sm block">{label}</span>
                  <span className="text-[10px] text-muted-foreground">{desc}</span>
                </div>
              </motion.button>
            ))}
          </div>
        </div>

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
                  <div className="bg-cyan-500 rounded-sm" />
                  <div className="bg-green-500 rounded-sm" />
                  <div className="bg-yellow-500 rounded-sm" />
                </div>
                <span className="text-sm">Discrete Bars</span>
              </motion.button>
            </div>
          </div>
        )}

        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-2 flex items-center space-x-2">
            <Info className="w-4 h-4 text-neon-purple" />
            <span>Quick Info</span>
          </h3>

          {stats && (
            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Trajectories:</span>
                <span className="text-white font-medium">{stats.total_trajectories}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Unique Endpoints:</span>
                <span className="text-white font-medium">{stats.unique_endpoints}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Grid Size:</span>
                <span className="text-white font-medium">{stats.grid_size || 'N/A'}</span>
              </div>
            </div>
          )}
        </div>

        {/* Peak Discovery Section - with accurate nearby counting */}
        {stats?.reward_peaks && stats.reward_peaks.length > 0 && (
          <div className="glass-dark rounded-lg p-4">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-neon-green" />
              <span>Endpoint Distribution</span>
            </h3>

            {(() => {
              // Calculate visits near each peak (within 1 cell radius)
              const peakStats = stats.reward_peaks.map((peak: any, idx: number) => {
                const px = peak.position[0]
                const py = peak.position[1]
                let nearbyVisits = 0
                let exactVisits = 0

                // Count visits within 1 cell of peak
                if (stats.terminal_distribution) {
                  Object.entries(stats.terminal_distribution).forEach(([key, count]) => {
                    const [x, y] = key.split(',').map(Number)
                    const dist = Math.max(Math.abs(x - px), Math.abs(y - py)) // Chebyshev distance
                    if (dist === 0) {
                      exactVisits = count as number
                      nearbyVisits += count as number
                    } else if (dist <= 1) {
                      nearbyVisits += count as number
                    }
                  })
                }

                return {
                  peak,
                  name: peak.name || `Peak ${String.fromCharCode(65 + idx)}`,
                  exactVisits,
                  nearbyVisits,
                  isDiscovered: nearbyVisits > 0
                }
              })

              // Define type for peak stats
              type PeakStat = typeof peakStats[number]

              // Calculate total visits accounted for by peaks
              const totalNearPeaks = peakStats.reduce((sum: number, p: PeakStat) => sum + p.nearbyVisits, 0)
              const otherVisits = stats.total_trajectories - totalNearPeaks
              const discoveredCount = peakStats.filter((p: PeakStat) => p.isDiscovered).length

              // Calculate balance score (how evenly distributed among discovered peaks)
              const discoveredPeaks = peakStats.filter((p: PeakStat) => p.isDiscovered)
              let balanceScore = 0
              if (discoveredPeaks.length > 1 && totalNearPeaks > 0) {
                const expectedShare = 1 / discoveredPeaks.length
                const actualShares = discoveredPeaks.map((p: PeakStat) => p.nearbyVisits / totalNearPeaks)
                const maxDeviation = Math.max(...actualShares.map((s: number) => Math.abs(s - expectedShare)))
                balanceScore = Math.max(0, 1 - maxDeviation / expectedShare) * 100
              } else if (discoveredPeaks.length === 1) {
                balanceScore = 100 // Single peak is "balanced"
              }

              return (
                <>
                  {/* Discovery status badge */}
                  <div className="flex items-center gap-2 mb-3 text-xs">
                    <span className={`px-2 py-1 rounded-full ${
                      discoveredCount === stats.reward_peaks.length
                        ? 'bg-neon-green/20 text-neon-green'
                        : discoveredCount > 0
                        ? 'bg-yellow-500/20 text-yellow-500'
                        : 'bg-red-500/20 text-red-500'
                    }`}>
                      {discoveredCount}/{stats.reward_peaks.length} peaks found
                    </span>
                  </div>

                  <div className="space-y-2">
                    {peakStats.map((p: PeakStat, idx: number) => {
                      const percentage = stats.total_trajectories > 0
                        ? ((p.nearbyVisits / stats.total_trajectories) * 100).toFixed(1)
                        : '0.0'

                      return (
                        <div
                          key={idx}
                          className={`p-2 rounded-lg border ${
                            p.isDiscovered
                              ? 'border-neon-green/30 bg-neon-green/5'
                              : 'border-red-500/30 bg-red-500/5'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            {p.isDiscovered ? (
                              <CheckCircle2 className="w-4 h-4 text-neon-green flex-shrink-0" />
                            ) : (
                              <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                            )}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-medium truncate">{p.name}</span>
                                <span className={`text-xs font-bold ${
                                  p.isDiscovered ? 'text-neon-green' : 'text-red-500'
                                }`}>
                                  {percentage}%
                                </span>
                              </div>
                              <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                                <span>[{p.peak.position[0]}, {p.peak.position[1]}] · R={p.peak.intensity}</span>
                                <span>{p.nearbyVisits} visits</span>
                              </div>
                            </div>
                          </div>

                          {/* Progress bar */}
                          <div className="mt-1.5 h-1 bg-dark-panel rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                p.isDiscovered ? 'bg-neon-green' : 'bg-red-500/50'
                              }`}
                              style={{ width: `${Math.min(parseFloat(percentage), 100)}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}

                    {/* Other endpoints (not near any peak) */}
                    {otherVisits > 0 && (
                      <div className="p-2 rounded-lg border border-muted-foreground/30 bg-muted-foreground/5">
                        <div className="flex items-center gap-2">
                          <div className="w-4 h-4 rounded-full bg-muted-foreground/30 flex items-center justify-center">
                            <span className="text-[8px]">?</span>
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-medium text-muted-foreground">Other locations</span>
                              <span className="text-xs font-bold text-muted-foreground">
                                {((otherVisits / stats.total_trajectories) * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="text-[10px] text-muted-foreground">
                              {otherVisits} visits not near any peak
                            </div>
                          </div>
                        </div>
                        <div className="mt-1.5 h-1 bg-dark-panel rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-muted-foreground/50"
                            style={{ width: `${(otherVisits / stats.total_trajectories) * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Summary metrics */}
                  <div className="mt-3 pt-3 border-t border-dark-border/50 space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Peak Coverage:</span>
                      <span className={`font-bold ${
                        discoveredCount === stats.reward_peaks.length
                          ? 'text-neon-green'
                          : discoveredCount > 0
                          ? 'text-yellow-500'
                          : 'text-red-500'
                      }`}>
                        {((discoveredCount / stats.reward_peaks.length) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Near-Peak Ratio:</span>
                      <span className={`font-bold ${
                        totalNearPeaks / stats.total_trajectories >= 0.8
                          ? 'text-neon-green'
                          : totalNearPeaks / stats.total_trajectories >= 0.5
                          ? 'text-yellow-500'
                          : 'text-red-500'
                      }`}>
                        {((totalNearPeaks / stats.total_trajectories) * 100).toFixed(0)}%
                      </span>
                    </div>
                    {discoveredPeaks.length > 1 && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Balance Score:</span>
                        <span className={`font-bold ${
                          balanceScore >= 70
                            ? 'text-neon-green'
                            : balanceScore >= 40
                            ? 'text-yellow-500'
                            : 'text-red-500'
                        }`}>
                          {balanceScore.toFixed(0)}%
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Explanation tooltip */}
                  <div className="mt-2 text-[10px] text-muted-foreground/70 italic">
                    Counts trajectories ending within 1 cell of each peak
                  </div>
                </>
              )
            })()}
          </div>
        )}
      </div>

      {/* 3D Canvas */}
      <div className="flex-1 relative">
        {/* Floating Legend */}
        <FloatingLegend viewMode={viewMode} />

        <Canvas
          camera={{ position: [10, 10, 10], fov: 45 }}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        >
          <color attach="background" args={['#0A0A0B']} />
          <fog attach="fog" args={['#0A0A0B', 25, 60]} />

          <ambientLight intensity={0.4} />
          <pointLight position={[10, 15, 10]} intensity={1.5} color="#aa88ff" />
          <pointLight position={[-10, 10, -10]} intensity={0.8} color="#00ff88" />
          <directionalLight position={[0, 20, 10]} intensity={0.4} />

          <Scene viewMode={viewMode} densityMode={densityMode} />

          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            target={[0, 0, 0]}
            minDistance={8}
            maxDistance={30}
            minPolarAngle={0.1}
            maxPolarAngle={Math.PI / 2.2}
          />

          <EffectComposer>
            <Bloom
              intensity={1.2}
              luminanceThreshold={0.25}
              luminanceSmoothing={0.9}
            />
          </EffectComposer>
        </Canvas>
      </div>
    </div>
  )
}
