import { useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html, Line } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'
import { motion } from 'framer-motion'
import { Info, Map, BarChart3, Activity, Target, Zap, TrendingUp } from 'lucide-react'

// =============================================================================
// DYNAMIC COORDINATE SYSTEM (matches GFlowNetDistribution3D)
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
// - Canvas Y=0 (top) maps to UV Y=1, which maps to world Z=-5 (front)
// - Canvas Y=size (bottom) maps to UV Y=0, which maps to world Z=+5 (back)
// - So: canvasY = uvY * size (NOT (1-uvY)*size)
// =============================================================================

const WORLD_SIZE = 10
const WORLD_HALF = 5

function stateToWorld(state: number, gridSize: number): number {
  if (gridSize <= 1) return 0
  return ((state - 1) / (gridSize - 1)) * WORLD_SIZE - WORLD_HALF
}

function stateToTextureUV(state: number, gridSize: number): number {
  if (gridSize <= 1) return 0.5
  return (state - 1) / (gridSize - 1)
}

interface FlowFieldData {
  resolution: [number, number, number]
  bounds: { x: [number, number]; y: [number, number] }
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
  gridSize: number
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

// =============================================================================
// REWARD HEATMAP - Ground texture showing reward distribution
// =============================================================================
function RewardHeatmap({ flowData }: { flowData: FlowFieldData }) {
  const texture = useMemo(() => {
    const size = 512
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!

    ctx.fillStyle = '#050510'
    ctx.fillRect(0, 0, size, size)

    flowData.reward_peaks.forEach(peak => {
      const uvX = stateToTextureUV(peak.position[0], flowData.gridSize)
      const uvY = stateToTextureUV(peak.position[1], flowData.gridSize)

      const x = uvX * size
      // FIXED: Correct canvas Y mapping to match world Z
      const y = uvY * size
      const radius = peak.intensity * 30

      for (let layer = 0; layer < 4; layer++) {
        const r = radius * (1 - layer * 0.15)
        const alpha = (0.25 - layer * 0.05) * (peak.intensity / 12)

        const gradient = ctx.createRadialGradient(x, y, 0, x, y, r)
        gradient.addColorStop(0, `rgba(0, 255, 136, ${alpha})`)
        gradient.addColorStop(0.4, `rgba(0, 220, 180, ${alpha * 0.7})`)
        gradient.addColorStop(0.7, `rgba(0, 150, 200, ${alpha * 0.4})`)
        gradient.addColorStop(1, 'rgba(0, 50, 100, 0)')

        ctx.fillStyle = gradient
        ctx.fillRect(0, 0, size, size)
      }
    })

    return new THREE.CanvasTexture(canvas)
  }, [flowData])

  return (
    <mesh position={[0, -0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[WORLD_SIZE, WORLD_SIZE]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={0.7}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}

// =============================================================================
// FLOW ARROWS - Policy direction vectors
// =============================================================================
function FlowArrows({ flowData }: { flowData: FlowFieldData }) {
  const arrows = useMemo(() => {
    return flowData.data
      .filter(point => point.magnitude > 0.01 || point.flow_value > 0.05)
      .map((point, i) => {
        const worldX = stateToWorld(point.position[0], flowData.gridSize)
        const worldZ = stateToWorld(point.position[1], flowData.gridSize)

        const dir = new THREE.Vector2(point.velocity[0], point.velocity[1])
        const dirLength = dir.length()

        const flowNorm = Math.max(0, Math.min(1, point.flow_value))
        const rewardNorm = Math.max(0, Math.min(1, point.reward / 10))
        const arrowLength = Math.max(0.15, Math.min(0.7, dirLength * 0.8 + flowNorm * 0.3))

        const color = new THREE.Color()
        if (flowNorm < 0.25) {
          color.lerpColors(new THREE.Color(0x444455), new THREE.Color(0x4488aa), flowNorm * 4)
        } else if (flowNorm < 0.5) {
          color.lerpColors(new THREE.Color(0x4488aa), new THREE.Color(0x6666dd), (flowNorm - 0.25) * 4)
        } else if (flowNorm < 0.75) {
          color.lerpColors(new THREE.Color(0x6666dd), new THREE.Color(0x9944dd), (flowNorm - 0.5) * 4)
        } else {
          color.lerpColors(new THREE.Color(0x9944dd), new THREE.Color(0xdd44aa), (flowNorm - 0.75) * 4)
        }

        const angle = dirLength > 0.001 ? Math.atan2(dir.y, dir.x) : 0

        return {
          key: i,
          position: [worldX, 0.02, worldZ] as [number, number, number],
          angle,
          length: arrowLength,
          color,
          opacity: 0.6 + flowNorm * 0.4,
          flowValue: flowNorm,
          rewardValue: rewardNorm,
          hasDirection: dirLength > 0.001,
        }
      })
  }, [flowData])

  return (
    <group>
      {arrows.map(arrow => (
        <group key={arrow.key} position={arrow.position}>
          {arrow.hasDirection ? (
            <group rotation={[0, -arrow.angle + Math.PI / 2, 0]}>
              <mesh position={[0, 0, arrow.length * 0.35]} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[0.03, 0.04, arrow.length * 0.6, 8]} />
                <meshBasicMaterial color={arrow.color} transparent opacity={arrow.opacity} />
              </mesh>
              <mesh position={[0, 0, arrow.length * 0.7]} rotation={[Math.PI / 2, 0, 0]}>
                <coneGeometry args={[0.1, 0.18, 6]} />
                <meshBasicMaterial color={arrow.color} transparent opacity={arrow.opacity} />
              </mesh>
            </group>
          ) : (
            <mesh>
              <sphereGeometry args={[0.06, 8, 8]} />
              <meshBasicMaterial color={arrow.color} transparent opacity={arrow.opacity * 0.5} />
            </mesh>
          )}

          {arrow.rewardValue > 0.3 && (
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
              <ringGeometry args={[0.12, 0.12 + arrow.rewardValue * 0.1, 16]} />
              <meshBasicMaterial
                color={new THREE.Color().lerpColors(
                  new THREE.Color(0x00aa44),
                  new THREE.Color(0x00ff88),
                  arrow.rewardValue
                )}
                transparent
                opacity={0.4}
                side={THREE.DoubleSide}
              />
            </mesh>
          )}
        </group>
      ))}
    </group>
  )
}

// =============================================================================
// TRAJECTORY PATHS - Recent trajectory lines
// =============================================================================
function TrajectoryPaths({ gridSize }: { gridSize: number }) {
  const { data: trajectories } = useQuery({
    queryKey: ['recent-trajectories-flow'],
    queryFn: async () => {
      const response = await api.trajectories.getRecent(20)
      return response.trajectories || []
    },
    staleTime: 2000,
    refetchInterval: 3000,
  })

  if (!trajectories || trajectories.length === 0) return <group />

  return (
    <group>
      {trajectories.slice(0, 15).map((traj: any, i: number) => {
        const points = traj.states.map((state: [number, number]) =>
          new THREE.Vector3(
            stateToWorld(state[0], gridSize),
            0.03 + i * 0.002,
            stateToWorld(state[1], gridSize)
          )
        )

        const recency = 1 - (i / 15)
        const rewardNorm = Math.min(1, (traj.total_reward || 0) / 80)

        const color = new THREE.Color().lerpColors(
          new THREE.Color(0x444466),
          new THREE.Color(0x00ddff),
          recency * 0.6 + rewardNorm * 0.4
        )

        return (
          <Line
            key={traj.id}
            points={points}
            color={color}
            lineWidth={1.5 + recency}
            opacity={0.25 + recency * 0.5}
            transparent
          />
        )
      })}
    </group>
  )
}

// =============================================================================
// STATE VISITATION - Heatmap of visited states
// =============================================================================
function StateVisitation({ stats, gridSize }: { stats: StateStats; gridSize: number }) {
  const texture = useMemo(() => {
    const size = 256
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!

    ctx.clearRect(0, 0, size, size)

    const maxVisits = stats.max_visits || 1
    Object.entries(stats.visitation_counts).forEach(([key, count]) => {
      const [stateX, stateY] = key.split(',').map(Number)

      const uvX = stateToTextureUV(stateX, gridSize)
      const uvY = stateToTextureUV(stateY, gridSize)

      const x = uvX * size
      // FIXED: Correct canvas Y mapping to match world Z
      const y = uvY * size
      const normalized = count / maxVisits

      const radius = 15
      for (let dx = -radius; dx <= radius; dx++) {
        for (let dy = -radius; dy <= radius; dy++) {
          const px = Math.floor(x + dx)
          const py = Math.floor(y + dy)
          if (px >= 0 && px < size && py >= 0 && py < size) {
            const dist2 = dx * dx + dy * dy
            const weight = Math.exp(-dist2 / (2 * 6 * 6)) * normalized

            ctx.fillStyle = `rgba(50, 150, 255, ${weight * 0.5})`
            ctx.fillRect(px, py, 1, 1)
          }
        }
      }
    })

    return new THREE.CanvasTexture(canvas)
  }, [stats, gridSize])

  return (
    <mesh position={[0, 0.005, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[WORLD_SIZE, WORLD_SIZE]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </mesh>
  )
}

// =============================================================================
// REWARD PEAK MARKER - Downward-pointing arrow (location pin style)
// =============================================================================
function RewardPeakMarker({ peak, gridSize }: { peak: FlowFieldData['reward_peaks'][0], gridSize: number }) {
  const worldX = stateToWorld(peak.position[0], gridSize)
  const worldZ = stateToWorld(peak.position[1], gridSize)

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

      {/* Arrow stem (thin cylinder going up) */}
      <mesh position={[0, 0.7, 0]}>
        <cylinderGeometry args={[0.04, 0.04, 0.8, 8]} />
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

      {/* Arrow head (cone pointing DOWN toward the target) */}
      <mesh position={[0, 0.25, 0]} rotation={[Math.PI, 0, 0]}>
        <coneGeometry args={[0.2, 0.5, 16]} />
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
      <Html position={[0, 1.3, 0]} distanceFactor={10}>
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
// REWARD PEAKS - Collection of peak markers
// =============================================================================
function RewardPeaks({ peaks, gridSize }: { peaks: FlowFieldData['reward_peaks']; gridSize: number }) {
  return (
    <group>
      {peaks.map((peak, i) => (
        <RewardPeakMarker key={i} peak={peak} gridSize={gridSize} />
      ))}
    </group>
  )
}

// =============================================================================
// FLOATING LEGEND - Clear explanation of visual elements
// =============================================================================
function FloatingLegend({ viewMode }: { viewMode: 'flow' | 'visits' | 'combined' }) {
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

          {/* Flow Arrows */}
          {(viewMode === 'flow' || viewMode === 'combined') && (
            <>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 flex items-center justify-center">
                  <div className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-b-[8px] border-b-purple-400" />
                </div>
                <span className="text-purple-300">Policy Direction</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-gradient-to-r from-gray-500 to-cyan-400 rounded" />
                <span className="text-cyan-300">Trajectory Paths</span>
              </div>
            </>
          )}

          {/* State Visits */}
          {(viewMode === 'visits' || viewMode === 'combined') && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-sm bg-gradient-to-br from-blue-800 to-blue-400 opacity-70" />
              <span className="text-blue-300">State Visitation</span>
            </div>
          )}

          {/* Explanatory note */}
          <div className="pt-1.5 mt-1.5 border-t border-dark-border text-[10px] text-muted-foreground">
            {viewMode === 'flow' && (
              <p>Arrows show learned policy direction. Brighter = higher flow value.</p>
            )}
            {viewMode === 'visits' && (
              <p>Blue intensity shows how often each state is visited during training.</p>
            )}
            {viewMode === 'combined' && (
              <p>Policy flows toward reward peaks; blue shows exploration coverage.</p>
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
      <gridHelper args={[WORLD_SIZE, 10, '#2A2A2D', '#1A1A1D']} position={[0, 0, 0]} />
      <Html position={[0, 2, 0]} center>
        <div className="text-sm text-muted-foreground bg-dark-panel/80 px-4 py-2 rounded-lg">
          Loading policy data...
        </div>
      </Html>
    </group>
  )
}

// =============================================================================
// MAIN SCENE
// =============================================================================
function Scene({ viewMode }: { viewMode: 'flow' | 'visits' | 'combined' }) {
  const { data: flowField } = useQuery({
    queryKey: ['flow-field'],
    queryFn: async () => {
      const data = await api.analysis.getFlowField()
      const trajs = await api.trajectories.getRecent(1)

      // Get grid size from trajectories domain
      const gridSizeArr = trajs.domain?.grid_size
      const gridSize = Array.isArray(gridSizeArr) ? gridSizeArr[0] : (gridSizeArr || 10)

      return {
        ...data,
        gridSize,
        reward_peaks: trajs.domain?.reward_peaks || data.reward_peaks || []
      } as FlowFieldData
    },
    staleTime: 2000,
    refetchInterval: 3000,
  })

  const { data: stateStats } = useQuery({
    queryKey: ['state-statistics'],
    queryFn: async () => {
      const data = await api.training.getState()
      return data as StateStats
    },
    staleTime: 2000,
    refetchInterval: 3000,
  })

  if (!flowField || !stateStats) {
    return <SceneLoadingPlaceholder />
  }

  const { gridSize } = flowField

  return (
    <group position={[0, 0, 0]}>
      <gridHelper args={[WORLD_SIZE, gridSize, '#2A2A2D', '#1A1A1D']} position={[0, 0, 0]} />

      <RewardHeatmap flowData={flowField} />
      <RewardPeaks peaks={flowField.reward_peaks} gridSize={gridSize} />

      {(viewMode === 'flow' || viewMode === 'combined') && (
        <>
          <FlowArrows flowData={flowField} />
          <TrajectoryPaths gridSize={gridSize} />
        </>
      )}

      {(viewMode === 'visits' || viewMode === 'combined') && (
        <StateVisitation stats={stateStats} gridSize={gridSize} />
      )}
    </group>
  )
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================
export function GFlowNetFlowField() {
  const [viewMode, setViewMode] = useState<'flow' | 'visits' | 'combined'>('combined')

  const { data: stateStats } = useQuery({
    queryKey: ['state-statistics'],
    queryFn: async () => {
      const data = await api.training.getState()
      return data as StateStats
    },
    staleTime: 2000,
    refetchInterval: 3000,
  })

  return (
    <div className="relative w-full h-full flex">
      <div className="w-80 p-4 space-y-4 overflow-y-auto bg-dark-bg/50">
        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Map className="w-4 h-4 text-neon-purple" />
            Policy Analysis
          </h3>
          <div className="space-y-2">
            {[
              { key: 'flow', label: 'Flow Directions', icon: Zap, desc: 'Policy action vectors' },
              { key: 'visits', label: 'State Visits', icon: Activity, desc: 'Visitation frequency' },
              { key: 'combined', label: 'Combined View', icon: Target, desc: 'All visualizations' }
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

        <div className="glass-dark rounded-lg p-4">
          <h3 className="text-sm font-medium mb-2 flex items-center space-x-2">
            <Info className="w-4 h-4 text-neon-purple" />
            <span>Policy Quality Guide</span>
          </h3>
          <div className="text-xs text-muted-foreground space-y-2">
            <div className="space-y-1">
              <p className="text-neon-blue font-medium">Good Policy:</p>
              <ul className="ml-2 space-y-0.5">
                <li>• Arrows point toward reward peaks</li>
                <li>• Multiple paths to different peaks</li>
                <li>• Smooth flow patterns</li>
              </ul>
            </div>
            <div className="space-y-1">
              <p className="text-neon-orange font-medium">Problems:</p>
              <ul className="ml-2 space-y-0.5">
                <li>• All arrows to one peak (mode collapse)</li>
                <li>• Circular or chaotic patterns</li>
                <li>• Very short arrows (low confidence)</li>
              </ul>
            </div>
          </div>
        </div>

        {stateStats && (
          <div className="glass-dark rounded-lg p-4">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-neon-blue" />
              Statistics
            </h3>
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
                <div className="text-xs text-muted-foreground">States</div>
              </div>
            </div>

            {stateStats.flow_statistics && (
              <div className="mt-3 pt-2 border-t border-dark-border space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Mean Flow:</span>
                  <span className="text-neon-purple font-medium">
                    {stateStats.flow_statistics.mean_flow.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Convergence:</span>
                  <span className="flex items-center gap-1">
                    <span className="text-neon-green font-medium">
                      {(stateStats.flow_statistics.convergence_ratio * 100).toFixed(1)}%
                    </span>
                    <TrendingUp className="w-3 h-3 text-neon-green" />
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex-1 relative">
        {/* Floating Legend */}
        <FloatingLegend viewMode={viewMode} />

        <Canvas
          camera={{ position: [0, 15, 12], fov: 45 }}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        >
          <color attach="background" args={['#0A0A0B']} />
          <fog attach="fog" args={['#0A0A0B', 20, 50]} />

          <ambientLight intensity={0.5} />
          <pointLight position={[10, 15, 10]} intensity={1.2} color="#aa88ff" />
          <pointLight position={[-10, 10, -10]} intensity={0.7} color="#00ff88" />
          <directionalLight position={[0, 20, 5]} intensity={0.4} />

          <Scene viewMode={viewMode} />

          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            target={[0, 0, 0]}
            minDistance={10}
            maxDistance={35}
            minPolarAngle={0.2}
            maxPolarAngle={Math.PI / 2.2}
          />

          <EffectComposer>
            <Bloom
              intensity={1.0}
              luminanceThreshold={0.3}
              luminanceSmoothing={0.9}
            />
          </EffectComposer>
        </Canvas>
      </div>
    </div>
  )
}
