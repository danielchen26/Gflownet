import { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Html, Line } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useQuery } from '@tanstack/react-query'
import axios from '../lib/axios'
import { motion } from 'framer-motion'
import { Play, Pause, SkipForward, Info } from 'lucide-react'

interface RewardPeak {
  position: [number, number]
  intensity: number
  name: string
}

interface Trajectory {
  id: string
  trajectory_id: number
  states: Array<{
    position: [number, number, number]
    grid_position: [number, number]
    is_terminal: boolean
  }>
  actions: Array<{ type: string }>
  rewards: number[]
  flows: number[]
  total_reward: number
  start_position: [number, number]
  end_position: [number, number]
}

interface TrajectoriesResponse {
  trajectories: Trajectory[]
  grid_size?: [number, number]
}

interface FlowFieldResponse {
  reward_peaks: RewardPeak[]
}

// Grid floor to show the state space
function GridFloor() {
  return (
    <group>
      {/* Grid lines */}
      {Array.from({ length: 11 }, (_, i) => (
        <group key={`grid-${i}`}>
          <Line
            points={[[0, i, 0], [10, i, 0]]}
            color="#2A2A2D"
            lineWidth={1}
          />
          <Line
            points={[[i, 0, 0], [i, 10, 0]]}
            color="#2A2A2D"
            lineWidth={1}
          />
        </group>
      ))}
      
      {/* Grid labels */}
      {[0, 5, 10].map(i => (
        <group key={`label-${i}`}>
          <Text
            position={[i, -0.5, 0]}
            fontSize={0.3}
            color="#666"
            anchorX="center"
          >
            {i}
          </Text>
          <Text
            position={[-0.5, i, 0]}
            fontSize={0.3}
            color="#666"
            anchorX="center"
          >
            {i}
          </Text>
        </group>
      ))}
    </group>
  )
}

// Reward peaks visualization
function RewardPeaks({ peaks }: { peaks: RewardPeak[] }) {
  return (
    <>
      {peaks.map((peak, i) => (
        <group key={i} position={[peak.position[0] - 1, peak.position[1] - 1, 0]}>
          {/* Glowing cylinder for reward */}
          <mesh>
            <cylinderGeometry args={[peak.intensity / 10, peak.intensity / 10, 0.1, 32]} />
            <meshBasicMaterial 
              color={i === 0 ? '#00FF88' : '#00D9FF'} 
              transparent 
              opacity={0.3} 
            />
          </mesh>
          
          {/* Glow effect */}
          <mesh scale={[1.5, 1.5, 1]}>
            <cylinderGeometry args={[peak.intensity / 10, peak.intensity / 10, 0.05, 32]} />
            <meshBasicMaterial 
              color={i === 0 ? '#00FF88' : '#00D9FF'} 
              transparent 
              opacity={0.1} 
            />
          </mesh>
          
          {/* Label */}
          <Html distanceFactor={10}>
            <div className="text-xs bg-dark-panel/80 px-2 py-1 rounded-md whitespace-nowrap">
              {peak.name}
              <div className="text-[10px] text-muted-foreground">
                Reward: {peak.intensity}
              </div>
            </div>
          </Html>
        </group>
      ))}
    </>
  )
}

// Single trajectory visualization
function TrajectoryPath({ trajectory, isActive, color, onClick }: { 
  trajectory: Trajectory
  isActive: boolean
  color: string
  onClick: () => void
}) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const [progress, setProgress] = useState(0)
  
  useFrame((state, delta) => {
    if (isActive) {
      setProgress((p) => (p + delta * 0.2) % 1)
    }
  })
  
  const points = useMemo(() => {
    return trajectory.states.map(s => 
      new THREE.Vector3(s.grid_position[0] - 1, s.grid_position[1] - 1, s.position[2])
    )
  }, [trajectory])
  
  const curve = useMemo(() => {
    return new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5)
  }, [points])
  
  // Color based on rewards with clearer gradient
  const colors = useMemo(() => {
    const maxReward = Math.max(...trajectory.rewards, 1)
    return trajectory.rewards.map(r => {
      const normalized = r / maxReward
      // Blue (low) -> Yellow (medium) -> Red (high)
      if (normalized < 0.5) {
        return new THREE.Color(0, 0, 1).lerp(new THREE.Color(1, 1, 0), normalized * 2)
      } else {
        return new THREE.Color(1, 1, 0).lerp(new THREE.Color(1, 0, 0), (normalized - 0.5) * 2)
      }
    })
  }, [trajectory])
  
  return (
    <group onClick={onClick}>
      {/* Trajectory line segments with color gradient */}
      {points.slice(0, -1).map((point, i) => {
        const nextPoint = points[i + 1]
        return (
          <Line
            key={i}
            points={[point, nextPoint]}
            color={colors[i]}
            lineWidth={isActive ? 4 : 2}
            opacity={isActive ? 1 : 0.3}
            transparent
          />
        )
      })}
      
      {/* State nodes */}
      {trajectory.states.map((state, i) => (
        <group key={i} position={[state.grid_position[0] - 1, state.grid_position[1] - 1, state.position[2]]}>
          <mesh>
            <sphereGeometry args={[isActive ? 0.15 : 0.08, 16, 16]} />
            <meshBasicMaterial color={colors[i]} />
          </mesh>
          
          {/* Show reward value for active trajectory */}
          {isActive && (
            <Html distanceFactor={10}>
              <div className="text-xs bg-dark-panel/90 px-1 rounded pointer-events-none">
                {trajectory.rewards[i].toFixed(1)}
              </div>
            </Html>
          )}
          
          {/* Start/End markers */}
          {i === 0 && (
            <>
              <mesh scale={[2, 2, 2]}>
                <sphereGeometry args={[0.15, 16, 16]} />
                <meshBasicMaterial color="#00FF88" transparent opacity={0.3} />
              </mesh>
              <Html distanceFactor={10}>
                <div className="text-xs text-neon-green font-bold">START</div>
              </Html>
            </>
          )}
          
          {state.is_terminal && (
            <>
              <mesh scale={[2, 2, 2]}>
                <sphereGeometry args={[0.15, 16, 16]} />
                <meshBasicMaterial color="#FF006E" transparent opacity={0.3} />
              </mesh>
              <Html distanceFactor={10}>
                <div className="text-xs text-neon-pink font-bold">
                  GOAL
                  <div className="text-[10px]">R={trajectory.total_reward.toFixed(1)}</div>
                </div>
              </Html>
            </>
          )}
        </group>
      ))}
      
      {/* Animated marker */}
      {isActive && (
        <mesh position={curve.getPointAt(progress)}>
          <sphereGeometry args={[0.2, 16, 16]} />
          <meshBasicMaterial color="#BD00FF" />
        </mesh>
      )}
    </group>
  )
}

// Main scene component
function Scene({ activeTrajectory, setActiveTrajectory }: { 
  activeTrajectory: number
  setActiveTrajectory: (idx: number) => void 
}) {
  
  const { data: trajectoriesData } = useQuery<TrajectoriesResponse>({
    queryKey: ['trajectories-3d'],
    queryFn: async () => {
      const response = await axios.get('/api/trajectories?view=3d')
      return response.data as TrajectoriesResponse
    },
  })
  
  const { data: flowField } = useQuery<FlowFieldResponse>({
    queryKey: ['flow-field'],
    queryFn: async () => {
      const response = await axios.get('/api/analysis/flow-field')
      return response.data as FlowFieldResponse
    },
  })
  
  const trajectories = trajectoriesData?.trajectories || []
  const peaks = flowField?.reward_peaks || []
  const gridSize = trajectoriesData?.grid_size?.[0] ?? 10
  const gridOffset = -((gridSize - 1) / 2)
  
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.2} />
      <pointLight position={[5, 5, 10]} intensity={1} color="#BD00FF" />
      <pointLight position={[0, 0, 5]} intensity={0.5} color="#00D9FF" />
      
      {/* Center the grid */}
      <group position={[gridOffset, gridOffset, 0]}>
        <GridFloor />
        <RewardPeaks peaks={peaks} />
        
        {/* Trajectories */}
        {trajectories.map((traj, i) => (
          <TrajectoryPath
            key={traj.id}
            trajectory={traj}
            isActive={i === activeTrajectory}
            color={i === activeTrajectory ? '#BD00FF' : '#666'}
            onClick={() => setActiveTrajectory(i)}
          />
        ))}
      </group>
    </>
  )
}

export function GFlowNetTrajectory3D() {
  const [isPlaying, setIsPlaying] = useState(true)
  const [selectedTrajectory, setSelectedTrajectory] = useState(0)
  
  const { data: trajectoriesData } = useQuery<TrajectoriesResponse>({
    queryKey: ['trajectories-3d'],
    queryFn: async () => {
      const response = await axios.get('/api/trajectories?view=3d')
      return response.data as TrajectoriesResponse
    },
  })
  
  const trajectories = trajectoriesData?.trajectories || []
  
  return (
    <div className="relative w-full h-full">
      <Canvas camera={{ position: [8, 8, 12], fov: 60 }}>
        <color attach="background" args={['#0A0A0B']} />
        <fog attach="fog" args={['#0A0A0B', 10, 50]} />
        
        <Scene activeTrajectory={selectedTrajectory} setActiveTrajectory={setSelectedTrajectory} />
        
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          target={[0, 0, 1]}
        />
        
        <EffectComposer>
          <Bloom
            intensity={1.2}
            luminanceThreshold={0.3}
            luminanceSmoothing={0.9}
          />
        </EffectComposer>
      </Canvas>
      
      {/* Info Panel */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="absolute top-4 left-4 glass-dark rounded-lg p-4 max-w-sm"
      >
        <h3 className="text-sm font-medium mb-2 flex items-center space-x-2">
          <Info className="w-4 h-4 text-neon-purple" />
          <span>GFlowNet Grid World Trajectories</span>
        </h3>
        <p className="text-xs text-muted-foreground mb-3">
          Agents learn to navigate from random starts (green) to high-reward goals (pink).
          Height represents time progression.
        </p>
        
        {/* Trajectory selector */}
        <div className="space-y-2">
          <label className="text-xs text-muted-foreground">Select Trajectory:</label>
          <select
            value={selectedTrajectory}
            onChange={(e) => setSelectedTrajectory(Number(e.target.value))}
            className="w-full bg-dark-panel border border-dark-border rounded-md px-2 py-1 text-sm"
          >
            {trajectories.map((traj, i) => (
              <option key={i} value={i}>
                Trajectory {traj.trajectory_id} (R={traj.total_reward.toFixed(1)})
              </option>
            ))}
          </select>
        </div>
        
        {trajectories[selectedTrajectory] && (
          <div className="mt-3 space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Start:</span>
              <span className="font-mono">
                ({trajectories[selectedTrajectory].start_position.map((v) => v - 1).join(',')})
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">End:</span>
              <span className="font-mono">
                ({trajectories[selectedTrajectory].end_position.map((v) => v - 1).join(',')})
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Length:</span>
              <span>{trajectories[selectedTrajectory].states.length} steps</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Total Reward:</span>
              <span className="text-neon-green">
                {trajectories[selectedTrajectory].total_reward.toFixed(2)}
              </span>
            </div>
          </div>
        )}
      </motion.div>
      
      {/* Legend */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute bottom-4 right-4 glass-dark rounded-lg p-3"
      >
        <h4 className="text-xs font-medium mb-2">Legend</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-neon-green"></div>
            <span>Start Position</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-neon-pink"></div>
            <span>Goal (Terminal)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-12 h-3 rounded bg-gradient-to-r from-blue-500 via-yellow-500 to-red-500"></div>
            <span>Reward (Low→High)</span>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
