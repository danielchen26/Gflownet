import { useRef, useMemo, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Trail, Float, MeshDistortMaterial, Environment } from '@react-three/drei'
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { motion } from 'framer-motion'

interface TrajectoryPoint {
  position: [number, number, number]
  reward: number
}

interface Trajectory {
  id: string
  states: Array<{
    position: [number, number, number]
    features: number[]
  }>
  rewards: number[]
  total_reward: number
}

// Particle system for trajectory visualization
function TrajectoryParticles({ trajectory }: { trajectory: Trajectory }) {
  const ref = useRef<THREE.Points>(null!)
  const particlesRef = useRef<THREE.BufferGeometry>(null!)
  
  const { positions, colors, sizes } = useMemo(() => {
    const positions = new Float32Array(trajectory.states.length * 3)
    const colors = new Float32Array(trajectory.states.length * 3)
    const sizes = new Float32Array(trajectory.states.length)
    
    trajectory.states.forEach((state, i) => {
      positions[i * 3] = state.position[0] - 5
      positions[i * 3 + 1] = state.position[1] - 5
      positions[i * 3 + 2] = state.position[2] - 2
      
      // Color gradient based on reward
      const normalizedReward = trajectory.rewards[i] / Math.max(...trajectory.rewards)
      const color = new THREE.Color()
      color.setHSL(0.7 - normalizedReward * 0.4, 1, 0.5 + normalizedReward * 0.3)
      
      colors[i * 3] = color.r
      colors[i * 3 + 1] = color.g
      colors[i * 3 + 2] = color.b
      
      sizes[i] = 0.1 + normalizedReward * 0.2
    })
    
    return { positions, colors, sizes }
  }, [trajectory])
  
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.elapsedTime * 0.05
    }
  })
  
  return (
    <points ref={ref}>
      <bufferGeometry ref={particlesRef}>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          count={sizes.length}
          array={sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        vertexColors
        size={0.2}
        sizeAttenuation
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

// Flowing trajectory line with glow
function TrajectoryFlow({ trajectory }: { trajectory: Trajectory }) {
  const curveRef = useRef<THREE.CatmullRomCurve3>()
  const tubeRef = useRef<THREE.Mesh>(null!)
  
  const curve = useMemo(() => {
    const points = trajectory.states.map(state => 
      new THREE.Vector3(state.position[0] - 5, state.position[1] - 5, state.position[2] - 2)
    )
    return new THREE.CatmullRomCurve3(points, false, 'smooth')
  }, [trajectory])
  
  useFrame((state) => {
    if (tubeRef.current && tubeRef.current.material) {
      const material = tubeRef.current.material as THREE.ShaderMaterial
      if (material.uniforms?.time) {
        material.uniforms.time.value = state.clock.elapsedTime
      }
    }
  })
  
  const flowMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        color1: { value: new THREE.Color('#BD00FF') },
        color2: { value: new THREE.Color('#00D9FF') },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vPosition;
        void main() {
          vUv = uv;
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec3 color1;
        uniform vec3 color2;
        varying vec2 vUv;
        varying vec3 vPosition;
        
        void main() {
          float flow = fract(vUv.x * 3.0 - time * 0.5);
          vec3 color = mix(color1, color2, flow);
          float alpha = sin(flow * 3.14159) * 0.8 + 0.2;
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
    })
  }, [])
  
  return (
    <mesh ref={tubeRef}>
      <tubeGeometry args={[curve, 100, 0.05, 8, false]} />
      <primitive object={flowMaterial} />
    </mesh>
  )
}

// Glowing orbs at key points
function KeyPointOrb({ position, color, size = 0.3 }: { position: [number, number, number], color: string, size?: number }) {
  const meshRef = useRef<THREE.Mesh>(null!)
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(size * (1 + Math.sin(state.clock.elapsedTime * 2) * 0.1))
    }
  })
  
  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <mesh ref={meshRef} position={position}>
        <sphereGeometry args={[size, 32, 32]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <mesh position={position}>
        <sphereGeometry args={[size * 1.5, 32, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.2} />
      </mesh>
    </Float>
  )
}

// Main 3D scene
function Scene() {
  const { data: trajectories } = useQuery({
    queryKey: ['trajectories'],
    queryFn: async () => {
      const response = await axios.get('/api/trajectories')
      return response.data.trajectories as Trajectory[]
    },
  })
  
  const latestTrajectory = trajectories?.[trajectories.length - 1]
  
  if (!latestTrajectory) {
    return (
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <MeshDistortMaterial color="#BD00FF" speed={2} distort={0.3} />
      </mesh>
    )
  }
  
  const startPos: [number, number, number] = [
    latestTrajectory.states[0].position[0] - 5,
    latestTrajectory.states[0].position[1] - 5,
    latestTrajectory.states[0].position[2] - 2,
  ]
  
  const endPos: [number, number, number] = [
    latestTrajectory.states[latestTrajectory.states.length - 1].position[0] - 5,
    latestTrajectory.states[latestTrajectory.states.length - 1].position[1] - 5,
    latestTrajectory.states[latestTrajectory.states.length - 1].position[2] - 2,
  ]
  
  return (
    <>
      <TrajectoryParticles trajectory={latestTrajectory} />
      <TrajectoryFlow trajectory={latestTrajectory} />
      <KeyPointOrb position={startPos} color="#00FF88" size={0.4} />
      <KeyPointOrb position={endPos} color="#FF006E" size={0.5} />
    </>
  )
}

export function TrajectoryViewer3D() {
  return (
    <div className="relative w-full h-full">
      <Canvas camera={{ position: [10, 10, 10], fov: 60 }}>
        <color attach="background" args={['#0A0A0B']} />
        <fog attach="fog" args={['#0A0A0B', 5, 50]} />
        
        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#BD00FF" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#00D9FF" />
        
        <Scene />
        
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate
          autoRotateSpeed={0.5}
        />
        
        <EffectComposer>
          <Bloom
            intensity={1.5}
            luminanceThreshold={0.3}
            luminanceSmoothing={0.9}
            height={300}
          />
          <ChromaticAberration offset={[0.002, 0.002]} />
        </EffectComposer>
      </Canvas>
      
      {/* Overlay UI */}
      <div className="absolute top-4 left-4 glass-dark rounded-lg px-4 py-2">
        <p className="text-sm text-muted-foreground">3D Trajectory Visualization</p>
        <p className="text-xs text-muted-foreground mt-1">Scroll to zoom • Drag to rotate</p>
      </div>
      
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="absolute bottom-4 left-4 flex items-center space-x-4"
      >
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-neon-green"></div>
          <span className="text-xs text-muted-foreground">Start</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-neon-pink"></div>
          <span className="text-xs text-muted-foreground">Goal</span>
        </div>
      </motion.div>
    </div>
  )
}