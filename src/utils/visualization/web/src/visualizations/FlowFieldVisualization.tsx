import { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Box, Sphere } from '@react-three/drei'
import { EffectComposer, Bloom, DepthOfField } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { motion } from 'framer-motion'
import { Layers, Eye, EyeOff } from 'lucide-react'

interface FlowPoint {
  position: [number, number, number]
  velocity: [number, number, number]
  magnitude: number
}

interface FlowFieldData {
  resolution: [number, number, number]
  bounds: {
    x: [number, number]
    y: [number, number]
    z: [number, number]
  }
  data: FlowPoint[]
}

// Custom shader for volumetric flow visualization
const VolumetricFlowMaterial = {
  uniforms: {
    time: { value: 0 },
    opacity: { value: 0.3 },
    colorA: { value: new THREE.Color('#BD00FF') },
    colorB: { value: new THREE.Color('#00D9FF') },
  },
  vertexShader: `
    varying vec3 vPosition;
    varying float vMagnitude;
    attribute float magnitude;
    
    void main() {
      vPosition = position;
      vMagnitude = magnitude;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = 5.0 + magnitude * 10.0;
    }
  `,
  fragmentShader: `
    uniform float time;
    uniform float opacity;
    uniform vec3 colorA;
    uniform vec3 colorB;
    varying vec3 vPosition;
    varying float vMagnitude;
    
    void main() {
      float dist = distance(gl_PointCoord, vec2(0.5));
      if (dist > 0.5) discard;
      
      float alpha = (1.0 - dist * 2.0) * opacity * vMagnitude;
      vec3 color = mix(colorA, colorB, vMagnitude);
      
      gl_FragColor = vec4(color, alpha);
    }
  `,
}

// Flow field particles
function FlowFieldParticles({ flowData }: { flowData: FlowFieldData }) {
  const ref = useRef<THREE.Points>(null!)
  const materialRef = useRef<THREE.ShaderMaterial>(null!)
  
  const { positions, magnitudes } = useMemo(() => {
    const positions = new Float32Array(flowData.data.length * 3)
    const magnitudes = new Float32Array(flowData.data.length)
    
    flowData.data.forEach((point, i) => {
      positions[i * 3] = point.position[0]
      positions[i * 3 + 1] = point.position[1]
      positions[i * 3 + 2] = point.position[2]
      magnitudes[i] = point.magnitude
    })
    
    return { positions, magnitudes }
  }, [flowData])
  
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime
    }
  })
  
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-magnitude"
          count={magnitudes.length}
          array={magnitudes}
          itemSize={1}
        />
      </bufferGeometry>
      <shaderMaterial
        ref={materialRef}
        {...VolumetricFlowMaterial}
        transparent
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  )
}

// Flow streamlines
function FlowStreamlines({ flowData }: { flowData: FlowFieldData }) {
  const linesRef = useRef<THREE.Group>(null!)
  
  const streamlines = useMemo(() => {
    const lines: THREE.Vector3[][] = []
    const step = 3
    
    // Sample starting points
    for (let x = 0; x < flowData.resolution[0]; x += step) {
      for (let y = 0; y < flowData.resolution[1]; y += step) {
        const line: THREE.Vector3[] = []
        let currentX = flowData.bounds.x[0] + (x / flowData.resolution[0]) * (flowData.bounds.x[1] - flowData.bounds.x[0])
        let currentY = flowData.bounds.y[0] + (y / flowData.resolution[1]) * (flowData.bounds.y[1] - flowData.bounds.y[0])
        let currentZ = 0
        
        // Trace streamline
        for (let step = 0; step < 20; step++) {
          line.push(new THREE.Vector3(currentX, currentY, currentZ))
          
          // Find nearest flow point (simplified)
          const idx = Math.floor(Math.random() * flowData.data.length)
          const flow = flowData.data[idx]
          
          currentX += flow.velocity[0] * 0.1
          currentY += flow.velocity[1] * 0.1
          currentZ += flow.velocity[2] * 0.1
          
          // Check bounds
          if (currentX < flowData.bounds.x[0] || currentX > flowData.bounds.x[1] ||
              currentY < flowData.bounds.y[0] || currentY > flowData.bounds.y[1]) {
            break
          }
        }
        
        if (line.length > 3) {
          lines.push(line)
        }
      }
    }
    
    return lines
  }, [flowData])
  
  useFrame((state) => {
    if (linesRef.current) {
      linesRef.current.rotation.z = state.clock.elapsedTime * 0.02
    }
  })
  
  return (
    <group ref={linesRef}>
      {streamlines.map((line, i) => (
        <mesh key={i}>
          <tubeGeometry args={[new THREE.CatmullRomCurve3(line), 20, 0.02, 8, false]} />
          <meshBasicMaterial
            color={new THREE.Color().setHSL(0.6 + i * 0.01, 0.8, 0.5)}
            transparent
            opacity={0.3}
          />
        </mesh>
      ))}
    </group>
  )
}

// Slicing plane for volumetric data
function SlicingPlane({ axis, position, bounds }: { axis: 'x' | 'y' | 'z', position: number, bounds: any }) {
  const planeRef = useRef<THREE.Mesh>(null!)
  
  const { planeGeometry, planePosition, planeRotation } = useMemo(() => {
    let geometry: [number, number, number, number]
    let position: [number, number, number]
    let rotation: [number, number, number]
    
    switch (axis) {
      case 'x':
        geometry = [0.1, bounds.y[1] - bounds.y[0], bounds.z[1] - bounds.z[0], 1]
        position = [position, 0, 0]
        rotation = [0, Math.PI / 2, 0]
        break
      case 'y':
        geometry = [bounds.x[1] - bounds.x[0], 0.1, bounds.z[1] - bounds.z[0], 1]
        position = [0, position, 0]
        rotation = [Math.PI / 2, 0, 0]
        break
      case 'z':
      default:
        geometry = [bounds.x[1] - bounds.x[0], bounds.y[1] - bounds.y[0], 0.1, 1]
        position = [0, 0, position]
        rotation = [0, 0, 0]
    }
    
    return { planeGeometry: geometry, planePosition: position, planeRotation: rotation }
  }, [axis, position, bounds])
  
  return (
    <mesh ref={planeRef} position={planePosition} rotation={planeRotation}>
      <planeGeometry args={planeGeometry} />
      <meshBasicMaterial color="#BD00FF" transparent opacity={0.2} side={THREE.DoubleSide} />
    </mesh>
  )
}

// Main scene
function FlowFieldScene() {
  const [showStreamlines, setShowStreamlines] = useState(true)
  const [showParticles, setShowParticles] = useState(true)
  const [slicePosition, setSlicePosition] = useState(0)
  
  const { data: flowData } = useQuery({
    queryKey: ['flow-field'],
    queryFn: async () => {
      const response = await axios.get('/api/analysis/flow-field')
      return response.data as FlowFieldData
    },
  })
  
  if (!flowData) {
    return <Box args={[1, 1, 1]}><meshBasicMaterial color="#BD00FF" /></Box>
  }
  
  return (
    <>
      {showParticles && <FlowFieldParticles flowData={flowData} />}
      {showStreamlines && <FlowStreamlines flowData={flowData} />}
      <SlicingPlane axis="z" position={slicePosition} bounds={flowData.bounds} />
      
      {/* Bounding box */}
      <Box args={[
        flowData.bounds.x[1] - flowData.bounds.x[0],
        flowData.bounds.y[1] - flowData.bounds.y[0],
        flowData.bounds.z[1] - flowData.bounds.z[0]
      ]} position={[0, 0, 0]}>
        <meshBasicMaterial color="#2A2A2D" wireframe />
      </Box>
    </>
  )
}

export function FlowFieldVisualization() {
  const [showStreamlines, setShowStreamlines] = useState(true)
  const [showParticles, setShowParticles] = useState(true)
  
  return (
    <div className="relative w-full h-full">
      <Canvas camera={{ position: [15, 15, 15], fov: 50 }}>
        <color attach="background" args={['#0A0A0B']} />
        <fog attach="fog" args={['#0A0A0B', 10, 100]} />
        
        <ambientLight intensity={0.1} />
        <pointLight position={[10, 10, 10]} intensity={0.5} color="#BD00FF" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#00D9FF" />
        
        <FlowFieldScene />
        
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate
          autoRotateSpeed={0.2}
        />
        
        <EffectComposer>
          <Bloom
            intensity={1.2}
            luminanceThreshold={0.2}
            luminanceSmoothing={0.9}
          />
          <DepthOfField
            focusDistance={0}
            focalLength={0.02}
            bokehScale={2}
            height={480}
          />
        </EffectComposer>
      </Canvas>
      
      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute top-4 right-4 glass-dark rounded-lg p-4 space-y-3"
      >
        <h3 className="text-sm font-medium mb-2">Visualization Controls</h3>
        
        <button
          onClick={() => setShowStreamlines(!showStreamlines)}
          className={`flex items-center space-x-2 text-sm ${showStreamlines ? 'text-neon-purple' : 'text-muted-foreground'}`}
        >
          {showStreamlines ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          <span>Streamlines</span>
        </button>
        
        <button
          onClick={() => setShowParticles(!showParticles)}
          className={`flex items-center space-x-2 text-sm ${showParticles ? 'text-neon-purple' : 'text-muted-foreground'}`}
        >
          {showParticles ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          <span>Particles</span>
        </button>
        
        <div className="pt-2">
          <label className="text-xs text-muted-foreground">Slice Position</label>
          <input
            type="range"
            min="-2"
            max="2"
            step="0.1"
            defaultValue="0"
            onChange={(e) => {/* Update slice position */}}
            className="w-full mt-1"
          />
        </div>
      </motion.div>
      
      {/* Legend */}
      <div className="absolute bottom-4 left-4 glass-dark rounded-lg px-4 py-2">
        <p className="text-sm text-muted-foreground">Flow Field Analysis</p>
        <div className="flex items-center space-x-4 mt-2">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-neon-purple to-neon-blue"></div>
            <span className="text-xs text-muted-foreground">Flow magnitude</span>
          </div>
        </div>
      </div>
    </div>
  )
}