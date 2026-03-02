import { useState, useEffect, useRef, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Box, Copy, Check, Camera, RotateCw, Loader2,
} from 'lucide-react'
import { PropertyRadarChart } from '../components/PropertyRadarChart'
import { getRDKit } from '../components/MoleculeViewer2D'
import { api, type Molecule } from '../services/api'

type RenderMode = 'ball-stick' | 'space-fill' | 'wireframe' | 'stick' | 'surface'
type ColorScheme = 'element' | 'reward' | 'qed' | 'logp'

interface MolecularViewer3DProps {
  problemConfig?: any
}

export function MolecularViewer3D({ problemConfig }: MolecularViewer3DProps) {
  const viewerRef = useRef<HTMLDivElement>(null)
  const viewerInstanceRef = useRef<any>(null)
  const [renderMode, setRenderMode] = useState<RenderMode>('ball-stick')
  const [colorScheme, setColorScheme] = useState<ColorScheme>('element')
  const [selectedMolecule, setSelectedMolecule] = useState<Molecule | null>(null)
  const [showHydrogen, setShowHydrogen] = useState(true)
  const [copied, setCopied] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [autoRotate, setAutoRotate] = useState(true)

  const { data: molecules } = useQuery({
    queryKey: ['molecules-for-viewer'],
    queryFn: () => api.molecular.getMolecules({ limit: 20, sort_by: 'reward' }),
    retry: 1,
  })

  // Initialize 3Dmol viewer
  useEffect(() => {
    if (!viewerRef.current || !selectedMolecule) return

    let viewer: any = null
    let cancelled = false
    setIsLoading(true)

    const init = async () => {
      try {
        const $3Dmol = await import('3dmol')
        if (cancelled) return

        viewerRef.current!.innerHTML = ''

        viewer = $3Dmol.createViewer(viewerRef.current!, {
          backgroundColor: '#0a0a0f',
          antialias: true,
        })
        viewerInstanceRef.current = viewer

        if (selectedMolecule.coords_3d && selectedMolecule.bonds) {
          // Build molecule from explicit coordinates
          const model = viewer.addModel()
          selectedMolecule.coords_3d.forEach((coord: any, i: number) => {
            model.addAtom({
              elem: coord.atom,
              x: coord.x, y: coord.y, z: coord.z,
              serial: i,
            })
          })
          selectedMolecule.bonds.forEach((bond: any) => {
            model.addBond(bond.from, bond.to, bond.order)
          })

          applyStyle(viewer, renderMode, showHydrogen, colorScheme, selectedMolecule)
          if (renderMode === 'surface') {
            viewer.addSurface($3Dmol.SurfaceType.VDW, {
              opacity: 0.85,
              color: 'white',
              smoothness: 2,
            })
          }
          viewer.zoomTo()
          viewer.render()
          if (autoRotate) viewer.spin(true)
        } else {
          // Generate 3D coordinates from SMILES using RDKit, then load as SDF
          try {
            const RDKit = await getRDKit()
            if (cancelled) return
            const mol = RDKit.get_mol(selectedMolecule.smiles)
            if (mol && mol.is_valid()) {
              const molblock = mol.get_molblock()
              mol.delete()

              if (molblock) {
                viewer.addModel(molblock, 'sdf')
                applyStyle(viewer, renderMode, showHydrogen, colorScheme, selectedMolecule)
                if (renderMode === 'surface') {
                  viewer.addSurface($3Dmol.SurfaceType.VDW, {
                    opacity: 0.85,
                    color: 'white',
                    smoothness: 2,
                  })
                }
                viewer.zoomTo()
                viewer.render()
                if (autoRotate) viewer.spin(true)
              }
            } else {
              if (mol) mol.delete()
              console.warn('Invalid SMILES for 3D:', selectedMolecule.smiles)
            }
          } catch (err) {
            console.warn('RDKit molblock generation failed:', err)
          }
        }
      } catch (err) {
        console.warn('3Dmol initialization failed:', err)
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    init()

    return () => {
      cancelled = true
      if (viewer) {
        try { viewer.spin(false); viewer.clear() } catch {}
      }
      viewerInstanceRef.current = null
    }
  }, [selectedMolecule, renderMode, showHydrogen, colorScheme])

  // Toggle auto-rotation
  useEffect(() => {
    const viewer = viewerInstanceRef.current
    if (viewer) {
      viewer.spin(autoRotate)
    }
  }, [autoRotate])

  const handleCopySmiles = useCallback(() => {
    if (selectedMolecule) {
      navigator.clipboard.writeText(selectedMolecule.smiles)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }
  }, [selectedMolecule])

  // Screenshot: export current view as PNG
  const handleScreenshot = useCallback(() => {
    const viewer = viewerInstanceRef.current
    if (!viewer) return
    try {
      const uri = viewer.pngURI()
      const link = document.createElement('a')
      link.download = `molecule_${selectedMolecule?.id ?? 'unknown'}.png`
      link.href = uri
      link.click()
    } catch (err) {
      console.warn('Screenshot failed:', err)
    }
  }, [selectedMolecule])

  // Pause auto-rotation on hover, resume on leave
  const handleMouseEnter = useCallback(() => {
    if (autoRotate && viewerInstanceRef.current) {
      viewerInstanceRef.current.spin(false)
    }
  }, [autoRotate])

  const handleMouseLeave = useCallback(() => {
    if (autoRotate && viewerInstanceRef.current) {
      viewerInstanceRef.current.spin(true)
    }
  }, [autoRotate])

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Context Banner */}
      <div className="mb-3 px-3 py-2 rounded-lg bg-neon-green/5 border border-neon-green/20">
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          <span className="text-neon-green font-medium">3D Structure Viewer</span> — Showing
          {molecules?.total ? <> top <span className="font-mono text-white">{molecules.molecules?.length ?? 0}</span> of {molecules.total} molecules</> : ' molecules'}
          {' '}ranked by reward score. These structures were generated by GFlowNet
          {problemConfig?.training_objective && <> ({problemConfig.training_objective.replace(/_/g, ' ')})</>}
          {' '}and stored in the session database. Select a molecule from the right panel to explore its 3D conformation.
        </p>
      </div>

      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold gradient-text">3D Molecular Viewer</h1>
        <div className="flex items-center gap-2">
          {/* Render Mode */}
          <select
            value={renderMode}
            onChange={(e) => setRenderMode(e.target.value as RenderMode)}
            className="px-2 py-1.5 rounded-md bg-dark-bg border border-dark-border text-xs"
          >
            <option value="ball-stick">Ball & Stick</option>
            <option value="space-fill">Space Filling</option>
            <option value="wireframe">Wireframe</option>
            <option value="stick">Stick</option>
            <option value="surface">Surface</option>
          </select>

          {/* Color Scheme */}
          <select
            value={colorScheme}
            onChange={(e) => setColorScheme(e.target.value as ColorScheme)}
            className="px-2 py-1.5 rounded-md bg-dark-bg border border-dark-border text-xs"
          >
            <option value="element">By Element</option>
            <option value="reward">By Reward</option>
            <option value="qed">By QED</option>
            <option value="logp">By LogP</option>
          </select>

          {/* Auto-Rotate Toggle */}
          <label className="flex items-center gap-1.5 text-[10px] text-muted-foreground cursor-pointer" title="Auto-rotate">
            <input
              type="checkbox"
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
              className="w-3 h-3 rounded accent-neon-purple"
            />
            <RotateCw className="w-3 h-3" />
          </label>

          {/* Hydrogen Toggle */}
          <label className="flex items-center gap-1.5 text-[10px] text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={showHydrogen}
              onChange={(e) => setShowHydrogen(e.target.checked)}
              className="w-3 h-3 rounded accent-neon-purple"
            />
            Show H
          </label>

          {/* Screenshot */}
          <button
            onClick={handleScreenshot}
            disabled={!selectedMolecule || isLoading}
            className="p-1.5 rounded-md bg-dark-bg border border-dark-border hover:border-neon-purple/50 text-muted-foreground hover:text-white transition-colors disabled:opacity-30"
            title="Save screenshot"
          >
            <Camera className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        {/* 3D Canvas */}
        <div className="flex-1 relative glass-dark rounded-xl overflow-hidden">
          {selectedMolecule ? (
            <>
              {/* Loading Spinner */}
              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-[#0a0a0f]/80 z-10">
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="w-8 h-8 text-neon-purple animate-spin" />
                    <span className="text-xs text-muted-foreground">Generating 3D coordinates...</span>
                  </div>
                </div>
              )}
              <div
                ref={viewerRef}
                className="w-full h-full"
                style={{ minHeight: 400 }}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
              />
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Box className="w-16 h-16 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">Select a molecule to view in 3D</p>
              <p className="text-[10px] text-muted-foreground/60 mt-1">
                Choose from the list on the right, or generate molecules first
              </p>
            </div>
          )}

          {/* Property Overlay */}
          {selectedMolecule && !isLoading && (
            <div className="absolute top-4 right-4 w-56 glass rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Properties</span>
                <button onClick={handleCopySmiles} className="text-muted-foreground hover:text-white">
                  {copied ? <Check className="w-3 h-3 text-neon-green" /> : <Copy className="w-3 h-3" />}
                </button>
              </div>
              <p className="text-[10px] font-mono text-white truncate">{selectedMolecule.smiles}</p>
              <div className="space-y-1">
                <PropRow label="Reward" value={selectedMolecule.reward?.toFixed(2)} color="#00FF88" />
                <PropRow label="MW" value={selectedMolecule.properties?.molecular_weight?.toFixed(0)} />
                <PropRow label="QED" value={selectedMolecule.properties?.qed?.toFixed(3)} />
                <PropRow label="LogP" value={selectedMolecule.properties?.logp?.toFixed(2)} />
                <PropRow label="SA" value={selectedMolecule.properties?.synthetic_accessibility?.toFixed(1)} />
                <PropRow label="TPSA" value={selectedMolecule.properties?.tpsa?.toFixed(0)} />
              </div>
            </div>
          )}
        </div>

        {/* Right: Molecule List */}
        <div className="w-64 glass-dark rounded-xl p-3 overflow-y-auto scrollbar-thin">
          <h3 className="text-xs font-semibold mb-2">Molecules</h3>
          {molecules?.molecules?.length > 0 ? (
            <div className="space-y-1.5">
              {molecules.molecules.map((mol: Molecule) => (
                <button
                  key={mol.id}
                  onClick={() => setSelectedMolecule(mol)}
                  className={`
                    w-full text-left p-2 rounded-md transition-all
                    ${selectedMolecule?.id === mol.id
                      ? 'bg-neon-purple/15 border border-neon-purple/30'
                      : 'hover:bg-white/5 border border-transparent'
                    }
                  `}
                >
                  <p className="text-[10px] font-mono truncate">{mol.smiles}</p>
                  <div className="flex items-center gap-2 text-[9px] text-muted-foreground mt-0.5">
                    <span className="text-neon-green">R: {mol.reward?.toFixed(1)}</span>
                    <span>QED: {mol.properties?.qed?.toFixed(2)}</span>
                    <span>MW: {mol.properties?.molecular_weight?.toFixed(0)}</span>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center py-8 text-center">
              <Box className="w-8 h-8 text-muted-foreground/30 mb-2" />
              <p className="text-[10px] text-muted-foreground">No molecules yet</p>
            </div>
          )}

          {/* Radar Chart for Selected */}
          {selectedMolecule?.properties && (
            <div className="mt-4 pt-3 border-t border-dark-border/30">
              <h4 className="text-[10px] font-semibold text-muted-foreground mb-1">Property Profile</h4>
              <PropertyRadarChart properties={selectedMolecule.properties} height={200} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Apply render style with optional property-based coloring
function applyStyle(viewer: any, mode: RenderMode, showH: boolean, colorScheme: ColorScheme, molecule: Molecule | null) {
  const style = getStyleObject(mode, showH)

  if (colorScheme === 'element' || !molecule) {
    viewer.setStyle({}, style)
    return
  }

  // Property-based coloring: map property value to a color gradient
  let value = 0
  let maxVal = 1
  if (colorScheme === 'reward') {
    value = molecule.reward ?? 0
    maxVal = 10
  } else if (colorScheme === 'qed') {
    value = molecule.properties?.qed ?? 0
    maxVal = 1
  } else if (colorScheme === 'logp') {
    value = molecule.properties?.logp ?? 0
    maxVal = 5
  }

  const t = Math.min(1, Math.max(0, value / maxVal))
  // Red (low) → Yellow (mid) → Green (high)
  const r = Math.round(t < 0.5 ? 255 : 255 * (1 - (t - 0.5) * 2))
  const g = Math.round(t < 0.5 ? 255 * t * 2 : 255)
  const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}00`

  // Apply style with uniform property color
  const coloredStyle: Record<string, unknown> = {}
  for (const [key, val] of Object.entries(style)) {
    coloredStyle[key] = { ...(val as Record<string, unknown>), color: hex }
  }
  viewer.setStyle({}, coloredStyle)
}

function getStyleObject(mode: RenderMode, _showH: boolean): Record<string, unknown> {
  switch (mode) {
    case 'ball-stick':
      return { stick: { radius: 0.15 }, sphere: { radius: 0.3 } }
    case 'space-fill':
      return { sphere: {} }
    case 'wireframe':
      return { line: {} }
    case 'stick':
      return { stick: { radius: 0.15 } }
    case 'surface':
      return { stick: { radius: 0.1 }, sphere: { radius: 0.2 } }
    default:
      return { stick: {}, sphere: { radius: 0.3 } }
  }
}

function PropRow({ label, value, color }: { label: string; value?: string; color?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[9px] text-muted-foreground">{label}</span>
      <span className="text-[10px] font-mono" style={{ color: color || 'inherit' }}>{value ?? '—'}</span>
    </div>
  )
}
