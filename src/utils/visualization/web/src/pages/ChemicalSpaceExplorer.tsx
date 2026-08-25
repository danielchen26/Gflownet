import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Compass, Layers, Filter, Download, Play, Pause } from 'lucide-react'
import * as d3 from 'd3'
import { MoleculeViewer2D } from '../components/MoleculeViewer2D'
import { PropertyRadarChart } from '../components/PropertyRadarChart'
import { api, type ChemicalSpacePoint } from '../services/api'
import type { ViewId } from '../components/Sidebar'

type ProjectionMethod = 'umap' | 'tsne' | 'pca'
type ColorBy = 'reward' | 'qed' | 'logp' | 'sa' | 'cluster' | 'epoch'

interface ChemicalSpaceExplorerProps {
  onNavigate?: (view: ViewId) => void
  problemConfig?: any
}

export function ChemicalSpaceExplorer({ onNavigate, problemConfig }: ChemicalSpaceExplorerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState({ w: 800, h: 600 })
  const [projection, setProjection] = useState<ProjectionMethod>('umap')
  const [colorBy, setColorBy] = useState<ColorBy>('reward')
  const [selectedPoint, setSelectedPoint] = useState<ChemicalSpacePoint | null>(null)
  const [hoveredPoint, setHoveredPoint] = useState<ChemicalSpacePoint | null>(null)
  const [showDensity, setShowDensity] = useState(false)
  const [animating, setAnimating] = useState(false)

  const { data: spaceData, isLoading } = useQuery({
    queryKey: ['chemical-space', projection],
    queryFn: () => api.molecular.getChemicalSpace({ method: projection, color_by: colorBy }),
    retry: 1,
    refetchInterval: 10000,
  })

  const points: ChemicalSpacePoint[] = spaceData?.points ?? []

  // Track container size for sharp rendering
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (width > 0 && height > 0) setCanvasSize({ w: Math.round(width), h: Math.round(height) })
    })
    ro.observe(container)
    return () => ro.disconnect()
  }, [])

  // D3 canvas rendering — HiDPI aware
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || points.length === 0) return

    const dpr = window.devicePixelRatio || 1
    const cssW = canvasSize.w
    const cssH = canvasSize.h

    // Set canvas buffer to device pixels for sharp rendering
    canvas.width = cssW * dpr
    canvas.height = cssH * dpr
    canvas.style.width = `${cssW}px`
    canvas.style.height = `${cssH}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)

    const padding = 50

    // Scales (operate in CSS coordinates)
    const xExtent = d3.extent(points, (p) => p.x) as [number, number]
    const yExtent = d3.extent(points, (p) => p.y) as [number, number]
    const xScale = d3.scaleLinear().domain(xExtent).range([padding, cssW - padding])
    const yScale = d3.scaleLinear().domain(yExtent).range([cssH - padding, padding])

    const colorScale = getColorScale(colorBy, points)

    // Clear
    ctx.clearRect(0, 0, cssW, cssH)

    // Draw density contours if enabled
    if (showDensity) {
      drawDensityContours(ctx, points, xScale, yScale, cssW, cssH)
    }

    // Draw axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.25)'
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(`${projection.toUpperCase()} Dimension 1`, cssW / 2, cssH - 8)
    ctx.save()
    ctx.translate(12, cssH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText(`${projection.toUpperCase()} Dimension 2`, 0, 0)
    ctx.restore()

    // Draw points — sorted so high-value on top
    const sorted = [...points].sort((a, b) => getColorValue(a, colorBy) - getColorValue(b, colorBy))
    sorted.forEach((p) => {
      const x = xScale(p.x)
      const y = yScale(p.y)
      const color = colorScale(getColorValue(p, colorBy))
      const radius = 3 + (p.reward / 10) * 4

      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.globalAlpha = 0.8
      ctx.fill()
      ctx.globalAlpha = 1

      // Highlight selected
      if (selectedPoint?.id === p.id) {
        ctx.strokeStyle = '#FFFFFF'
        ctx.lineWidth = 2.5
        ctx.stroke()
      }
    })

    // Draw hovered point highlight
    if (hoveredPoint) {
      const x = xScale(hoveredPoint.x)
      const y = yScale(hoveredPoint.y)
      ctx.beginPath()
      ctx.arc(x, y, 10, 0, Math.PI * 2)
      ctx.strokeStyle = '#FFFFFF'
      ctx.lineWidth = 2
      ctx.stroke()
    }
  }, [points, colorBy, showDensity, selectedPoint, hoveredPoint, canvasSize, projection])

  // Handle canvas click — in CSS coordinates
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || points.length === 0) return

    const rect = canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    const padding = 50
    const cssW = canvasSize.w
    const cssH = canvasSize.h
    const xExtent = d3.extent(points, (p) => p.x) as [number, number]
    const yExtent = d3.extent(points, (p) => p.y) as [number, number]
    const xScale = d3.scaleLinear().domain(xExtent).range([padding, cssW - padding])
    const yScale = d3.scaleLinear().domain(yExtent).range([cssH - padding, padding])

    let closest: ChemicalSpacePoint | null = null
    let minDist = Infinity

    points.forEach((p) => {
      const dx = xScale(p.x) - mouseX
      const dy = yScale(p.y) - mouseY
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < 15 && dist < minDist) {
        minDist = dist
        closest = p
      }
    })

    setSelectedPoint(closest)
  }, [points, canvasSize])

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Context Banner */}
      <div className="mb-3 px-3 py-2 rounded-lg bg-neon-purple/5 border border-neon-purple/20">
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          <span className="text-neon-purple font-medium">Chemical Space Projection</span> — Each point represents a generated molecule, projected into 2D
          using <span className="font-mono text-white">{projection.toUpperCase()}</span> on molecular fingerprints.
          Nearby points are structurally similar. Point size reflects reward score, color maps to <span className="font-mono text-white">{colorBy}</span>.
          {points.length > 0 ? <> Displaying <span className="font-mono text-white">{points.length}</span> molecules.</> : ' No molecules generated yet — start training to populate this view.'}
          {' '}Click any point to inspect its structure and properties.
        </p>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold gradient-text">Chemical Space Explorer</h1>
        <div className="flex items-center gap-2">
          <select
            value={projection}
            onChange={(e) => setProjection(e.target.value as ProjectionMethod)}
            className="px-2 py-1.5 rounded-md bg-dark-bg border border-dark-border text-xs"
          >
            <option value="umap">UMAP</option>
            <option value="tsne">t-SNE</option>
            <option value="pca">PCA</option>
          </select>

          <select
            value={colorBy}
            onChange={(e) => setColorBy(e.target.value as ColorBy)}
            className="px-2 py-1.5 rounded-md bg-dark-bg border border-dark-border text-xs"
          >
            <option value="reward">Color: Reward</option>
            <option value="qed">Color: QED</option>
            <option value="logp">Color: LogP</option>
            <option value="sa">Color: SA</option>
            <option value="cluster">Color: Cluster</option>
            <option value="epoch">Color: Epoch</option>
          </select>

          <label className="flex items-center gap-1.5 text-[10px] text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={showDensity}
              onChange={(e) => setShowDensity(e.target.checked)}
              className="w-3 h-3 rounded accent-neon-purple"
            />
            <Layers className="w-3 h-3" />
            Density
          </label>

          <button
            onClick={() => setAnimating(!animating)}
            className="p-1.5 rounded-md bg-white/5 hover:bg-white/10 text-muted-foreground transition-colors"
            title="Animate generation timeline"
          >
            {animating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex gap-4 min-h-0">
        {/* Canvas */}
        <div ref={containerRef} className="flex-1 relative glass-dark rounded-xl overflow-hidden">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-sm text-muted-foreground">Loading chemical space...</div>
            </div>
          ) : points.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Compass className="w-16 h-16 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">No molecules in chemical space yet</p>
              <p className="text-[10px] text-muted-foreground/60 mt-1">Generate molecules to populate the space</p>
            </div>
          ) : (
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              style={{ cursor: 'crosshair', position: 'absolute', top: 0, left: 0 }}
            />
          )}

          {/* Legend */}
          <div className="absolute bottom-4 left-4 glass rounded-md p-2">
            <div className="text-[9px] text-muted-foreground mb-1">{colorBy.toUpperCase()}</div>
            <div className="flex items-center gap-1">
              <div className="w-20 h-2 rounded-full" style={{
                background: `linear-gradient(to right, ${getColorRange(colorBy).join(', ')})`
              }} />
              <span className="text-[8px] text-muted-foreground">Low → High</span>
            </div>
          </div>

          {/* Stats Badge */}
          <div className="absolute top-4 left-4 glass rounded-md px-2.5 py-1.5 text-[10px]">
            <span className="text-white font-mono font-medium">{points.length}</span>
            <span className="text-muted-foreground"> molecules</span>
            <span className="text-muted-foreground/50 mx-1.5">|</span>
            <span className="text-muted-foreground">Source: </span>
            <span className="text-neon-purple font-medium">GFlowNet Training + DB</span>
          </div>

          {/* Projection Badge */}
          <div className="absolute top-4 right-4 glass rounded-md px-2.5 py-1.5 text-[10px]">
            <span className="text-neon-blue font-mono">{projection.toUpperCase()}</span>
            <span className="text-muted-foreground"> projection</span>
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-72 glass-dark rounded-xl p-3 overflow-y-auto scrollbar-thin">
          {selectedPoint ? (
            <div className="space-y-3">
              <h3 className="text-xs font-semibold">Selected Molecule</h3>

              {/* 2D Preview */}
              <div className="bg-white/5 rounded-lg p-2 flex justify-center">
                <MoleculeViewer2D smiles={selectedPoint.smiles} width={200} height={200} />
              </div>

              {/* SMILES */}
              <div className="bg-white/5 rounded-md p-2">
                <span className="text-[9px] text-muted-foreground">SMILES</span>
                <p className="text-[10px] font-mono break-all">{selectedPoint.smiles}</p>
              </div>

              {/* Properties */}
              <div className="space-y-1">
                <PropRow label="Reward" value={selectedPoint.reward?.toFixed(2)} highlight />
                <PropRow label="MW" value={selectedPoint.properties?.molecular_weight?.toFixed(0)} />
                <PropRow label="QED" value={selectedPoint.properties?.qed?.toFixed(3)} />
                <PropRow label="LogP" value={selectedPoint.properties?.logp?.toFixed(2)} />
                <PropRow label="SA" value={selectedPoint.properties?.synthetic_accessibility?.toFixed(1)} />
                <PropRow label="TPSA" value={selectedPoint.properties?.tpsa?.toFixed(0)} />
              </div>

              {/* Radar Chart */}
              {selectedPoint.properties && (
                <PropertyRadarChart properties={selectedPoint.properties} height={180} />
              )}

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={() => onNavigate?.('structure')}
                  className="flex-1 px-2 py-1.5 rounded-md bg-neon-purple/20 text-neon-purple text-[10px] hover:bg-neon-purple/30 transition-colors"
                >
                  View 3D
                </button>
                <button className="flex-1 px-2 py-1.5 rounded-md bg-white/5 text-muted-foreground text-[10px] hover:bg-white/10 transition-colors">
                  Export
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <Compass className="w-8 h-8 text-muted-foreground/30 mb-2" />
              <p className="text-xs text-muted-foreground">Click a point to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function getColorScale(colorBy: ColorBy, points: ChemicalSpacePoint[]) {
  const values = points.map((p) => getColorValue(p, colorBy))
  const extent = d3.extent(values) as [number, number]

  if (colorBy === 'cluster') {
    return d3.scaleOrdinal(d3.schemeCategory10).domain(values.map(String)) as any
  }

  return d3.scaleSequential(d3.interpolateViridis).domain(extent)
}

function getColorValue(point: ChemicalSpacePoint, colorBy: ColorBy): number {
  switch (colorBy) {
    case 'reward': return point.reward
    case 'qed': return point.properties?.qed ?? 0
    case 'logp': return point.properties?.logp ?? 0
    case 'sa': return point.properties?.synthetic_accessibility ?? 0
    case 'cluster': return point.cluster_id ?? 0
    case 'epoch': return point.generation_epoch ?? 0
    default: return point.reward
  }
}

function getColorRange(colorBy: ColorBy): string[] {
  return ['#440154', '#31688e', '#35b779', '#fde725']
}

function drawDensityContours(
  ctx: CanvasRenderingContext2D,
  points: ChemicalSpacePoint[],
  xScale: d3.ScaleLinear<number, number>,
  yScale: d3.ScaleLinear<number, number>,
  width: number,
  height: number
) {
  // Simple kernel density estimation visualization
  const gridSize = 50
  const grid = Array.from({ length: gridSize }, () => new Float32Array(gridSize))
  const bandwidth = 2

  points.forEach((p) => {
    const gx = Math.floor(((xScale(p.x) - 40) / (width - 80)) * gridSize)
    const gy = Math.floor(((yScale(p.y) - 40) / (height - 80)) * gridSize)

    for (let dx = -3; dx <= 3; dx++) {
      for (let dy = -3; dy <= 3; dy++) {
        const nx = gx + dx
        const ny = gy + dy
        if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) {
          const dist = Math.sqrt(dx * dx + dy * dy)
          grid[nx][ny] += Math.exp(-(dist * dist) / (2 * bandwidth * bandwidth))
        }
      }
    }
  })

  // Render density as semi-transparent color
  const maxDensity = Math.max(...grid.flatMap((row) => Array.from(row)))
  const cellW = (width - 80) / gridSize
  const cellH = (height - 80) / gridSize

  for (let x = 0; x < gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      const density = grid[x][y] / maxDensity
      if (density > 0.05) {
        ctx.fillStyle = `rgba(189, 0, 255, ${density * 0.15})`
        ctx.fillRect(40 + x * cellW, 40 + y * cellH, cellW, cellH)
      }
    }
  }
}

function PropRow({ label, value, highlight }: { label: string; value?: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[9px] text-muted-foreground">{label}</span>
      <span className={`text-[10px] font-mono ${highlight ? 'text-neon-green font-bold' : ''}`}>{value ?? '—'}</span>
    </div>
  )
}
