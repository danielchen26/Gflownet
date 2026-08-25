import { useState, useEffect, useMemo, useRef } from 'react'
import { Target, RefreshCw, Loader2, Info } from 'lucide-react'
import { molecularApi, ParetoPoint, ParetoFrontResponse } from '../services/api'

interface ParetoFrontExplorerProps {
  autoRefresh?: boolean
  refreshInterval?: number
  onSelectMolecule?: (point: ParetoPoint) => void
}

const OBJECTIVE_LABELS: Record<string, string> = {
  qed: 'QED (Drug-likeness)',
  sa: 'SA (Synth. Access.)',
  logp: 'LogP (Lipophilicity)',
  mw: 'MW (Molecular Weight)',
}

const OBJECTIVE_COLORS: Record<string, string> = {
  qed: '#10B981',
  sa: '#3B82F6',
  logp: '#F59E0B',
  mw: '#EC4899',
}

export default function ParetoFrontExplorer({
  autoRefresh = false,
  refreshInterval = 15000,
  onSelectMolecule,
}: ParetoFrontExplorerProps) {
  const [data, setData] = useState<ParetoFrontResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [xAxis, setXAxis] = useState<string>('qed')
  const [yAxis, setYAxis] = useState<string>('sa')
  const [hoveredPoint, setHoveredPoint] = useState<ParetoPoint | null>(null)
  const [selectedPoint, setSelectedPoint] = useState<ParetoPoint | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const fetchParetoFront = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await molecularApi.getParetoFront()
      setData(result)
    } catch (err) {
      setError('Failed to load Pareto front')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchParetoFront()
    if (autoRefresh) {
      const interval = setInterval(fetchParetoFront, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refreshInterval])

  // Available objectives from data
  const objectiveNames = data?.objective_names ?? ['qed', 'sa', 'logp', 'mw']

  // Compute chart layout
  const margin = { top: 20, right: 20, bottom: 40, left: 50 }
  const chartWidth = 400
  const chartHeight = 280
  const innerW = chartWidth - margin.left - margin.right
  const innerH = chartHeight - margin.top - margin.bottom

  // Scale points to chart coordinates
  const { points, xRange, yRange } = useMemo(() => {
    if (!data?.points?.length) return { points: [], xRange: [0, 1] as [number, number], yRange: [0, 1] as [number, number] }

    const xVals = data.points.map(p => p.objectives[xAxis] ?? 0)
    const yVals = data.points.map(p => p.objectives[yAxis] ?? 0)

    const xMin = Math.min(...xVals)
    const xMax = Math.max(...xVals)
    const yMin = Math.min(...yVals)
    const yMax = Math.max(...yVals)

    // Add 5% padding
    const xPad = (xMax - xMin) * 0.05 || 0.1
    const yPad = (yMax - yMin) * 0.05 || 0.1

    const xRange: [number, number] = [xMin - xPad, xMax + xPad]
    const yRange: [number, number] = [yMin - yPad, yMax + yPad]

    const scaleX = (v: number) => ((v - xRange[0]) / (xRange[1] - xRange[0])) * innerW
    const scaleY = (v: number) => innerH - ((v - yRange[0]) / (yRange[1] - yRange[0])) * innerH

    const mapped = data.points.map(p => ({
      ...p,
      cx: scaleX(p.objectives[xAxis] ?? 0),
      cy: scaleY(p.objectives[yAxis] ?? 0),
    }))

    return { points: mapped, xRange, yRange }
  }, [data, xAxis, yAxis, innerW, innerH])

  // Separate Pareto-optimal from dominated
  const paretoPoints = points.filter(p => p.is_pareto_optimal)
  const dominatedPoints = points.filter(p => !p.is_pareto_optimal)

  // Pareto front line (sorted by x axis)
  const paretoLine = useMemo(() => {
    if (!paretoPoints.length) return ''
    const sorted = [...paretoPoints].sort((a, b) => a.cx - b.cx)
    return sorted.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.cx},${p.cy}`).join(' ')
  }, [paretoPoints])

  const handlePointClick = (point: ParetoPoint) => {
    setSelectedPoint(point)
    onSelectMolecule?.(point)
  }

  // Axis tick generation
  const xTicks = useMemo(() => {
    const count = 5
    const step = (xRange[1] - xRange[0]) / (count - 1)
    return Array.from({ length: count }, (_, i) => xRange[0] + i * step)
  }, [xRange])

  const yTicks = useMemo(() => {
    const count = 5
    const step = (yRange[1] - yRange[0]) / (count - 1)
    return Array.from({ length: count }, (_, i) => yRange[0] + i * step)
  }, [yRange])

  if (loading && !data) {
    return (
      <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#9CA3AF' }}>
        <Loader2 style={{ width: 20, height: 20, animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
        Loading Pareto front...
      </div>
    )
  }

  if (error) {
    return <div style={{ padding: '16px', color: '#EF4444', fontSize: '13px' }}>{error}</div>
  }

  const scaleX = (v: number) => ((v - xRange[0]) / (xRange[1] - xRange[0])) * innerW
  const scaleY = (v: number) => innerH - ((v - yRange[0]) / (yRange[1] - yRange[0])) * innerH

  return (
    <div style={{ padding: '16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#E5E7EB', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Target style={{ width: 14, height: 14, color: '#8B5CF6' }} />
          Pareto Front
          {data && (
            <span style={{
              fontSize: '10px', fontWeight: 500, padding: '2px 6px',
              background: '#8B5CF620', color: '#8B5CF6', borderRadius: '4px',
            }}>
              {paretoPoints.length} optimal / {points.length} total
            </span>
          )}
        </h3>
        <button
          onClick={fetchParetoFront}
          disabled={loading}
          style={{
            padding: '4px 8px', fontSize: '11px', background: '#374151',
            border: '1px solid #4B5563', borderRadius: '4px', color: '#9CA3AF',
            cursor: loading ? 'wait' : 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
          }}
        >
          <RefreshCw style={{ width: 10, height: 10, ...(loading ? { animation: 'spin 1s linear infinite' } : {}) }} />
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {/* Axis selectors */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ fontSize: '11px', color: '#9CA3AF' }}>X:</span>
          <select
            value={xAxis}
            onChange={e => setXAxis(e.target.value)}
            style={{
              fontSize: '11px', padding: '2px 6px', background: '#1F2937',
              border: '1px solid #374151', borderRadius: '4px', color: '#E5E7EB',
            }}
          >
            {objectiveNames.map(name => (
              <option key={name} value={name}>{OBJECTIVE_LABELS[name] || name}</option>
            ))}
          </select>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ fontSize: '11px', color: '#9CA3AF' }}>Y:</span>
          <select
            value={yAxis}
            onChange={e => setYAxis(e.target.value)}
            style={{
              fontSize: '11px', padding: '2px 6px', background: '#1F2937',
              border: '1px solid #374151', borderRadius: '4px', color: '#E5E7EB',
            }}
          >
            {objectiveNames.map(name => (
              <option key={name} value={name}>{OBJECTIVE_LABELS[name] || name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* SVG Chart */}
      {points.length > 0 ? (
        <svg
          ref={svgRef}
          width={chartWidth}
          height={chartHeight}
          style={{ background: '#111827', borderRadius: '8px', border: '1px solid #374151' }}
        >
          <g transform={`translate(${margin.left},${margin.top})`}>
            {/* Grid lines */}
            {xTicks.map(tick => (
              <line
                key={`xg-${tick}`}
                x1={scaleX(tick)} y1={0}
                x2={scaleX(tick)} y2={innerH}
                stroke="#1F2937" strokeWidth={1}
              />
            ))}
            {yTicks.map(tick => (
              <line
                key={`yg-${tick}`}
                x1={0} y1={scaleY(tick)}
                x2={innerW} y2={scaleY(tick)}
                stroke="#1F2937" strokeWidth={1}
              />
            ))}

            {/* Pareto front line */}
            {paretoLine && (
              <path
                d={paretoLine}
                fill="none"
                stroke="#8B5CF6"
                strokeWidth={1.5}
                strokeDasharray="4,3"
                opacity={0.6}
              />
            )}

            {/* Dominated points */}
            {dominatedPoints.map(p => (
              <circle
                key={p.id}
                cx={p.cx}
                cy={p.cy}
                r={hoveredPoint?.id === p.id ? 5 : 3.5}
                fill="#4B556380"
                stroke="#6B7280"
                strokeWidth={hoveredPoint?.id === p.id ? 1.5 : 0.5}
                style={{ cursor: 'pointer', transition: 'all 0.15s' }}
                onMouseEnter={() => setHoveredPoint(p)}
                onMouseLeave={() => setHoveredPoint(null)}
                onClick={() => handlePointClick(p)}
              />
            ))}

            {/* Pareto-optimal points */}
            {paretoPoints.map(p => (
              <circle
                key={p.id}
                cx={p.cx}
                cy={p.cy}
                r={hoveredPoint?.id === p.id || selectedPoint?.id === p.id ? 6 : 4.5}
                fill={selectedPoint?.id === p.id ? '#A78BFA' : '#8B5CF6'}
                stroke={selectedPoint?.id === p.id ? '#E9D5FF' : '#C4B5FD'}
                strokeWidth={selectedPoint?.id === p.id ? 2 : 1}
                style={{ cursor: 'pointer', transition: 'all 0.15s' }}
                onMouseEnter={() => setHoveredPoint(p)}
                onMouseLeave={() => setHoveredPoint(null)}
                onClick={() => handlePointClick(p)}
              />
            ))}

            {/* X axis ticks and labels */}
            {xTicks.map(tick => (
              <g key={`xt-${tick}`} transform={`translate(${scaleX(tick)},${innerH})`}>
                <line y2={4} stroke="#4B5563" />
                <text y={14} textAnchor="middle" fill="#6B7280" fontSize={9} fontFamily="monospace">
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Y axis ticks and labels */}
            {yTicks.map(tick => (
              <g key={`yt-${tick}`} transform={`translate(0,${scaleY(tick)})`}>
                <line x2={-4} stroke="#4B5563" />
                <text x={-8} textAnchor="end" dominantBaseline="middle" fill="#6B7280" fontSize={9} fontFamily="monospace">
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Axis labels */}
            <text
              x={innerW / 2} y={innerH + 32}
              textAnchor="middle" fill={OBJECTIVE_COLORS[xAxis] || '#9CA3AF'}
              fontSize={11} fontWeight={500}
            >
              {OBJECTIVE_LABELS[xAxis] || xAxis}
            </text>
            <text
              transform={`translate(-36,${innerH / 2}) rotate(-90)`}
              textAnchor="middle" fill={OBJECTIVE_COLORS[yAxis] || '#9CA3AF'}
              fontSize={11} fontWeight={500}
            >
              {OBJECTIVE_LABELS[yAxis] || yAxis}
            </text>
          </g>
        </svg>
      ) : (
        <div style={{
          width: chartWidth, height: chartHeight, background: '#111827',
          borderRadius: '8px', border: '1px solid #374151',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        }}>
          <Target style={{ width: 24, height: 24, color: '#374151', marginBottom: '8px' }} />
          <span style={{ fontSize: '12px', color: '#6B7280' }}>No Pareto data yet</span>
          <span style={{ fontSize: '10px', color: '#4B5563', marginTop: '4px' }}>Start MOGFN training to see the Pareto front</span>
        </div>
      )}

      {/* Hover tooltip */}
      {hoveredPoint && (
        <div style={{
          marginTop: '8px', padding: '8px 10px', background: '#1F2937',
          borderRadius: '6px', border: '1px solid #374151',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#E5E7EB' }}>
              {hoveredPoint.smiles.length > 30 ? hoveredPoint.smiles.slice(0, 30) + '...' : hoveredPoint.smiles}
            </span>
            {hoveredPoint.is_pareto_optimal && (
              <span style={{
                fontSize: '9px', padding: '1px 5px', background: '#8B5CF620',
                color: '#8B5CF6', borderRadius: '3px', fontWeight: 600,
              }}>
                Pareto Optimal
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: '12px', fontSize: '10px' }}>
            {objectiveNames.map(name => (
              <span key={name} style={{ color: OBJECTIVE_COLORS[name] || '#9CA3AF' }}>
                {name.toUpperCase()}: <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                  {(hoveredPoint.objectives[name] ?? 0).toFixed(3)}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Hypervolume indicator */}
      {data?.hypervolume != null && (
        <div style={{
          marginTop: '8px', padding: '6px 10px', background: '#1F2937',
          borderRadius: '6px', border: '1px solid #374151',
          display: 'flex', alignItems: 'center', gap: '6px',
        }}>
          <Info style={{ width: 12, height: 12, color: '#6B7280' }} />
          <span style={{ fontSize: '11px', color: '#9CA3AF' }}>
            Hypervolume: <span style={{ fontFamily: 'monospace', color: '#8B5CF6', fontWeight: 600 }}>
              {data.hypervolume.toFixed(4)}
            </span>
          </span>
        </div>
      )}
    </div>
  )
}
