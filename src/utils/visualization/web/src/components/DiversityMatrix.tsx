import { useState, useEffect, useRef, useCallback } from 'react'
import { molecularApi } from '../services/api'

interface DiversityMatrixProps {
  maxMolecules?: number
}

/** Color interpolation from blue (0) → white (0.5) → red (1) for Tanimoto similarity */
function similarityColor(sim: number): string {
  if (sim <= 0.5) {
    const t = sim / 0.5
    const r = Math.round(30 + t * 225)
    const g = Math.round(58 + t * 197)
    const b = Math.round(138 - t * 10)
    return `rgb(${r},${g},${b})`
  } else {
    const t = (sim - 0.5) / 0.5
    const r = Math.round(255)
    const g = Math.round(255 - t * 186)
    const b = Math.round(128 - t * 59)
    return `rgb(${r},${g},${b})`
  }
}

export default function DiversityMatrix({ maxMolecules = 50 }: DiversityMatrixProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [matrix, setMatrix] = useState<number[][] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number; sim: number } | null>(null)

  const fetchMatrix = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await molecularApi.getDiversity({ sample_size: maxMolecules })
      if (data.similarity_matrix) {
        let mat = data.similarity_matrix
        // Handle flat 1D array from backend — reshape to 2D
        if (mat.length > 0 && !Array.isArray(mat[0])) {
          const n = Math.round(Math.sqrt(mat.length))
          const flat = mat as unknown as number[]
          mat = Array.from({ length: n }, (_, i) => flat.slice(i * n, (i + 1) * n))
        }
        setMatrix(mat)
      } else {
        setError('No similarity matrix returned')
      }
    } catch {
      setError('Failed to compute similarity matrix')
    } finally {
      setLoading(false)
    }
  }, [maxMolecules])

  useEffect(() => {
    fetchMatrix()
  }, [fetchMatrix])

  // Render matrix to canvas
  useEffect(() => {
    if (!matrix || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const n = matrix.length
    const cellSize = Math.max(2, Math.min(8, Math.floor(300 / n)))
    const size = n * cellSize

    canvas.width = size
    canvas.height = size

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        ctx.fillStyle = similarityColor(matrix[i][j])
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize)
      }
    }
  }, [matrix])

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!matrix || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const n = matrix.length
    const cellSize = Math.max(2, Math.min(8, Math.floor(300 / n)))
    const j = Math.floor((e.clientX - rect.left) / cellSize)
    const i = Math.floor((e.clientY - rect.top) / cellSize)
    if (i >= 0 && i < n && j >= 0 && j < n) {
      setHoveredCell({ i, j, sim: matrix[i][j] })
    } else {
      setHoveredCell(null)
    }
  }

  if (loading && !matrix) {
    return <div style={{ padding: '16px', color: '#9CA3AF' }}>Computing similarity matrix...</div>
  }

  if (error) {
    return <div style={{ padding: '16px', color: '#EF4444' }}>{error}</div>
  }

  if (!matrix) return null

  return (
    <div style={{ padding: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#E5E7EB' }}>
          Tanimoto Similarity Matrix
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', color: '#9CA3AF' }}>{matrix.length} molecules</span>
          <button
            onClick={fetchMatrix}
            disabled={loading}
            style={{
              padding: '4px 8px',
              fontSize: '11px',
              background: '#374151',
              border: '1px solid #4B5563',
              borderRadius: '4px',
              color: '#9CA3AF',
              cursor: loading ? 'wait' : 'pointer',
            }}
          >
            {loading ? 'Computing...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div style={{ position: 'relative', display: 'inline-block' }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredCell(null)}
          style={{
            borderRadius: '4px',
            border: '1px solid #374151',
            imageRendering: 'pixelated',
            maxWidth: '100%',
            height: 'auto',
          }}
        />

        {/* Tooltip */}
        {hoveredCell && (
          <div
            style={{
              position: 'absolute',
              top: '4px',
              right: '4px',
              padding: '4px 8px',
              background: 'rgba(17, 24, 39, 0.9)',
              border: '1px solid #374151',
              borderRadius: '4px',
              fontSize: '11px',
              color: '#E5E7EB',
              pointerEvents: 'none',
            }}
          >
            Mol {hoveredCell.i + 1} vs {hoveredCell.j + 1}: <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{(hoveredCell.sim ?? 0).toFixed(4)}</span>
          </div>
        )}
      </div>

      {/* Color legend */}
      <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ fontSize: '10px', color: '#9CA3AF' }}>0.0</span>
        <div
          style={{
            flex: 1,
            height: '8px',
            borderRadius: '4px',
            background: 'linear-gradient(to right, rgb(30,58,138), rgb(255,255,128), rgb(255,69,69))',
          }}
        />
        <span style={{ fontSize: '10px', color: '#9CA3AF' }}>1.0</span>
        <span style={{ fontSize: '10px', color: '#6B7280' }}>Tanimoto</span>
      </div>
    </div>
  )
}
