import { useState, useEffect } from 'react'
import { molecularApi, DiversityStats as DiversityStatsType } from '../services/api'

interface DiversityStatsProps {
  autoRefresh?: boolean
  refreshInterval?: number
}

export default function DiversityStats({ autoRefresh = false, refreshInterval = 30000 }: DiversityStatsProps) {
  const [stats, setStats] = useState<DiversityStatsType | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchDiversity = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await molecularApi.getDiversity({ sample_size: 500 })
      if (data.stats) {
        setStats(data.stats)
      }
    } catch (err) {
      setError('Failed to compute diversity')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDiversity()
    if (autoRefresh) {
      const interval = setInterval(fetchDiversity, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refreshInterval])

  if (loading && !stats) {
    return <div style={{ padding: '16px', color: '#9CA3AF' }}>Computing diversity metrics...</div>
  }

  if (error) {
    return <div style={{ padding: '16px', color: '#EF4444' }}>{error}</div>
  }

  if (!stats) return null

  const metrics = [
    { label: 'Internal Diversity 1', value: stats.internal_diversity_1, format: (v: number) => v.toFixed(4) },
    { label: 'Internal Diversity 2', value: stats.internal_diversity_2, format: (v: number) => v.toFixed(4) },
    { label: 'Mean Pairwise Tanimoto', value: stats.mean_pairwise, format: (v: number) => v.toFixed(4) },
    { label: 'Median NN Distance', value: stats.median_nn_distance, format: (v: number) => v.toFixed(4) },
    { label: 'Unique Scaffolds', value: stats.n_unique_scaffolds || 0, format: (v: number) => v.toString() },
    { label: 'Scaffold Entropy', value: stats.scaffold_entropy || 0, format: (v: number) => v.toFixed(3) },
    { label: 'Molecules Analyzed', value: stats.n_molecules, format: (v: number) => v.toLocaleString() },
  ]

  return (
    <div style={{ padding: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#E5E7EB' }}>
          Diversity Analysis
        </h3>
        <button
          onClick={fetchDiversity}
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

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        {metrics.map((m) => (
          <div
            key={m.label}
            style={{
              padding: '8px 10px',
              background: '#1F2937',
              borderRadius: '6px',
              border: '1px solid #374151',
            }}
          >
            <div style={{ fontSize: '11px', color: '#9CA3AF', marginBottom: '2px' }}>{m.label}</div>
            <div style={{ fontSize: '16px', fontWeight: 600, color: '#F9FAFB', fontFamily: 'monospace' }}>
              {m.format(m.value)}
            </div>
          </div>
        ))}
      </div>

      {/* Diversity quality indicator */}
      <div style={{ marginTop: '12px', padding: '8px 10px', background: '#1F2937', borderRadius: '6px', border: '1px solid #374151' }}>
        <div style={{ fontSize: '11px', color: '#9CA3AF', marginBottom: '4px' }}>Diversity Quality</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              flex: 1,
              height: '6px',
              background: '#374151',
              borderRadius: '3px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${Math.min(stats.internal_diversity_1 * 100, 100)}%`,
                height: '100%',
                background: stats.internal_diversity_1 > 0.85 ? '#10B981' : stats.internal_diversity_1 > 0.7 ? '#F59E0B' : '#EF4444',
                borderRadius: '3px',
                transition: 'width 0.3s ease',
              }}
            />
          </div>
          <span style={{ fontSize: '12px', color: '#E5E7EB', minWidth: '40px' }}>
            {(stats.internal_diversity_1 * 100).toFixed(1)}%
          </span>
        </div>
        <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '2px' }}>
          MOSES benchmark target: IntDiv1 &ge; 0.85
        </div>
      </div>
    </div>
  )
}
