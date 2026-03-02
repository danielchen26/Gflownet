import { useState, useEffect } from 'react'
import { FlaskConical, ArrowRight, Loader2, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react'
import { molecularApi, SynthesisRouteResponse, SynthesisStep } from '../services/api'

interface SynthesisRouteProps {
  moleculeId: string | null
  compact?: boolean
}

const CLASS_COLORS: Record<string, string> = {
  'C-C bond formation': '#8B5CF6',
  'C-N bond formation': '#3B82F6',
  'C-O bond formation': '#10B981',
  'ring formation': '#F59E0B',
  'functional group': '#EF4444',
  'protection': '#6366F1',
  'deprotection': '#EC4899',
  'reduction': '#14B8A6',
  'oxidation': '#F97316',
}

export default function SynthesisRoute({ moleculeId, compact = false }: SynthesisRouteProps) {
  const [data, setData] = useState<SynthesisRouteResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (!moleculeId) {
      setData(null)
      return
    }
    const fetchRoute = async () => {
      setLoading(true)
      setError(null)
      try {
        const result = await molecularApi.getSynthesisRoute(moleculeId)
        setData(result)
      } catch (err) {
        setError('Failed to load synthesis route')
      } finally {
        setLoading(false)
      }
    }
    fetchRoute()
  }, [moleculeId])

  if (!moleculeId) return null

  if (loading) {
    return (
      <div style={{ padding: '12px', color: '#9CA3AF', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <Loader2 style={{ width: 12, height: 12, animation: 'spin 1s linear infinite' }} />
        Loading synthesis route...
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: '12px', color: '#EF4444', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <AlertCircle style={{ width: 12, height: 12 }} />
        {error}
      </div>
    )
  }

  if (!data || !data.has_synthesis) {
    return (
      <div style={{ padding: compact ? '8px 12px' : '12px', fontSize: '11px', color: '#6B7280' }}>
        No synthesis route available
      </div>
    )
  }

  const yieldPct = data.cumulative_yield != null ? (data.cumulative_yield * 100).toFixed(1) : null

  return (
    <div style={{ padding: compact ? '8px 12px' : '12px 16px' }}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          width: '100%', background: 'none', border: 'none', cursor: 'pointer',
          padding: 0, marginBottom: expanded ? '8px' : 0,
        }}
      >
        <span style={{
          fontSize: compact ? '12px' : '13px', fontWeight: 600, color: '#E5E7EB',
          display: 'flex', alignItems: 'center', gap: '6px',
        }}>
          <FlaskConical style={{ width: 13, height: 13, color: '#8B5CF6' }} />
          Synthesis Route
          <span style={{
            fontSize: '9px', padding: '1px 5px', borderRadius: '3px',
            background: '#8B5CF620', color: '#8B5CF6',
          }}>
            {data.n_steps} step{data.n_steps !== 1 ? 's' : ''}
          </span>
          {yieldPct && (
            <span style={{
              fontSize: '9px', padding: '1px 5px', borderRadius: '3px',
              background: parseFloat(yieldPct) > 50 ? '#10B98120' : '#F59E0B20',
              color: parseFloat(yieldPct) > 50 ? '#10B981' : '#F59E0B',
            }}>
              ~{yieldPct}% yield
            </span>
          )}
        </span>
        {expanded
          ? <ChevronUp style={{ width: 12, height: 12, color: '#6B7280' }} />
          : <ChevronDown style={{ width: 12, height: 12, color: '#6B7280' }} />
        }
      </button>

      {/* Expanded: synthesis steps */}
      {expanded && (
        <div style={{ marginTop: '4px' }}>
          {data.steps.map((step: SynthesisStep, i: number) => {
            const classColor = CLASS_COLORS[step.reaction_class] || '#6B7280'
            return (
              <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '6px' }}>
                {/* Step number */}
                <div style={{
                  width: '20px', height: '20px', borderRadius: '50%', flexShrink: 0,
                  background: `${classColor}20`, border: `1px solid ${classColor}50`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '9px', fontWeight: 700, color: classColor,
                }}>
                  {step.step}
                </div>

                {/* Step details */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '11px', fontWeight: 600, color: '#E5E7EB' }}>
                    {step.reaction_name}
                  </div>
                  <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginTop: '2px', flexWrap: 'wrap' }}>
                    <span style={{
                      fontSize: '9px', padding: '1px 4px', borderRadius: '2px',
                      background: `${classColor}15`, color: classColor,
                    }}>
                      {step.reaction_class}
                    </span>
                    <span style={{ fontSize: '9px', color: '#6B7280' }}>
                      ~{(step.yield_estimate * 100).toFixed(0)}% yield
                    </span>
                  </div>
                  {step.intermediate && (
                    <div style={{
                      fontSize: '9px', fontFamily: 'monospace', color: '#9CA3AF',
                      marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {step.intermediate}
                    </div>
                  )}
                </div>

                {/* Arrow to next step */}
                {i < data.steps.length - 1 && (
                  <ArrowRight style={{ width: 10, height: 10, color: '#4B5563', flexShrink: 0, marginTop: '5px' }} />
                )}
              </div>
            )
          })}

          {/* Final product */}
          <div style={{
            marginTop: '8px', padding: '6px 8px', borderRadius: '4px',
            background: '#10B98110', border: '1px solid #10B98130',
          }}>
            <div style={{ fontSize: '10px', color: '#10B981', fontWeight: 600 }}>Final Product</div>
            <div style={{
              fontSize: '10px', fontFamily: 'monospace', color: '#D1D5DB',
              marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {data.smiles}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
