import { useState, useEffect } from 'react'
import { Zap, RefreshCw, Loader2, AlertCircle, CheckCircle2, Play } from 'lucide-react'
import { molecularApi } from '../services/api'
import type { OracleStatusResponse, OracleAvailableResponse, OracleEvaluateResponse } from '../services/api'

const ORACLE_COLORS: Record<string, string> = {
  DRD2: '#06B6D4',
  GSK3B: '#22D3EE',
  JNK3: '#67E8F9',
}

const DEFAULT_ORACLES = ['DRD2', 'GSK3B', 'JNK3']

export default function OraclePanel() {
  // State
  const [availableOracles, setAvailableOracles] = useState<string[]>(DEFAULT_ORACLES)
  const [selectedOracles, setSelectedOracles] = useState<Set<string>>(new Set(DEFAULT_ORACLES))
  const [weights, setWeights] = useState<Record<string, number>>({ DRD2: 1.0, GSK3B: 1.0, JNK3: 1.0 })
  const [status, setStatus] = useState<OracleStatusResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Evaluate state
  const [evalSmiles, setEvalSmiles] = useState('')
  const [evalResult, setEvalResult] = useState<OracleEvaluateResponse | null>(null)
  const [evaluating, setEvaluating] = useState(false)

  // Fetch available oracles
  const fetchAvailable = async () => {
    try {
      const result: OracleAvailableResponse = await molecularApi.getOraclesAvailable()
      if (result.oracles?.length) {
        setAvailableOracles(result.oracles)
      }
    } catch {
      // Use defaults if endpoint unavailable
    }
  }

  // Fetch oracle status (budget, etc.)
  const fetchStatus = async () => {
    setLoading(true)
    setError(null)
    try {
      const result: OracleStatusResponse = await molecularApi.getOracleStatus()
      setStatus(result)
    } catch {
      setError('Failed to load oracle status')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAvailable()
    fetchStatus()
  }, [])

  // Toggle oracle selection
  const toggleOracle = (oracle: string) => {
    setSelectedOracles(prev => {
      const next = new Set(prev)
      if (next.has(oracle)) {
        next.delete(oracle)
      } else {
        next.add(oracle)
      }
      return next
    })
  }

  // Update weight for an oracle
  const updateWeight = (oracle: string, value: number) => {
    setWeights(prev => ({ ...prev, [oracle]: value }))
  }

  // Evaluate SMILES against selected oracles
  const handleEvaluate = async () => {
    if (!evalSmiles.trim() || selectedOracles.size === 0) return
    setEvaluating(true)
    setEvalResult(null)
    setError(null)
    try {
      const result: OracleEvaluateResponse = await molecularApi.evaluateOracle({
        smiles: evalSmiles.trim(),
        oracles: Array.from(selectedOracles),
      })
      setEvalResult(result)
      // Refresh status to update budget meter
      fetchStatus()
    } catch {
      setError('Evaluation failed')
    } finally {
      setEvaluating(false)
    }
  }

  // Score color helper
  const getScoreColor = (score: number): string => {
    if (score > 0.7) return '#10B981'
    if (score > 0.3) return '#F59E0B'
    return '#EF4444'
  }

  // Budget percentage
  const budgetUsed = status?.budget_used ?? 0
  const budgetTotal = status?.budget_total ?? 10000
  const budgetPct = budgetTotal > 0 ? (budgetUsed / budgetTotal) * 100 : 0

  if (loading && !status) {
    return (
      <div style={{ padding: '16px', color: '#9CA3AF', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <Loader2 style={{ width: 12, height: 12, animation: 'spin 1s linear infinite' }} />
        Loading oracle configuration...
      </div>
    )
  }

  return (
    <div style={{ padding: '16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{
          margin: 0, fontSize: '14px', fontWeight: 600, color: '#E5E7EB',
          display: 'flex', alignItems: 'center', gap: '6px',
        }}>
          <Zap style={{ width: 14, height: 14, color: '#06B6D4' }} />
          TDC Oracles
          {status?.active ? (
            <span style={{
              fontSize: '9px', padding: '1px 5px', background: 'rgba(6,182,212,0.12)',
              color: '#06B6D4', borderRadius: '3px', fontWeight: 600,
            }}>
              Active
            </span>
          ) : (
            <span style={{
              fontSize: '9px', padding: '1px 5px', background: 'rgba(107,114,128,0.2)',
              color: '#6B7280', borderRadius: '3px', fontWeight: 600,
            }}>
              Inactive
            </span>
          )}
          {status?.benchmark_mode && (
            <span style={{
              fontSize: '9px', padding: '1px 5px', background: 'rgba(245,158,11,0.12)',
              color: '#F59E0B', borderRadius: '3px', fontWeight: 600,
            }}>
              Benchmark
            </span>
          )}
        </h3>
        <button
          onClick={fetchStatus}
          disabled={loading}
          style={{
            padding: '3px 6px', fontSize: '10px', background: '#374151',
            border: '1px solid #4B5563', borderRadius: '4px', color: '#9CA3AF', cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: '4px',
          }}
        >
          <RefreshCw style={{ width: 10, height: 10, ...(loading ? { animation: 'spin 1s linear infinite' } : {}) }} />
        </button>
      </div>

      {/* Budget Meter */}
      <div style={{
        marginBottom: '14px', padding: '10px', borderRadius: '8px',
        background: 'rgba(6,182,212,0.05)', border: '1px solid rgba(6,182,212,0.2)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
          <span style={{ fontSize: '11px', color: '#9CA3AF' }}>Oracle Budget</span>
          <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#06B6D4', fontWeight: 600 }}>
            {budgetUsed.toLocaleString()} / {budgetTotal.toLocaleString()}
          </span>
        </div>
        <div style={{ height: '6px', background: '#1F2937', borderRadius: '3px', overflow: 'hidden' }}>
          <div style={{
            width: `${Math.min(budgetPct, 100)}%`,
            height: '100%',
            background: budgetPct > 90 ? '#EF4444' : budgetPct > 70 ? '#F59E0B' : '#06B6D4',
            borderRadius: '3px',
            transition: 'width 0.3s',
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
          <span style={{ fontSize: '9px', color: '#6B7280' }}>
            {status?.budget_remaining?.toLocaleString() ?? (budgetTotal - budgetUsed).toLocaleString()} remaining
          </span>
          <span style={{ fontSize: '9px', color: '#6B7280' }}>
            {status?.cache_size != null ? `Cache: ${status.cache_size}` : ''}
          </span>
        </div>
      </div>

      {/* Oracle Selector */}
      <div style={{ marginBottom: '14px' }}>
        <div style={{ fontSize: '11px', color: '#9CA3AF', marginBottom: '6px' }}>Select Oracles</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {availableOracles.map(oracle => {
            const isSelected = selectedOracles.has(oracle)
            const color = ORACLE_COLORS[oracle] || '#06B6D4'
            return (
              <div key={oracle} style={{
                padding: '8px 10px', borderRadius: '6px',
                background: isSelected ? `${color}08` : '#1F2937',
                border: `1px solid ${isSelected ? `${color}40` : '#374151'}`,
                transition: 'all 0.15s',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {/* Checkbox */}
                  <div
                    onClick={() => toggleOracle(oracle)}
                    style={{
                      width: '16px', height: '16px', borderRadius: '3px', cursor: 'pointer',
                      border: `1.5px solid ${isSelected ? color : '#4B5563'}`,
                      background: isSelected ? `${color}25` : 'transparent',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      transition: 'all 0.15s',
                    }}
                  >
                    {isSelected && (
                      <CheckCircle2 style={{ width: 10, height: 10, color }} />
                    )}
                  </div>

                  {/* Label */}
                  <span style={{
                    fontSize: '12px', fontWeight: 600,
                    color: isSelected ? '#E5E7EB' : '#6B7280',
                    flex: 1,
                  }}>
                    {oracle}
                  </span>

                  {/* Weight badge */}
                  <span style={{
                    fontSize: '9px', fontFamily: 'monospace', fontWeight: 600,
                    padding: '1px 5px', borderRadius: '3px',
                    background: isSelected ? `${color}15` : '#37415150',
                    color: isSelected ? color : '#4B5563',
                  }}>
                    w={weights[oracle]?.toFixed(1) ?? '1.0'}
                  </span>
                </div>

                {/* Weight slider (only shown when selected) */}
                {isSelected && (
                  <div style={{ marginTop: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '9px', color: '#6B7280', minWidth: '38px' }}>Weight</span>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.1}
                      value={weights[oracle] ?? 1.0}
                      onChange={e => updateWeight(oracle, parseFloat(e.target.value))}
                      style={{
                        flex: 1, height: '4px', accentColor: color,
                        cursor: 'pointer',
                      }}
                    />
                    <span style={{
                      fontSize: '10px', fontFamily: 'monospace', color,
                      minWidth: '24px', textAlign: 'right',
                    }}>
                      {(weights[oracle] ?? 1.0).toFixed(1)}
                    </span>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Quick Evaluate */}
      <div style={{ marginBottom: '10px' }}>
        <div style={{ fontSize: '11px', color: '#9CA3AF', marginBottom: '4px' }}>Quick Evaluate</div>
        <div style={{ display: 'flex', gap: '4px' }}>
          <input
            type="text"
            value={evalSmiles}
            onChange={e => setEvalSmiles(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleEvaluate()}
            placeholder="Enter SMILES..."
            style={{
              flex: 1, padding: '6px 8px', fontSize: '11px', fontFamily: 'monospace',
              background: '#1F2937', border: '1px solid #374151', borderRadius: '4px',
              color: '#E5E7EB',
            }}
          />
          <button
            onClick={handleEvaluate}
            disabled={evaluating || !evalSmiles.trim() || selectedOracles.size === 0}
            style={{
              padding: '6px 10px', fontSize: '11px', fontWeight: 600,
              background: evaluating ? '#374151' : 'rgba(6,182,212,0.12)',
              color: evaluating ? '#9CA3AF' : '#06B6D4',
              border: `1px solid ${evaluating ? '#4B5563' : 'rgba(6,182,212,0.4)'}`,
              borderRadius: '4px',
              cursor: evaluating || !evalSmiles.trim() || selectedOracles.size === 0 ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', gap: '4px',
            }}
          >
            {evaluating ? (
              <Loader2 style={{ width: 10, height: 10, animation: 'spin 1s linear infinite' }} />
            ) : (
              <Play style={{ width: 10, height: 10 }} />
            )}
            {evaluating ? '...' : 'Evaluate'}
          </button>
        </div>
      </div>

      {/* Score Display */}
      {evalResult && (
        <div style={{
          padding: '10px', background: '#1F2937', borderRadius: '8px',
          border: '1px solid #374151', marginBottom: '10px',
        }}>
          <div style={{ fontSize: '10px', fontFamily: 'monospace', color: '#9CA3AF', marginBottom: '8px', wordBreak: 'break-all' }}>
            {evalResult.smiles}
            {evalResult.cached && (
              <span style={{
                marginLeft: '6px', fontSize: '9px', padding: '1px 5px',
                background: 'rgba(107,114,128,0.2)', color: '#6B7280', borderRadius: '3px',
              }}>
                cached
              </span>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {Object.entries(evalResult.scores).map(([oracle, score]) => {
              const color = getScoreColor(score)
              const oracleColor = ORACLE_COLORS[oracle] || '#06B6D4'
              return (
                <div key={oracle}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                    <span style={{ fontSize: '11px', color: oracleColor, fontWeight: 600 }}>
                      {oracle}
                    </span>
                    <span style={{
                      fontSize: '13px', fontWeight: 700, fontFamily: 'monospace', color,
                    }}>
                      {score.toFixed(4)}
                    </span>
                  </div>
                  {/* Score bar */}
                  <div style={{ height: '4px', background: '#374151', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.min(score * 100, 100)}%`,
                      height: '100%',
                      background: color,
                      borderRadius: '2px',
                      transition: 'width 0.3s',
                    }} />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Weighted aggregate */}
          {Object.keys(evalResult.scores).length > 1 && (
            <div style={{
              marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #374151',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <span style={{ fontSize: '11px', color: '#9CA3AF' }}>Weighted Average</span>
              <span style={{
                fontSize: '13px', fontWeight: 700, fontFamily: 'monospace',
                color: getScoreColor(computeWeightedAvg(evalResult.scores, weights)),
              }}>
                {computeWeightedAvg(evalResult.scores, weights).toFixed(4)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ fontSize: '10px', color: '#EF4444', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '6px' }}>
          <AlertCircle style={{ width: 10, height: 10 }} />
          {error}
        </div>
      )}
    </div>
  )
}

/** Compute weighted average from scores and weights */
function computeWeightedAvg(scores: Record<string, number>, weights: Record<string, number>): number {
  let totalWeight = 0
  let weightedSum = 0
  for (const [oracle, score] of Object.entries(scores)) {
    const w = weights[oracle] ?? 1.0
    weightedSum += score * w
    totalWeight += w
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 0
}
