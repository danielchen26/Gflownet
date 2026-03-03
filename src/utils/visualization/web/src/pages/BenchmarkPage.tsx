import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, Trophy, RefreshCw, Loader2, CheckCircle2, Clock, XCircle } from 'lucide-react'
import type { ViewId } from '../components/Sidebar'

// PMO Benchmark Tasks (23 canonical PMO tasks)
const PMO_TASKS = [
  { id: 'albuterol_similarity', name: 'Albuterol Similarity', category: 'Similarity' },
  { id: 'amlodipine_mpo', name: 'Amlodipine MPO', category: 'MPO' },
  { id: 'celecoxib_rediscovery', name: 'Celecoxib Rediscovery', category: 'Rediscovery' },
  { id: 'deco_hop', name: 'Deco Hop', category: 'Hop' },
  { id: 'drd2', name: 'DRD2', category: 'Bioactivity' },
  { id: 'fexofenadine_mpo', name: 'Fexofenadine MPO', category: 'MPO' },
  { id: 'gsk3b', name: 'GSK3B', category: 'Bioactivity' },
  { id: 'isomers_c7h8n2o2', name: 'Isomers C7H8N2O2', category: 'Isomers' },
  { id: 'isomers_c9h10n2o2pf2cl', name: 'Isomers C9H10N2O2PF2Cl', category: 'Isomers' },
  { id: 'jnk3', name: 'JNK3', category: 'Bioactivity' },
  { id: 'median1', name: 'Median 1', category: 'Median' },
  { id: 'median2', name: 'Median 2', category: 'Median' },
  { id: 'mestranol_similarity', name: 'Mestranol Similarity', category: 'Similarity' },
  { id: 'osimertinib_mpo', name: 'Osimertinib MPO', category: 'MPO' },
  { id: 'perindopril_mpo', name: 'Perindopril MPO', category: 'MPO' },
  { id: 'qed', name: 'QED', category: 'Property' },
  { id: 'ranolazine_mpo', name: 'Ranolazine MPO', category: 'MPO' },
  { id: 'scaffold_hop', name: 'Scaffold Hop', category: 'Hop' },
  { id: 'sitagliptin_mpo', name: 'Sitagliptin MPO', category: 'MPO' },
  { id: 'thiothixene_rediscovery', name: 'Thiothixene Rediscovery', category: 'Rediscovery' },
  { id: 'troglitazone_rediscovery', name: 'Troglitazone Rediscovery', category: 'Rediscovery' },
  { id: 'valsartan_smarts', name: 'Valsartan SMARTS', category: 'SMARTS' },
  { id: 'zaleplon_mpo', name: 'Zaleplon MPO', category: 'MPO' },
]

// SOTA comparison data from PMO paper
const SOTA_METHODS = [
  { name: 'Genetic GFN', score: 16.2, color: '#06B6D4', isSelf: true },
  { name: 'REINVENT', score: 15.2, color: '#9CA3AF', isSelf: false },
  { name: 'Mol GA', score: 15.7, color: '#9CA3AF', isSelf: false },
  { name: 'Graph GA', score: 14.8, color: '#9CA3AF', isSelf: false },
  { name: 'SMILES GA', score: 14.3, color: '#9CA3AF', isSelf: false },
]

interface TaskResult {
  task_id: string
  auc_top10: number
  diversity: number
  best_score: number
  n_oracle_calls: number
}

type BenchmarkStatus = 'not_started' | 'running' | 'complete'

interface BenchmarkPageProps {
  onNavigate?: (view: ViewId) => void
}

export function BenchmarkPage({ onNavigate }: BenchmarkPageProps) {
  const [benchmarkStatus, setBenchmarkStatus] = useState<BenchmarkStatus>('not_started')
  const [results, setResults] = useState<TaskResult[]>([])
  const [loading, setLoading] = useState(false)
  const [totalScore, setTotalScore] = useState<number | null>(null)

  // Simulated fetch -- in production this would call a real benchmark API
  const fetchResults = async () => {
    setLoading(true)
    try {
      // Placeholder: would call GET /api/v2/benchmark/results
      // For now, mark as not started if no results exist
      await new Promise(resolve => setTimeout(resolve, 500))
      setBenchmarkStatus('not_started')
    } catch {
      // silently fail
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchResults()
  }, [])

  // Compute total score
  useEffect(() => {
    if (results.length > 0) {
      const sum = results.reduce((acc, r) => acc + r.auc_top10, 0)
      setTotalScore(sum)
    }
  }, [results])

  // Score color
  const getScoreColor = (score: number): string => {
    if (score > 0.7) return '#10B981'
    if (score > 0.3) return '#F59E0B'
    return '#EF4444'
  }

  // Category badge color
  const getCategoryColor = (category: string): string => {
    const map: Record<string, string> = {
      Bioactivity: '#06B6D4',
      MPO: '#8B5CF6',
      Similarity: '#10B981',
      Rediscovery: '#EF4444',
      Hop: '#F59E0B',
      Isomers: '#EC4899',
      Median: '#3B82F6',
      Property: '#22D3EE',
      SMARTS: '#6366F1',
    }
    return map[category] || '#6B7280'
  }

  // Status icon
  const StatusIcon = () => {
    if (benchmarkStatus === 'complete') return <CheckCircle2 style={{ width: 14, height: 14, color: '#10B981' }} />
    if (benchmarkStatus === 'running') return <Loader2 style={{ width: 14, height: 14, color: '#06B6D4', animation: 'spin 1s linear infinite' }} />
    return <Clock style={{ width: 14, height: 14, color: '#6B7280' }} />
  }

  const statusLabel = benchmarkStatus === 'complete' ? 'Complete' : benchmarkStatus === 'running' ? 'Running...' : 'Not Yet Run'
  const statusColor = benchmarkStatus === 'complete' ? '#10B981' : benchmarkStatus === 'running' ? '#06B6D4' : '#6B7280'

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Header with back button */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {onNavigate && (
          <button
            onClick={() => onNavigate('toolkit')}
            style={{
              padding: '6px 8px', borderRadius: '6px', cursor: 'pointer',
              background: '#1F2937', border: '1px solid #374151', color: '#9CA3AF',
              display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px',
            }}
          >
            <ArrowLeft style={{ width: 14, height: 14 }} />
            Back
          </button>
        )}
        <div>
          <h1 className="text-2xl font-bold gradient-text">PMO Benchmark</h1>
          <p className="text-xs text-muted-foreground mt-1">
            Practical Molecular Optimization -- 23 task evaluation suite
          </p>
        </div>
      </div>

      {/* Status + SOTA Comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        {/* Status Card */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            padding: '16px', borderRadius: '10px',
            background: 'rgba(6,182,212,0.05)', border: '1px solid rgba(6,182,212,0.2)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <StatusIcon />
            <span style={{ fontSize: '13px', fontWeight: 600, color: statusColor }}>
              {statusLabel}
            </span>
            <button
              onClick={fetchResults}
              disabled={loading}
              style={{
                marginLeft: 'auto', padding: '3px 6px', fontSize: '10px',
                background: '#374151', border: '1px solid #4B5563', borderRadius: '4px',
                color: '#9CA3AF', cursor: 'pointer',
              }}
            >
              <RefreshCw style={{ width: 10, height: 10, ...(loading ? { animation: 'spin 1s linear infinite' } : {}) }} />
            </button>
          </div>

          {/* Total Score */}
          <div style={{
            padding: '12px', borderRadius: '8px', background: '#111827',
            border: '1px solid #1F2937', textAlign: 'center',
          }}>
            <div style={{ fontSize: '10px', color: '#9CA3AF', marginBottom: '4px' }}>
              Total Score (Sum of AUC Top-10)
            </div>
            <div style={{
              fontSize: '28px', fontWeight: 800, fontFamily: 'monospace',
              color: totalScore != null ? '#06B6D4' : '#374151',
            }}>
              {totalScore != null ? totalScore.toFixed(1) : '--.-'}
            </div>
            <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '4px' }}>
              out of {PMO_TASKS.length}.0 max
            </div>
          </div>
        </motion.div>

        {/* SOTA Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          style={{
            padding: '16px', borderRadius: '10px',
            background: '#111827', border: '1px solid #1F2937',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '12px' }}>
            <Trophy style={{ width: 14, height: 14, color: '#F59E0B' }} />
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#E5E7EB' }}>SOTA Comparison</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {SOTA_METHODS.map(method => {
              const barWidth = (method.score / 20) * 100 // Max ~20 for display
              return (
                <div key={method.name}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                    <span style={{
                      fontSize: '11px', fontWeight: method.isSelf ? 700 : 500,
                      color: method.isSelf ? '#06B6D4' : '#9CA3AF',
                    }}>
                      {method.name}
                      {method.isSelf && (
                        <span style={{
                          marginLeft: '6px', fontSize: '9px', padding: '1px 5px',
                          background: 'rgba(6,182,212,0.12)', color: '#06B6D4',
                          borderRadius: '3px', fontWeight: 600,
                        }}>
                          Ours
                        </span>
                      )}
                    </span>
                    <span style={{
                      fontSize: '12px', fontWeight: 700, fontFamily: 'monospace',
                      color: method.isSelf ? '#06B6D4' : '#E5E7EB',
                    }}>
                      {method.score.toFixed(1)}
                    </span>
                  </div>
                  <div style={{ height: '4px', background: '#1F2937', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${barWidth}%`, height: '100%',
                      background: method.isSelf ? '#06B6D4' : '#4B5563',
                      borderRadius: '2px', transition: 'width 0.3s',
                    }} />
                  </div>
                </div>
              )
            })}
          </div>
        </motion.div>
      </div>

      {/* Task Table */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        style={{
          background: '#111827', borderRadius: '12px',
          border: '1px solid #1F2937', overflow: 'hidden',
        }}
      >
        <div style={{ padding: '14px 16px', borderBottom: '1px solid #1F2937', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: '#E5E7EB' }}>
            Task Results
          </span>
          <span style={{
            fontSize: '10px', padding: '2px 6px', borderRadius: '4px',
            background: 'rgba(6,182,212,0.1)', color: '#06B6D4', fontWeight: 600,
          }}>
            {PMO_TASKS.length} tasks
          </span>
        </div>

        {/* Table header */}
        <div style={{
          display: 'grid', gridTemplateColumns: '2fr 90px 90px 90px 90px 60px',
          gap: '4px', padding: '8px 16px',
          borderBottom: '1px solid #1F2937', background: '#0D1117',
        }}>
          <span style={{ fontSize: '10px', fontWeight: 600, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Task</span>
          <span style={{ fontSize: '10px', fontWeight: 600, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'right' }}>AUC Top-10</span>
          <span style={{ fontSize: '10px', fontWeight: 600, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'right' }}>Diversity</span>
          <span style={{ fontSize: '10px', fontWeight: 600, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'right' }}>Best Score</span>
          <span style={{ fontSize: '10px', fontWeight: 600, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'right' }}>Calls</span>
          <span style={{ fontSize: '10px', fontWeight: 600, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'center' }}>Status</span>
        </div>

        {/* Task rows */}
        <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
          {PMO_TASKS.map((task, idx) => {
            const result = results.find(r => r.task_id === task.id)
            const catColor = getCategoryColor(task.category)
            return (
              <div
                key={task.id}
                style={{
                  display: 'grid', gridTemplateColumns: '2fr 90px 90px 90px 90px 60px',
                  gap: '4px', padding: '8px 16px',
                  borderBottom: idx < PMO_TASKS.length - 1 ? '1px solid #1F293780' : 'none',
                  background: idx % 2 === 0 ? 'transparent' : '#0D111708',
                }}
              >
                {/* Task name + category */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{
                    fontSize: '8px', fontWeight: 700, padding: '1px 4px',
                    borderRadius: '2px', background: `${catColor}15`, color: catColor,
                    textTransform: 'uppercase', letterSpacing: '0.03em',
                  }}>
                    {task.category}
                  </span>
                  <span style={{ fontSize: '11px', color: '#E5E7EB' }}>{task.name}</span>
                </div>

                {/* AUC Top-10 */}
                <span style={{
                  fontSize: '11px', fontFamily: 'monospace', textAlign: 'right',
                  color: result ? getScoreColor(result.auc_top10) : '#374151',
                  fontWeight: result ? 600 : 400,
                }}>
                  {result ? result.auc_top10.toFixed(3) : '---'}
                </span>

                {/* Diversity */}
                <span style={{
                  fontSize: '11px', fontFamily: 'monospace', textAlign: 'right',
                  color: result ? '#9CA3AF' : '#374151',
                }}>
                  {result ? result.diversity.toFixed(3) : '---'}
                </span>

                {/* Best Score */}
                <span style={{
                  fontSize: '11px', fontFamily: 'monospace', textAlign: 'right',
                  color: result ? getScoreColor(result.best_score) : '#374151',
                  fontWeight: result ? 600 : 400,
                }}>
                  {result ? result.best_score.toFixed(3) : '---'}
                </span>

                {/* Oracle Calls */}
                <span style={{
                  fontSize: '11px', fontFamily: 'monospace', textAlign: 'right',
                  color: result ? '#6B7280' : '#374151',
                }}>
                  {result ? result.n_oracle_calls.toLocaleString() : '---'}
                </span>

                {/* Status */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {result ? (
                    <CheckCircle2 style={{ width: 12, height: 12, color: '#10B981' }} />
                  ) : benchmarkStatus === 'running' ? (
                    <Loader2 style={{ width: 12, height: 12, color: '#06B6D4', animation: 'spin 1s linear infinite' }} />
                  ) : (
                    <XCircle style={{ width: 12, height: 12, color: '#374151' }} />
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </motion.div>
    </div>
  )
}

export default BenchmarkPage
