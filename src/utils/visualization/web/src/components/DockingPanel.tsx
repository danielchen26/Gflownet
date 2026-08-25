import { useState, useEffect } from 'react'
import { Crosshair, RefreshCw, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react'
import { molecularApi, DockingTarget, DockingTargetsResponse } from '../services/api'

interface DockingPanelProps {
  compact?: boolean
  onTargetChange?: (targetId: string) => void
}

export default function DockingPanel({ compact = false, onTargetChange }: DockingPanelProps) {
  const [data, setData] = useState<DockingTargetsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dockSmiles, setDockSmiles] = useState('')
  const [dockResult, setDockResult] = useState<{ score: number; method: string; runtime?: number } | null>(null)
  const [docking, setDocking] = useState(false)

  const fetchTargets = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await molecularApi.getDockingTargets()
      setData(result)
    } catch (err) {
      setError('Failed to load docking targets')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchTargets() }, [])

  const handleTargetChange = async (targetId: string) => {
    try {
      await molecularApi.setDockingTarget(targetId)
      onTargetChange?.(targetId)
      fetchTargets()
    } catch (err) {
      setError('Failed to set target')
    }
  }

  const handleDock = async () => {
    if (!dockSmiles.trim()) return
    setDocking(true)
    setDockResult(null)
    try {
      const result = await molecularApi.dockMolecule({
        smiles: dockSmiles.trim(),
        method: data?.proxy_available ? 'proxy' : 'vina',
      })
      if (result.error) {
        setError(result.error)
      } else {
        setDockResult({
          score: result.normalized_score ?? 0,
          method: result.method,
          runtime: result.runtime_ms,
        })
      }
    } catch (err) {
      setError('Docking failed')
    } finally {
      setDocking(false)
    }
  }

  if (loading && !data) {
    return (
      <div style={{ padding: '16px', color: '#9CA3AF', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <Loader2 style={{ width: 12, height: 12, animation: 'spin 1s linear infinite' }} />
        Loading docking targets...
      </div>
    )
  }

  return (
    <div style={{ padding: compact ? '12px' : '16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <h3 style={{ margin: 0, fontSize: compact ? '13px' : '14px', fontWeight: 600, color: '#E5E7EB', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Crosshair style={{ width: 14, height: 14, color: '#EF4444' }} />
          Docking
          {data?.docking_available ? (
            <span style={{ fontSize: '9px', padding: '1px 5px', background: '#10B98120', color: '#10B981', borderRadius: '3px' }}>
              Vina
            </span>
          ) : (
            <span style={{ fontSize: '9px', padding: '1px 5px', background: '#F59E0B20', color: '#F59E0B', borderRadius: '3px' }}>
              Proxy Only
            </span>
          )}
        </h3>
        <button
          onClick={fetchTargets}
          disabled={loading}
          style={{
            padding: '3px 6px', fontSize: '10px', background: '#374151',
            border: '1px solid #4B5563', borderRadius: '4px', color: '#9CA3AF', cursor: 'pointer',
          }}
        >
          <RefreshCw style={{ width: 10, height: 10 }} />
        </button>
      </div>

      {/* Target selector */}
      <div style={{ marginBottom: '10px' }}>
        <div style={{ fontSize: '11px', color: '#9CA3AF', marginBottom: '4px' }}>Target Protein</div>
        <select
          value={data?.active_target ?? ''}
          onChange={e => handleTargetChange(e.target.value)}
          style={{
            width: '100%', padding: '4px 8px', fontSize: '11px',
            background: '#1F2937', border: '1px solid #374151', borderRadius: '4px',
            color: '#E5E7EB',
          }}
        >
          {(data?.targets ?? []).map((t: DockingTarget) => (
            <option key={t.id} value={t.id}>
              {t.name} ({t.pdb_id})
              {!t.has_receptor ? ' [no receptor]' : ''}
            </option>
          ))}
        </select>
        {data?.active_target && (
          <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '2px' }}>
            {data.targets.find(t => t.id === data.active_target)?.description || ''}
          </div>
        )}
      </div>

      {/* Status indicators */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '10px', fontSize: '10px' }}>
        <StatusBadge
          label="Vina"
          active={data?.docking_available ?? false}
        />
        <StatusBadge
          label="Proxy"
          active={data?.proxy_available ?? false}
        />
      </div>

      {/* Quick dock */}
      <div style={{ marginBottom: '8px' }}>
        <div style={{ fontSize: '11px', color: '#9CA3AF', marginBottom: '4px' }}>Quick Dock</div>
        <div style={{ display: 'flex', gap: '4px' }}>
          <input
            type="text"
            value={dockSmiles}
            onChange={e => setDockSmiles(e.target.value)}
            placeholder="Enter SMILES..."
            style={{
              flex: 1, padding: '4px 6px', fontSize: '10px', fontFamily: 'monospace',
              background: '#1F2937', border: '1px solid #374151', borderRadius: '4px',
              color: '#E5E7EB',
            }}
          />
          <button
            onClick={handleDock}
            disabled={docking || !dockSmiles.trim()}
            style={{
              padding: '4px 8px', fontSize: '10px', fontWeight: 600,
              background: docking ? '#374151' : '#EF444420',
              color: docking ? '#9CA3AF' : '#EF4444',
              border: `1px solid ${docking ? '#4B5563' : '#EF444450'}`,
              borderRadius: '4px', cursor: docking ? 'wait' : 'pointer',
            }}
          >
            {docking ? '...' : 'Dock'}
          </button>
        </div>
      </div>

      {/* Dock result */}
      {dockResult && (
        <div style={{
          padding: '8px', background: '#1F2937', borderRadius: '6px',
          border: '1px solid #374151', marginBottom: '8px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: '#9CA3AF' }}>Score:</span>
            <span style={{
              fontSize: '14px', fontWeight: 700, fontFamily: 'monospace',
              color: dockResult.score > 0.7 ? '#10B981' : dockResult.score > 0.4 ? '#F59E0B' : '#EF4444',
            }}>
              {(dockResult.score * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ fontSize: '9px', color: '#6B7280', marginTop: '2px' }}>
            Method: {dockResult.method}
            {dockResult.runtime ? ` | ${dockResult.runtime}ms` : ''}
          </div>
          {/* Score bar */}
          <div style={{ marginTop: '4px', height: '4px', background: '#374151', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{
              width: `${dockResult.score * 100}%`, height: '100%',
              background: dockResult.score > 0.7 ? '#10B981' : dockResult.score > 0.4 ? '#F59E0B' : '#EF4444',
              borderRadius: '2px', transition: 'width 0.3s',
            }} />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ fontSize: '10px', color: '#EF4444', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <AlertCircle style={{ width: 10, height: 10 }} />
          {error}
        </div>
      )}
    </div>
  )
}

function StatusBadge({ label, active }: { label: string; active: boolean }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '3px',
      padding: '2px 6px', borderRadius: '3px', fontSize: '10px',
      background: active ? '#10B98115' : '#37415150',
      color: active ? '#10B981' : '#6B7280',
      border: `1px solid ${active ? '#10B98130' : '#4B556330'}`,
    }}>
      {active ? <CheckCircle2 style={{ width: 8, height: 8 }} /> : <AlertCircle style={{ width: 8, height: 8 }} />}
      {label}
    </span>
  )
}
