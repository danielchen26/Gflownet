import { useState, useCallback, useEffect } from 'react'
import { Sliders, RotateCcw, Loader2 } from 'lucide-react'
import { molecularApi, ParetoGenerateResponse } from '../services/api'

interface PreferenceSlidersProps {
  objectives?: string[]
  onGenerate?: (molecules: ParetoGenerateResponse) => void
  compact?: boolean
}

const DEFAULT_OBJECTIVES = ['QED', 'SA', 'LogP', 'MW']

const PRESETS: Record<string, { label: string; weights: number[]; color: string }> = {
  uniform: { label: 'Balanced', weights: [0.25, 0.25, 0.25, 0.25], color: '#8B5CF6' },
  druglike: { label: 'Drug-like', weights: [0.45, 0.35, 0.15, 0.05], color: '#10B981' },
  potent: { label: 'Potent Binder', weights: [0.6, 0.1, 0.2, 0.1], color: '#F59E0B' },
  synthesizable: { label: 'Easy Synthesis', weights: [0.1, 0.7, 0.1, 0.1], color: '#3B82F6' },
  diverse: { label: 'Diverse', weights: [0.15, 0.15, 0.35, 0.35], color: '#EC4899' },
}

const OBJECTIVE_COLORS: Record<string, string> = {
  QED: '#10B981',
  SA: '#3B82F6',
  LogP: '#F59E0B',
  MW: '#EC4899',
}

export default function PreferenceSliders({
  objectives = DEFAULT_OBJECTIVES,
  onGenerate,
  compact = false,
}: PreferenceSlidersProps) {
  const [weights, setWeights] = useState<number[]>(() => {
    const n = objectives.length
    return Array(n).fill(1 / n)
  })
  const [generating, setGenerating] = useState(false)
  const [nMolecules, setNMolecules] = useState(20)
  const [error, setError] = useState<string | null>(null)

  // Reset weights when objectives change
  useEffect(() => {
    const n = objectives.length
    setWeights(Array(n).fill(1 / n))
  }, [objectives.length])

  // Adjust a single slider — redistribute remaining weight proportionally
  const handleSliderChange = useCallback((index: number, newValue: number) => {
    setWeights(prev => {
      const updated = [...prev]
      const oldValue = updated[index]
      const clampedNew = Math.max(0, Math.min(1, newValue))
      updated[index] = clampedNew

      const delta = clampedNew - oldValue
      const othersSum = prev.reduce((s, w, i) => i !== index ? s + w : s, 0)

      if (othersSum > 0.001) {
        // Redistribute proportionally
        for (let i = 0; i < updated.length; i++) {
          if (i !== index) {
            updated[i] = Math.max(0, updated[i] - (delta * (prev[i] / othersSum)))
          }
        }
      }

      // Normalize to ensure sum = 1
      const sum = updated.reduce((s, w) => s + w, 0)
      if (sum > 0.001) {
        for (let i = 0; i < updated.length; i++) {
          updated[i] = updated[i] / sum
        }
      }

      return updated
    })
  }, [])

  const applyPreset = useCallback((presetKey: string) => {
    const preset = PRESETS[presetKey]
    if (!preset) return
    // Adapt preset weights to match objective count
    const adapted = objectives.map((_, i) => preset.weights[i] ?? (1 / objectives.length))
    const sum = adapted.reduce((s, w) => s + w, 0)
    setWeights(adapted.map(w => w / sum))
  }, [objectives])

  const handleGenerate = async () => {
    setGenerating(true)
    setError(null)
    try {
      const prefs: Record<string, number> = {}
      objectives.forEach((obj, i) => {
        prefs[obj.toLowerCase()] = weights[i]
      })
      const result = await molecularApi.generatePareto({
        preferences: prefs,
        n_molecules: nMolecules,
      })
      if (result.error) {
        setError(result.error)
      } else {
        onGenerate?.(result)
      }
    } catch (err: any) {
      const msg = err?.response?.data?.error || 'Pareto generation failed. Start a MOGFN training session first.'
      setError(msg)
    } finally {
      setGenerating(false)
    }
  }

  const resetWeights = () => {
    const n = objectives.length
    setWeights(Array(n).fill(1 / n))
  }

  return (
    <div style={{ padding: compact ? '12px' : '16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ margin: 0, fontSize: compact ? '13px' : '14px', fontWeight: 600, color: '#E5E7EB', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Sliders style={{ width: 14, height: 14, color: '#8B5CF6' }} />
          Preference Weights
        </h3>
        <button
          onClick={resetWeights}
          style={{
            padding: '3px 6px', fontSize: '10px', background: '#374151',
            border: '1px solid #4B5563', borderRadius: '4px', color: '#9CA3AF',
            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '3px',
          }}
        >
          <RotateCcw style={{ width: 10, height: 10 }} />
          Reset
        </button>
      </div>

      {/* Preset buttons */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '12px' }}>
        {Object.entries(PRESETS).map(([key, preset]) => (
          <button
            key={key}
            onClick={() => applyPreset(key)}
            style={{
              padding: '3px 8px', fontSize: '10px', fontWeight: 500,
              background: `${preset.color}15`, color: preset.color,
              border: `1px solid ${preset.color}40`, borderRadius: '4px',
              cursor: 'pointer', transition: 'all 0.15s',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = `${preset.color}30` }}
            onMouseLeave={e => { e.currentTarget.style.background = `${preset.color}15` }}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Weight sliders */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '12px' }}>
        {objectives.map((obj, idx) => {
          const color = OBJECTIVE_COLORS[obj] || '#8B5CF6'
          const pct = (weights[idx] * 100).toFixed(1)
          return (
            <div key={obj}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                <span style={{ fontSize: '11px', color: '#9CA3AF', fontWeight: 500 }}>{obj}</span>
                <span style={{ fontSize: '11px', fontFamily: 'monospace', color, fontWeight: 600 }}>{pct}%</span>
              </div>
              <div style={{ position: 'relative', height: '20px' }}>
                {/* Background bar */}
                <div style={{
                  position: 'absolute', top: '8px', left: 0, right: 0, height: '4px',
                  background: '#374151', borderRadius: '2px',
                }} />
                {/* Fill bar */}
                <div style={{
                  position: 'absolute', top: '8px', left: 0, height: '4px',
                  width: `${weights[idx] * 100}%`, background: color, borderRadius: '2px',
                  transition: 'width 0.1s ease',
                }} />
                <input
                  type="range"
                  min={0} max={1} step={0.01}
                  value={weights[idx]}
                  onChange={e => handleSliderChange(idx, parseFloat(e.target.value))}
                  style={{
                    position: 'absolute', top: 0, left: 0, width: '100%', height: '20px',
                    opacity: 0, cursor: 'pointer',
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* Weight distribution bar */}
      <div style={{
        display: 'flex', height: '6px', borderRadius: '3px', overflow: 'hidden', marginBottom: '12px',
      }}>
        {objectives.map((obj, idx) => (
          <div
            key={obj}
            style={{
              width: `${weights[idx] * 100}%`,
              background: OBJECTIVE_COLORS[obj] || '#8B5CF6',
              transition: 'width 0.15s ease',
            }}
          />
        ))}
      </div>

      {/* Generate controls */}
      <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
        <input
          type="number"
          min={5} max={100} step={5}
          value={nMolecules}
          onChange={e => setNMolecules(Math.max(5, Math.min(100, Number(e.target.value))))}
          style={{
            width: '48px', padding: '4px 6px', fontSize: '11px', fontFamily: 'monospace',
            background: '#1F2937', border: '1px solid #374151', borderRadius: '4px',
            color: '#E5E7EB', textAlign: 'center',
          }}
        />
        <button
          onClick={handleGenerate}
          disabled={generating}
          style={{
            flex: 1, padding: '6px 12px', fontSize: '12px', fontWeight: 600,
            background: generating ? '#374151' : '#8B5CF620',
            color: generating ? '#9CA3AF' : '#8B5CF6',
            border: `1px solid ${generating ? '#4B5563' : '#8B5CF650'}`,
            borderRadius: '6px', cursor: generating ? 'wait' : 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
            transition: 'all 0.15s',
          }}
        >
          {generating ? (
            <>
              <Loader2 style={{ width: 12, height: 12, animation: 'spin 1s linear infinite' }} />
              Generating...
            </>
          ) : (
            `Generate ${nMolecules} Molecules`
          )}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <div style={{
          marginTop: '8px', padding: '6px 10px', fontSize: '11px',
          background: '#EF444415', color: '#EF4444', borderRadius: '6px',
          border: '1px solid #EF444430',
        }}>
          {error}
        </div>
      )}
    </div>
  )
}
