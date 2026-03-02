import { useState, useEffect, useMemo } from 'react'
import { Puzzle, RefreshCw, Loader2, Search, Filter } from 'lucide-react'
import { molecularApi, FragmentInfo, FragmentLibraryResponse } from '../services/api'

interface FragmentBrowserProps {
  compact?: boolean
}

const CATEGORY_COLORS: Record<string, string> = {
  ring: '#8B5CF6',
  functional_group: '#3B82F6',
  linker: '#F59E0B',
  starter: '#10B981',
  unknown: '#6B7280',
}

const CATEGORY_LABELS: Record<string, string> = {
  ring: 'Ring System',
  functional_group: 'Functional Group',
  linker: 'Linker',
  starter: 'Starter',
  unknown: 'Other',
}

export default function FragmentBrowser({ compact = false }: FragmentBrowserProps) {
  const [data, setData] = useState<FragmentLibraryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [selectedFragment, setSelectedFragment] = useState<FragmentInfo | null>(null)

  const fetchFragments = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await molecularApi.getFragmentsCurrent()
      setData(result)
    } catch {
      setError('Failed to load fragment library')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchFragments() }, [])

  // Category counts
  const categoryCounts = useMemo(() => {
    if (!data?.fragments) return {}
    const counts: Record<string, number> = {}
    data.fragments.forEach(f => {
      const cat = f.category || 'unknown'
      counts[cat] = (counts[cat] || 0) + 1
    })
    return counts
  }, [data])

  // Filter fragments
  const filteredFragments = useMemo(() => {
    if (!data?.fragments) return []
    return data.fragments.filter(f => {
      if (categoryFilter !== 'all' && f.category !== categoryFilter) return false
      if (search) {
        const q = search.toLowerCase()
        return f.smiles.toLowerCase().includes(q) ||
               f.name.toLowerCase().includes(q) ||
               f.category.toLowerCase().includes(q)
      }
      return true
    })
  }, [data, search, categoryFilter])

  if (loading && !data) {
    return (
      <div style={{ padding: '24px', color: '#9CA3AF', display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center' }}>
        <Loader2 style={{ width: 16, height: 16, animation: 'spin 1s linear infinite' }} />
        Loading fragment library...
      </div>
    )
  }

  if (error) {
    return <div style={{ padding: '16px', color: '#EF4444', fontSize: '13px' }}>{error}</div>
  }

  if (!data) return null

  return (
    <div style={{ padding: compact ? '12px' : '16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#E5E7EB', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Puzzle style={{ width: 14, height: 14, color: '#F59E0B' }} />
          Fragment Library
          <span style={{
            fontSize: '10px', fontWeight: 500, padding: '2px 6px',
            background: '#F59E0B20', color: '#F59E0B', borderRadius: '4px',
          }}>
            {data.n_fragments} fragments
          </span>
        </h3>
        <button
          onClick={fetchFragments}
          disabled={loading}
          style={{
            padding: '4px 8px', fontSize: '11px', background: '#374151',
            border: '1px solid #4B5563', borderRadius: '4px', color: '#9CA3AF',
            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
          }}
        >
          <RefreshCw style={{ width: 10, height: 10, ...(loading ? { animation: 'spin 1s linear infinite' } : {}) }} />
        </button>
      </div>

      {/* Category filter pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '10px' }}>
        <button
          onClick={() => setCategoryFilter('all')}
          style={{
            padding: '3px 8px', fontSize: '10px', fontWeight: 500, borderRadius: '4px', cursor: 'pointer',
            background: categoryFilter === 'all' ? '#E5E7EB20' : 'transparent',
            color: categoryFilter === 'all' ? '#E5E7EB' : '#6B7280',
            border: `1px solid ${categoryFilter === 'all' ? '#E5E7EB40' : '#374151'}`,
          }}
        >
          All ({data.n_fragments})
        </button>
        {Object.entries(categoryCounts).map(([cat, count]) => {
          const color = CATEGORY_COLORS[cat] || '#6B7280'
          const active = categoryFilter === cat
          return (
            <button
              key={cat}
              onClick={() => setCategoryFilter(active ? 'all' : cat)}
              style={{
                padding: '3px 8px', fontSize: '10px', fontWeight: 500, borderRadius: '4px', cursor: 'pointer',
                background: active ? `${color}25` : 'transparent',
                color: active ? color : '#6B7280',
                border: `1px solid ${active ? `${color}50` : '#374151'}`,
              }}
            >
              {CATEGORY_LABELS[cat] || cat} ({count})
            </button>
          )
        })}
      </div>

      {/* Search */}
      <div style={{ position: 'relative', marginBottom: '10px' }}>
        <Search style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', width: 12, height: 12, color: '#6B7280' }} />
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search fragments by name or SMILES..."
          style={{
            width: '100%', padding: '6px 8px 6px 26px', fontSize: '11px',
            background: '#1F2937', border: '1px solid #374151', borderRadius: '6px',
            color: '#E5E7EB',
          }}
        />
      </div>

      {/* Fragment grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: compact ? 'repeat(2, 1fr)' : 'repeat(auto-fill, minmax(140px, 1fr))',
        gap: '6px',
        maxHeight: compact ? '300px' : '500px',
        overflowY: 'auto',
      }}>
        {filteredFragments.map(frag => {
          const catColor = CATEGORY_COLORS[frag.category] || '#6B7280'
          const isSelected = selectedFragment?.id === frag.id
          return (
            <button
              key={frag.id}
              onClick={() => setSelectedFragment(isSelected ? null : frag)}
              style={{
                padding: '8px', textAlign: 'left', cursor: 'pointer',
                background: isSelected ? `${catColor}15` : '#1F2937',
                border: `1px solid ${isSelected ? `${catColor}50` : '#374151'}`,
                borderRadius: '6px', transition: 'all 0.15s',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
                <span style={{ fontSize: '10px', fontWeight: 600, color: '#E5E7EB' }}>
                  #{frag.id}
                </span>
                {frag.is_starter && (
                  <span style={{
                    fontSize: '8px', padding: '1px 4px', borderRadius: '2px',
                    background: '#10B98120', color: '#10B981',
                  }}>
                    Starter
                  </span>
                )}
              </div>
              <div style={{
                fontSize: '10px', fontFamily: 'monospace', color: '#9CA3AF',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                marginBottom: '4px',
              }}>
                {frag.smiles}
              </div>
              <div style={{ display: 'flex', gap: '4px', alignItems: 'center', flexWrap: 'wrap' }}>
                <span style={{
                  fontSize: '8px', padding: '1px 4px', borderRadius: '2px',
                  background: `${catColor}15`, color: catColor,
                }}>
                  {CATEGORY_LABELS[frag.category] || frag.category}
                </span>
                {frag.n_attachments > 1 && (
                  <span style={{
                    fontSize: '8px', padding: '1px 4px', borderRadius: '2px',
                    background: '#37415180', color: '#9CA3AF',
                  }}>
                    {frag.n_attachments} attach
                  </span>
                )}
              </div>
            </button>
          )
        })}
      </div>

      {filteredFragments.length === 0 && (
        <div style={{ padding: '20px', textAlign: 'center', color: '#6B7280', fontSize: '12px' }}>
          No fragments match your search
        </div>
      )}

      {/* Selected fragment detail */}
      {selectedFragment && (
        <div style={{
          marginTop: '10px', padding: '10px', background: '#1F2937',
          borderRadius: '6px', border: '1px solid #374151',
        }}>
          <div style={{ fontSize: '12px', fontWeight: 600, color: '#E5E7EB', marginBottom: '6px' }}>
            {selectedFragment.name}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', fontSize: '11px' }}>
            <div>
              <span style={{ color: '#6B7280' }}>SMILES: </span>
              <span style={{ fontFamily: 'monospace', color: '#D1D5DB' }}>{selectedFragment.smiles}</span>
            </div>
            <div>
              <span style={{ color: '#6B7280' }}>Category: </span>
              <span style={{ color: CATEGORY_COLORS[selectedFragment.category] || '#9CA3AF' }}>
                {CATEGORY_LABELS[selectedFragment.category] || selectedFragment.category}
              </span>
            </div>
            <div>
              <span style={{ color: '#6B7280' }}>Attachments: </span>
              <span style={{ color: '#D1D5DB' }}>{selectedFragment.n_attachments}</span>
            </div>
            <div>
              <span style={{ color: '#6B7280' }}>Starter: </span>
              <span style={{ color: selectedFragment.is_starter ? '#10B981' : '#6B7280' }}>
                {selectedFragment.is_starter ? 'Yes' : 'No'}
              </span>
            </div>
            {selectedFragment.brics_labels.length > 0 && (
              <div style={{ gridColumn: '1 / -1' }}>
                <span style={{ color: '#6B7280' }}>BRICS Labels: </span>
                <span style={{ color: '#D1D5DB' }}>
                  {selectedFragment.brics_labels.join(', ')}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
