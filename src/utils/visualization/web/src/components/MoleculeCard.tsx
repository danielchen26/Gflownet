import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Star, ExternalLink, Copy, Check, Maximize2, Minimize2 } from 'lucide-react'
import { useState, useCallback } from 'react'
import { MoleculeViewer2D } from './MoleculeViewer2D'
import type { Molecule, MolecularProperties } from '../services/api'

interface MoleculeCardProps {
  molecule: Molecule
  rank?: number
  selected?: boolean
  onSelect?: (id: string) => void
  onClick?: (id: string) => void
  size?: 'sm' | 'md' | 'lg'
  showProperties?: boolean
  className?: string
}

export const MoleculeCard = React.memo(function MoleculeCard({
  molecule,
  rank,
  selected = false,
  onSelect,
  onClick,
  size = 'md',
  showProperties = true,
  className = '',
}: MoleculeCardProps) {
  const [copied, setCopied] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const sizes = {
    sm: { viewer: 100, text: 'text-[9px]', pad: 'p-1.5' },
    md: { viewer: 150, text: 'text-[10px]', pad: 'p-2.5' },
    lg: { viewer: 200, text: 'text-xs', pad: 'p-3' },
  }
  const s = sizes[size]
  const viewerSize = expanded ? Math.round(s.viewer * 1.8) : s.viewer

  const handleCopy = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    navigator.clipboard.writeText(molecule.smiles)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [molecule.smiles])

  const rewardColor = getRewardColor(molecule.reward)

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`
        glass-dark rounded-lg ${s.pad} cursor-pointer
        transition-all duration-150
        ${selected ? 'ring-2 ring-neon-purple' : 'hover:ring-1 hover:ring-neon-purple/30'}
        ${className}
      `}
      onClick={() => onClick?.(molecule.id)}
    >
      {/* Rank Badge */}
      {rank != null && (
        <div className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded-md bg-dark-bg/80 text-[9px] font-bold text-muted-foreground">
          #{rank}
        </div>
      )}

      {/* Selection Checkbox */}
      {onSelect && (
        <button
          onClick={(e) => { e.stopPropagation(); onSelect(molecule.id) }}
          className={`absolute top-1.5 right-1.5 w-4 h-4 rounded border flex items-center justify-center transition-colors
            ${selected ? 'bg-neon-purple border-neon-purple' : 'border-dark-border hover:border-neon-purple/50'}
          `}
        >
          {selected && <Check className="w-3 h-3 text-white" />}
        </button>
      )}

      {/* 2D Structure (expandable) */}
      <motion.div
        className="flex justify-center mb-2 bg-white/5 rounded-md overflow-hidden relative group/viewer"
        animate={{ height: viewerSize }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        <MoleculeViewer2D
          smiles={molecule.smiles}
          width={viewerSize}
          height={viewerSize}
        />
        {/* Expand/Collapse toggle */}
        <button
          onClick={(e) => { e.stopPropagation(); setExpanded(!expanded) }}
          className="absolute top-1 right-1 p-0.5 rounded bg-dark-bg/60 text-muted-foreground hover:text-white opacity-0 group-hover/viewer:opacity-100 transition-opacity"
          title={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
        </button>
      </motion.div>

      {/* Properties */}
      {showProperties && (
        <div className="space-y-1">
          {/* Reward */}
          <div className="flex items-center justify-between">
            <span className={`${s.text} text-muted-foreground`}>Reward</span>
            <span className={`${s.text} font-bold font-mono`} style={{ color: rewardColor }}>
              {molecule.reward.toFixed(2)}
            </span>
          </div>

          {/* Key Properties */}
          <PropertyRow label="QED" value={molecule.properties.qed} format={2} size={s.text} good={molecule.properties.qed > 0.5} />
          <PropertyRow label="MW" value={molecule.properties.molecular_weight} format={0} size={s.text} unit="Da" />
          <PropertyRow label="LogP" value={molecule.properties.logp} format={1} size={s.text} />
          <PropertyRow label="SA" value={molecule.properties.synthetic_accessibility} format={1} size={s.text} good={molecule.properties.synthetic_accessibility < 5} />

          {/* SMILES */}
          <div className="flex items-center mt-1.5 pt-1.5 border-t border-dark-border/30">
            <span className={`${s.text} font-mono text-muted-foreground truncate flex-1`}>
              {molecule.smiles}
            </span>
            <button onClick={handleCopy} className="ml-1 text-muted-foreground hover:text-white">
              {copied ? <Check className="w-3 h-3 text-neon-green" /> : <Copy className="w-3 h-3" />}
            </button>
          </div>
        </div>
      )}
    </motion.div>
  )
})

function PropertyRow({
  label,
  value,
  format,
  size,
  unit,
  good,
}: {
  label: string
  value: number
  format: number
  size: string
  unit?: string
  good?: boolean
}) {
  return (
    <div className="flex items-center justify-between">
      <span className={`${size} text-muted-foreground`}>{label}</span>
      <span className={`${size} font-mono ${good != null ? (good ? 'text-neon-green' : 'text-neon-orange') : 'text-white'}`}>
        {value.toFixed(format)}{unit ? ` ${unit}` : ''}
      </span>
    </div>
  )
}

function getRewardColor(reward: number): string {
  if (reward >= 8) return '#00FF88'
  if (reward >= 5) return '#FFD700'
  if (reward >= 3) return '#FFA07A'
  return '#FF6B6B'
}

// Molecule Gallery - grid of molecule cards
interface MoleculeGalleryProps {
  molecules: Molecule[]
  selectedIds?: Set<string>
  onSelect?: (id: string) => void
  onClick?: (id: string) => void
  columns?: number
  size?: 'sm' | 'md' | 'lg'
}

export function MoleculeGallery({
  molecules,
  selectedIds,
  onSelect,
  onClick,
  columns = 4,
  size = 'md',
}: MoleculeGalleryProps) {
  return (
    <div
      className="grid gap-3"
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {molecules.map((mol, i) => (
        <MoleculeCard
          key={mol.id}
          molecule={mol}
          rank={i + 1}
          selected={selectedIds?.has(mol.id)}
          onSelect={onSelect}
          onClick={onClick}
          size={size}
        />
      ))}
    </div>
  )
}
