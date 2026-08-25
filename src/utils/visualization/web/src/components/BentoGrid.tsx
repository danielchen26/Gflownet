import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface BentoGridProps {
  children: ReactNode
  columns?: number
  className?: string
}

export function BentoGrid({ children, columns = 4, className = '' }: BentoGridProps) {
  return (
    <div
      className={`grid gap-4 ${className}`}
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {children}
    </div>
  )
}

interface BentoCardProps {
  children: ReactNode
  colSpan?: number
  rowSpan?: number
  className?: string
  onClick?: () => void
  variant?: 'default' | 'gradient' | 'accent'
}

export function BentoCard({
  children,
  colSpan = 1,
  rowSpan = 1,
  className = '',
  onClick,
  variant = 'default',
}: BentoCardProps) {
  const variantClasses = {
    default: 'glass-dark',
    gradient: 'bg-gradient-to-br from-neon-purple/10 to-neon-blue/10 border border-neon-purple/20',
    accent: 'bg-gradient-to-br from-neon-green/10 to-neon-blue/10 border border-neon-green/20',
  }

  return (
    <motion.div
      whileHover={onClick ? { scale: 1.01 } : undefined}
      whileTap={onClick ? { scale: 0.99 } : undefined}
      className={`
        rounded-xl p-4 ${variantClasses[variant]}
        ${onClick ? 'cursor-pointer' : ''}
        ${className}
      `}
      style={{
        gridColumn: `span ${colSpan}`,
        gridRow: `span ${rowSpan}`,
      }}
      onClick={onClick}
    >
      {children}
    </motion.div>
  )
}

// Hero Metric Card with animated counter
interface MetricCardProps {
  label: string
  value: number | string
  icon: ReactNode
  trend?: { value: number; direction: 'up' | 'down' }
  sparkline?: number[]
  color?: string
}

export function MetricCard({ label, value, icon, trend, sparkline, color }: MetricCardProps) {
  return (
    <BentoCard className="flex flex-col justify-between min-h-[120px]">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">{label}</span>
        <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center" style={{ color }}>
          {icon}
        </div>
      </div>

      <div className="mt-2">
        <div className="text-2xl font-bold font-mono" style={{ color }}>
          {typeof value === 'number' ? value.toLocaleString() : value}
        </div>

        {trend && (
          <div className={`flex items-center mt-1 text-[10px] font-medium ${
            trend.direction === 'up' ? 'text-neon-green' : 'text-red-400'
          }`}>
            <span>{trend.direction === 'up' ? '+' : ''}{trend.value}%</span>
          </div>
        )}
      </div>

      {/* Mini Sparkline */}
      {sparkline && sparkline.length > 0 && (
        <div className="mt-2 flex items-end gap-0.5 h-6">
          {sparkline.map((v, i) => {
            const max = Math.max(...sparkline)
            const height = max > 0 ? (v / max) * 100 : 0
            return (
              <div
                key={i}
                className="flex-1 rounded-t-sm"
                style={{
                  height: `${Math.max(4, height)}%`,
                  backgroundColor: color || 'rgb(var(--neon-purple))',
                  opacity: 0.3 + (i / sparkline.length) * 0.7,
                }}
              />
            )
          })}
        </div>
      )}
    </BentoCard>
  )
}
