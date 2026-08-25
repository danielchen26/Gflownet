import { useCallback, useRef } from 'react'

interface PropertySliderRangeProps {
  label: string
  min: number
  max: number
  value: [number, number]
  onChange: (value: [number, number]) => void
  step?: number
  unit?: string
  format?: (v: number) => string
  className?: string
}

export function PropertySliderRange({
  label,
  min,
  max,
  value,
  onChange,
  step = 1,
  unit = '',
  format = (v) => v.toString(),
  className = '',
}: PropertySliderRangeProps) {
  const trackRef = useRef<HTMLDivElement>(null)

  const leftPercent = ((value[0] - min) / (max - min)) * 100
  const rightPercent = ((value[1] - min) / (max - min)) * 100

  const handleMinChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newMin = parseFloat(e.target.value)
      onChange([Math.min(newMin, value[1] - step), value[1]])
    },
    [onChange, value, step]
  )

  const handleMaxChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newMax = parseFloat(e.target.value)
      onChange([value[0], Math.max(newMax, value[0] + step)])
    },
    [onChange, value, step]
  )

  return (
    <div className={`space-y-1 ${className}`}>
      <div className="flex items-center justify-between">
        <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">{label}</label>
        <span className="text-[10px] font-mono text-white">
          {format(value[0])}{unit} – {format(value[1])}{unit}
        </span>
      </div>

      <div className="relative h-6" ref={trackRef}>
        {/* Track background */}
        <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1 rounded-full bg-dark-border" />

        {/* Active range */}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-1 rounded-full bg-gradient-to-r from-neon-purple to-neon-blue"
          style={{ left: `${leftPercent}%`, width: `${rightPercent - leftPercent}%` }}
        />

        {/* Min slider */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value[0]}
          onChange={handleMinChange}
          className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer z-10 pointer-events-auto"
          style={{ clipPath: `inset(0 ${100 - (leftPercent + rightPercent) / 2}% 0 0)` }}
        />

        {/* Max slider */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value[1]}
          onChange={handleMaxChange}
          className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer z-10 pointer-events-auto"
          style={{ clipPath: `inset(0 0 0 ${(leftPercent + rightPercent) / 2}%)` }}
        />

        {/* Thumb indicators */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-white border-2 border-neon-purple shadow-md pointer-events-none"
          style={{ left: `calc(${leftPercent}% - 6px)` }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-white border-2 border-neon-blue shadow-md pointer-events-none"
          style={{ left: `calc(${rightPercent}% - 6px)` }}
        />
      </div>
    </div>
  )
}
