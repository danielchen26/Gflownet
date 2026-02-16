import { useState, useRef, useEffect } from 'react'
import { Palette } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTheme, THEMES, type ThemeName } from '../contexts/ThemeContext'

export function ThemeSelector() {
  const { theme, setTheme } = useTheme()
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Close on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const currentTheme = THEMES.find(t => t.id === theme)

  return (
    <div ref={containerRef} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1 rounded-full bg-dark-panel/80 border border-dark-border/50 hover:border-dark-border transition-colors"
        title="Change theme"
      >
        <Palette className="w-3.5 h-3.5 text-muted-foreground" />
        <div className="flex gap-0.5">
          {currentTheme?.colors.map((color, i) => (
            <div
              key={i}
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: color }}
            />
          ))}
        </div>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 top-full mt-2 z-[200] w-56 rounded-lg shadow-xl overflow-hidden border border-dark-border"
            style={{
              backdropFilter: 'none',
              WebkitBackdropFilter: 'none',
              backgroundColor: 'rgb(var(--dark-panel))',
              isolation: 'isolate',
            }}
          >
            {THEMES.map((t) => (
              <button
                key={t.id}
                onClick={() => {
                  setTheme(t.id as ThemeName)
                  setIsOpen(false)
                }}
                className={`w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors ${
                  theme === t.id
                    ? 'bg-neon-purple/10 text-white'
                    : 'text-muted-foreground hover:text-white hover:bg-white/5'
                }`}
              >
                <div className="flex gap-1 shrink-0">
                  {t.colors.map((color, i) => (
                    <div
                      key={i}
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium">{t.label}</div>
                  <div className="text-[9px] opacity-50 truncate">{t.description}</div>
                </div>
                {theme === t.id && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-neon-purple shrink-0" />
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
