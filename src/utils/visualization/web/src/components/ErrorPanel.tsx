import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, X, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'

interface ErrorPanelProps {
  errorCount: number
  lastError: string | null
  consecutiveFailures?: number
  onClear?: () => void
  onRetry?: () => void
}

export function ErrorPanel({
  errorCount,
  lastError,
  consecutiveFailures = 0,
  onClear,
  onRetry
}: ErrorPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  if (errorCount === 0 && !lastError) {
    return null
  }

  const isRecovering = consecutiveFailures > 0 && consecutiveFailures < 3
  const isCritical = consecutiveFailures >= 3

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20, height: 0 }}
        animate={{ opacity: 1, y: 0, height: 'auto' }}
        exit={{ opacity: 0, y: -20, height: 0 }}
        className={`rounded-lg border ${
          isCritical
            ? 'bg-red-900/30 border-red-500/50'
            : isRecovering
              ? 'bg-yellow-900/30 border-yellow-500/50'
              : 'bg-orange-900/30 border-orange-500/50'
        }`}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between p-3 cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className={`w-4 h-4 ${
              isCritical ? 'text-red-400' : isRecovering ? 'text-yellow-400' : 'text-orange-400'
            }`} />
            <span className="text-sm font-medium">
              {isCritical
                ? 'Critical Error'
                : isRecovering
                  ? 'Recovering...'
                  : `${errorCount} Error${errorCount > 1 ? 's' : ''}`
              }
            </span>
            {consecutiveFailures > 0 && (
              <span className="text-xs text-muted-foreground">
                ({consecutiveFailures} consecutive)
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {onRetry && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onRetry()
                }}
                className="p-1 hover:bg-dark-panel/50 rounded"
                title="Retry"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            )}
            {onClear && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onClear()
                }}
                className="p-1 hover:bg-dark-panel/50 rounded"
                title="Clear Errors"
              >
                <X className="w-3 h-3" />
              </button>
            )}
            {isExpanded ? (
              <ChevronUp className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            )}
          </div>
        </div>

        {/* Expanded Error Details */}
        <AnimatePresence>
          {isExpanded && lastError && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="border-t border-dark-border/30"
            >
              <div className="p-3">
                <p className="text-xs text-muted-foreground mb-1">Last Error:</p>
                <pre className="text-xs bg-dark-panel/50 p-2 rounded overflow-x-auto max-h-32 overflow-y-auto">
                  {lastError}
                </pre>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Recovery Status */}
        {isRecovering && (
          <div className="px-3 pb-3">
            <div className="flex items-center gap-2">
              <RefreshCw className="w-3 h-3 text-yellow-400 animate-spin" />
              <span className="text-xs text-yellow-400">
                Auto-recovery in progress...
              </span>
            </div>
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  )
}
