import { useState, useCallback, useEffect } from 'react'
import { Check, X, Copy, ClipboardPaste } from 'lucide-react'
import { MoleculeViewer2D } from './MoleculeViewer2D'
import { api } from '../services/api'

interface SMILESInputProps {
  value: string
  onChange: (smiles: string) => void
  placeholder?: string
  showPreview?: boolean
  previewSize?: number
  className?: string
  label?: string
}

export function SMILESInput({
  value,
  onChange,
  placeholder = 'Enter SMILES (e.g., c1ccccc1)',
  showPreview = true,
  previewSize = 150,
  className = '',
  label,
}: SMILESInputProps) {
  const [isValid, setIsValid] = useState<boolean | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // Debounced validation
  useEffect(() => {
    if (!value) {
      setIsValid(null)
      setValidationError(null)
      return
    }

    const timer = setTimeout(async () => {
      try {
        const result = await api.molecular.validateSmiles(value)
        setIsValid(result.valid)
        setValidationError(result.valid ? null : result.error || 'Invalid SMILES')
      } catch {
        // If backend is down, do basic client-side check
        const basicValid = /^[A-Za-z0-9@+\-\[\]\(\)\\\/=#%.*~]+$/.test(value)
        setIsValid(basicValid)
        setValidationError(basicValid ? null : 'Invalid characters in SMILES')
      }
    }, 300)

    return () => clearTimeout(timer)
  }, [value])

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(value)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [value])

  const handlePaste = useCallback(async () => {
    const text = await navigator.clipboard.readText()
    onChange(text.trim())
  }, [onChange])

  return (
    <div className={`space-y-2 ${className}`}>
      {label && (
        <label className="text-xs font-medium text-muted-foreground">{label}</label>
      )}

      <div className="flex items-start gap-3">
        {/* Input */}
        <div className="flex-1">
          <div className="relative">
            <input
              type="text"
              value={value}
              onChange={(e) => onChange(e.target.value)}
              placeholder={placeholder}
              className={`
                w-full px-3 py-2 rounded-lg bg-dark-bg/80 border text-sm font-mono
                outline-none transition-colors
                ${isValid === true
                  ? 'border-neon-green/50 focus:border-neon-green'
                  : isValid === false
                    ? 'border-red-500/50 focus:border-red-500'
                    : 'border-dark-border focus:border-neon-purple'
                }
              `}
              spellCheck={false}
            />

            {/* Status indicator */}
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center space-x-1">
              {isValid === true && <Check className="w-4 h-4 text-neon-green" />}
              {isValid === false && <X className="w-4 h-4 text-red-500" />}
              {value && (
                <button onClick={handleCopy} className="p-0.5 hover:text-white text-muted-foreground transition-colors">
                  {copied ? <Check className="w-3 h-3 text-neon-green" /> : <Copy className="w-3 h-3" />}
                </button>
              )}
              <button onClick={handlePaste} className="p-0.5 hover:text-white text-muted-foreground transition-colors">
                <ClipboardPaste className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Validation message */}
          {validationError && (
            <p className="mt-1 text-[10px] text-red-400">{validationError}</p>
          )}
        </div>

        {/* Live 2D Preview */}
        {showPreview && value && isValid !== false && (
          <div className="flex-shrink-0 rounded-lg overflow-hidden border border-dark-border/50 bg-white/5">
            <MoleculeViewer2D smiles={value} width={previewSize} height={previewSize} />
          </div>
        )}
      </div>
    </div>
  )
}
