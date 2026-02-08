import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Settings,
  Info,
  Plus,
  Trash2,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle
} from 'lucide-react'
import { DomainOption, DomainConfigSchema } from './DomainSelector'

interface DomainConfig {
  [key: string]: unknown
}

interface DomainConfigPanelProps {
  domain: DomainOption
  config: DomainConfig
  onConfigChange: (config: DomainConfig) => void
  onValidationChange?: (isValid: boolean) => void
}

interface FieldProps {
  name: string
  schema: {
    type: string
    description?: string
    default?: unknown
    minimum?: number
    maximum?: number
    enum?: string[]
  }
  value: unknown
  onChange: (value: unknown) => void
  required?: boolean
}

// Field components for different types
function NumberField({ name, schema, value, onChange, required }: FieldProps) {
  const numValue = typeof value === 'number' ? value : (schema.default as number) || 0

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium flex items-center gap-1">
          {formatLabel(name)}
          {required && <span className="text-neon-pink">*</span>}
        </label>
        <span className="text-xs text-muted-foreground">
          {schema.minimum !== undefined && schema.maximum !== undefined
            ? `${schema.minimum} - ${schema.maximum}`
            : schema.type === 'integer' ? 'Integer' : 'Number'}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <input
          type="number"
          value={numValue}
          min={schema.minimum}
          max={schema.maximum}
          step={schema.type === 'integer' ? 1 : 0.1}
          onChange={(e) => {
            const val = schema.type === 'integer'
              ? parseInt(e.target.value, 10)
              : parseFloat(e.target.value)
            onChange(isNaN(val) ? schema.default : val)
          }}
          className="flex-1 px-3 py-2 bg-dark-card border border-dark-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-neon-purple/50 focus:border-neon-purple transition-all"
        />
        {schema.minimum !== undefined && schema.maximum !== undefined && (
          <input
            type="range"
            min={schema.minimum}
            max={schema.maximum}
            step={schema.type === 'integer' ? 1 : (schema.maximum - schema.minimum) / 100}
            value={numValue}
            onChange={(e) => {
              const val = schema.type === 'integer'
                ? parseInt(e.target.value, 10)
                : parseFloat(e.target.value)
              onChange(val)
            }}
            className="w-32 accent-neon-purple"
          />
        )}
      </div>
      {schema.description && (
        <p className="text-xs text-muted-foreground">{schema.description}</p>
      )}
    </div>
  )
}

function StringField({ name, schema, value, onChange, required }: FieldProps) {
  const strValue = typeof value === 'string' ? value : (schema.default as string) || ''

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium flex items-center gap-1">
          {formatLabel(name)}
          {required && <span className="text-neon-pink">*</span>}
        </label>
      </div>
      {schema.enum ? (
        <select
          value={strValue}
          onChange={(e) => onChange(e.target.value)}
          className="w-full px-3 py-2 bg-dark-card border border-dark-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-neon-purple/50 focus:border-neon-purple transition-all"
        >
          <option value="">Select {formatLabel(name)}...</option>
          {schema.enum.map((opt) => (
            <option key={opt} value={opt}>
              {formatLabel(opt)}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="text"
          value={strValue}
          onChange={(e) => onChange(e.target.value)}
          placeholder={schema.description || `Enter ${formatLabel(name).toLowerCase()}`}
          className="w-full px-3 py-2 bg-dark-card border border-dark-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-neon-purple/50 focus:border-neon-purple transition-all"
        />
      )}
      {schema.description && !schema.enum && (
        <p className="text-xs text-muted-foreground">{schema.description}</p>
      )}
    </div>
  )
}

function ArrayField({ name, schema, value, onChange, required }: FieldProps) {
  const arrValue = Array.isArray(value) ? value : (schema.default as unknown[]) || []
  const [newItem, setNewItem] = useState('')

  const addItem = () => {
    if (newItem.trim()) {
      onChange([...arrValue, newItem.trim()])
      setNewItem('')
    }
  }

  const removeItem = (index: number) => {
    onChange(arrValue.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium flex items-center gap-1">
          {formatLabel(name)}
          {required && <span className="text-neon-pink">*</span>}
        </label>
        <span className="text-xs text-muted-foreground">
          {arrValue.length} item{arrValue.length !== 1 ? 's' : ''}
        </span>
      </div>
      {schema.description && (
        <p className="text-xs text-muted-foreground">{schema.description}</p>
      )}
      <div className="flex gap-2">
        <input
          type="text"
          value={newItem}
          onChange={(e) => setNewItem(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && addItem()}
          placeholder="Add item..."
          className="flex-1 px-3 py-2 bg-dark-card border border-dark-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-neon-purple/50 focus:border-neon-purple transition-all"
        />
        <button
          onClick={addItem}
          disabled={!newItem.trim()}
          className="p-2 bg-neon-purple/20 hover:bg-neon-purple/30 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Plus className="w-4 h-4 text-neon-purple" />
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        <AnimatePresence>
          {arrValue.map((item, index) => (
            <motion.div
              key={`${item}-${index}`}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="flex items-center gap-1 px-2 py-1 bg-dark-card border border-dark-border rounded-lg"
            >
              <span className="text-sm">{String(item)}</span>
              <button
                onClick={() => removeItem(index)}
                className="p-0.5 hover:bg-red-500/20 rounded transition-colors"
              >
                <Trash2 className="w-3 h-3 text-red-400" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  )
}

function ObjectField({ name, schema, value, onChange, required }: FieldProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const objValue = typeof value === 'object' && value !== null ? value : {}

  return (
    <div className="space-y-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between w-full p-2 bg-dark-card border border-dark-border rounded-lg hover:border-neon-purple/50 transition-colors"
      >
        <span className="text-sm font-medium flex items-center gap-1">
          {formatLabel(name)}
          {required && <span className="text-neon-pink">*</span>}
        </span>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        )}
      </button>
      {schema.description && (
        <p className="text-xs text-muted-foreground">{schema.description}</p>
      )}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="pl-4 border-l-2 border-dark-border space-y-3"
          >
            <p className="text-xs text-muted-foreground italic">
              Object configuration will be available when domain is fully implemented.
            </p>
            <pre className="text-xs bg-dark-card p-2 rounded overflow-x-auto">
              {JSON.stringify(objValue, null, 2)}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function BooleanField({ name, schema, value, onChange, required }: FieldProps) {
  const boolValue = typeof value === 'boolean' ? value : (schema.default as boolean) || false

  return (
    <div className="flex items-center justify-between p-3 bg-dark-card border border-dark-border rounded-lg">
      <div className="space-y-0.5">
        <label className="text-sm font-medium flex items-center gap-1">
          {formatLabel(name)}
          {required && <span className="text-neon-pink">*</span>}
        </label>
        {schema.description && (
          <p className="text-xs text-muted-foreground">{schema.description}</p>
        )}
      </div>
      <button
        onClick={() => onChange(!boolValue)}
        className={`relative w-12 h-6 rounded-full transition-colors ${
          boolValue ? 'bg-neon-green' : 'bg-dark-border'
        }`}
      >
        <motion.div
          animate={{ x: boolValue ? 24 : 2 }}
          className="absolute top-1 w-4 h-4 bg-white rounded-full shadow"
        />
      </button>
    </div>
  )
}

// Helper to format field labels
function formatLabel(name: string): string {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

export function DomainConfigPanel({
  domain,
  config,
  onConfigChange,
  onValidationChange
}: DomainConfigPanelProps) {
  const [localConfig, setLocalConfig] = useState<DomainConfig>(config)
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})
  const [isValid, setIsValid] = useState(true)

  // Update local config when prop changes
  useEffect(() => {
    setLocalConfig(config)
  }, [config])

  // Validate config
  const validate = useCallback(() => {
    const errors: Record<string, string> = {}
    const schema = domain.configSchema

    // Check required fields
    schema.required?.forEach(field => {
      const value = localConfig[field]
      if (value === undefined || value === null || value === '') {
        errors[field] = `${formatLabel(field)} is required`
      }
    })

    // Check field-specific validations
    Object.entries(schema.properties).forEach(([field, fieldSchema]) => {
      const value = localConfig[field]
      if (value !== undefined && value !== null) {
        if (fieldSchema.type === 'integer' || fieldSchema.type === 'number') {
          const numVal = Number(value)
          if (fieldSchema.minimum !== undefined && numVal < fieldSchema.minimum) {
            errors[field] = `Must be at least ${fieldSchema.minimum}`
          }
          if (fieldSchema.maximum !== undefined && numVal > fieldSchema.maximum) {
            errors[field] = `Must be at most ${fieldSchema.maximum}`
          }
        }
      }
    })

    setValidationErrors(errors)
    const valid = Object.keys(errors).length === 0
    setIsValid(valid)
    onValidationChange?.(valid)

    return valid
  }, [localConfig, domain.configSchema, onValidationChange])

  // Validate on config change
  useEffect(() => {
    validate()
  }, [validate])

  // Handle field change
  const handleFieldChange = (field: string, value: unknown) => {
    const newConfig = { ...localConfig, [field]: value }
    setLocalConfig(newConfig)
    onConfigChange(newConfig)
  }

  // Render field based on type
  const renderField = (name: string, schema: FieldProps['schema']) => {
    const value = localConfig[name]
    const required = domain.configSchema.required?.includes(name)
    const error = validationErrors[name]

    const props: FieldProps = {
      name,
      schema,
      value,
      onChange: (val) => handleFieldChange(name, val),
      required
    }

    let field: React.ReactNode
    switch (schema.type) {
      case 'integer':
      case 'number':
        field = <NumberField {...props} />
        break
      case 'string':
        field = <StringField {...props} />
        break
      case 'array':
        field = <ArrayField {...props} />
        break
      case 'object':
        field = <ObjectField {...props} />
        break
      case 'boolean':
        field = <BooleanField {...props} />
        break
      default:
        field = <StringField {...props} />
    }

    return (
      <div key={name} className="space-y-1">
        {field}
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-xs text-red-400 flex items-center gap-1"
          >
            <AlertCircle className="w-3 h-3" />
            {error}
          </motion.p>
        )}
      </div>
    )
  }

  const Icon = domain.icon

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-neon-purple to-neon-blue">
            <Icon className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="font-semibold">{domain.name} Configuration</h3>
            <p className="text-xs text-muted-foreground">
              Configure domain-specific parameters
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isValid ? (
            <div className="flex items-center gap-1 text-neon-green text-xs">
              <CheckCircle className="w-4 h-4" />
              Valid
            </div>
          ) : (
            <div className="flex items-center gap-1 text-red-400 text-xs">
              <AlertCircle className="w-4 h-4" />
              {Object.keys(validationErrors).length} error(s)
            </div>
          )}
        </div>
      </div>

      {/* Info Banner */}
      <div className="flex items-start gap-2 p-3 bg-neon-blue/10 border border-neon-blue/30 rounded-lg">
        <Info className="w-4 h-4 text-neon-blue mt-0.5 shrink-0" />
        <p className="text-xs text-muted-foreground">
          {domain.description}
        </p>
      </div>

      {/* Configuration Fields */}
      <div className="space-y-4">
        {Object.entries(domain.configSchema.properties).map(([name, schema]) =>
          renderField(name, schema)
        )}
      </div>

      {/* Config Preview */}
      <details className="group">
        <summary className="flex items-center gap-2 cursor-pointer text-sm text-muted-foreground hover:text-white transition-colors">
          <Settings className="w-4 h-4" />
          <span>View JSON Configuration</span>
          <ChevronDown className="w-4 h-4 group-open:rotate-180 transition-transform" />
        </summary>
        <motion.pre
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="mt-2 p-3 bg-dark-card border border-dark-border rounded-lg text-xs overflow-x-auto"
        >
          {JSON.stringify(localConfig, null, 2)}
        </motion.pre>
      </details>
    </motion.div>
  )
}
