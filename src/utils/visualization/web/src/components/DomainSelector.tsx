import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Grid3X3,
  Share2,
  Type,
  Atom,
  Settings,
  Search,
  Star,
  Layers,
  Sparkles,
  ChevronRight
} from 'lucide-react'

// Types
export type RendererType = 'grid2d' | 'graph3d' | 'sequence' | 'molecule3d' | 'custom'

export interface DomainConfigSchema {
  type: string
  properties: Record<string, {
    type: string
    description?: string
    default?: unknown
    minimum?: number
    maximum?: number
    enum?: string[]
  }>
  required?: string[]
}

export interface DomainOption {
  id: string
  name: string
  description: string
  icon: React.ElementType
  renderer: RendererType
  isBuiltIn: boolean
  configSchema: DomainConfigSchema
  tags: string[]
  isPopular?: boolean
}

// Schema definitions for built-in domains
const gridWorldSchema: DomainConfigSchema = {
  type: 'object',
  properties: {
    grid_size: {
      type: 'integer',
      description: 'Size of the grid (NxN)',
      default: 8,
      minimum: 4,
      maximum: 32
    },
    reward_positions: {
      type: 'array',
      description: 'Positions and values of reward peaks'
    },
    obstacles: {
      type: 'array',
      description: 'Positions of obstacles'
    }
  },
  required: ['grid_size']
}

const dagSchema: DomainConfigSchema = {
  type: 'object',
  properties: {
    node_count: {
      type: 'integer',
      description: 'Number of nodes in the DAG',
      default: 10,
      minimum: 3,
      maximum: 100
    },
    edge_density: {
      type: 'number',
      description: 'Probability of edge creation',
      default: 0.3,
      minimum: 0.1,
      maximum: 0.9
    },
    source_node: {
      type: 'integer',
      description: 'Starting node ID',
      default: 0
    },
    sink_nodes: {
      type: 'array',
      description: 'Terminal node IDs'
    }
  },
  required: ['node_count']
}

const sequenceSchema: DomainConfigSchema = {
  type: 'object',
  properties: {
    vocabulary: {
      type: 'array',
      description: 'Available tokens/characters',
      default: ['A', 'B', 'C', 'D']
    },
    max_length: {
      type: 'integer',
      description: 'Maximum sequence length',
      default: 10,
      minimum: 1,
      maximum: 100
    },
    target_sequence: {
      type: 'string',
      description: 'Target sequence for reward calculation'
    },
    scoring_method: {
      type: 'string',
      description: 'How to score generated sequences',
      enum: ['exact_match', 'partial_match', 'edit_distance', 'custom']
    }
  },
  required: ['vocabulary', 'max_length']
}

const moleculeSchema: DomainConfigSchema = {
  type: 'object',
  properties: {
    allowed_atoms: {
      type: 'array',
      description: 'Allowed atom types',
      default: ['C', 'N', 'O', 'S']
    },
    max_atoms: {
      type: 'integer',
      description: 'Maximum number of atoms',
      default: 20,
      minimum: 2,
      maximum: 100
    },
    target_properties: {
      type: 'object',
      description: 'Target molecular properties (MW, logP, etc.)'
    },
    scoring_function: {
      type: 'string',
      description: 'Scoring function for molecules',
      enum: ['qed', 'logp', 'binding_affinity', 'custom']
    }
  },
  required: ['allowed_atoms', 'max_atoms']
}

const customSchema: DomainConfigSchema = {
  type: 'object',
  properties: {
    state_type: {
      type: 'string',
      description: 'Type of state representation',
      enum: ['discrete', 'continuous', 'composite']
    },
    action_type: {
      type: 'string',
      description: 'Type of action space',
      enum: ['discrete', 'continuous']
    },
    state_features: {
      type: 'array',
      description: 'Feature definitions for states'
    },
    actions: {
      type: 'array',
      description: 'Action definitions'
    },
    reward_function: {
      type: 'string',
      description: 'Custom reward function code'
    }
  },
  required: ['state_type', 'action_type']
}

// Built-in domains
export const BUILT_IN_DOMAINS: DomainOption[] = [
  {
    id: 'grid_world',
    name: 'Grid World',
    description: '2D grid navigation with configurable rewards and obstacles. Perfect for learning GFlowNet fundamentals.',
    icon: Grid3X3,
    renderer: 'grid2d',
    isBuiltIn: true,
    configSchema: gridWorldSchema,
    tags: ['navigation', '2d', 'beginner'],
    isPopular: true
  },
  {
    id: 'dag',
    name: 'DAG / Graph',
    description: 'Directed acyclic graph traversal with customizable topology and node rewards.',
    icon: Share2,
    renderer: 'graph3d',
    isBuiltIn: true,
    configSchema: dagSchema,
    tags: ['graph', 'network', 'traversal'],
    isPopular: true
  },
  {
    id: 'sequence',
    name: 'Sequence Generation',
    description: 'Token-by-token sequence building with vocabulary constraints and target matching.',
    icon: Type,
    renderer: 'sequence',
    isBuiltIn: true,
    configSchema: sequenceSchema,
    tags: ['text', 'generation', 'nlp'],
    isPopular: true
  },
  {
    id: 'molecule',
    name: 'Molecular Design',
    description: 'Atom-by-atom molecule construction with property optimization. Ideal for drug discovery.',
    icon: Atom,
    renderer: 'molecule3d',
    isBuiltIn: true,
    configSchema: moleculeSchema,
    tags: ['chemistry', 'drug-discovery', 'advanced'],
    isPopular: false
  },
  {
    id: 'custom',
    name: 'Custom Domain',
    description: 'Define your own state/action space with custom reward functions and transitions.',
    icon: Settings,
    renderer: 'custom',
    isBuiltIn: false,
    configSchema: customSchema,
    tags: ['custom', 'advanced', 'flexible'],
    isPopular: false
  }
]

type TabType = 'popular' | 'all' | 'custom'

interface DomainSelectorProps {
  selectedDomain?: string
  onDomainSelect: (domain: DomainOption) => void
  showTabs?: boolean
}

export function DomainSelector({
  selectedDomain,
  onDomainSelect,
  showTabs = true
}: DomainSelectorProps) {
  const [activeTab, setActiveTab] = useState<TabType>('popular')
  const [searchQuery, setSearchQuery] = useState('')
  const [hoveredDomain, setHoveredDomain] = useState<string | null>(null)

  // Filter domains based on tab and search
  const filteredDomains = useMemo(() => {
    let domains = BUILT_IN_DOMAINS

    // Filter by tab
    if (activeTab === 'popular') {
      domains = domains.filter(d => d.isPopular)
    } else if (activeTab === 'custom') {
      domains = domains.filter(d => !d.isBuiltIn)
    }

    // Filter by search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      domains = domains.filter(d =>
        d.name.toLowerCase().includes(query) ||
        d.description.toLowerCase().includes(query) ||
        d.tags.some(tag => tag.toLowerCase().includes(query))
      )
    }

    return domains
  }, [activeTab, searchQuery])

  const tabs: { id: TabType; label: string; icon: React.ElementType }[] = [
    { id: 'popular', label: 'Popular', icon: Star },
    { id: 'all', label: 'All Domains', icon: Layers },
    { id: 'custom', label: 'Custom', icon: Sparkles }
  ]

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Select Domain</h2>
          <p className="text-sm text-muted-foreground">
            Choose a domain type for your GFlowNet experiment
          </p>
        </div>
      </div>

      {/* Search and Tabs */}
      <div className="space-y-3">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search domains..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-dark-card border border-dark-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-neon-purple/50 focus:border-neon-purple transition-all"
          />
        </div>

        {/* Tabs */}
        {showTabs && (
          <div className="flex gap-2 p-1 bg-dark-card/50 rounded-lg">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all flex-1 justify-center ${
                    activeTab === tab.id
                      ? 'bg-gradient-to-r from-neon-purple/20 to-neon-blue/20 text-white border border-neon-purple/30'
                      : 'text-muted-foreground hover:text-white hover:bg-dark-card'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* Domain Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnimatePresence mode="popLayout">
          {filteredDomains.map((domain, index) => {
            const Icon = domain.icon
            const isSelected = selectedDomain === domain.id
            const isHovered = hoveredDomain === domain.id

            return (
              <motion.div
                key={domain.id}
                layout
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ delay: index * 0.05 }}
                onMouseEnter={() => setHoveredDomain(domain.id)}
                onMouseLeave={() => setHoveredDomain(null)}
                onClick={() => onDomainSelect(domain)}
                className={`relative cursor-pointer rounded-xl border p-4 transition-all ${
                  isSelected
                    ? 'bg-gradient-to-br from-neon-purple/20 to-neon-blue/10 border-neon-purple shadow-lg shadow-neon-purple/20'
                    : 'glass-dark border-dark-border hover:border-neon-purple/50 hover:shadow-md'
                }`}
              >
                {/* Popular badge */}
                {domain.isPopular && (
                  <div className="absolute -top-2 -right-2">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="bg-gradient-to-r from-neon-orange-from to-neon-orange-to text-white text-xs px-2 py-0.5 rounded-full font-medium"
                    >
                      Popular
                    </motion.div>
                  </div>
                )}

                {/* Icon and Title */}
                <div className="flex items-start gap-3">
                  <div className={`p-3 rounded-xl ${
                    isSelected
                      ? 'bg-gradient-to-br from-neon-purple to-neon-blue'
                      : 'bg-gradient-to-br from-dark-card to-dark-border'
                  }`}>
                    <Icon className={`w-6 h-6 ${isSelected ? 'text-white' : 'text-neon-purple'}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-white">{domain.name}</h3>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {domain.renderer.toUpperCase()} Renderer
                    </p>
                  </div>
                  <motion.div
                    animate={{ x: isHovered ? 4 : 0, opacity: isHovered ? 1 : 0.5 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronRight className="w-5 h-5 text-muted-foreground" />
                  </motion.div>
                </div>

                {/* Description */}
                <p className="text-sm text-muted-foreground mt-3 line-clamp-2">
                  {domain.description}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {domain.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 text-xs rounded-full bg-dark-card/80 text-muted-foreground border border-dark-border"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                {/* Selection indicator */}
                {isSelected && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="absolute bottom-3 right-3"
                  >
                    <div className="w-6 h-6 rounded-full bg-neon-green flex items-center justify-center">
                      <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>

      {/* Empty state */}
      {filteredDomains.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <Search className="w-12 h-12 text-muted-foreground mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium text-muted-foreground">No domains found</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Try adjusting your search or filter criteria
          </p>
        </motion.div>
      )}
    </div>
  )
}
