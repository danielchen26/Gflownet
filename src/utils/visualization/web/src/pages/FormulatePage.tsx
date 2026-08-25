import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Bot, Sparkles, ArrowRight, ChevronRight, Lightbulb,
  Grid3X3, Atom, Share2, Type, Settings,
  Box, Shuffle, Target, Zap, CheckCircle2,
} from 'lucide-react'
import type { ViewId } from '../components/Sidebar'

interface FormulatePageProps {
  onNavigate: (view: ViewId) => void
  onDomainSelect: (domainId: string) => void
}

/* ─── GFlowNet Framework Concepts ─── */

const FRAMEWORK_PILLARS = [
  {
    id: 'state-space',
    label: 'State Space (S)',
    icon: Box,
    color: '#BD00FF',
    description: 'The set of all possible partial and complete objects your GFlowNet can construct.',
    examples: [
      { domain: 'Grid World', detail: 'All (x, y) positions on an N×N grid' },
      { domain: 'Molecular', detail: 'All partial molecules built from fragment library' },
      { domain: 'DAG', detail: 'All sub-graphs of the target DAG' },
      { domain: 'Sequence', detail: 'All prefixes of length ≤ L over vocabulary V' },
    ],
  },
  {
    id: 'action-space',
    label: 'Action Space (A)',
    icon: Shuffle,
    color: '#00D9FF',
    description: 'Transitions between states. Each action adds a component, moves, or terminates construction.',
    examples: [
      { domain: 'Grid World', detail: 'Move right or down on the grid' },
      { domain: 'Molecular', detail: 'Attach fragment at available bond site' },
      { domain: 'DAG', detail: 'Add an edge to the current sub-graph' },
      { domain: 'Sequence', detail: 'Append next token from vocabulary' },
    ],
  },
  {
    id: 'reward',
    label: 'Reward Function R(x)',
    icon: Target,
    color: '#00FF88',
    description: 'Scores terminal states. GFlowNet samples x proportionally to R(x), discovering diverse high-reward solutions.',
    examples: [
      { domain: 'Grid World', detail: 'Multi-modal peaks at configurable positions' },
      { domain: 'Molecular', detail: 'QED × SA score × property constraints' },
      { domain: 'DAG', detail: 'Bayesian posterior or graph score' },
      { domain: 'Sequence', detail: 'Target similarity or language model score' },
    ],
  },
  {
    id: 'objective',
    label: 'Training Objective',
    icon: Zap,
    color: '#F59E0B',
    description: 'The loss function that enforces flow conservation. Different objectives have different trade-offs.',
    examples: [
      { domain: 'TB', detail: 'Trajectory Balance — most common, robust, single log-ratio' },
      { domain: 'DB', detail: 'Detailed Balance — local flow matching at each transition' },
      { domain: 'STB', detail: 'Sub-Trajectory Balance — generalizes TB with partial credit' },
      { domain: 'FM', detail: 'Flow Matching — enforces flow in = flow out at each state' },
      { domain: 'TLM', detail: 'Trajectory Likelihood Max. — maximum likelihood variant' },
    ],
  },
]

const DOMAIN_TEMPLATES = [
  {
    id: 'grid_world',
    name: 'Grid World',
    icon: Grid3X3,
    color: '#BD00FF',
    tagline: 'Best for learning GFlowNet fundamentals',
    stateSpace: 'N×N grid positions',
    actionSpace: 'Move right / down',
    reward: 'Configurable multi-modal peaks',
    recommended: 'TB with epsilon exploration',
  },
  {
    id: 'molecule',
    name: 'Molecular Design',
    icon: Atom,
    color: '#00FF88',
    tagline: 'Drug-like molecule generation',
    stateSpace: 'Partial molecules (50 fragments)',
    actionSpace: 'Attach fragment at bond site',
    reward: 'QED × SA × property targets',
    recommended: 'TB with replay buffer',
  },
  {
    id: 'dag',
    name: 'DAG / Graph',
    icon: Share2,
    color: '#00D9FF',
    tagline: 'Directed acyclic graph construction',
    stateSpace: 'Sub-graphs of target topology',
    actionSpace: 'Add edge to current graph',
    reward: 'Bayesian score / posterior',
    recommended: 'DB or FM objective',
  },
  {
    id: 'sequence',
    name: 'Sequence',
    icon: Type,
    color: '#F59E0B',
    tagline: 'Token-by-token generation',
    stateSpace: 'Prefixes of length ≤ L',
    actionSpace: 'Append token from vocabulary',
    reward: 'Target similarity / LM score',
    recommended: 'TB or STB objective',
  },
  {
    id: 'custom',
    name: 'Custom Domain',
    icon: Settings,
    color: '#FF6B6B',
    tagline: 'Define your own problem',
    stateSpace: 'Your custom state representation',
    actionSpace: 'Your custom transitions',
    reward: 'Your custom reward function',
    recommended: 'Depends on problem structure',
  },
]

export function FormulatePage({ onNavigate, onDomainSelect }: FormulatePageProps) {
  const [expandedPillar, setExpandedPillar] = useState<string | null>(null)

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-purple to-neon-blue flex items-center justify-center">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold gradient-text">Formulate Your GFlowNet Problem</h1>
            <p className="text-xs text-muted-foreground">
              Define the optimization framework — GFlowNet works across any domain with discrete state/action spaces
            </p>
          </div>
        </div>
      </div>

      {/* Core Concept Banner */}
      <div className="glass-dark rounded-xl p-5 border border-neon-purple/20">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-lg bg-neon-purple/10 flex items-center justify-center flex-shrink-0 mt-0.5">
            <Sparkles className="w-5 h-5 text-neon-purple" />
          </div>
          <div>
            <h3 className="text-sm font-semibold mb-1">What is GFlowNet?</h3>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              <strong className="text-white">Generative Flow Networks</strong> learn to sample diverse, high-quality objects
              proportionally to a reward function R(x). Unlike reinforcement learning (which collapses to a single optimum),
              GFlowNet discovers the <em>full distribution</em> of good solutions — ideal for drug discovery, material design,
              combinatorial optimization, and any problem where diversity matters.
            </p>
            <div className="flex items-center gap-4 mt-3 text-[10px]">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded bg-neon-green/20 flex items-center justify-center">
                  <CheckCircle2 className="w-3 h-3 text-neon-green" />
                </div>
                <span className="text-muted-foreground">Diverse sampling P(x) ∝ R(x)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded bg-neon-green/20 flex items-center justify-center">
                  <CheckCircle2 className="w-3 h-3 text-neon-green" />
                </div>
                <span className="text-muted-foreground">Amortized generation (fast at test time)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded bg-neon-green/20 flex items-center justify-center">
                  <CheckCircle2 className="w-3 h-3 text-neon-green" />
                </div>
                <span className="text-muted-foreground">Works with any discrete DAG structure</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Framework Pillars — the 4 components every GFlowNet problem needs */}
      <div>
        <h2 className="text-lg font-semibold mb-1">GFlowNet Framework</h2>
        <p className="text-[10px] text-muted-foreground mb-4">
          Every GFlowNet problem is defined by four components. Click each to see domain-specific examples.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {FRAMEWORK_PILLARS.map((pillar, i) => {
            const Icon = pillar.icon
            const isExpanded = expandedPillar === pillar.id
            return (
              <motion.button
                key={pillar.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                onClick={() => setExpandedPillar(isExpanded ? null : pillar.id)}
                className="glass-dark rounded-xl p-4 text-left border border-dark-border hover:border-neon-purple/30 transition-all"
              >
                <div className="flex items-center gap-3 mb-2">
                  <div
                    className="w-9 h-9 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${pillar.color}15` }}
                  >
                    <Icon className="w-4 h-4" style={{ color: pillar.color }} />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-sm font-semibold">{pillar.label}</h3>
                  </div>
                  <ChevronRight
                    className={`w-4 h-4 text-muted-foreground transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                  />
                </div>
                <p className="text-[10px] text-muted-foreground leading-relaxed">{pillar.description}</p>

                {isExpanded && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-3 pt-3 border-t border-dark-border/50 space-y-1.5"
                  >
                    {pillar.examples.map((ex) => (
                      <div key={ex.domain} className="flex items-start gap-2 text-[10px]">
                        <span className="font-medium text-white min-w-[70px]">{ex.domain}</span>
                        <span className="text-muted-foreground">{ex.detail}</span>
                      </div>
                    ))}
                  </motion.div>
                )}
              </motion.button>
            )
          })}
        </div>
      </div>

      {/* Domain Templates */}
      <div>
        <h2 className="text-lg font-semibold mb-1">Choose a Domain Template</h2>
        <p className="text-[10px] text-muted-foreground mb-4">
          Select a pre-built domain or define your own. Each template pre-configures the state/action space and reward function.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
          {DOMAIN_TEMPLATES.map((tmpl, i) => {
            const Icon = tmpl.icon
            return (
              <motion.button
                key={tmpl.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onDomainSelect(tmpl.id)}
                className="glass-dark rounded-xl p-4 text-left border border-dark-border hover:border-neon-purple/40 transition-all group"
              >
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center mb-3"
                  style={{ backgroundColor: `${tmpl.color}15` }}
                >
                  <Icon className="w-5 h-5" style={{ color: tmpl.color }} />
                </div>
                <h3 className="text-sm font-semibold mb-0.5">{tmpl.name}</h3>
                <p className="text-[9px] text-muted-foreground mb-3">{tmpl.tagline}</p>
                <div className="space-y-1 text-[9px]">
                  <div><span className="text-muted-foreground/60">S:</span> <span className="text-muted-foreground">{tmpl.stateSpace}</span></div>
                  <div><span className="text-muted-foreground/60">A:</span> <span className="text-muted-foreground">{tmpl.actionSpace}</span></div>
                  <div><span className="text-muted-foreground/60">R:</span> <span className="text-muted-foreground">{tmpl.reward}</span></div>
                </div>
                <div className="mt-3 flex items-center gap-1 text-[9px] text-neon-purple opacity-0 group-hover:opacity-100 transition-opacity">
                  <ArrowRight className="w-3 h-3" /> Configure this domain
                </div>
              </motion.button>
            )
          })}
        </div>
      </div>

      {/* AI Guidance Tip */}
      <div className="glass-dark rounded-xl p-4 border border-neon-orange/20">
        <div className="flex items-start gap-3">
          <Lightbulb className="w-5 h-5 text-neon-orange flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold mb-1">Need help formulating your problem?</h3>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              Use the <strong className="text-white">AI Assistant</strong> (bot icon, bottom-right) for contextual guidance.
              It can help you choose the right domain, training objective, and hyperparameters for your specific use case.
              For novel domains, start with Grid World to validate your approach, then scale up.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
