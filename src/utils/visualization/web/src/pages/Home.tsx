import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Grid3X3, Share2, Type, Atom, Settings, Activity,
  Sparkles, TrendingUp, Plus, Play, Upload, ArrowRight,
  FlaskConical, BarChart3, Brain, Box, ChevronRight, Workflow,
  Search, Sliders, Monitor, LineChart, RotateCcw,
} from 'lucide-react'
import { BentoGrid, BentoCard, MetricCard } from '../components/BentoGrid'
import { api, type TrainingState } from '../services/api'
import type { ViewId } from '../components/Sidebar'

interface HomeProps {
  onNavigate: (view: ViewId) => void
  onDomainSelect: (domainId: string) => void
}

const DOMAIN_CARDS = [
  {
    id: 'grid_world',
    name: 'Grid World',
    description: '2D grid navigation with configurable rewards. Perfect for learning GFlowNet fundamentals.',
    icon: Grid3X3,
    color: '#BD00FF',
    tags: ['Beginner', '2D'],
  },
  {
    id: 'dag',
    name: 'DAG / Graph',
    description: 'Directed acyclic graph traversal with customizable topology and node rewards.',
    icon: Share2,
    color: '#00D9FF',
    tags: ['Graph', 'Network'],
  },
  {
    id: 'sequence',
    name: 'Sequence Generation',
    description: 'Token-by-token sequence building with vocabulary constraints and target matching.',
    icon: Type,
    color: '#F59E0B',
    tags: ['NLP', 'Text'],
  },
  {
    id: 'molecule',
    name: 'Molecular Design',
    description: 'Atom-by-atom molecule construction with property optimization and ADMET filters.',
    icon: Atom,
    color: '#00FF88',
    tags: ['Chemistry', 'Drug Discovery'],
  },
  {
    id: 'custom',
    name: 'Custom Domain',
    description: 'Define your own state/action space with custom reward functions.',
    icon: Settings,
    color: '#FF6B6B',
    tags: ['Advanced', 'Flexible'],
  },
]

const ANALYSIS_TOOLS = [
  { id: 'candidates' as ViewId, label: 'Candidates', icon: BarChart3, desc: 'Browse & export generated solutions' },
  { id: 'structure' as ViewId, label: 'Structure Viewer', icon: Box, desc: 'Inspect candidate structures' },
  { id: 'interpret' as ViewId, label: 'Interpretability', icon: Brain, desc: 'Attribution maps & reward decomposition' },
]

export function Home({ onNavigate, onDomainSelect }: HomeProps) {
  const { data: trainingState } = useQuery<TrainingState>({
    queryKey: ['training-state'],
    queryFn: api.training.getState,
    refetchInterval: 5000,
    retry: 1,
  })

  const isTraining = trainingState?.is_training && !trainingState?.is_paused

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Platform Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">GFlowNet Interactive Lab</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Train generative flow networks across multiple domains — from grid worlds to molecular design
          </p>
        </div>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onNavigate('configure')}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-neon-purple to-neon-blue text-white text-sm font-medium shadow-lg shadow-neon-purple/20"
        >
          <Plus className="w-4 h-4" />
          New Experiment
        </motion.button>
      </div>

      {/* Active Experiment Banner */}
      {isTraining && trainingState && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-dark rounded-xl p-4 border border-neon-green/30"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="px-2 py-0.5 rounded-full bg-neon-green/20 text-neon-green text-[10px] font-medium animate-pulse">
                Running
              </div>
              <h3 className="text-sm font-semibold">Active Experiment</h3>
              <span className="text-xs text-muted-foreground">
                Episode {trainingState.current_iteration}/{trainingState.total_iterations}
              </span>
            </div>
            <button
              onClick={() => onNavigate('train')}
              className="flex items-center gap-1 text-xs text-neon-purple hover:text-neon-blue transition-colors"
            >
              View Dashboard <ArrowRight className="w-3 h-3" />
            </button>
          </div>          <div className="w-full h-2 rounded-full bg-dark-border overflow-hidden">
            <motion.div
              className="h-full rounded-full bg-gradient-to-r from-neon-purple to-neon-blue"
              initial={{ width: 0 }}
              animate={{
                width: `${(trainingState.current_iteration / trainingState.total_iterations) * 100}%`,
              }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <div className="grid grid-cols-4 gap-4 mt-3">
            <div>
              <span className="text-[10px] text-muted-foreground">Loss</span>
              <p className="text-sm font-mono font-bold">{trainingState.latest_loss?.toFixed(4) ?? '—'}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground">Mean Reward</span>
              <p className="text-sm font-mono font-bold text-neon-green">{trainingState.metrics?.mean_reward?.toFixed(3) ?? '—'}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground">Gradient Norm</span>
              <p className="text-sm font-mono font-bold">{trainingState.latest_gradient_norm?.toFixed(4) ?? '—'}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground">Diversity</span>
              <p className="text-sm font-mono font-bold">{trainingState.metrics?.diversity_ratio?.toFixed(3) ?? '—'}</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Domain Cards — the core of the platform */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold">Choose a Domain</h2>
            <p className="text-xs text-muted-foreground">Select a problem domain to configure and train your GFlowNet</p>
          </div>
          <button
            onClick={() => onNavigate('configure')}
            className="text-xs text-muted-foreground hover:text-white transition-colors flex items-center gap-1"
          >
            View all domains <ChevronRight className="w-3 h-3" />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
          {DOMAIN_CARDS.map((domain, i) => {
            const Icon = domain.icon
            return (
              <motion.button
                key={domain.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onDomainSelect(domain.id)}
                className="glass-dark rounded-xl p-4 text-left border border-dark-border hover:border-neon-purple/40 transition-all group"
              >
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center mb-3"
                  style={{ backgroundColor: `${domain.color}15` }}
                >
                  <Icon className="w-5 h-5" style={{ color: domain.color }} />
                </div>
                <h3 className="text-sm font-semibold mb-1 group-hover:text-white transition-colors">{domain.name}</h3>
                <p className="text-[10px] text-muted-foreground leading-relaxed mb-2 line-clamp-2">
                  {domain.description}
                </p>
                <div className="flex gap-1">
                  {domain.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-1.5 py-0.5 text-[8px] rounded-full bg-white/5 text-muted-foreground"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </motion.button>
            )
          })}
        </div>
      </div>

      {/* Platform Metrics */}
      <BentoGrid columns={4} className="grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Experiments Run"
          value={trainingState?.has_session ? 1 : 0}
          icon={<Activity className="w-4 h-4" />}
          color="#00D9FF"
        />
        <MetricCard
          label="Candidates Generated"
          value={trainingState?.total_molecules ?? 0}
          icon={<FlaskConical className="w-4 h-4" />}
          color="#BD00FF"
        />
        <MetricCard
          label="Best Reward"
          value={(trainingState?.latest_reward ?? 0).toFixed(2)}
          icon={<Sparkles className="w-4 h-4" />}
          color="#00FF88"
        />
        <MetricCard
          label="Diversity Index"
          value={trainingState?.metrics?.diversity_ratio?.toFixed(3) ?? '—'}
          icon={<TrendingUp className="w-4 h-4" />}
          color="#F59E0B"
        />
      </BentoGrid>

      {/* Quick Actions — two columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Getting Started */}
        <div className="glass-dark rounded-xl p-5 border border-dark-border">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <Play className="w-4 h-4 text-neon-purple" />
            Quick Start
          </h3>
          <div className="space-y-2">
            <button
              onClick={() => onDomainSelect('grid_world')}
              className="w-full flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-left group"
            >
              <div className="w-8 h-8 rounded-lg bg-neon-purple/10 flex items-center justify-center">
                <Grid3X3 className="w-4 h-4 text-neon-purple" />
              </div>
              <div className="flex-1">
                <span className="text-xs font-medium">Grid World Demo</span>
                <p className="text-[10px] text-muted-foreground">Learn GFlowNet basics with an 8x8 grid</p>
              </div>
              <ArrowRight className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
            <button
              onClick={() => onDomainSelect('molecule')}
              className="w-full flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-left group"
            >
              <div className="w-8 h-8 rounded-lg bg-neon-green/10 flex items-center justify-center">
                <Atom className="w-4 h-4 text-neon-green" />
              </div>
              <div className="flex-1">
                <span className="text-xs font-medium">Molecular Generation</span>
                <p className="text-[10px] text-muted-foreground">Generate drug-like molecules with GFlowNet</p>
              </div>
              <ArrowRight className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
            <button
              onClick={() => onNavigate('configure')}
              className="w-full flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-left group"
            >
              <div className="w-8 h-8 rounded-lg bg-neon-blue/10 flex items-center justify-center">
                <Upload className="w-4 h-4 text-neon-blue" />
              </div>
              <div className="flex-1">
                <span className="text-xs font-medium">Import Dataset</span>
                <p className="text-[10px] text-muted-foreground">Load SMILES, SDF, or custom state files</p>
              </div>
              <ArrowRight className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
          </div>
        </div>

        {/* Analysis Tools */}
        <div className="glass-dark rounded-xl p-5 border border-dark-border">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-neon-blue" />
            Analysis Tools
          </h3>
          <div className="space-y-2">
            {ANALYSIS_TOOLS.map((tool) => {
              const Icon = tool.icon
              return (
                <button
                  key={tool.id}
                  onClick={() => onNavigate(tool.id)}
                  className="w-full flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-left group"
                >
                  <div className="w-8 h-8 rounded-lg bg-neon-blue/10 flex items-center justify-center">
                    <Icon className="w-4 h-4 text-neon-blue" />
                  </div>
                  <div className="flex-1">
                    <span className="text-xs font-medium">{tool.label}</span>
                    <p className="text-[10px] text-muted-foreground">{tool.desc}</p>
                  </div>
                  <ArrowRight className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Workflow Overview */}
      <div className="glass-dark rounded-xl p-5 border border-dark-border">
        <h3 className="text-sm font-semibold mb-1 flex items-center gap-2">
          <Workflow className="w-4 h-4 text-neon-purple" />
          How It Works
        </h3>
        <p className="text-[10px] text-muted-foreground mb-4">
          GFlowNet samples diverse, high-quality candidates proportional to a reward function R(x) — unlike RL which finds a single optimum.
        </p>
        <div className="flex items-start gap-0">
          {[
            { step: 1, label: 'Choose Domain', desc: 'Select your problem space', icon: Search, color: '#BD00FF', view: 'home' as ViewId },
            { step: 2, label: 'Configure', desc: 'Set parameters & objective', icon: Sliders, color: '#00D9FF', view: 'configure' as ViewId },
            { step: 3, label: 'Train', desc: 'Run GFlowNet & monitor', icon: Monitor, color: '#00FF88', view: 'train' as ViewId },
            { step: 4, label: 'Explore', desc: 'Browse candidates & landscape', icon: LineChart, color: '#F59E0B', view: 'candidates' as ViewId },
            { step: 5, label: 'Iterate', desc: 'Refine & retrain', icon: RotateCcw, color: '#FF6B6B', view: 'configure' as ViewId },
          ].map((ws, i, arr) => {
            const Icon = ws.icon
            return (
              <div key={ws.step} className="flex items-start flex-1 min-w-0">
                <button
                  onClick={() => onNavigate(ws.view)}
                  className="flex flex-col items-center text-center px-2 py-2 rounded-lg hover:bg-white/5 transition-colors group flex-1"
                >
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center mb-2 transition-transform group-hover:scale-110"
                    style={{ backgroundColor: `${ws.color}15`, border: `1px solid ${ws.color}40` }}
                  >
                    <Icon className="w-4 h-4" style={{ color: ws.color }} />
                  </div>
                  <span className="text-[11px] font-semibold text-white">{ws.label}</span>
                  <span className="text-[9px] text-muted-foreground mt-0.5 leading-tight">{ws.desc}</span>
                </button>
                {i < arr.length - 1 && (
                  <div className="flex items-center pt-5 px-0">
                    <ChevronRight className="w-4 h-4 text-dark-border flex-shrink-0" />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Molecular Design Capabilities — Five Gaps */}
      <div className="glass-dark rounded-xl p-5 border border-dark-border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Atom className="w-4 h-4 text-neon-green" />
              Molecular Design Capabilities
            </h3>
            <p className="text-[10px] text-muted-foreground mt-0.5">
              Advanced tools for drug-like molecule generation and multi-objective optimization
            </p>
          </div>
          <button
            onClick={() => onNavigate('toolkit')}
            className="text-xs text-neon-purple hover:text-neon-blue transition-colors flex items-center gap-1"
          >
            Open Toolkit <ArrowRight className="w-3 h-3" />
          </button>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {[
            { label: 'Diversity Analysis', desc: 'Tanimoto similarity, scaffold diversity, nearest-neighbor metrics', color: '#00D9FF', badge: 'Gap 1' },
            { label: 'Docking Reward', desc: 'AutoDock Vina binding affinity with proxy MLP for fast scoring', color: '#BD00FF', badge: 'Gap 2' },
            { label: 'Fragment Library', desc: '50 BRICS fragments with category metadata and compatibility rules', color: '#F59E0B', badge: 'Gap 3' },
            { label: 'Synthesis Routes', desc: '17 reaction templates ensuring synthesizable molecule designs', color: '#00FF88', badge: 'Gap 4' },
            { label: 'Multi-Objective', desc: 'MOGFN Pareto-front exploration with preference-conditioned sampling', color: '#FF6B6B', badge: 'Gap 5' },
          ].map((cap) => (
            <button
              key={cap.label}
              onClick={() => onNavigate('toolkit')}
              className="p-3 rounded-lg bg-white/5 border border-dark-border/50 hover:border-neon-purple/30 transition-all text-left group"
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{ backgroundColor: `${cap.color}20`, color: cap.color }}>
                  {cap.badge}
                </span>
              </div>
              <h4 className="text-xs font-semibold mb-1 group-hover:text-white transition-colors">{cap.label}</h4>
              <p className="text-[9px] text-muted-foreground leading-relaxed line-clamp-2">{cap.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Supported Objectives */}
      <div className="glass-dark rounded-xl p-5 border border-dark-border">
        <h3 className="text-sm font-semibold mb-3">Supported Training Objectives</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          {[
            { short: 'TB', name: 'Trajectory Balance', color: 'neon-purple' },
            { short: 'STB', name: 'Sub-Trajectory Balance', color: 'neon-blue' },
            { short: 'DB', name: 'Detailed Balance', color: 'neon-green' },
            { short: 'FM', name: 'Flow Matching', color: 'neon-orange' },
            { short: 'TLM', name: 'Trajectory Likelihood Max.', color: 'neon-cyan' },
          ].map((obj) => (
            <div
              key={obj.short}
              className="p-2.5 rounded-lg bg-white/5 border border-dark-border/50"
            >
              <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded bg-${obj.color}/20 text-${obj.color}`}>
                {obj.short}
              </span>
              <p className="text-[10px] text-muted-foreground mt-1">{obj.name}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
