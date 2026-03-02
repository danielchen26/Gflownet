import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Home,
  Activity,
  Box,
  Map,
  BarChart3,
  Brain,
  Network,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Sliders,
  Bot,
} from 'lucide-react'
import { useThemeLayout } from '../contexts/ThemeContext'
import { ThemeSelector } from './ThemeSelector'

export type ViewId =
  | 'home'
  | 'formulate'
  | 'configure'
  | 'train'
  | 'candidates'
  | 'structure'
  | 'landscape'
  | 'interpret'
  | 'flow'

interface SidebarProps {
  activeView: ViewId
  onViewChange: (view: ViewId) => void
  isConnected: boolean
  domainType?: string
}

const NAV_ITEMS: Array<{
  id: ViewId
  label: string
  icon: typeof Home
  section: 'framework' | 'explore' | 'insights'
  desc?: string
}> = [
  // Framework — the GFlowNet workflow stages
  { id: 'home', label: 'Home', icon: Home, section: 'framework', desc: 'Overview & workflow' },
  { id: 'formulate', label: 'Formulate', icon: Bot, section: 'framework', desc: 'AI-guided problem setup' },
  { id: 'configure', label: 'Configure', icon: Sliders, section: 'framework', desc: 'Parameters & objective' },
  { id: 'train', label: 'Train', icon: Activity, section: 'framework', desc: 'Run & monitor' },

  // Explore — domain-adaptive result exploration
  { id: 'candidates', label: 'Candidates', icon: BarChart3, section: 'explore', desc: 'Generated solutions' },
  { id: 'structure', label: 'Structure', icon: Box, section: 'explore', desc: 'Inspect candidates' },
  { id: 'landscape', label: 'Landscape', icon: Map, section: 'explore', desc: 'Solution space' },

  // Insights — model understanding
  { id: 'interpret', label: 'Interpret', icon: Brain, section: 'insights', desc: 'Attribution & decomposition' },
  { id: 'flow', label: 'Flow', icon: Network, section: 'insights', desc: 'Learned flow field' },
]

const SECTION_LABELS: Record<string, string> = {
  framework: 'Framework',
  explore: 'Explore',
  insights: 'Insights',
}

export function Sidebar({ activeView, onViewChange, isConnected }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)
  const layout = useThemeLayout()

  const sections = ['framework', 'explore', 'insights'] as const

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 56 : 200 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="relative z-20 h-full border-r border-dark-border/50 bg-dark-panel/80 backdrop-blur-sm flex flex-col"
    >
      {/* Logo */}
      <div className={`flex items-center ${collapsed ? 'justify-center px-2' : 'px-4'} py-4 border-b border-dark-border/30`}>
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-purple to-neon-blue flex items-center justify-center flex-shrink-0">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              className="ml-3 overflow-hidden whitespace-nowrap"
            >
              <h1 className={`${layout.headingSize} font-bold gradient-text`}>GFlowNet</h1>
              <p className={`${layout.tinySize} text-muted-foreground`}>Optimization Lab</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2 scrollbar-thin">
        {sections.map((section) => {
          const items = NAV_ITEMS.filter((i) => i.section === section)
          return (
            <NavSection key={section} label={SECTION_LABELS[section]} collapsed={collapsed}>
              {items.map((item) => (
                <NavItem
                  key={item.id}
                  item={item}
                  isActive={activeView === item.id}
                  collapsed={collapsed}
                  onClick={() => onViewChange(item.id)}
                />
              ))}
            </NavSection>
          )
        })}
      </nav>

      {/* Bottom: Theme + Connection */}
      <div className="border-t border-dark-border/30 p-2 space-y-2">
        <div className={`flex items-center ${collapsed ? 'justify-center' : 'px-2'}`}>
          <ThemeSelector />
        </div>
        <div className={`flex items-center ${collapsed ? 'justify-center' : 'px-2'} space-x-2`}>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-neon-green animate-pulse' : 'bg-red-500'}`} />
          <AnimatePresence>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-[10px] text-muted-foreground"
              >
                {isConnected ? 'Connected' : 'Disconnected'}
              </motion.span>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Collapse Button */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-dark-panel border border-dark-border flex items-center justify-center text-muted-foreground hover:text-white transition-colors z-30"
      >
        {collapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronLeft className="w-3 h-3" />}
      </button>
    </motion.aside>
  )
}

function NavSection({
  label,
  collapsed,
  children,
}: {
  label: string
  collapsed: boolean
  children: React.ReactNode
}) {
  return (
    <div className="mb-2">
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="px-4 py-1"
          >
            <span className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground/60">
              {label}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
      {collapsed && <div className="h-px bg-dark-border/30 mx-2 my-1" />}
      <div className="space-y-0.5 px-2">{children}</div>
    </div>
  )
}

function NavItem({
  item,
  isActive,
  collapsed,
  onClick,
}: {
  item: (typeof NAV_ITEMS)[number]
  isActive: boolean
  collapsed: boolean
  onClick: () => void
}) {
  const Icon = item.icon

  return (
    <button
      onClick={onClick}
      title={collapsed ? `${item.label} — ${item.desc}` : undefined}
      className={`
        relative w-full flex items-center ${collapsed ? 'justify-center px-2' : 'px-3'} py-2 rounded-md
        transition-all duration-150 group
        ${isActive
          ? 'bg-neon-purple/15 text-white'
          : 'text-muted-foreground hover:text-white hover:bg-white/5'
        }
      `}
    >
      {isActive && (
        <motion.div
          layoutId="sidebarActive"
          className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-r bg-gradient-to-b from-neon-purple to-neon-blue"
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        />
      )}
      <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-neon-purple' : ''}`} />
      <AnimatePresence>
        {!collapsed && (
          <motion.span
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: 'auto' }}
            exit={{ opacity: 0, width: 0 }}
            className="ml-3 text-xs font-medium whitespace-nowrap overflow-hidden"
          >
            {item.label}
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  )
}
