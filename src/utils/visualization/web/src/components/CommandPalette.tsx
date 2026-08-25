import { useEffect, useState } from 'react'
import { Command } from 'cmdk'
import {
  Home,
  Bot,
  Activity,
  Box,
  Map,
  BarChart3,
  Brain,
  Network,
  Sliders,
  Search,
} from 'lucide-react'
import type { ViewId } from './Sidebar'

interface CommandPaletteProps {
  onNavigate: (view: ViewId) => void
}

const COMMANDS: Array<{ id: ViewId; label: string; icon: typeof Home; keywords: string }> = [
  { id: 'home', label: 'Home / Overview', icon: Home, keywords: 'dashboard overview workflow' },
  { id: 'formulate', label: 'Formulate Problem', icon: Bot, keywords: 'ai formulate problem state action reward define' },
  { id: 'configure', label: 'Configure Experiment', icon: Sliders, keywords: 'configure setup grid world molecule molecular design generate create parameters' },
  { id: 'train', label: 'Train & Monitor', icon: Activity, keywords: 'train loss reward monitor' },
  { id: 'candidates', label: 'Candidates / Results', icon: BarChart3, keywords: 'results candidates molecules solutions export browse' },
  { id: 'structure', label: 'Structure Viewer', icon: Box, keywords: '3d structure viewer inspect molecule grid graph' },
  { id: 'landscape', label: 'Solution Landscape', icon: Map, keywords: 'landscape space umap tsne pca scatter distribution chemical' },
  { id: 'interpret', label: 'Interpret Model', icon: Brain, keywords: 'interpret attribution saliency explain decomposition' },
  { id: 'flow', label: 'Flow Field', icon: Network, keywords: 'flow policy arrows field balance' },
]

export function CommandPalette({ onNavigate }: CommandPaletteProps) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  return (
    <>
      {open && (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />

          {/* Dialog */}
          <div className="relative w-full max-w-lg bg-dark-panel border border-dark-border rounded-xl shadow-2xl overflow-hidden">
            <Command className="w-full" loop>
              <div className="flex items-center px-4 border-b border-dark-border/50">
                <Search className="w-4 h-4 text-muted-foreground mr-3" />
                <Command.Input
                  placeholder="Type a command or search..."
                  className="w-full py-3 bg-transparent text-sm text-white outline-none placeholder:text-muted-foreground"
                  autoFocus
                />
                <kbd className="ml-2 px-1.5 py-0.5 text-[10px] text-muted-foreground bg-dark-border/50 rounded">
                  ESC
                </kbd>
              </div>

              <Command.List className="max-h-64 overflow-y-auto p-2 scrollbar-thin">
                <Command.Empty className="py-6 text-center text-sm text-muted-foreground">
                  No results found.
                </Command.Empty>

                <Command.Group heading="Navigation" className="text-[10px] uppercase tracking-wider text-muted-foreground/60 px-2 py-1">
                  {COMMANDS.map((cmd) => {
                    const Icon = cmd.icon
                    return (
                      <Command.Item
                        key={cmd.id}
                        value={`${cmd.label} ${cmd.keywords}`}
                        onSelect={() => {
                          onNavigate(cmd.id)
                          setOpen(false)
                        }}
                        className="flex items-center px-3 py-2 rounded-md text-sm text-muted-foreground cursor-pointer data-[selected=true]:bg-neon-purple/15 data-[selected=true]:text-white transition-colors"
                      >
                        <Icon className="w-4 h-4 mr-3 opacity-60" />
                        <span>{cmd.label}</span>
                      </Command.Item>
                    )
                  })}
                </Command.Group>
              </Command.List>
            </Command>
          </div>
        </div>
      )}
    </>
  )
}
