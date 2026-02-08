import { useState, useRef, useEffect } from 'react'
import { Info } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface InfoTooltipProps {
  content: string | React.ReactNode
  title?: string
  citation?: string  // e.g., "Malkin et al. 2022"
  position?: 'top' | 'bottom' | 'left' | 'right'
  size?: 'sm' | 'md' | 'lg'
}

export function InfoTooltip({
  content,
  title,
  citation,
  position = 'top',
  size = 'sm'
}: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const triggerRef = useRef<HTMLButtonElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  // Calculate tooltip position to keep it in viewport
  useEffect(() => {
    if (isVisible && triggerRef.current && tooltipRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect()
      const tooltipRect = tooltipRef.current.getBoundingClientRect()
      const viewportWidth = window.innerWidth
      const viewportHeight = window.innerHeight

      let x = 0
      let y = 0

      // Adjust horizontal position
      if (position === 'left' || position === 'right') {
        x = position === 'left' ? -tooltipRect.width - 8 : triggerRect.width + 8
        y = -(tooltipRect.height / 2) + (triggerRect.height / 2)
      } else {
        x = -(tooltipRect.width / 2) + (triggerRect.width / 2)
        y = position === 'top' ? -tooltipRect.height - 8 : triggerRect.height + 8
      }

      // Keep tooltip in viewport
      const absoluteX = triggerRect.left + x
      const absoluteY = triggerRect.top + y

      if (absoluteX < 8) x += (8 - absoluteX)
      if (absoluteX + tooltipRect.width > viewportWidth - 8) {
        x -= (absoluteX + tooltipRect.width - viewportWidth + 8)
      }
      if (absoluteY < 8) y += (8 - absoluteY)
      if (absoluteY + tooltipRect.height > viewportHeight - 8) {
        y -= (absoluteY + tooltipRect.height - viewportHeight + 8)
      }

      setTooltipPosition({ x, y })
    }
  }, [isVisible, position])

  const iconSize = size === 'sm' ? 'w-3 h-3' : size === 'md' ? 'w-4 h-4' : 'w-5 h-5'
  const tooltipWidth = size === 'sm' ? 'max-w-[200px]' : size === 'md' ? 'max-w-[280px]' : 'max-w-[360px]'

  return (
    <div className="relative inline-flex items-center">
      <button
        ref={triggerRef}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        className={`${iconSize} text-muted-foreground hover:text-neon-blue transition-colors cursor-help`}
        aria-label="More information"
      >
        <Info className={iconSize} />
      </button>

      <AnimatePresence>
        {isVisible && (
          <motion.div
            ref={tooltipRef}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={`absolute z-50 ${tooltipWidth} p-3 bg-dark-panel border border-dark-border rounded-lg shadow-xl pointer-events-none`}
            style={{
              left: tooltipPosition.x,
              top: tooltipPosition.y,
            }}
          >
            {title && (
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-semibold text-white">{title}</span>
                {citation && (
                  <span className="px-1.5 py-0.5 text-[9px] bg-neon-blue/20 text-neon-blue rounded">
                    {citation}
                  </span>
                )}
              </div>
            )}
            <div className="text-[11px] text-muted-foreground leading-relaxed">
              {content}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// Pre-defined tooltips for common GFlowNet concepts
export const TOOLTIPS = {
  // Training Objectives
  TRAJECTORY_BALANCE: {
    title: 'Trajectory Balance',
    citation: 'Bengio et al. 2021',
    content: 'The original GFlowNet objective. Enforces: Z · ∏P_F(aᵢ|sᵢ) = R(sₙ) · ∏P_B(aᵢ|sᵢ₊₁). Uses learnable partition function Z for proper normalization.'
  },
  SUB_TRAJECTORY_BALANCE: {
    title: 'Sub-Trajectory Balance',
    citation: 'Malkin et al. 2022',
    content: 'Provides O(T²) learning signals from all sub-trajectories instead of just full trajectories. Better credit assignment for long sequences. Especially useful when paths to reward states are long.'
  },
  DETAILED_BALANCE: {
    title: 'Detailed Balance',
    citation: 'Bengio et al. 2021',
    content: 'Local balance conditions at each transition: F(s) · P_F(a|s) = F(s′) · P_B(a|s′). Requires backward policy training. More local gradients than TB.'
  },
  FLOW_MATCHING: {
    title: 'Flow Matching',
    citation: 'Bengio et al. 2023',
    content: 'Directly estimates flow F(s) at each state using a flow estimator network. Minimizes (Z(s) - F(s))². Trades some accuracy for computational efficiency on large state spaces.'
  },
  TRAJECTORY_LIKELIHOOD_MAXIMIZATION: {
    title: 'TLM',
    citation: 'ICLR 2025',
    content: 'Trains backward policy P_B to encode path count information. Max-entropy backward: P_B(s|s′) ∝ n(s)/n(s′) where n(s) = #paths. Directly compensates for structural path asymmetry (e.g., 70:1 ratio).'
  },

  // Mode Collapse Prevention
  EPSILON_EXPLORATION: {
    title: 'ε-Uniform Exploration',
    citation: 'Malkin et al. 2022',
    content: 'Mixes policy with uniform random actions: P(a|s) = (1-ε) · P_F(a|s) + ε · Uniform. Ensures non-zero probability for all actions, enabling discovery of minority modes.'
  },
  EPSILON_DECAY: {
    title: 'Epsilon Annealing',
    content: 'Linearly decreases ε from initial value to 0 during training. High exploration early (mode discovery) → exploitation later (convergence). Formula: ε_t = ε_0 · (1 - t/T)'
  },
  ENTROPY_REGULARIZATION: {
    title: 'Entropy Regularization',
    citation: 'AISTATS 2024',
    content: 'Adds policy entropy bonus to loss: L = L_TB - λ · H(π). Prevents premature convergence to deterministic policies. Higher entropy = more diverse sampling.'
  },
  Z_LEARNING_RATE: {
    title: 'Z Learning Rate Multiplier',
    citation: 'Jain et al. 2023',
    content: 'Trains partition function Z faster than policy network. Z convergence is often the bottleneck in TB training. 10× multiplier recommended based on peptide generation experiments.'
  },
  EXPERIENCE_REPLAY: {
    title: 'Experience Replay Buffer',
    citation: 'JMLR 2023',
    content: 'Stores past trajectories for off-policy learning. Replay ratio controls mix of fresh vs. stored samples. Helps remember minority modes discovered earlier. Priority sampling emphasizes high-reward trajectories.'
  },
  TLM_BACKWARD_WEIGHT: {
    title: 'Backward Weight (λ)',
    content: 'Weight for backward policy loss in TLM objective. Higher λ = stronger backward policy training. The backward policy learns to encode path count structure for mode coverage.'
  },
  TLM_ENTROPY_COEFF: {
    title: 'Backward Entropy',
    content: 'Entropy regularization for backward policy. Prevents P_B from collapsing to deterministic. Maintains exploration in the backward direction.'
  },

  // Basic Parameters
  LEARNING_RATE: {
    title: 'Learning Rate',
    content: 'Controls step size for gradient updates. Too high → unstable training, NaN losses. Too low → slow convergence. Start with 0.001, adjust based on loss curves.'
  },
  BATCH_SIZE: {
    title: 'Batch Size',
    content: 'Number of trajectories sampled per training step. Larger batches → more stable gradients, slower iteration. Smaller batches → faster iteration, more noise. 16-32 typical for grid worlds.'
  },
  HIDDEN_DIM: {
    title: 'Hidden Dimension',
    content: 'Size of hidden layers in policy networks. Larger → more expressive, risk of overfitting. Smaller → faster, may underfit. 64 sufficient for small grids, 128-256 for complex domains.'
  },
}
