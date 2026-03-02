// Consistent color scheme for GFlowNet visualizations

export const COLORS = {
  // Primary theme colors
  primary: {
    purple: '#BD00FF',     // Main purple
    green: '#00FF88',      // Success/reward green
    blue: '#00D9FF',       // Info blue
    pink: '#FF006E',       // Warning/terminal pink
    orange: '#FFA07A',     // Exploration orange
  },

  // Chemistry atom colors (CPK convention)
  atom: {
    carbon: '#404040',
    nitrogen: '#3050F8',
    oxygen: '#FF0D0D',
    sulfur: '#FFFF30',
    phosphorus: '#FF8000',
    halogen: '#1FF01F',
    hydrogen: '#FFFFFF',
    iron: '#E06633',
    default: '#FF1493',
  },

  // Molecular property indicators
  property: {
    good: '#10B981',       // Emerald - within range
    warning: '#F59E0B',    // Amber - borderline
    bad: '#EF4444',        // Red - out of range
  },

  // Reward gradient for molecular scoring
  molecularReward: {
    low: '#3B82F6',        // Blue
    mid: '#EAB308',        // Yellow
    high: '#EF4444',       // Red
  },
  
  // Reward-related colors
  reward: {
    high: '#00FF88',       // High reward - bright green
    medium: '#FFD700',     // Medium reward - gold
    low: '#FF6B6B',        // Low reward - red
    background: '#0A1F1F', // Dark green background
  },
  
  // Trajectory colors (gradient from high to low reward)
  trajectory: {
    best: '#00FF88',       // Best trajectory - green
    good: '#7FFF00',       // Good trajectory - lime
    average: '#FFD700',    // Average trajectory - gold
    poor: '#FF8C00',       // Poor trajectory - orange
    worst: '#FF4500',      // Worst trajectory - red
  },
  
  // State/flow colors
  flow: {
    high: '#BD00FF',       // High flow - purple
    medium: '#9932CC',     // Medium flow - dark orchid
    low: '#6A5ACD',        // Low flow - slate blue
    none: '#2E2E2E',       // No flow - dark gray
  },
  
  // UI colors
  ui: {
    glass: 'rgba(42, 42, 45, 0.8)',
    border: 'rgba(255, 255, 255, 0.1)',
    muted: '#666666',
    text: '#FFFFFF',
    background: '#0A0A0B',
  },
  
  // Gradients
  gradients: {
    reward: 'linear-gradient(to right, #FF4500, #FFD700, #00FF88)',
    flow: 'linear-gradient(to right, #2E2E2E, #6A5ACD, #BD00FF)',
    density: 'linear-gradient(to top, #1A1A1D, #BD00FF)',
  }
}

// Helper functions for color interpolation
export function interpolateRewardColor(normalizedReward: number): string {
  // normalizedReward should be between 0 and 1
  const clamped = Math.max(0, Math.min(1, normalizedReward))
  
  if (clamped < 0.5) {
    // Red to Gold
    const t = clamped * 2
    return interpolateHex('#FF4500', '#FFD700', t)
  } else {
    // Gold to Green
    const t = (clamped - 0.5) * 2
    return interpolateHex('#FFD700', '#00FF88', t)
  }
}

export function interpolateFlowColor(normalizedFlow: number): string {
  // normalizedFlow should be between 0 and 1
  const clamped = Math.max(0, Math.min(1, normalizedFlow))
  
  if (clamped < 0.5) {
    // Dark gray to Slate blue
    const t = clamped * 2
    return interpolateHex('#2E2E2E', '#6A5ACD', t)
  } else {
    // Slate blue to Purple
    const t = (clamped - 0.5) * 2
    return interpolateHex('#6A5ACD', '#BD00FF', t)
  }
}

function interpolateHex(color1: string, color2: string, t: number): string {
  const r1 = parseInt(color1.slice(1, 3), 16)
  const g1 = parseInt(color1.slice(3, 5), 16)
  const b1 = parseInt(color1.slice(5, 7), 16)
  
  const r2 = parseInt(color2.slice(1, 3), 16)
  const g2 = parseInt(color2.slice(3, 5), 16)
  const b2 = parseInt(color2.slice(5, 7), 16)
  
  const r = Math.round(r1 + (r2 - r1) * t)
  const g = Math.round(g1 + (g2 - g1) * t)
  const b = Math.round(b1 + (b2 - b1) * t)
  
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

// HSL color utilities for smooth transitions
export function interpolateHSL(hue1: number, hue2: number, sat: number, light: number, t: number): string {
  const hue = hue1 + (hue2 - hue1) * t
  return `hsl(${hue}, ${sat}%, ${light}%)`
}

export function getRewardHSL(normalizedReward: number): string {
  // Red (0°) to Green (120°)
  const hue = normalizedReward * 120
  return `hsl(${hue}, 100%, 50%)`
}

export function getFlowHSL(normalizedFlow: number): string {
  // Purple (280°) to Blue (240°)
  const hue = 280 - (normalizedFlow * 40)
  return `hsl(${hue}, 100%, 50%)`
}