import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'

export type ThemeName = 'cyberpunk' | 'cream' | 'glass' | 'midnight'

export interface ThemeOption {
  id: ThemeName
  label: string
  description: string
  font: string
  colors: [string, string, string] // 3 preview swatch colors
  isLight?: boolean
}

export const THEMES: ThemeOption[] = [
  {
    id: 'cyberpunk',
    label: 'Cyberpunk',
    description: 'Electric neons, sharp glow',
    font: 'Inter',
    colors: ['#BD00FF', '#00D9FF', '#00FF88'],
  },
  {
    id: 'cream',
    label: 'Cream',
    description: 'Warm light, soft shadows',
    font: 'DM Sans',
    colors: ['#D47564', '#5CA386', '#C49B6E'],
    isLight: true,
  },
  {
    id: 'glass',
    label: 'Glass',
    description: 'Amber glow, translucent panels',
    font: 'DM Sans',
    colors: ['#F59E0B', '#EA580C', '#EAB308'],
  },
  {
    id: 'midnight',
    label: 'Midnight',
    description: 'Teal accents, data-dense',
    font: 'Space Grotesk',
    colors: ['#14B8A6', '#06B6D4', '#10B981'],
  },
]

interface ThemeContextValue {
  theme: ThemeName
  setTheme: (theme: ThemeName) => void
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: 'cyberpunk',
  setTheme: () => {},
})

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeName>(() => {
    const stored = localStorage.getItem('gflownet-theme')
    // Validate stored theme is still a valid option
    if (stored && ['cyberpunk', 'cream', 'glass', 'midnight'].includes(stored)) {
      return stored as ThemeName
    }
    return 'cyberpunk'
  })

  const setTheme = (newTheme: ThemeName) => {
    setThemeState(newTheme)
    localStorage.setItem('gflownet-theme', newTheme)
  }

  // Apply data-theme attribute to document root
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'cyberpunk') {
      root.removeAttribute('data-theme')
    } else {
      root.setAttribute('data-theme', theme)
    }
  }, [theme])

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  return useContext(ThemeContext)
}

// Chart colors — hex values per theme for Recharts (which needs strings, not CSS vars)
const CHART_COLORS: Record<ThemeName, {
  primary: string    // Loss line, main accent
  secondary: string  // TB loss, secondary accent
  tertiary: string   // Flow loss, third accent
  green: string      // Reward curves
  grid: string       // CartesianGrid stroke
  axis: string       // Axis stroke + tick fill
  brushFill: string  // Brush background
  gradientStart: string // Progress ring gradient start
  gradientEnd: string   // Progress ring gradient end
}> = {
  cyberpunk: {
    primary: '#BD00FF',
    secondary: '#00D9FF',
    tertiary: '#FF006E',
    green: '#00FF88',
    grid: '#2A2A2D',
    axis: '#666',
    brushFill: '#1a1a1d',
    gradientStart: '#BD00FF',
    gradientEnd: '#00D9FF',
  },
  cream: {
    primary: '#D47564',
    secondary: '#8A9BA8',
    tertiary: '#C27878',
    green: '#5CA386',
    grid: '#E0D8D0',
    axis: '#948E88',
    brushFill: '#F0EBE4',
    gradientStart: '#D47564',
    gradientEnd: '#5CA386',
  },
  glass: {
    primary: '#F59E0B',
    secondary: '#EA580C',
    tertiary: '#DC3838',
    green: '#EAB308',
    grid: '#372D23',
    axis: '#B4A591',
    brushFill: '#1A1510',
    gradientStart: '#F59E0B',
    gradientEnd: '#EA580C',
  },
  midnight: {
    primary: '#14B8A6',
    secondary: '#06B6D4',
    tertiary: '#3B82F6',
    green: '#10B981',
    grid: '#1E2D41',
    axis: '#8291A5',
    brushFill: '#0E1621',
    gradientStart: '#14B8A6',
    gradientEnd: '#06B6D4',
  },
}

export function useChartColors() {
  const { theme } = useTheme()
  return CHART_COLORS[theme]
}

// Layout configuration per theme — drives structural differences in components
interface ThemeLayout {
  // Card & section padding classes
  cardPad: string       // Tailwind padding for metric cards
  sectionPad: string    // Tailwind padding for large sections
  innerPad: string      // Tailwind padding for inner elements
  // Gap classes
  gap: string           // Tailwind gap for grid items
  sectionGap: string    // Tailwind gap between major sections
  // Text sizes
  headingSize: string   // Section headings
  labelSize: string     // Field labels
  valueSize: string     // Metric values
  tinySize: string      // Smallest text
  titleSize: string     // Page titles
  // Grid density
  metricsColumns: string // Grid columns for metrics grids
  configColumns: string  // Grid columns for config panels
  // Flags
  compact: boolean
  spacious: boolean
  // Progress ring size
  ringSize: number
}

const THEME_LAYOUTS: Record<ThemeName, ThemeLayout> = {
  cyberpunk: {
    cardPad: 'p-3',
    sectionPad: 'p-6',
    innerPad: 'p-2.5',
    gap: 'gap-2',
    sectionGap: 'gap-6',
    headingSize: 'text-xs',
    labelSize: 'text-[11px]',
    valueSize: 'text-lg',
    tinySize: 'text-[9px]',
    titleSize: 'text-xl',
    metricsColumns: 'grid-cols-3',
    configColumns: 'grid-cols-3 lg:grid-cols-5',
    compact: false,
    spacious: false,
    ringSize: 80,
  },
  cream: {
    cardPad: 'p-5',
    sectionPad: 'p-8',
    innerPad: 'p-4',
    gap: 'gap-4',
    sectionGap: 'gap-8',
    headingSize: 'text-sm',
    labelSize: 'text-xs',
    valueSize: 'text-xl',
    tinySize: 'text-[10px]',
    titleSize: 'text-2xl',
    metricsColumns: 'grid-cols-2',
    configColumns: 'grid-cols-2 lg:grid-cols-3',
    compact: false,
    spacious: true,
    ringSize: 100,
  },
  glass: {
    cardPad: 'p-4',
    sectionPad: 'p-6',
    innerPad: 'p-3',
    gap: 'gap-3',
    sectionGap: 'gap-6',
    headingSize: 'text-xs',
    labelSize: 'text-[11px]',
    valueSize: 'text-lg',
    tinySize: 'text-[9px]',
    titleSize: 'text-xl',
    metricsColumns: 'grid-cols-3',
    configColumns: 'grid-cols-3 lg:grid-cols-5',
    compact: false,
    spacious: false,
    ringSize: 80,
  },
  midnight: {
    cardPad: 'p-1.5',
    sectionPad: 'p-3',
    innerPad: 'p-1.5',
    gap: 'gap-1',
    sectionGap: 'gap-3',
    headingSize: 'text-[10px]',
    labelSize: 'text-[9px]',
    valueSize: 'text-sm',
    tinySize: 'text-[8px]',
    titleSize: 'text-base',
    metricsColumns: 'grid-cols-4 lg:grid-cols-6',
    configColumns: 'grid-cols-4 lg:grid-cols-6',
    compact: true,
    spacious: false,
    ringSize: 56,
  },
}

export function useThemeLayout() {
  const { theme } = useTheme()
  return THEME_LAYOUTS[theme]
}
