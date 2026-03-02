import { useEffect, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'

interface MoleculeViewer2DProps {
  smiles: string
  width?: number
  height?: number
  className?: string
  highlightAtoms?: number[]
  highlightColors?: string[]
  onClick?: () => void
}

// RDKit WASM module singleton
let rdkitPromise: Promise<any> | null = null
let rdkitModule: any = null

export async function getRDKit() {
  if (rdkitModule) return rdkitModule
  if (!rdkitPromise) {
    rdkitPromise = (async () => {
      // Strategy 1: Load from public directory (most reliable for Vite)
      try {
        const script = document.createElement('script')
        script.src = '/RDKit_minimal.js'
        await new Promise<void>((resolve, reject) => {
          script.onload = () => resolve()
          script.onerror = () => reject(new Error('Script load failed'))
          document.head.appendChild(script)
        })
        const initFn = (window as any).initRDKitModule
        if (initFn) {
          const mod = await initFn({
            locateFile: (file: string) => `/${file}`,
          })
          return mod
        }
      } catch {
        // Strategy 1 failed, try strategy 2
      }

      // Strategy 2: Dynamic import from npm package
      try {
        const rdkitImport: any = await import('@rdkit/rdkit')
        const initFn = rdkitImport.default ?? rdkitImport
        if (typeof initFn === 'function') {
          const mod = await initFn({
            locateFile: (file: string) => `/${file}`,
          })
          return mod
        }
      } catch {
        // Strategy 2 failed
      }

      throw new Error('Could not load RDKit WASM module')
    })()

    rdkitPromise.then((mod) => { rdkitModule = mod }).catch(() => { rdkitPromise = null })
  }
  return rdkitPromise
}

/**
 * Post-process RDKit SVG for dark theme:
 * - Make the white background rect transparent
 * - Replace black (#000000, #000) strokes/fills with light colors
 * - Ensure bonds and atom labels are visible on dark backgrounds
 */
function fixSvgForDarkTheme(svg: string): string {
  return svg
    // Remove white background rect (RDKit adds fill:#FFFFFF on the first rect)
    .replace(
      /(<rect\s+style='[^']*?)fill:#FFFFFF([^']*?')/g,
      '$1fill:none$2'
    )
    .replace(
      /(<rect\s+style="[^"]*?)fill:#FFFFFF([^"]*?")/g,
      '$1fill:none$2'
    )
    // Replace black stroke colors with light gray
    .replace(/stroke='#000000'/g, "stroke='#E0E0E0'")
    .replace(/stroke="#000000"/g, 'stroke="#E0E0E0"')
    .replace(/stroke='#000'/g, "stroke='#E0E0E0'")
    .replace(/stroke="#000"/g, 'stroke="#E0E0E0"')
    // Replace black fill colors (atom labels) with white
    .replace(/fill='#000000'/g, "fill='#FFFFFF'")
    .replace(/fill="#000000"/g, 'fill="#FFFFFF"')
    .replace(/fill='#000'/g, "fill='#FFFFFF'")
    .replace(/fill="#000"/g, 'fill="#FFFFFF"')
    // Handle black stroke in style attributes
    .replace(/style='([^']*)stroke:\s*#000(?:000)?/g, (match) =>
      match.replace(/#000(?:000)?/, '#E0E0E0')
    )
    .replace(/style="([^"]*)stroke:\s*#000(?:000)?/g, (match) =>
      match.replace(/#000(?:000)?/, '#E0E0E0')
    )
    // Handle black fill in style attributes
    .replace(/style='([^']*)fill:\s*#000(?:000)?/g, (match) =>
      match.replace(/#000(?:000)?/, '#FFFFFF')
    )
    .replace(/style="([^"]*)fill:\s*#000(?:000)?/g, (match) =>
      match.replace(/#000(?:000)?/, '#FFFFFF')
    )
}

export function MoleculeViewer2D({
  smiles,
  width = 200,
  height = 200,
  className = '',
  highlightAtoms,
  highlightColors,
  onClick,
}: MoleculeViewer2DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [svgContent, setSvgContent] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!smiles) {
      setSvgContent('')
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    getRDKit()
      .then((RDKit) => {
        if (cancelled) return

        const mol = RDKit.get_mol(smiles)
        if (!mol || !mol.is_valid()) {
          setError('Invalid SMILES')
          setLoading(false)
          if (mol) mol.delete()
          return
        }

        const drawOpts: Record<string, unknown> = {
          width,
          height,
          bondLineWidth: 1.5,
          addAtomIndices: false,
          addStereoAnnotation: true,
        }

        if (highlightAtoms && highlightAtoms.length > 0) {
          drawOpts.atoms = highlightAtoms
          if (highlightColors) {
            drawOpts.highlightColour = highlightColors
          }
        }

        try {
          let svg = mol.get_svg_with_highlights(JSON.stringify(drawOpts))
          svg = fixSvgForDarkTheme(svg)
          if (!cancelled) {
            setSvgContent(svg)
            setLoading(false)
          }
        } catch {
          // Fallback to basic SVG rendering
          try {
            let svg = mol.get_svg(width, height)
            svg = fixSvgForDarkTheme(svg)
            if (!cancelled) {
              setSvgContent(svg)
              setLoading(false)
            }
          } catch {
            if (!cancelled) {
              setError('Render failed')
              setLoading(false)
            }
          }
        }

        mol.delete()
      })
      .catch(() => {
        if (!cancelled) {
          // Fallback: show SMILES text if RDKit not available
          setError('RDKit not loaded')
          setLoading(false)
        }
      })

    return () => { cancelled = true }
  }, [smiles, width, height, highlightAtoms, highlightColors])

  if (loading) {
    return (
      <div
        className={`flex items-center justify-center bg-dark-bg/50 rounded-lg ${className}`}
        style={{ width, height }}
      >
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div
        className={`flex flex-col items-center justify-center bg-dark-bg/50 rounded-lg text-muted-foreground ${className}`}
        style={{ width, height }}
        onClick={onClick}
      >
        <div className="text-[10px] mb-1">{error}</div>
        <div className="text-[9px] font-mono truncate max-w-full px-2">{smiles}</div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={`relative ${onClick ? 'cursor-pointer hover:ring-1 hover:ring-neon-purple/30' : ''} rounded-lg overflow-hidden ${className}`}
      style={{ width, height }}
      onClick={onClick}
      dangerouslySetInnerHTML={{ __html: svgContent }}
    />
  )
}
