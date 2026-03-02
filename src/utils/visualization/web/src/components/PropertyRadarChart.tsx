import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts'
import { useChartColors } from '../contexts/ThemeContext'
import type { MolecularProperties } from '../services/api'

interface PropertyRadarChartProps {
  properties: MolecularProperties
  compareProperties?: MolecularProperties
  className?: string
  height?: number
}

// Normalize each property to 0-1 range based on typical drug-like ranges
function normalizeProperties(props: MolecularProperties) {
  return [
    { property: 'QED', value: props.qed, fullMark: 1 },
    { property: 'MW', value: Math.min(props.molecular_weight / 800, 1), fullMark: 1 },
    { property: 'LogP', value: Math.min(Math.max((props.logp + 2) / 9, 0), 1), fullMark: 1 },
    { property: 'SA', value: Math.min(props.synthetic_accessibility / 10, 1), fullMark: 1 },
    { property: 'TPSA', value: Math.min(props.tpsa / 200, 1), fullMark: 1 },
    { property: 'HBD', value: Math.min(props.hbd / 10, 1), fullMark: 1 },
    { property: 'HBA', value: Math.min(props.hba / 15, 1), fullMark: 1 },
    { property: 'RotBonds', value: Math.min(props.rotatable_bonds / 15, 1), fullMark: 1 },
  ]
}

export function PropertyRadarChart({
  properties,
  compareProperties,
  className = '',
  height = 300,
}: PropertyRadarChartProps) {
  const colors = useChartColors()
  const data = normalizeProperties(properties)

  // Merge comparison data if provided
  if (compareProperties) {
    const compareData = normalizeProperties(compareProperties)
    data.forEach((d, i) => {
      ;(d as any).compare = compareData[i].value
    })
  }

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid stroke={colors.grid} strokeOpacity={0.5} />
          <PolarAngleAxis
            dataKey="property"
            tick={{ fill: colors.axis, fontSize: 10 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 1]}
            tick={false}
            axisLine={false}
          />
          <Radar
            name="Molecule"
            dataKey="value"
            stroke={colors.primary}
            fill={colors.primary}
            fillOpacity={0.2}
            strokeWidth={2}
          />
          {compareProperties && (
            <Radar
              name="Comparison"
              dataKey="compare"
              stroke={colors.secondary}
              fill={colors.secondary}
              fillOpacity={0.1}
              strokeWidth={2}
              strokeDasharray="4 4"
            />
          )}
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgb(var(--dark-panel))',
              border: '1px solid rgb(var(--dark-border))',
              borderRadius: '8px',
              fontSize: '11px',
            }}
          />
          {compareProperties && <Legend wrapperStyle={{ fontSize: '10px' }} />}
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
