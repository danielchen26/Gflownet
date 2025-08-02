/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Base colors from CSS variables
        background: 'rgb(var(--background) / <alpha-value>)',
        foreground: 'rgb(var(--foreground) / <alpha-value>)',
        card: {
          DEFAULT: 'rgb(var(--card) / <alpha-value>)',
          foreground: 'rgb(var(--card-foreground) / <alpha-value>)'
        },
        primary: {
          DEFAULT: 'rgb(var(--primary) / <alpha-value>)',
          foreground: 'rgb(var(--primary-foreground) / <alpha-value>)'
        },
        secondary: {
          DEFAULT: 'rgb(var(--secondary) / <alpha-value>)',
          foreground: 'rgb(var(--secondary-foreground) / <alpha-value>)'
        },
        accent: {
          DEFAULT: 'rgb(var(--accent) / <alpha-value>)',
          foreground: 'rgb(var(--accent-foreground) / <alpha-value>)'
        },
        muted: {
          DEFAULT: 'rgb(var(--muted) / <alpha-value>)',
          foreground: 'rgb(var(--muted-foreground) / <alpha-value>)'
        },
        border: 'rgb(var(--border) / <alpha-value>)',
        ring: 'rgb(var(--ring) / <alpha-value>)',
        
        // Dark theme colors
        dark: {
          bg: '#0A0A0B',
          panel: '#1A1A1D',
          border: '#2A2A2D',
        },
        // Neon accent colors
        neon: {
          purple: '#BD00FF',
          blue: '#00D9FF',
          pink: '#FF006E',
          green: '#00FF88',
          orange: '#FF8E53',
        },
        // Gradient colors
        gradient: {
          purple: {
            from: '#8B5CF6',
            via: '#A78BFA',
            to: '#D946EF',
          },
          orange: {
            from: '#FF6B35',
            via: '#FF8E53',
            to: '#FFA07A',
          },
          blue: {
            from: '#3B82F6',
            via: '#60A5FA',
            to: '#06B6D4',
          },
          green: {
            from: '#10B981',
            via: '#34D399',
            to: '#6EE7B7',
          },
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 3s ease-in-out infinite',
        'pulse-neon': 'pulse-neon 1.5s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%': { 
            'box-shadow': '0 0 5px rgba(189, 0, 255, 0.5), 0 0 10px rgba(189, 0, 255, 0.5), 0 0 15px rgba(189, 0, 255, 0.5)' 
          },
          '100%': { 
            'box-shadow': '0 0 10px rgba(189, 0, 255, 0.8), 0 0 20px rgba(189, 0, 255, 0.8), 0 0 30px rgba(189, 0, 255, 0.8)' 
          },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        'pulse-neon': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
}