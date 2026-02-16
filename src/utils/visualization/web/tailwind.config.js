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

        // Dark theme colors — CSS variable driven
        dark: {
          bg: 'rgb(var(--dark-bg) / <alpha-value>)',
          panel: 'rgb(var(--dark-panel) / <alpha-value>)',
          border: 'rgb(var(--dark-border) / <alpha-value>)',
        },
        // Neon accent colors — CSS variable driven
        neon: {
          purple: 'rgb(var(--neon-purple) / <alpha-value>)',
          blue: 'rgb(var(--neon-blue) / <alpha-value>)',
          pink: 'rgb(var(--neon-pink) / <alpha-value>)',
          green: 'rgb(var(--neon-green) / <alpha-value>)',
          orange: 'rgb(var(--neon-orange) / <alpha-value>)',
          cyan: 'rgb(var(--neon-cyan) / <alpha-value>)',
        },
        // Gradient colors — CSS variable driven
        gradient: {
          purple: {
            from: 'rgb(var(--gradient-purple-from) / <alpha-value>)',
            via: 'rgb(var(--gradient-purple-via) / <alpha-value>)',
            to: 'rgb(var(--gradient-purple-to) / <alpha-value>)',
          },
          orange: {
            from: 'rgb(var(--gradient-orange-from) / <alpha-value>)',
            via: 'rgb(var(--gradient-orange-via) / <alpha-value>)',
            to: 'rgb(var(--gradient-orange-to) / <alpha-value>)',
          },
          blue: {
            from: 'rgb(var(--gradient-blue-from) / <alpha-value>)',
            via: 'rgb(var(--gradient-blue-via) / <alpha-value>)',
            to: 'rgb(var(--gradient-blue-to) / <alpha-value>)',
          },
          green: {
            from: 'rgb(var(--gradient-green-from) / <alpha-value>)',
            via: 'rgb(var(--gradient-green-via) / <alpha-value>)',
            to: 'rgb(var(--gradient-green-to) / <alpha-value>)',
          },
        },
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace'],
      },
      borderRadius: {
        none: '0',
        sm: 'var(--radius-sm, 0.125rem)',
        DEFAULT: 'var(--radius, 0.25rem)',
        md: 'var(--radius-md, 0.375rem)',
        lg: 'var(--radius-lg, 0.5rem)',
        xl: 'var(--radius-xl, 0.75rem)',
        '2xl': 'var(--radius-2xl, 1rem)',
        '3xl': 'var(--radius-3xl, 1.5rem)',
        full: '9999px',
      },
      animation: {
        'glow': 'glow var(--animation-speed, 2s) ease-in-out infinite alternate',
        'float': 'float var(--animation-speed-slow, 3s) ease-in-out infinite',
        'pulse-neon': 'pulse-neon 1.5s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%': {
            'box-shadow': '0 0 5px rgb(var(--neon-purple) / 0.5), 0 0 10px rgb(var(--neon-purple) / 0.5), 0 0 15px rgb(var(--neon-purple) / 0.5)'
          },
          '100%': {
            'box-shadow': '0 0 10px rgb(var(--neon-purple) / 0.8), 0 0 20px rgb(var(--neon-purple) / 0.8), 0 0 30px rgb(var(--neon-purple) / 0.8)'
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
