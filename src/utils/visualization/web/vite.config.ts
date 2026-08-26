import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      // useBackendHealth polls GET /health (unified_server.jl:342), which is
      // NOT under /api and so needs its own proxy entry.
      '/health': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      // No '/ws' entry: the backend registers no @websocket route, and the
      // useWebSocket stub that pretended otherwise has been removed.
    },
  },
})