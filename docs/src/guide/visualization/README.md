# GFlowNet Modern Web Visualization System

## Overview

This is a cutting-edge web-based visualization system for GFlowNet that leverages modern web technologies to create beautiful, interactive, and performant data visualizations that rival the best data visualization platforms.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Julia Backend                   │
│  ┌─────────────┐        ┌──────────────────┐   │
│  │  GFlowNet   │───────▶│   Oxygen.jl API   │   │
│  │    Core     │        │   (REST + WS)     │   │
│  └─────────────┘        └──────────────────┘   │
└─────────────────────────────────────────────────┘
                          │
                          │ JSON/WebSocket
                          ▼
┌─────────────────────────────────────────────────┐
│                  Web Frontend                    │
│  ┌─────────────┐        ┌──────────────────┐   │
│  │   React +   │───────▶│   D3.js +       │   │
│  │ TypeScript  │        │   Three.js      │   │
│  └─────────────┘        └──────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Features

### 🎨 Beautiful Visualizations
- **3D Trajectory Visualization**: WebGL-powered particle systems with glow effects
- **Flow Field Analysis**: Volumetric rendering with interactive slicing
- **Real-time Dashboards**: Smooth 60fps animations with spring physics
- **Modern UI/UX**: Glassmorphism, neon accents, dark/light themes

### 🚀 Performance
- **GPU Acceleration**: WebGL shaders for millions of particles
- **Efficient Rendering**: Instanced rendering and LOD systems
- **Real-time Updates**: WebSocket streaming with < 50ms latency
- **Optimized Data Transfer**: Binary protocols for large datasets

### 🛠️ Technology Stack
- **Backend**: Julia + Oxygen.jl for high-performance API
- **Frontend**: React 18 + TypeScript for type-safe UI
- **Visualization**: D3.js (2D) + Three.js (3D) + React Three Fiber
- **Styling**: Tailwind CSS + Framer Motion for animations
- **State**: Zustand + React Query for efficient data management

## Quick Start

### 1. Install Dependencies

```bash
# Julia dependencies (from project root)
julia --project -e 'using Pkg; Pkg.add(["Oxygen", "HTTP", "JSON3"])'

# Web dependencies
cd src/utils/visualization/web
npm install
```

### 2. Start the API Server

```bash
# From project root
julia --project -e 'include("src/utils/visualization/api/server.jl")'
```

The API server will start at `http://localhost:8080`

### 3. Start the Web Dashboard

```bash
# In another terminal
cd src/utils/visualization/web
npm run dev
```

The web dashboard will be available at `http://localhost:5173`

## API Endpoints

### REST API

- `GET /health` - Health check
- `GET /api/trajectories` - List all trajectories
- `GET /api/trajectories/:id` - Get specific trajectory
- `POST /api/trajectories/sample` - Sample new trajectory
- `GET /api/training/history` - Get training history
- `GET /api/training/metrics` - Get current metrics
- `GET /api/analysis/flow-field` - Get flow field data

### WebSocket Events

Connect to `ws://localhost:8080/ws` for real-time updates:

```javascript
// Subscribe to channels
ws.send({
  type: 'subscribe',
  channels: ['training', 'trajectories']
})

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  if (data.type === 'training.update') {
    // Handle training update
  }
}
```

## Development

### Project Structure

```
visualization/
├── api/
│   └── server.jl          # Oxygen.jl API server
├── web/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── visualizations/# D3/Three.js visualizations
│   │   ├── hooks/         # Custom React hooks
│   │   ├── stores/        # Zustand stores
│   │   └── utils/         # Utilities
│   ├── package.json
│   └── vite.config.ts
├── README.md              # This file
└── visualization.jl       # Julia exports
```

### Adding New Visualizations

1. Create visualization component in `web/src/visualizations/`
2. Add API endpoint in `api/server.jl` if needed
3. Connect via React component in `web/src/components/`

### Styling Guide

We use a modern, premium design system:

- **Colors**: Dark theme with vibrant neon accents
- **Typography**: Inter for UI, JetBrains Mono for code
- **Effects**: Subtle glassmorphism, smooth gradients
- **Animations**: 60fps spring-based physics

## Examples

### Basic Trajectory Visualization

```javascript
import { TrajectoryViewer } from '@/visualizations/TrajectoryViewer'

function MyComponent() {
  return (
    <TrajectoryViewer
      trajectories={data}
      style="particles"
      effects={{
        glow: true,
        trails: true,
        particles: true
      }}
    />
  )
}
```

### Real-time Training Monitor

```javascript
import { TrainingMonitor } from '@/components/TrainingMonitor'

function Dashboard() {
  return (
    <TrainingMonitor
      realTime={true}
      metrics={['loss', 'reward']}
      smoothing={0.9}
    />
  )
}
```

## Performance Optimization

1. **Large Datasets**: Use data decimation and LOD
2. **Mobile**: Automatic quality reduction
3. **Slow Networks**: Progressive loading with placeholders
4. **Memory**: Cleanup Three.js geometries and materials

## Deployment

### Docker

```bash
docker-compose up
```

### Manual

1. Build frontend: `npm run build`
2. Serve API: `julia --project server.jl --host 0.0.0.0`
3. Serve static files with nginx/caddy

## Troubleshooting

### Common Issues

1. **CORS Errors**: API server includes CORS headers by default
2. **WebSocket Connection**: Check firewall settings
3. **Performance**: Enable GPU acceleration in browser
4. **Memory Leaks**: Dispose Three.js objects properly

### Debug Mode

Enable debug logging:

```javascript
// Frontend
localStorage.setItem('debug', 'gflownet:*')

// Backend
ENV["JULIA_DEBUG"] = "GFlowNet"
```

## Contributing

1. Follow the TypeScript style guide
2. Write tests for new visualizations
3. Optimize for 60fps performance
4. Ensure mobile compatibility
5. Document new components

## License

Same as GFlowNet.jl main project

---

*Creating beautiful visualizations for the future of AI* ✨