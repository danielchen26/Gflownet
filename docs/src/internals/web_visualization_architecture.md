# GFlowNet Web Visualization Architecture

## Overview

The GFlowNet.jl web visualization system provides real-time, interactive insights into GFlowNet training dynamics and learned policies. It consists of a React/TypeScript frontend with Three.js 3D graphics and a Julia backend REST API.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Frontend (React)                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Problem   │  │   Training   │  │  3D Visualiz.   │   │
│  │    Setup    │  │   Monitor    │  │   Components    │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│  ┌──────┴─────────────────┴────────────────────┴────────┐   │
│  │              React Query + Axios Client              │   │
│  └──────────────────────────┬───────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────┘
                              │ HTTP/REST
┌─────────────────────────────┼───────────────────────────────┐
│                       Backend (Julia)                         │
├─────────────────────────────┴───────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Oxygen.jl API  │  │   Training   │  │  Trajectory  │   │
│  │     Server      │  │  Simulator   │  │  Generator   │   │
│  └────────┬────────┘  └──────┬───────┘  └──────┬───────┘   │
│           │                   │                  │           │
│  ┌────────┴───────────────────┴──────────────────┴──────┐   │
│  │           GFlowNet Core (Training, Sampling)         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Frontend Architecture

### Technology Stack
- **React 18** with TypeScript for UI framework
- **Three.js** with React Three Fiber for 3D visualization
- **Recharts** for 2D training charts
- **TailwindCSS** for styling
- **Framer Motion** for animations
- **React Query** for server state management

### Component Structure

```typescript
src/
├── components/
│   ├── ProblemSetup.tsx         # Configure training parameters
│   ├── GFlowNetTrainingDashboard.tsx  # Real-time training monitor
│   └── Layout.tsx               # App layout and navigation
├── visualizations/
│   ├── GFlowNetDistribution3D.tsx    # 3D trajectory density
│   ├── GFlowNetFlowField.tsx        # Policy flow arrows
│   └── TrajectoryView2D.tsx         # 2D trajectory paths
├── lib/
│   ├── axios.ts                 # HTTP client config
│   └── queryClient.ts           # React Query setup
└── utils/
    └── colors.ts                # Color schemes and gradients
```

### Key Features

#### 1. Real-Time Updates
- Training metrics poll every 250ms for smooth number transitions
- Chart data refreshes every 500ms during active training
- WebSocket-ready architecture (currently using polling)

#### 2. 3D Visualization
- **Density Heatmaps**: High-resolution (256x256) with Gaussian smoothing
- **Trajectory Paths**: Color-coded by reward value
- **Reward Landscape**: Gradient visualization of reward structure
- **Camera Controls**: OrbitControls with auto-rotation

#### 3. Performance Optimizations
- Memoized computations for expensive 3D calculations
- Texture-based rendering for smooth heatmaps
- LOD (Level of Detail) for trajectory rendering
- WebGL power preference set to "high-performance"

## Backend Architecture

### Technology Stack
- **Julia** with Oxygen.jl for REST API
- **JSON3.jl** for JSON serialization
- **HTTP.jl** for server infrastructure
- **CORS** enabled for local development

### API Endpoints

```julia
# Training Control
POST /api/training/reset        # Reset training state
GET  /api/training/state        # Current training status
GET  /api/training/history      # Full training history
GET  /api/training/metrics      # Latest metrics only

# Trajectory Data
GET  /api/trajectories          # List recent trajectories
GET  /api/trajectories/{id}     # Single trajectory details
POST /api/trajectories/sample   # Generate new trajectory
GET  /api/trajectories/all      # All trajectories for 3D view

# Analysis
GET  /api/analysis/flow-field   # Policy flow vectors
GET  /api/analysis/state-statistics  # State visitation counts
GET  /api/analysis/distribution # Distribution statistics
```

### Data Formats

#### Training State
```json
{
  "is_training": true,
  "current_episode": 150,
  "elapsed_time": 75.0,
  "current_loss": 0.543,
  "current_reward": 18.7,
  "current_exploration": 0.15
}
```

#### Trajectory Format
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "states": [[1.5, 2.3], [2.1, 2.8], ...],
  "rewards": [0.5, 1.2, ...],
  "total_reward": 25.3,
  "trajectory_type": "exploitation"
}
```

#### Flow Field Data
```json
{
  "resolution": [15, 15, 1],
  "bounds": {"x": [1, 10], "y": [1, 10]},
  "data": [{
    "position": [5.0, 5.0, 0],
    "velocity": [0.3, -0.2, 0],
    "magnitude": 0.36,
    "reward": 2.5,
    "flow_value": 8.3
  }, ...]
}
```

## Data Flow

### Training Monitor Flow
```
1. Frontend polls /api/training/state (250ms)
2. Backend calculates progress based on elapsed time
3. Dynamic data generation simulates training
4. Frontend updates numbers with smooth transitions
5. Charts poll /api/training/history (500ms)
6. Recharts renders with animation
```

### 3D Visualization Flow
```
1. Frontend requests /api/trajectories/all
2. Backend generates diverse trajectory set
3. Frontend processes into density texture
4. Three.js renders with WebGL
5. User interacts with OrbitControls
6. Updates trigger re-computation
```

## Integration Points

### Connecting to Real GFlowNet

Replace `simple_server.jl` with `gflownet_server.jl`:

```julia
# Real GFlowNet Integration
function create_gflownet_server(model::GFlowNetModel, env::GFlowEnvironment)
    # Hook into training callbacks
    @post "/api/training/step" function()
        metrics = train_step!(model, env)
        return json(metrics)
    end
    
    # Sample real trajectories
    @get "/api/trajectories/sample" function()
        trajectory = sample_trajectory(model, env)
        return json(serialize_trajectory(trajectory))
    end
    
    # Compute actual flow field
    @get "/api/analysis/flow-field" function()
        flow_field = compute_policy_flow(model, env)
        return json(flow_field)
    end
end
```

### Domain Adaptation

For new domains, implement:

1. **State Serialization**: Convert domain states to [x, y] coordinates
2. **Reward Mapping**: Project rewards to visual space
3. **Action Visualization**: Map actions to flow vectors

Example for molecule domain:
```julia
function serialize_molecule_state(mol::Molecule)
    # Use 2D embedding (e.g., t-SNE, UMAP)
    coords = embed_2d(mol)
    return [coords.x, coords.y]
end
```

## Performance Considerations

### Frontend Optimization
- Use `useMemo` for expensive calculations
- Implement virtual scrolling for large trajectory lists
- Debounce user inputs in problem setup
- Lazy load 3D components

### Backend Optimization
- Cache trajectory computations
- Use streaming for large datasets
- Implement pagination for trajectory lists
- Pre-compute flow fields

### Scalability
- WebSocket support for real-time updates
- Worker threads for trajectory generation
- GPU acceleration for flow computation
- CDN for static assets

## Development Workflow

### Setup
```bash
# Backend
cd src/utils/visualization/api
julia --project=. simple_server.jl

# Frontend
cd src/utils/visualization/web
npm install
npm run dev
```

### Testing
```bash
# Frontend tests
npm test

# Backend tests
julia --project=. test/visualization_api_tests.jl
```

### Building for Production
```bash
# Frontend build
npm run build

# Serve with Julia
julia --project=. serve_production.jl
```

## Future Enhancements

1. **WebSocket Integration**: Real-time bidirectional communication
2. **Multi-Domain Support**: Templates for different problem types
3. **Collaborative Features**: Share visualizations via URLs
4. **Export Capabilities**: Save visualizations as videos/images
5. **Advanced Analytics**: Convergence diagnostics, mode discovery
6. **Mobile Support**: Responsive design for tablets

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend has CORS headers set
2. **Performance Issues**: Reduce trajectory count or grid resolution
3. **WebGL Errors**: Check browser compatibility
4. **Memory Leaks**: Dispose Three.js objects properly

### Debug Mode

Enable debug logging:
```typescript
// Frontend
localStorage.setItem('debug', 'gflownet:*')

// Backend
ENV["GFLOWNET_DEBUG"] = "true"
```

## Conclusion

The web visualization architecture provides a flexible, performant foundation for understanding GFlowNet training dynamics. Its modular design allows easy extension to new domains while maintaining real-time responsiveness.