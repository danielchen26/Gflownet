# GFlowNet Interactive Visualization

A beautiful, real-time web-based visualization system for understanding GFlowNet training dynamics and learned policies.

## Overview

This visualization provides comprehensive insights into:
- **Training Progress**: Real-time metrics with dynamic updates
- **3D Distribution**: Trajectory density and posterior probability visualization
- **Policy Flow**: Learned action policies and flow fields
- **Interactive Setup**: Configure reward landscapes and training parameters

## Quick Start

```bash
# Run the visualization
julia show_visualization.jl
```

This will:
1. Start the API server at `http://localhost:8080`
2. Launch the web dashboard at `http://localhost:5173`
3. Open your browser automatically

## Features

### 🎯 Problem Setup
- Interactive grid for placing reward peaks
- Configure training objectives (TB, SubTB, DB, FM)
- Adjust exploration rates and episode counts
- Quick presets for common scenarios

### 📊 Training Monitor
- **Real-time Metrics**: Loss, reward, and convergence updated every 250ms
- **Training Curves**: 
  - Multi-loss visualization (Total, TB, Flow)
  - Reward and exploration rate tracking
  - Smooth animations and transitions
- **2D Trajectory View**: Live sampling visualization

### 🌐 3D Distribution View
- **Trajectory Density**: Smooth heatmap showing state visitation frequency
- **Posterior Distribution**: Contour lines for P(x|R)
- **Reward Landscape**: Gradient visualization of reward structure
- **Multiple View Modes**: Density, posterior, or combined views

### ⚡ Policy Flow View
- **Flow Arrows**: Direction and magnitude of learned policy
- **Color Coding**: Visual indication of flow values
- **State Values**: Ring indicators for state value estimates
- **Comprehensive Guide**: Built-in explanation of policy quality indicators

## Architecture

```
visualization/
├── show_visualization.jl    # Main entry point
├── Project.toml            # Julia dependencies
└── README.md              # This file

src/utils/visualization/
├── api/
│   └── gflownet_server.jl # GFlowNet visualization server
└── web/
    ├── src/
    │   ├── components/    # React components
    │   ├── visualizations/ # 3D visualizations
    │   └── App.tsx        # Main app
    └── package.json       # Node dependencies
```

## Technical Details

### Frontend
- **React** with TypeScript for UI
- **Three.js** with React Three Fiber for 3D graphics
- **Recharts** for 2D charts
- **TailwindCSS** for styling
- **Framer Motion** for animations

### Backend
- **Julia** with Oxygen.jl for REST API
- **Dynamic simulation** of GFlowNet training
- **CORS enabled** for local development

### Key Visualizations

1. **Training Progress**
   - Updates every 250ms for smooth number transitions
   - Charts refresh every 500ms during active training
   - Simulates training at 2 episodes/second

2. **3D Rendering**
   - High-resolution (256x256) density heatmaps
   - Gaussian smoothing for continuous visualization
   - Optimized camera angles with optional auto-rotation

3. **Policy Visualization**
   - Arrow length indicates action confidence
   - Color gradient shows flow values
   - Background heatmap displays reward structure

## Customization

### Adding New Domains

1. Modify the reward function in `simple_server.jl`
2. Update grid size and parameters in `ProblemSetup.tsx`
3. Adjust visualization bounds in 3D components

### Server Details

The `gflownet_server.jl` handles:
- Problem configuration from the setup page
- Dynamic training simulation
- Real-time metrics and trajectory generation
- Flow field and state statistics calculation

## Development

```bash
# Frontend development
cd src/utils/visualization/web
npm install
npm run dev

# Backend development
julia --project=. src/utils/visualization/api/gflownet_server.jl
```

## Troubleshooting

- **WebSocket errors**: Expected - the server uses polling for simplicity
- **Blank visualizations**: Ensure both servers are running
- **Performance issues**: Reduce trajectory count or grid resolution

## Future Enhancements

- [ ] WebSocket support for true real-time updates
- [ ] Multiple domain templates
- [ ] Export visualization as video
- [ ] Integration with TensorBoard
- [ ] 3D trajectory replay controls

## Credits

Created as part of the GFlowNet.jl project for better understanding and debugging of GFlowNet training dynamics.