# GFlowNet.jl Visualization System Changelog

## Version 2.0.0 - Major Update (January 2025)

### Overview
This major update significantly enhances the GFlowNet.jl web visualization system with improved layouts, better 3D rendering, comprehensive bug fixes, and new interactive features.

### Monitor Tab Improvements

#### Layout Optimization
- **Two-Row Design**: Eliminated scrolling with a fixed two-row layout
  - Row 1: Trajectory visualization (66%) + Real-time metrics (33%)
  - Row 2: Full-width training progress charts
- **Space Utilization**: Reduced padding and font sizes for better content density
- **Responsive Design**: Improved handling of different screen sizes

#### Real-Time Features
- **Trajectory Sampling Window**: Restored with live updates
- **Metrics Updates**: Smooth transitions every 250ms
- **Chart Refresh**: Training progress updates every 500ms
- **No Layout Shifts**: Fixed positioning prevents jitter during updates

### Training Dashboard Enhancements

#### Full History Display
- **Complete Data**: Removed artificial slice limitations
- **All Episodes**: Shows training history from episode 0 to current
- **Long-Term Trends**: Better visualization of convergence patterns
- **Memory Efficient**: Optimized rendering for thousands of data points

#### Synchronized Zoom
- **Recharts Brush**: Interactive zoom component on any chart
- **Cross-Chart Sync**: Zooming one chart updates all others
- **Reset Button**: Quick return to full view
- **Smooth Interactions**: No lag or delays during zoom operations

#### Chart Improvements
- **Removed**: exploration_rate parameter (not used in GFlowNets)
- **Added**: Loss components breakdown chart showing:
  - Trajectory balance loss
  - Flow matching loss
  - Regularization terms
- **Layout**: Reduced margins for more chart area
- **Colors**: Improved color scheme for better distinction

### 3D Visualization Overhaul

#### Smooth Density Surface
- **Gaussian Smoothing**: Reward-weighted kernel for natural appearance
  ```javascript
  kernel_weight = exp(-distance²/(2σ²)) * (1 + reward/max_reward)
  ```
- **Visual Style**: Natural hill-like landscape
- **Color Gradient**: Blue → Cyan → Green → Yellow → Orange
- **Resolution**: 256x256 texture for smooth rendering
- **GPU Acceleration**: WebGL texture mapping for performance

#### Discrete Bars Visualization
- **Toggle Control**: Switch between smooth surface and discrete bars
- **Bar Height**: Proportional to endpoint density
- **Bar Color**: Indicates average reward at location
- **Use Case**: Better for understanding discrete state visitation

#### Posterior Probability Display
- **3D Spheres**: Replaced contour lines with interactive spheres
- **Size Mapping**: Sphere radius ∝ P(s_T)
- **Color Gradient**: Purple (low reward) → Pink (high reward)
- **Interactive Labels**: Show P and R̄ for significant endpoints
- **Threshold**: Only displays endpoints with P > 0.01

#### Coordinate System Fix
- **Y-Up System**: Proper 3D coordinate convention
- **Ground Plane**: Grid renders on horizontal XZ plane
- **Alignment**: All elements properly positioned
- **Camera**: Optimal viewing angle for 3D perspective
- **Depth**: Correct rendering order and occlusion

### Bug Fixes

#### Navigation Issues
- **Black Screen Fix**: Resolved "Start Training" button navigation bug
- **Component Lifecycle**: Proper cleanup and initialization
- **Resource Management**: Three.js objects properly disposed
- **Route Transitions**: Smooth transitions between views

#### 3D Rendering Fixes
- **Plane Alignment**: Grid and density surface properly aligned
- **Depth Buffer**: Correct configuration for proper occlusion
- **Z-Fighting**: Eliminated overlapping element conflicts
- **Transparency**: Proper sorting for transparent spheres

#### Component Issues
- **Missing References**: Removed DistributionComparison component calls
- **Import Statements**: Updated all module imports
- **TypeScript Errors**: Fixed type definition issues
- **Props Validation**: Corrected component property types

### Performance Optimizations

#### Rendering Efficiency
- **Memoization**: React.memo for expensive 3D calculations
- **Frustum Culling**: Only render visible spheres
- **Texture Caching**: Reuse density textures when possible
- **Batch Updates**: Group state changes to reduce re-renders

#### Data Management
- **Circular Buffer**: Efficient trajectory storage
- **Incremental Updates**: Only update changed density regions
- **Lazy Loading**: 3D components load on demand
- **WebGL Optimization**: High-performance GPU settings

### Technical Details

#### Frontend Stack
- React 18 with TypeScript
- Three.js + React Three Fiber
- Recharts for 2D charts
- TailwindCSS for styling
- Framer Motion for animations

#### Backend Integration
- Julia with Oxygen.jl
- REST API endpoints
- CORS enabled
- Dynamic training simulation

#### Key Files Modified
- `GFlowNetTrainingDashboard.tsx`: Full history, synchronized zoom
- `GFlowNetDistribution3D.tsx`: Smooth surface, discrete bars, spheres
- `TrainingMonitor.tsx`: Two-row layout, trajectory window
- `web_visualization_architecture.md`: Updated documentation

### Migration Notes

#### For Developers
1. Update density calculation to use reward weighting
2. Replace contour visualization with sphere rendering
3. Use Y-up coordinate system for all 3D elements
4. Implement synchronized zoom with Recharts Brush

#### For Users
1. Clear browser cache for latest updates
2. New toggle controls in 3D view
3. Use brush component for zooming training charts
4. Check sphere labels for posterior probabilities

### Future Enhancements
- WebSocket support for real-time updates
- Export functionality for visualizations
- Multi-domain visualization templates
- Mobile responsive design
- Advanced analytics dashboard

---

## Version 1.0.0 - Initial Release (August 2024)

### Features
- Basic training monitor with metrics
- Simple 3D trajectory visualization
- Policy flow field display
- Problem setup interface

### Known Issues (Fixed in v2.0.0)
- Monitor tab required scrolling
- Limited training history display
- Basic 3D rendering without smoothing
- Navigation bugs
- Missing trajectory sampling window