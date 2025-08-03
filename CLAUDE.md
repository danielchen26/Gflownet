# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Principles

- Remember forever, do not write Co-Authored-By: Claude <noreply@anthropic.com> or anything related to AI or Claude in the future and now
- Remember never show Co-Authored-By: Claude <noreply@anthropic.com> Commit FLOW_MATCHING implementation in the end

## Repository Overview

GFlowNet.jl is a production-ready Julia implementation of Generative Flow Networks (GFlowNets) - a breakthrough class of generative models that learn to sample diverse, high-quality objects proportionally to a reward function. The codebase follows modern ML package design with clean separation of concerns and comprehensive tooling.

## Recently Completed Features (January 2025)

### Newly Implemented
1. **DETAILED_BALANCE Training Objective** ✅
   - Full implementation with joint backward policy representation
   - Zygote-compatible gradient computation
   - Comprehensive test coverage

2. **Complete Flow Computation** ✅
   - Recursive flow calculation with memoization
   - Edge flows and partition function computation
   - Proper caching with Zygote compatibility

3. **Learnable Partition Function** ✅
   - LEARNABLE_ESTIMATION option for trainable Z
   - Improves exploration and theoretical correctness

4. **Backward Policy with Joint Representation** ✅
   - P_B(s|s') using concatenated state features
   - Required for DETAILED_BALANCE
   - Enables better credit assignment
   - Added validation functions for normalization checks

5. **FLOW_MATCHING Objective** ✅
   - Complete implementation minimizing (Z(s) - F(s))²
   - Uses flow estimator network
   - Full test coverage

6. **Multi-Start GFlowNets** ✅
   - Support for multiple initial states
   - Per-initial-state partition functions
   - Initial state sampling based on Z values

7. **Training Code Reorganization** ✅
   - Moved all training functions from core/interface.jl to training/
   - Clean separation: interface.jl now only has model creation
   - Better modularity and maintainability

8. **SUB_TRAJECTORY_BALANCE Objective** ✅
   - Implemented sub-trajectory balance loss for O(T²) learning signals
   - Better credit assignment for long trajectories
   - Domain-agnostic implementation
   - Configurable sub-trajectory length

9. **DIRECT_FLOW_OBJECTIVE Training Method** ✅
   - Neural network directly estimates F(s) instead of recursive computation
   - Added flow estimator network support with include_flow_estimator parameter
   - Trades accuracy for computational efficiency
   - Ideal for large state spaces where recursive flow is expensive

10. **Interactive Web Visualization System** ✅ (January 2025)
   - Beautiful real-time visualization for GFlowNet training and analysis
   - Three main views: Training Monitor, 3D Distribution, Policy Flow
   - React + Three.js frontend with smooth animations
   - Julia backend with dynamic training simulation
   - Located in `examples/core_features/visualization/`
   
11. **Visualization System Major Update** ✅ (January 2025)
   - **Monitor Tab**: Two-row layout, real-time trajectory sampling, optimized space usage
   - **Training Dashboard**: Full history with synchronized zoom, loss components breakdown
   - **3D Visualization**: Smooth density surfaces, discrete bars toggle, sphere-based posteriors
   - **Performance**: Memoization, frustum culling, incremental updates
   - **Bug Fixes**: Navigation issues, 3D alignment, proper Y-up coordinate system

## Project Roadmap and Vision

GFlowNet.jl has a comprehensive development roadmap focused on making it the premier production-ready implementation of Generative Flow Networks. See [ROADMAP.md](ROADMAP.md) for detailed development phases, timelines, and success metrics.

### Key Upcoming Priorities
1. **GPU Acceleration**: Full GPU pipeline for 10-100x speedup
2. **Continuous Domains**: Support for continuous state/action spaces
3. **Advanced Domains**: Molecular design, protein engineering, industrial applications
4. **Developer Experience**: AutoML integration, debugging tools, model zoo
5. **Ecosystem Integration**: PyTorch/JAX bridges, cloud deployment, MLflow

## Agent Coordination System

The project uses specialized AI agents for different aspects of development. A new hierarchical coordination system has been proposed to improve efficiency. See [.claude/system_design/](.claude/system_design/) for design documents and proposals.

### Current Specialized Agents

#### Master Orchestrator
- **gflownet-master-orchestrator**: Analyzes queries, decomposes tasks, and intelligently coordinates all other agents

#### Specialist Agents
- **gflownet-architecture-analyzer**: System design and architecture analysis
- **gflownet-debugger**: Bug diagnosis and fixing
- **gflownet-mathematician**: Theoretical foundations and proofs
- **gflownet-performance-optimizer**: GPU acceleration and optimization
- **gflownet-domain-implementer**: New domain applications
- **gflownet-testing-validator**: Test design and validation
- **gflownet-training-expert**: Training configuration and hyperparameters
- **gflownet-documentation**: Documentation and tutorials

## Development Guidance

- Remember that example folders are associated with different domains and for the examples related to the core development, you should put them into the core features sub folder
- Remember you should put the test in the relevent folder not just scatter it in the test folder, we already have a Hierarchical folders
- You should remember that every time we update the documentation and changes in the architecture we should automatically update the agents and all files in the @.claude 

### Visualization System Notes

#### Architecture
- **Frontend**: React 18 + TypeScript + Three.js/React Three Fiber for 3D visualization
- **Backend**: Julia with Oxygen.jl providing REST API endpoints
- **Real-time Updates**: Polling-based system (250ms for metrics, 500ms for charts)
- **3D Rendering**: High-resolution (256x256) textures with reward-weighted Gaussian smoothing

#### Key Components
1. **Training Monitor**: 
   - Two-row layout without scrolling
   - Real-time trajectory sampling window
   - Live metrics with smooth transitions
   - Full-width training progress charts
   
2. **3D Distribution View**: 
   - Smooth density surface with natural hill appearance
   - Toggle between smooth surface and discrete bars
   - Sphere-based posterior probability display
   - Proper Y-up coordinate system with XZ ground plane
   
3. **Training Dashboard**:
   - Full training history display (no slicing)
   - Synchronized zoom with Recharts Brush
   - Loss components breakdown chart
   - Removed unused exploration_rate parameter
   
4. **Problem Setup**: 
   - Interactive configuration for reward peaks
   - Training parameter adjustment
   - Real-time preview of reward landscape

#### File Structure
```
src/utils/visualization/
├── api/
│   ├── simple_server.jl    # Mock API with simulated training
│   └── gflownet_server.jl  # Real GFlowNet integration (template)
└── web/
    ├── src/
    │   ├── components/     # React UI components
    │   └── visualizations/ # 3D visualization components
    └── package.json

examples/core_features/visualization/
├── show_visualization.jl   # Main entry point
├── README.md              # Comprehensive usage guide
└── Project.toml
```

#### Important Implementation Details
- Dynamic training simulation starts automatically on server launch
- Training progress based on elapsed time (2 episodes/second)
- All 3D elements use proper Y-up coordinate system with XZ ground plane
- Policy flow arrows use simplified geometry for WebGL compatibility
- CORS enabled for local development

#### Visualization Updates (January 2025)
- **Density Calculation**: Reward-weighted Gaussian kernel for natural appearance
- **Sphere Rendering**: Size proportional to P(s_T), color gradient by reward
- **Coordinate Fix**: Y-up system with grid on XZ plane, proper depth sorting
- **Performance**: Memoization, frustum culling, texture caching
- **UI Improvements**: Reduced padding, optimized layouts, synchronized controls
- **Bug Fixes**: Navigation black screen, component lifecycle, proper cleanup

## Testing Strategy

- Unit tests for core functionality in `test/`
- Integration tests for domain implementations
- Property-based testing for mathematical properties
- Performance benchmarks for critical paths
- Visualization tests using mock data

## Documentation

- API documentation using Julia docstrings
- Conceptual guides in `docs/src/guide/`
- Mathematical theory in `docs/src/theory/`
- Internal documentation in `docs/src/internals/`
  - Architecture design in `internals/architecture/`
  - Implementation notes in `internals/implementation_notes/`
  - Development guidelines in `internals/development_guides/`
- Interactive examples with visualization
- Visualization changelog in `src/utils/visualization/CHANGELOG.md`

## Code Style

- Follow Julia style guide
- Use meaningful variable names
- Document all public functions
- Keep functions focused and small
- Prefer composition over inheritance