# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Continuity

When starting a conversation:
1. Check `.claude/sessions/` for recent session logs
2. Load the latest session context if user requests it
3. Continue from previous TODOs and decisions
4. See `.claude/docs/CONVERSATION_PERSISTENCE.md` for details

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

GFlowNet.jl has a comprehensive development roadmap focused on making it the premier production-ready implementation of Generative Flow Networks. See [docs/src/internals/development_guides/roadmap.md](docs/src/internals/development_guides/roadmap.md) for detailed development phases, timelines, and success metrics.

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

## Claude Code Skills

The `.claude/skills/` directory contains workflow-oriented skills for GFlowNet development. These provide active, step-by-step guidance that should be invoked via the Skill tool.

### Available Skills

#### `systematic-debugging`
**When to use**: Encountering bugs, training failures, or Zygote errors
**What it does**: Evidence-first debugging methodology that checks previous successful runs before making changes

**Key workflow**:
1. Gather success evidence (check results/, git history)
2. Compare working vs broken versions
3. Identify root cause (Zygote mutations, type issues, numerical instability)
4. Fix root cause only, keep everything else intact

#### `domain-implementation`
**When to use**: Implementing a new GFlowNet domain or application
**What it does**: Step-by-step workflow for creating domains with all required interfaces

**Key workflow**:
1. Define state and action types
2. Implement 5 required interface functions (state_to_features, is_applicable, apply_action, is_terminal_state, reward)
3. Create model using high-level API (never manual networks!)
4. Configure training with TrainingConfig
5. Write comprehensive tests

#### `code-review`
**When to use**: Before commits, after implementation, or reviewing generated code
**What it does**: Comprehensive code quality checklist with Zygote compatibility verification

**Key checks**:
- ✅ Zygote compatibility (no mutations!)
- ✅ Clean comment style (no development tags)
- ✅ Proper type system usage
- ✅ Numerical stability (positive rewards, safe operations)
- ✅ Example directory structure compliance

#### `testing-strategy`
**When to use**: Writing tests for new features or domains
**What it does**: Comprehensive testing workflow with property-based tests and benchmarks

**Key tests**:
- Mathematical properties (flow conservation, probability normalization)
- Zygote compatibility (gradient computation)
- Training objectives (all 6 objectives)
- Interface compliance (all required functions)
- Performance benchmarks

### Using Skills

Skills are invoked automatically by Claude Code when appropriate. You can also explicitly request them:

```
user: "The grid world training is failing with a Zygote error"
# Claude automatically invokes systematic-debugging skill
# Creates TodoWrite checklist for debugging workflow
```

### Skills vs Reference Documentation

**Skills** (`.claude/skills/` directory):
- Workflow-oriented procedures
- Create TodoWrite checklists
- Invoked actively via Skill tool
- Guide HOW to do something

**Reference Docs** (`docs/src/reference/`):
- Fact-oriented specifications
- Read passively for information
- Provide context and API details
- Explain WHAT something is

### Critical Context Files

Critical development rules are maintained in `.claude/critical_context/` and must be consulted before any code generation or modification:

#### 1. Zygote Compatibility ([.claude/critical_context/zygote_compatibility.md](.claude/critical_context/zygote_compatibility.md))
**ALWAYS CHECK BEFORE**: Writing any domain logic, implementing state transitions, or creating differentiable functions

**Key Rules**:
- ❌ NO mutations in differentiable functions (`+=`, `push!`, array mutations)
- ✅ Pure functional transformations only
- ✅ Use conditional expressions instead of mutations
- ✅ Debugging: Look for `+=`, `-=`, `push!`, `append!` when Zygote errors occur

**When to Read**: Before implementing `apply_action()`, `state_to_features()`, or any function called during backpropagation

#### 2. High-Level API Usage ([.claude/critical_context/high_level_api.md](.claude/critical_context/high_level_api.md))
**ALWAYS CHECK BEFORE**: Creating models, writing examples, or training GFlowNets

**Key Rules**:
- ❌ NEVER manually define neural networks with `Chain()` or `Dense()`
- ✅ Always use `create_gflownet()` for model creation
- ✅ Always use `train_gflownet()` for training
- ✅ Use configuration helpers: `create_default_config()`, `create_fast_config()`, etc.

**When to Read**: Before writing any example, creating models, or showing users how to train

#### Quick Reference
See [docs/src/reference/quick_reference.md](docs/src/reference/quick_reference.md) for a condensed overview of critical rules, current project state, key file locations, and common pitfalls.

## Invoking Mechanism: How .claude/ Structure Works

The `.claude/` directory contains different types of guidance that are invoked at different times:

### 1. Critical Context (`.claude/critical_context/`) - ALWAYS AVAILABLE
**When Invoked**: Automatically loaded at session start, consulted before any code generation
**Purpose**: Non-negotiable development rules that must ALWAYS be followed
**Files**:
- `zygote_compatibility.md` - AD/Zygote compatibility rules
- `high_level_api.md` - API usage requirements

**Usage Pattern**:
```
Before writing ANY Julia code → Check zygote_compatibility.md
Before creating ANY model or example → Check high_level_api.md
```

### 2. Skills (`.claude/skills/`) - ACTIVELY INVOKED
**When Invoked**: Explicitly invoked via Skill tool when matching workflow is needed
**Purpose**: Step-by-step workflows for complex tasks
**Files**:
- `systematic-debugging.md` - Debug bugs and errors
- `domain-implementation.md` - Implement new domains
- `code-review.md` - Review code quality
- `testing-strategy.md` - Write comprehensive tests

**Usage Pattern**:
```
User reports bug → Invoke systematic-debugging skill
User wants new domain → Invoke domain-implementation skill
Before commit → Invoke code-review skill
Writing tests → Invoke testing-strategy skill
```

### 3. Reference Docs (`docs/src/reference/`) - PASSIVELY READ
**When Invoked**: Read when specific factual information is needed
**Purpose**: Fact-oriented specifications and API details
**Files**:
- `architecture.md` - System architecture overview
- `project_structure.md` - Directory layout and file organization
- `core_concepts.md` - GFlowNet mathematical foundations
- `quick_reference.md` - Quick lookup for common patterns

**Usage Pattern**:
```
Need to understand codebase structure → Read project_structure.md
Need mathematical background → Read core_concepts.md
Need quick lookup → Read quick_reference.md
```

### 4. Agent Definitions (`.claude/agents/`) - TASK-SPECIFIC
**When Invoked**: Via Task tool when specialized expertise is needed
**Purpose**: Domain-specific agents for complex multi-step tasks
**Usage Pattern**:
```
Complex debugging → Launch gflownet-debugger agent
Performance optimization → Launch gflownet-performance-optimizer agent
Architecture questions → Launch gflownet-architecture-analyzer agent
```

### 5. System Design (`.claude/system_design/`) - COORDINATION
**When Invoked**: When designing agent workflows or improving development process
**Purpose**: Agent coordination protocols and system design docs

### 6. Session Logs (`.claude/sessions/`) - CONTINUITY
**When Invoked**: At session start if user requests continuation
**Purpose**: Preserve context across sessions
**See**: `.claude/docs/CONVERSATION_PERSISTENCE.md`

### Comprehensive Workflow Example

```
User: "I want to implement a molecule generation domain"

Step 1: Check critical context (ALWAYS)
  → Read .claude/critical_context/zygote_compatibility.md
  → Read .claude/critical_context/high_level_api.md
  → Internalize: No mutations, use create_gflownet()

Step 2: Invoke appropriate skill (ACTIVE)
  → Invoke domain-implementation skill
  → Get step-by-step workflow with TodoWrite checklist

Step 3: Consult reference docs as needed (PASSIVE)
  → Read docs/src/reference/project_structure.md to find where to put files
  → Read docs/src/reference/core_concepts.md for domain design patterns

Step 4: Implement following critical context rules
  → Write pure functional apply_action() (no mutations!)
  → Use create_gflownet() API (no manual networks!)

Step 5: Before commit
  → Invoke code-review skill
  → Verify Zygote compatibility, API usage, code quality

Step 6: Write tests
  → Invoke testing-strategy skill
  → Add mathematical property tests, Zygote tests, etc.
```

This structure ensures:
- **Critical rules are NEVER violated** (always available context)
- **Complex workflows are SYSTEMATIC** (skills with checklists)
- **Information is ACCESSIBLE** (reference docs for lookup)
- **Expertise is AVAILABLE** (agents for specialized tasks)

## Development Guidance

- Remember that example folders are associated with different domains and for the examples related to the core development, you should put them into the core features sub folder
- Remember you should put the test in the relevent folder not just scatter it in the test folder, we already have a Hierarchical folders
- You should remember that every time we update the documentation and changes in the architecture we should automatically update the agents and all files in the @.claude
- **Use skills proactively**: Invoke `systematic-debugging` when encountering bugs, `domain-implementation` for new domains, `code-review` before commits, `testing-strategy` when writing tests 

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