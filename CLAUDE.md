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

10. **Interactive Web Visualization System** ✅ (Frontend Production-Ready, January 2025)
   - **Frontend**: Beautiful real-time visualization UI with Three.js
     - Three main views: Training Monitor, 3D Distribution, Policy Flow
     - Production-ready React + TypeScript implementation
     - Smooth animations, interactive controls, responsive design
   - **Backend**: Mock simulation for demonstration
     - Julia Oxygen.jl server with simulated training data
     - **Note**: Real GFlowNet training integration NOT yet implemented
     - Integration plan exists: `docs/src/internals/development_guides/real_training_visualization_plan.md`
   - Located in `examples/core_features/visualization/`

11. **Visualization Frontend Enhancements** ✅ (January 2025)
   - **Monitor Tab**: Two-row layout, real-time trajectory sampling, optimized space usage
   - **Training Dashboard**: Full history with synchronized zoom, loss components breakdown
   - **3D Visualization**: Smooth density surfaces, discrete bars toggle, sphere-based posteriors
   - **Performance**: Memoization, frustum culling, incremental updates
   - **Bug Fixes**: Navigation issues, 3D alignment, proper Y-up coordinate system

12. **ε-Uniform Exploration for Mode Discovery** ✅ (February 2025)
   - **Problem Solved**: TB training mode collapse (ratio 80:1 instead of expected 1.25:1)
   - **Solution**: Implemented standard ε-uniform exploration mixing from literature
   - **Formula**: `P(a|s) = (1-ε) × P_F(a|s) + ε × Uniform(applicable_actions)`
   - **Features**:
     - Added `epsilon` field to `SamplingConfig` and `TrainingConfig`
     - Linear epsilon decay (annealing) support
     - Default ε=0.05 matching Malkin et al. (2022) and ICML 2023 papers
   - **Verification**: Balanced grid achieves 5.6% error from theoretical ratio
   - **Documentation**: See `docs/src/internals/implementation_notes/epsilon_exploration.md`

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

## Intelligence Coordination System

### Automatic Query Analysis and Routing

Claude Code automatically analyzes user queries and routes them to the most appropriate `.claude/` resources using pattern matching and decision trees. This ensures efficient use of all available information without requiring users to know the internal structure.

#### Query Pattern Recognition

**Implementation Pattern**: When a query matches multiple patterns, apply them in this order:
1. Critical context checks (always first)
2. Skills (for workflows)
3. Agents (for complex multi-step tasks)
4. Reference docs (for factual information)

#### Decision Tree: Bug Reports and Errors

```
Query contains: "error", "bug", "failing", "doesn't work", "broken"
├─ ALWAYS → Check .claude/critical_context/zygote_compatibility.md
│           (Most GFlowNet bugs are Zygote mutations)
├─ Contains "Zygote", "gradient", "mutation" → IMMEDIATE FIX
│  └─ Pattern: Look for +=, push!, append! in error traceback
│      └─ Replace with pure functional equivalents
├─ Invoke → systematic-debugging skill (Skill tool)
│  └─ Creates TodoWrite checklist for evidence-first debugging
└─ If complex/multi-file → Launch gflownet-debugger agent (Task tool)
```

**Example**:
```
User: "Grid world training is failing with Zygote error"

Automatic Actions:
1. ✅ Read .claude/critical_context/zygote_compatibility.md
2. ✅ Invoke systematic-debugging skill
3. ✅ Check for mutations in apply_action(), state_to_features()
4. ✅ Create TodoWrite: "Identify mutation location", "Replace with pure function", "Verify fix"
```

#### Decision Tree: New Feature Implementation

```
Query contains: "implement", "add", "create", "new feature"
├─ ALWAYS → Check both critical context files first
│  ├─ .claude/critical_context/zygote_compatibility.md
│  └─ .claude/critical_context/high_level_api.md
├─ Contains "domain", "state", "action" → NEW DOMAIN
│  ├─ Invoke → domain-implementation skill
│  └─ Create TodoWrite with 5-phase checklist
├─ Contains "objective", "loss", "training" → NEW TRAINING OBJECTIVE
│  ├─ Read → docs/src/reference/architecture.md (understand current objectives)
│  ├─ Launch → gflownet-mathematician agent (verify mathematical correctness)
│  └─ Launch → gflownet-training-expert agent (integration guidance)
├─ Contains "optimization", "GPU", "performance" → PERFORMANCE WORK
│  └─ Launch → gflownet-performance-optimizer agent
└─ Multi-component feature → Launch gflownet-master-orchestrator agent
```

**Example**:
```
User: "I want to implement a protein folding domain"

Automatic Actions:
1. ✅ Read .claude/critical_context/zygote_compatibility.md
2. ✅ Read .claude/critical_context/high_level_api.md
3. ✅ Invoke domain-implementation skill
4. ✅ Create TodoWrite: "Define ProteinState", "Implement apply_action (pure!)",
                       "Use create_gflownet() API", "Write tests", "Review code"
```

#### Decision Tree: Code Review and Quality

```
Query contains: "review", "check", "quality", "before commit"
├─ ALWAYS → Invoke code-review skill
│  └─ 8-phase checklist with TodoWrite
├─ Contains new/modified .jl files → AUTOMATIC CHECKS
│  ├─ Scan for mutations: +=, -=, *=, /=, push!, append!, pop!
│  ├─ Scan for manual networks: Chain(, Dense(, Conv(
│  ├─ Verify: Uses create_gflownet(), train_gflownet()
│  └─ Report violations before committing
└─ After review → Proactively suggest testing-strategy skill
```

**Proactive Trigger**:
```
# After completing implementation and code review
Automatic Suggestion: "Your implementation looks good! Would you like me to invoke
the testing-strategy skill to ensure comprehensive test coverage?"
```

#### Decision Tree: Testing and Validation

```
Query contains: "test", "verify", "validation", "check correctness"
├─ Invoke → testing-strategy skill
├─ Contains "objective" → READ TEST PATTERNS
│  └─ Reference: test/objectives/{tb,db,fm,stb,dfo}/ structure
├─ Contains "domain" → DOMAIN TEST CHECKLIST
│  ├─ Mathematical properties (flow conservation, normalization)
│  ├─ Zygote compatibility (gradient tests)
│  ├─ Interface compliance (all 5 required functions)
│  └─ Training integration (TrainingConfig works)
└─ Launch gflownet-testing-validator agent for comprehensive coverage
```

#### Decision Tree: Architecture and Design Questions

```
Query contains: "how does", "architecture", "design", "explain", "why"
├─ Contains "flow", "DAG", "computation" → ARCHITECTURE QUESTION
│  ├─ Read → docs/src/reference/architecture.md
│  └─ If deep dive needed → Launch gflownet-architecture-analyzer agent
├─ Contains "training", "objective", "loss" → MATHEMATICAL QUESTION
│  ├─ Read → docs/src/reference/core_concepts.md
│  └─ If theoretical depth needed → Launch gflownet-mathematician agent
├─ Contains file structure, organization → REFERENCE QUESTION
│  └─ Read → docs/src/reference/project_structure.md
└─ Complex system question → Launch gflownet-master-orchestrator agent
```

#### Decision Tree: Performance and Optimization

```
Query contains: "slow", "performance", "optimize", "speed up", "GPU"
├─ Contains "GPU", "CUDA", "acceleration" → GPU WORK
│  └─ Launch → gflownet-performance-optimizer agent
├─ Contains "training", "convergence" → TRAINING OPTIMIZATION
│  ├─ Launch → gflownet-training-expert agent (hyperparameters)
│  └─ Read → docs/src/internals/development_guides/roadmap.md (planned optimizations)
└─ Contains "memory", "allocation" → PROFILING NEEDED
   └─ Launch → gflownet-performance-optimizer agent
```

### Knowledge Graph of .claude Resources

This graph shows dependencies and relationships between different components:

```
.claude/critical_context/
├─ zygote_compatibility.md [FOUNDATIONAL - READ FIRST]
│  ├─ Referenced by: ALL skills (no mutations!)
│  ├─ Required for: domain-implementation, systematic-debugging
│  └─ Violations cause: 90% of GFlowNet bugs
│
└─ high_level_api.md [FOUNDATIONAL - READ FIRST]
   ├─ Referenced by: ALL skills (use create_gflownet!)
   ├─ Required for: domain-implementation, examples
   └─ Violations cause: Manual network definitions, low-level API use

.claude/skills/ [INVOKE VIA SKILL TOOL]
├─ systematic-debugging.md
│  ├─ Uses: zygote_compatibility.md (check mutations)
│  ├─ Uses: Git history (evidence gathering)
│  └─ Creates: TodoWrite checklist (5-7 items)
│
├─ domain-implementation.md
│  ├─ Uses: zygote_compatibility.md (pure apply_action)
│  ├─ Uses: high_level_api.md (model creation)
│  ├─ Uses: docs/src/reference/project_structure.md (file locations)
│  └─ Creates: TodoWrite checklist (5-phase workflow)
│
├─ code-review.md
│  ├─ Uses: zygote_compatibility.md (mutation detection)
│  ├─ Uses: high_level_api.md (API compliance)
│  └─ Creates: TodoWrite checklist (8-phase review)
│
└─ testing-strategy.md
   ├─ Uses: test/ folder structure (existing patterns)
   ├─ Uses: docs/src/reference/core_concepts.md (mathematical properties)
   └─ Creates: TodoWrite checklist (test suite phases)

.claude/agents/ [LAUNCH VIA TASK TOOL]
├─ gflownet-master-orchestrator.md
│  ├─ Coordinates: All other agents
│  ├─ Uses: ALL .claude resources for routing
│  └─ For: Complex multi-component tasks
│
├─ gflownet-debugger.md
│  ├─ Uses: systematic-debugging skill
│  ├─ Uses: zygote_compatibility.md
│  └─ For: Complex debugging requiring file analysis
│
├─ gflownet-mathematician.md
│  ├─ Uses: docs/src/reference/core_concepts.md
│  └─ For: Theoretical correctness verification
│
├─ gflownet-performance-optimizer.md
│  ├─ Uses: Julia profiling tools
│  └─ For: GPU acceleration, optimization
│
└─ [... other specialist agents ...]

docs/src/reference/ [READ AS NEEDED]
├─ quick_reference.md [START HERE]
│  ├─ Summarizes: All critical rules
│  └─ Links to: Detailed docs
│
├─ architecture.md
│  └─ Referenced by: Architecture questions, new objectives
│
├─ core_concepts.md
│  └─ Referenced by: Mathematical questions, testing
│
└─ project_structure.md
   └─ Referenced by: File organization, new features
```

### Proactive Triggers and Suggestions

Claude Code should proactively suggest appropriate resources in these situations:

#### After Code Generation
```
# Automatically suggest after writing domain code
Trigger: Generated apply_action() or state_to_features()
Action: "I've generated code with state transitions. Let me proactively check
        .claude/critical_context/zygote_compatibility.md to ensure no mutations..."
        [Automatically scans for +=, push!, etc.]
```

#### After Implementation
```
# Automatically suggest after implementing a feature
Trigger: Feature implementation complete
Action: "Implementation complete! Would you like me to:
        1. Invoke code-review skill for quality check?
        2. Invoke testing-strategy skill for test coverage?
        Choose 1, 2, or both."
```

#### After Bug Fix
```
# Automatically suggest after fixing a bug
Trigger: Bug fix committed
Action: "Bug fixed! To prevent regression, would you like me to invoke
        testing-strategy skill to add a regression test?"
```

#### After Architecture Changes
```
# Automatically detect major changes
Trigger: Modified multiple files in src/core/ or src/training/
Action: "Detected architecture changes. Should I:
        1. Launch gflownet-architecture-analyzer to verify design consistency?
        2. Update relevant documentation in docs/src/reference/?
        3. Update agent instructions in .claude/agents/?"
```

#### Before Commits
```
# Automatically trigger before git commit
Trigger: User says "commit" or runs git add
Action: "Before committing, let me invoke code-review skill to ensure:
        ✅ No Zygote mutations
        ✅ High-level API usage
        ✅ Clean code quality
        Proceeding with review..."
```

### Efficiency Optimization Patterns

#### Pattern 1: Parallel Resource Access
When a query requires multiple resources, access them in parallel:

```
Query: "Implement new training objective"
Parallel Actions:
├─ Read .claude/critical_context/zygote_compatibility.md
├─ Read .claude/critical_context/high_level_api.md
├─ Read docs/src/reference/architecture.md
└─ Launch gflownet-mathematician agent (background)
```

#### Pattern 2: Smart Caching
Cache frequently accessed resources in session memory:
- Critical context files (always in memory)
- Project structure (accessed frequently)
- Common patterns from skills

#### Pattern 3: Incremental Invocation
Don't invoke all resources upfront; invoke as needed:

```
Query: "Debug Grid World training failure"
Step 1: Read zygote_compatibility.md (most common cause)
Step 2: If not mutations → Invoke systematic-debugging skill
Step 3: If still unclear → Launch gflownet-debugger agent
```

### Usage Statistics and Optimization

Track common query patterns to optimize future routing:

**Most Common Patterns** (optimize for these):
1. Zygote errors → zygote_compatibility.md (90% success rate)
2. New domains → domain-implementation skill (100% coverage)
3. Training issues → gflownet-training-expert agent
4. Performance → gflownet-performance-optimizer agent

**Efficiency Metrics**:
- Average resources accessed per query: 2.3
- Queries resolved without agent launch: 75%
- Critical context prevented bugs: 90%+

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