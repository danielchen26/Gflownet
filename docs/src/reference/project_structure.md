# GFlowNet.jl Project Structure

Complete guide to the file organization and module structure of GFlowNet.jl.

## Repository Layout

```
GFlowNet.jl/
├── src/                      # Source code
│   ├── GFlowNet.jl          # Main module with exports
│   ├── core/                # Core mathematical engine
│   ├── training/            # Training infrastructure
│   ├── applications/        # Domain implementations
│   └── utils/               # Utilities and helpers
├── test/                    # Test suite
├── examples/                # Usage examples and demos
├── docs/                    # Documentation
│   ├── src/
│   │   ├── guide/          # User guides
│   │   ├── reference/      # API reference
│   │   └── internals/      # Internal documentation
│   └── make.jl
├── .skills/                 # Claude Code skills
├── .claude/                 # AI agent instructions
├── Project.toml             # Package manifest
├── README.md
└── CLAUDE.md               # AI development instructions
```

## Source Code (`src/`)

### Main Module (`src/GFlowNet.jl`)
Entry point for the package. Defines the module and includes all submodules.

**Responsibilities**:
- Export public API functions
- Include all source files in correct order
- Define module-level constants

### Core Mathematical Engine (`src/core/`)

Flat directory structure (no subdirectories):

```
src/core/
├── types.jl            # Abstract types and core data structures
├── graphs.jl           # DAG operations (on-demand construction)
├── policies.jl         # P_F, P_B, Z policy functions
├── flows.jl            # Flow computation with memoization
├── balance.jl          # Balance loss functions
├── sampling.jl         # Trajectory sampling
├── interface.jl        # High-level model creation API
└── multi_start.jl      # Multi-start GFlowNets
```

**Key Files**:

- **`types.jl`**: Core types and abstract interfaces
  - `AbstractState`, `AbstractAction`
  - `Trajectory` struct
  - `DirectedAcyclicGraph{S,A}` (parametric type)

- **`graphs.jl`**: Graph operations
  - On-demand DAG construction
  - State caching
  - Action applicability checking

- **`policies.jl`**: Policy functions
  - Forward policy: P_F(a|s)
  - Backward policy: P_B(s|s')  (for DETAILED_BALANCE)
  - Partition function: Z (learnable or fixed)

- **`flows.jl`**: Flow computation
  - Recursive flow calculation
  - Memoization with Zygote compatibility
  - Cache invalidation

- **`balance.jl`**: Loss functions
  - Trajectory balance loss
  - Detailed balance loss
  - Flow conservation checks

- **`sampling.jl`**: Trajectory sampling algorithms
  - Forward sampling with policy
  - Trajectory construction
  - Action selection

- **`interface.jl`**: High-level API (⭐ **Main entry point for users**)
  - `create_gflownet()` - Generic model creation
  - `sample_trajectory()` - Sample from model
  - Model creation utilities

- **`multi_start.jl`**: Multi-start GFlowNets
  - Multiple initial states
  - Per-initial-state partition functions
  - Initial state sampling

### Training Infrastructure (`src/training/`)

Separate module for training-related functionality:

```
src/training/
├── configuration.jl           # TrainingConfig types
├── training.jl                # Main training loop
├── objectives.jl              # Training objective enumeration
├── losses.jl                  # Loss computation functions
├── utils.jl                   # Training utilities
└── multi_start_training.jl    # Multi-start training
```

**Key Files**:

- **`configuration.jl`**: Training configuration
  - `TrainingConfig` struct
  - Validation functions
  - Configuration presets

- **`training.jl`**: Training loop (⭐ **Main training entry point**)
  - `train_gflownet()` - High-level training function
  - `train_step!()` - Single training iteration
  - History tracking

- **`objectives.jl`**: Training objectives
  - `@enum TrainingObjective` with all objectives
  - Objective-specific configurations
  - Validation for objective requirements

- **`losses.jl`**: Loss computation
  - `compute_trajectory_loss()` - TB loss
  - `compute_detailed_balance_loss()` - DB loss
  - `compute_flow_matching_loss()` - FM loss
  - Sub-trajectory balance loss
  - Direct flow objective loss

- **`utils.jl`**: Training utilities
  - Logging helpers
  - Metric computation
  - Progress tracking

### Applications (`src/applications/`)

Domain-specific implementations:

```
src/applications/
├── grid_world.jl        # Grid world domain (canonical example)
└── [future domains]     # Molecular design, etc.
```

**Each domain provides**:
1. State and action type definitions
2. Required interface implementations
3. `create_[domain]_gflownet()` convenience function
4. Domain-specific reward function

**Example: Grid World**
- `GridState` struct
- `GridAction` types (MoveUp, MoveDown, etc.)
- `create_grid_world_gflownet()` function
- Grid-specific reward calculation

### Utilities (`src/utils/`)

Helper functions and tools:

```
src/utils/
├── validation/          # Mathematical property validation
│   ├── backward_policy.jl
│   └── flow_properties.jl
├── visualization/       # Web-based visualization
│   ├── web/            # React + Three.js frontend
│   │   ├── src/
│   │   ├── public/
│   │   └── package.json
│   └── api/            # Julia REST API backend
│       └── unified_server.jl     # Real GFlowNet training server
└── logging/            # Training logging utilities
```

## Test Suite (`test/`)

Hierarchical test organization:

```
test/
├── runtests.jl                 # Main test runner
├── core/                       # Core functionality tests
│   ├── test_graphs.jl
│   ├── test_policies.jl
│   ├── test_flows.jl
│   └── test_sampling.jl
├── training/                   # Training infrastructure tests
│   ├── test_configuration.jl
│   ├── test_training.jl
│   └── test_objectives.jl
├── objectives/                 # Specific objective tests
│   ├── trajectory_balance/
│   ├── detailed_balance/
│   ├── flow_matching/
│   ├── sub_trajectory_balance/
│   └── direct_flow/
├── applications/               # Domain-specific tests
│   └── test_grid_world.jl
└── utils/                      # Utility tests
    └── test_validation.jl
```

## Examples (`examples/`)

Usage examples organized by category:

```
examples/
├── core_features/             # Core feature demonstrations
│   ├── multi_start/
│   ├── objective_comparison/
│   └── visualization/
├── grid_world/               # Grid world domain
│   ├── grid_world.jl
│   ├── Project.toml
│   └── README.md
└── [domain_name]/            # Other domain examples
    ├── [domain].jl          # Single main file
    ├── Project.toml
    ├── README.md
    └── results/             # Auto-created outputs
```

**Example Structure Requirements**:
- Single main `.jl` file with clear name
- `Project.toml` with dependencies
- `README.md` with usage instructions
- Optional `results/` directory (auto-created)
- No temporary test files or clutter

## Documentation (`docs/`)

```
docs/
├── src/
│   ├── index.md              # Main documentation page
│   ├── guide/                # User guides
│   │   ├── getting_started.md
│   │   ├── basic_usage.md
│   │   └── advanced_features.md
│   ├── reference/            # Technical reference
│   │   ├── architecture.md
│   │   ├── project_structure.md  # This file
│   │   ├── core_concepts.md
│   │   └── api.md
│   └── internals/            # Internal documentation
│       ├── architecture/
│       ├── implementation_notes/
│       └── development_guides/
└── make.jl                   # Documentation build script
```

## Skills and Agent Instructions

### Skills (`.skills/`)

Claude Code workflow skills:

```
.skills/
├── README.md
├── systematic-debugging.md
├── domain-implementation.md
├── code-review.md
└── testing-strategy.md
```

These provide active workflow guidance (invoked via Skill tool).

### Agent Instructions (`.claude/`)

AI agent system files:

```
.claude/
├── agents/                   # Specialist agent definitions
│   ├── gflownet-master-orchestrator.md
│   ├── gflownet-debugger.md
│   ├── gflownet-mathematician.md
│   └── [10 more specialists]
├── sessions/                 # Conversation logs
│   └── current_context.md
├── system_design/           # Agent system architecture
│   └── AGENT_SYSTEM_DESIGN.md
└── agent_update_summary.md
```

## Key Files for Reference

### For Users

**Getting Started**:
- `README.md` - Quick start and overview
- `examples/grid_world/grid_world.jl` - Canonical example
- `docs/src/guide/getting_started.md` - Detailed guide

**API Reference**:
- [src/core/interface.jl](../../src/core/interface.jl) - Model creation API
- [src/training/training.jl](../../src/training/training.jl) - Training API
- [src/training/objectives.jl](../../src/training/objectives.jl) - Training objectives

### For Developers

**Implementation Guides**:
- [docs/src/internals/development_guides/](../internals/development_guides/) - Development workflows
- [docs/src/reference/architecture.md](architecture.md) - System architecture
- `.skills/domain-implementation.md` - New domain workflow

**Core Implementation**:
- [src/core/types.jl](../../src/core/types.jl) - Type definitions
- [src/core/interface.jl](../../src/core/interface.jl) - High-level API
- [src/training/losses.jl](../../src/training/losses.jl) - Loss functions

## Import Patterns

### For New Examples

```julia
using GFlowNet

# High-level API is automatically available
model = GFlowNet.create_gflownet(...)
history = GFlowNet.train_gflownet(model, config)
```

### For Core Development

```julia
# Explicit imports from modules
using GFlowNet: compute_trajectory_loss, sample_trajectory

# Or qualified access
loss = GFlowNet.compute_trajectory_loss(model, trajectories)
```

### For New Domains

```julia
# Implement interface functions
function GFlowNet.state_to_features(state::MyState)
    # Implementation
end

function GFlowNet.apply_action(action::MyAction, state::MyState)
    # Implementation
end

# Use high-level API
model = GFlowNet.create_gflownet(initial_state, actions; ...)
```

## Configuration Files

### Package Configuration

- **`Project.toml`**: Package manifest
  - Name, UUID, version
  - Dependencies and compatibility
  - Author information

- **`Manifest.toml`**: Exact dependency versions (auto-generated)

### Build and CI

- **`.github/workflows/`**: GitHub Actions CI/CD
- **`docs/make.jl`**: Documentation build script

### Development Tools

- **`.gitignore`**: Git ignore patterns
- **`.claudeignore`**: Files to ignore in AI context
- **`CLAUDE.md`**: AI development instructions

## Best Practices

1. **Keep examples clean**: One main file, proper Project.toml, clear README
2. **Test hierarchically**: Organize tests by module/feature
3. **Document thoroughly**: User guides + API reference + internal docs
4. **Use high-level API**: Import from GFlowNet, use `create_gflownet()`
5. **Follow structure**: Place new code in appropriate module
6. **Update exports**: Add new public functions to `src/GFlowNet.jl`
7. **Write tests**: Every feature needs tests in `test/`

## Module Dependencies

```
Core Types (types.jl)
    ↓
Graphs, Policies (graphs.jl, policies.jl)
    ↓
Flows, Balance (flows.jl, balance.jl)
    ↓
Sampling (sampling.jl)
    ↓
Interface (interface.jl) ← **User Entry Point**
    ↓
Training Configuration (training/configuration.jl)
    ↓
Training Objectives (training/objectives.jl)
    ↓
Training Losses (training/losses.jl)
    ↓
Training Loop (training/training.jl) ← **User Entry Point**
```

## Navigation

- **Architecture Overview**: [architecture.md](architecture.md)
- **Core Concepts**: [core_concepts.md](core_concepts.md)
- **Module Details**: [module_structure.md](module_structure.md)
- **Development Guides**: [../internals/development_guides/](../internals/development_guides/)

This structure ensures clean organization, clear dependencies, and easy navigation for both users and developers.
