# Training Code Reorganization Plan

## Current Structure Analysis

### Files with Training Logic:

1. **src/training/**
   - `configuration.jl` - Training configuration types (TrainingConfig, TrainingHistory, etc.)

2. **src/core/**
   - `interface.jl` - Contains:
     - `train_gflownet()` - Main training loop
     - `train_step!()` - Single training step
     - `compute_trajectory_loss()` - Loss computation
     - `sample_trajectory()` - Trajectory sampling
   - `objectives.jl` - Contains:
     - `ObjectiveConfig` - Objective configuration
     - Various objective helper functions
   - `multi_start_training.jl` - Multi-start specific training
   - `balance.jl` - Balance loss functions (TB, DB, FM)

## Proposed Reorganization

### 1. Move to `src/training/`:

- **training.jl** (new) - Main training loop
  - `train_gflownet()` 
  - `train_step!()`
  - Training utilities
  
- **losses.jl** (new) - Loss computation
  - `compute_trajectory_loss()`
  - `compute_single_trajectory_loss()`
  - Loss aggregation functions
  
- **objectives.jl** (move from core/)
  - Keep objective configurations
  - Objective-specific helpers
  
- **multi_start_training.jl** (move from core/)
  - Multi-start training specifics
  
### 2. Keep in `src/core/`:

- **interface.jl** - High-level API only
  - `create_gflownet()`
  - `sample_trajectory()` (it's used outside training)
  - Model creation functions
  
- **balance.jl** - Mathematical definitions
  - Balance condition definitions
  - Mathematical loss functions
  - These are core mathematical concepts, not just training

### 3. Keep in current locations:

- **sampling.jl** - Trajectory sampling (used beyond training)
- **types.jl** - Core type definitions
- **policies.jl** - Policy implementations

## Benefits:

1. **Clear Separation**: Training logic in `training/`, core math in `core/`
2. **Easier Navigation**: All training-related code in one place
3. **Better Modularity**: Can swap training implementations easily
4. **Consistent with Julia Patterns**: Similar to Flux.jl organization

## Implementation Steps:

1. Create new files in `src/training/`
2. Move functions systematically
3. Update imports in moved files
4. Update includes in GFlowNet.jl
5. Run tests to ensure nothing breaks
6. Update documentation references