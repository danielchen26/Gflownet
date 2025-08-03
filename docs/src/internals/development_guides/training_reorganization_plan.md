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

1. ✅ Create new files in `src/training/`
   - ✅ `training.jl` - Created with main training loop
   - ✅ `losses.jl` - Created with loss computation functions
   - ✅ `utils.jl` - Created with training utilities
   - ✅ `multi_start_training.jl` - Already moved from core/
   
2. ⏳ Move functions systematically
   - ✅ Copied `train_gflownet()` to training.jl
   - ✅ Copied `train_step!()` to training.jl
   - ✅ Copied `compute_trajectory_loss()` to losses.jl
   - ✅ Copied `compute_single_trajectory_loss()` to losses.jl
   - ⏳ Need to move `objectives.jl` from core/ to training/
   - ⏳ Need to remove training functions from interface.jl
   
3. ⏳ Update imports in moved files
   - ✅ Added necessary imports to new files
   - ⏳ Need to verify all imports work correctly
   
4. ⏳ Update includes in GFlowNet.jl
   - ⏳ Add includes for new training files
   - ⏳ Update order to avoid circular dependencies
   
5. ⏳ Run tests to ensure nothing breaks
   
6. ⏳ Update documentation references
   - ⏳ Update any references to old file locations
   - ⏳ Update this plan when complete

## Current Status - COMPLETED ✅

**All Steps Completed:**

1. ✅ **Created new training files:**
   - `training/training.jl` - Main training loop (train_gflownet, train_step!)
   - `training/losses.jl` - Loss computation (compute_trajectory_loss, etc.)
   - `training/utils.jl` - Training utilities (gradient_norm, validation)
   - `training/multi_start_training.jl` - Already in correct location

2. ✅ **Moved objectives.jl** from core/ to training/

3. ✅ **Cleaned interface.jl:**
   - Removed all training functions
   - Kept only model creation and sampling
   - Interface is now purely high-level API

4. ✅ **Updated module structure:**
   - Training config loads first (needed by interface)
   - Interface loads before training (training depends on it)
   - All imports properly organized

5. ✅ **Tested and verified:**
   - Module loads successfully
   - Training runs without errors
   - All tests still pass

## Final Structure

```
src/
├── training/
│   ├── configuration.jl    # Training types (TrainingConfig, etc.)
│   ├── objectives.jl       # Objective configurations
│   ├── training.jl         # Main training loop
│   ├── losses.jl           # Loss computation
│   ├── utils.jl            # Training utilities
│   └── multi_start_training.jl  # Multi-start specific
└── core/
    ├── interface.jl        # Model creation & sampling only
    ├── balance.jl          # Mathematical loss definitions
    └── ...
```

## Benefits Achieved

1. ✅ **Clear separation**: Training in training/, core math in core/
2. ✅ **Better organization**: Easy to find training-related code
3. ✅ **Improved modularity**: Can swap training implementations
4. ✅ **Consistent structure**: Similar to other Julia ML packages