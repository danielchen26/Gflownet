# Feature Acquisition Archive

This directory contains the historical evolution of the feature acquisition example, preserving different implementation approaches and improvements over time.

## 📚 Version History

### Version 1 (v1/) - Original Implementation
**Date**: Early development  
**Status**: Legacy implementation with known limitations

**Key Characteristics:**
- Basic feature acquisition concept
- Manual training loops using `compute_loss_and_grad`
- Simple DAG creation approach
- Basic reward calculation

**Files:**
- `main_v1.jl` - Original implementation
- `run_v1.jl` - Runner script
- `v1_results/` - Generated outputs

**Known Issues:**
- DAG creation challenges with cycle detection
- Basic reward handling
- Legacy training patterns

### Version 2 (v2/) - Enhanced Implementation  
**Date**: Mid-development  
**Status**: Improved core functionality

**Key Improvements:**
- Improved DAG creation with step counter to ensure acyclicity
- Type-based reward system (`FeatureExperimentReward`)
- Better error handling and logging
- Enhanced state representation

**Files:**
- `main_v2.jl` - Enhanced implementation
- `run_v2.jl` / `run_v2_direct.jl` - Runner scripts
- `v2_results/` - Generated outputs

**Technical Advances:**
- Concrete reward type extending `RewardFunction`
- Comprehensive error handling
- Resolved cycle detection issues

### Version 3 (v3/) - Modern Implementation
**Date**: Latest development  
**Status**: Current best practices (used by main.jl)

**Key Features:**
- **Modern Training Interface**: Uses `TrainingConfig` and `train_gflownet()`
- **Advanced Objectives**: `ADAPTIVE_SUB_TB` for sparse critical decisions
- **Intelligent Z Estimation**: `ADAPTIVE_ESTIMATION` for complex spaces
- **Robust Implementation**: Comprehensive error handling

**Files:**
- `main_v3.jl` - Modernized implementation
- `run_v3.jl` / `run_v3_direct.jl` - Runner scripts  
- `v3_results/` - Generated outputs
- `v3_output.log` - Training log

**Modern Configuration:**
```julia
config = TrainingConfig(
    objective = ADAPTIVE_SUB_TB,
    partition_function_method = ADAPTIVE_ESTIMATION,
    batch_size = 32,
    n_iterations = 100,
    sub_trajectory_config = Dict(
        :difficulty_threshold => 0.05
    )
)
```

## 🎯 Evolution Summary

| Aspect | V1 | V2 | V3 |
|--------|----|----|----| 
| **Training** | Manual loops | Manual loops | Modern interface |
| **Objectives** | Basic TB | Basic TB | Adaptive Sub-TB |
| **Z Method** | Simple | Simple | Adaptive |
| **DAG Creation** | Basic | Step counter | Robust |
| **Reward System** | Basic | Type-based | Modern |
| **Error Handling** | Minimal | Enhanced | Comprehensive |

## 🔄 Running Archived Versions

Each version is self-contained with its own dependencies:

```bash
# Version 1
cd archive/v1
julia run_v1.jl

# Version 2  
cd archive/v2
julia run_v2.jl

# Version 3
cd archive/v3
julia run_v3.jl
```

## 📊 Comparing Results

Each version generates its own results directory:
- `v1_results/` - Basic outputs, may have quality issues
- `v2_results/` - Improved quality and reliability
- `v3_results/` - Modern outputs with advanced metrics

## 🧪 Research Value

These archived versions are valuable for:

1. **Understanding Evolution**: See how the implementation improved over time
2. **Methodology Comparison**: Compare training approaches and their effectiveness
3. **Debugging Reference**: Historical context for troubleshooting
4. **Research Reproducibility**: Ability to reproduce historical results
5. **Educational Value**: Learning progression from basic to advanced patterns

## 📁 Additional Archive Contents

### Legacy Visualizations
- `legacy_figs/` - Historical visualization outputs with duplicate reports

### Comparison Scripts
- `run_both_versions.jl` - Script for comparing V1 vs V2
- `run_both.jl` - General comparison runner

### Development History
See the main directory for:
- `VERSION_DIFFERENCES.md` - Detailed technical comparison
- `fix_summary.md` - Summary of specific fixes applied
- `visualization.jl` - Advanced visualization tools used across versions

## 🚀 Migration Guide

When updating from older versions:

1. **From V1 → V2**: Focus on DAG creation and reward system improvements
2. **From V2 → V3**: Migrate to modern training interface and advanced objectives  
3. **From Any → Current**: Use `main.jl` with the latest patterns

## 💡 Lessons Learned

Key insights from this evolution:

1. **DAG Complexity**: Feature acquisition creates complex state graphs requiring careful design
2. **Training Objectives**: Advanced objectives (Sub-TB) crucial for sparse reward landscapes
3. **Modern Patterns**: Configuration-based training significantly improves maintainability
4. **Error Handling**: Comprehensive error handling essential for complex domains
5. **Adaptive Methods**: Automatic method selection (Z estimation) improves robustness

This archive preserves the complete development journey of one of the most challenging GFlowNet applications. 