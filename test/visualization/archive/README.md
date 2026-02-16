# Archived Visualization Tests

This directory contains development/debugging test files that were created during the exploration of training configurations and epsilon exploration strategies.

## Why Archived

These files are not part of the main test suite (`runtests.jl`) and represent:
- Development exploration of different epsilon values
- Debugging investigations of mode discovery
- Temporary comparison tests between objectives

## Files

### Epsilon Exploration Variants
- `test_epsilon_aggressive.jl` - ε=0.2, no decay
- `test_epsilon_balanced.jl` - Balanced epsilon settings
- `test_epsilon_exploration.jl` - Exploration investigation
- `test_epsilon_extended.jl` - Extended training with epsilon
- `test_high_epsilon.jl` - High epsilon (0.3-0.5) comparison

### Objective Comparison Tests
- `test_db_vs_tb_exploration.jl` - Detailed Balance vs Trajectory Balance
- `test_tb_comprehensive.jl` - Comprehensive TB testing
- `test_tb_pure_peaks.jl` - TB with pure peak configurations

### Development/Debug Files
- `test_comprehensive_visualization.jl` - Old visualization test (Aug 2025)
- `manual_integration_test.jl` - Manual integration testing
- `quick_subtb_test.jl` - Quick SubTB validation

### Test Outputs
- `results/` - Training curves and output files

## Usage

These files can still be run individually for debugging:
```bash
julia --project=. test/visualization/archive/test_epsilon_aggressive.jl
```

## Date Archived
February 3, 2026
