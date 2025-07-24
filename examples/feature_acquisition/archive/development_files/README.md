# Development Files Archive

This directory contains development files that were moved from the main feature_acquisition directory during the consolidation process.

## Files Archived Here

### Duplicate Implementation
- **`feature_acquisition_modern.jl`** - Duplicate modern implementation (functionality merged into main.jl)

### Development Utilities
- **`cleanup.jl`** - Utility script for cleaning generated files
- **`visualization.jl`** - Advanced visualization tools and plotting functions
- **`report_generation.jl`** - Report generation utilities and templates

### Development Documentation
- **`fix_summary.md`** - Summary of fixes and improvements applied during development
- **`VERSION_DIFFERENCES.md`** - Detailed comparison between different implementation versions

### Generated Data
- **`feature_acquisition_metrics.csv`** - Training metrics from development runs

## Why These Were Archived

These files were moved to clean up the main directory and provide a single, consolidated entry point (`main.jl`) that incorporates the best features from all versions. The main directory now contains only:

- `main.jl` - Modern, consolidated implementation
- `DOCUMENTATION.md` - Comprehensive technical documentation
- `README.md` - Quick start guide
- `Project.toml` / `Manifest.toml` - Dependencies

## Accessing Archived Functionality

If you need functionality from these archived files:

1. **Visualization**: The core plotting functionality is now integrated into `main.jl`
2. **Reports**: Basic reporting is included in the main implementation
3. **Alternative Implementations**: See the versioned archives in `v1/`, `v2/`, `v3/` directories

## Development History

This archive preserves the development process that led to the final consolidated implementation, showing the evolution of:

- Training interface modernization
- Reward function improvements
- Visualization enhancements
- Documentation consolidation

---

*These files are preserved for reference but are not needed for running the current feature acquisition example.* 