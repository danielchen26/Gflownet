# Known Limitations

This document lists current limitations in GFlowNet.jl and planned improvements.

*Last updated: February 2025*

## Performance Limitations

### 1. GPU Acceleration

**Status**: Limited support

**Current State**:
- Neural networks run on GPU via Lux.jl
- Trajectory sampling is CPU-only
- No batched environment operations

**Impact**:
- Slower training for GPU-friendly domains
- CPU becomes bottleneck for simple environments

**Future Plan**:
- Implement GPU kernels for trajectory sampling
- Batch environment operations
- Full GPU pipeline

### 2. Distributed Training

**Status**: Not implemented

**Limitation**: Single machine, single GPU only.

**Impact**:
- Cannot scale to very large problems
- Limited by single machine memory
- Longer training times

**Future Plan**:
- Multi-GPU support via data parallelism
- Distributed trajectory sampling
- Model parallelism for large networks

## Domain Limitations

### 3. Continuous State Spaces

**Status**: Experimental only

**Current State**:
- Framework assumes discrete actions
- State discretization required
- No continuous normalizing flows

**Impact**:
- Cannot handle truly continuous domains
- Discretization introduces approximation
- Limited resolution

**Future Plan**:
- Continuous action support
- Continuous flow networks
- Integration with normalizing flows

### 4. Non-Markovian Trajectories

**Status**: Not supported

**Limitation**: Assumes Markov property for state transitions.

**Impact**:
- Cannot handle history-dependent rewards
- No support for recurrent policies
- Limited sequence modeling

**Future Plan**:
- RNN/Transformer policies
- History embedding
- Non-Markovian objectives

## Algorithmic Limitations

### 5. Variance Reduction

**Status**: Basic only

**Current State**:
- No baseline functions
- No importance sampling
- Limited control variates

**Impact**:
- Higher variance gradients
- Slower convergence
- Less stable training

**Future Plan**:
- Learned baselines
- Importance weighted objectives
- Advanced variance reduction

### 6. Off-Policy Learning

**Status**: On-policy only

**Limitation**: Cannot reuse old trajectories efficiently.

**Impact**:
- Sample inefficient
- Wastes experience
- Slower learning

**Future Plan**:
- Importance sampling
- Experience replay
- Off-policy corrections

## Quality of Life Limitations

### 7. Hyperparameter Tuning

**Status**: Manual only

**Limitation**: No automatic hyperparameter optimization.

**Impact**:
- Requires manual search
- Suboptimal configurations
- Time consuming

**Future Plan**:
- Optuna integration
- Automatic tuning
- Domain-specific defaults

### 8. Model Persistence

**Status**: Basic save/load

**Limitations**:
- No versioning
- No partial checkpoints
- Limited metadata

**Impact**:
- Risk of incompatibility
- Large checkpoint files
- Manual tracking

**Future Plan**:
- Versioned checkpoints
- Incremental saves
- Automatic metadata

## Compatibility Limitations

### 9. Julia Version Support

**Status**: Julia 1.9+ required

**Why**: Depends on modern Julia features.

**Impact**: Cannot use with older Julia versions.

### 10. Package Ecosystem

**Status**: Limited integrations

**Missing**:
- MLFlow tracking
- Weights & Biases
- Standard benchmarks

**Future Plan**: Add common ML tool integrations.

## Summary Table

| Feature | Status | Priority | Workaround Available |
|---------|--------|----------|---------------------|
| GPU Sampling | Partial | High | No |
| Distributed Training | Not Implemented | Low | No |
| Continuous Spaces | Experimental | Medium | Yes (discretize) |
| Variance Reduction | Basic | Medium | No |
| Off-Policy Learning | Not Implemented | Medium | No |
| AutoML/Tuning | Not Implemented | Low | Yes (manual) |

## Implemented Features (No Longer Limitations)

The following were previously limitations but are now fully implemented:

- **Flow Functions** - Complete recursive flow computation with memoization
- **TRAJECTORY_BALANCE** - Full implementation with learnable Z
- **DETAILED_BALANCE** - Joint backward policy representation
- **FLOW_MATCHING** - Complete implementation
- **SUB_TRAJECTORY_BALANCE** - O(T^2) learning signals for better credit assignment
- **Multi-Start GFlowNets** - Multiple initial states with per-state partition functions
- **Epsilon-Uniform Exploration** - Standard exploration mixing for mode discovery
- **Web Visualization** - Interactive 3D training monitor with real GFlowNet training

## Reporting Issues

If you encounter limitations not listed here:

1. Check GitHub issues for known problems
2. Create detailed bug report with:
   - Minimal reproducible example
   - Expected behavior
   - Actual behavior
   - System information
3. Consider contributing a fix!

## See Also
- [Roadmap](roadmap.md) - Future development plans
- [Quick Reference](../../reference/quick_reference.md) - API overview
