# Known Limitations

This document lists current limitations in GFlowNet.jl and planned improvements.

## Core Functionality Limitations

### 1. Flow Computation Functions

**Status**: ✅ Fully Implemented (January 2025)

**Implemented Functions**:
- `flow(model, state)` - Unified interface for flow computation
- `compute_recursive_flow(model, state)` - Recursive flow using F(s) = Σ P_F(s'|s) * F(s')
- `compute_recursive_flow_memoized(model, state)` - With Zygote-compatible caching
- `partition_function(model)` - Computes Z = F(s₀)
- `edge_flow(model, source, target)` - Edge flow F(s→s') = P_F(s'|s) * F(s)
- `flow_analysis(model, state)` - Comprehensive flow debugging

**Implementation Details**:
- Uses on-demand computation (no explicit DAG needed)
- Includes memoization with proper Zygote handling
- Cache operations wrapped in `Zygote.@ignore`
- Properly handles terminal states: F(s) = R(s)

**Current Limitations**:
- DIRECT_FLOW method requires flow estimator network training (not yet implemented)
- Performance may degrade for very deep state spaces

### 2. Advanced Training Objectives

**Status**: Partially implemented

**Implemented**:
- `TRAJECTORY_BALANCE` ✅ - Complete with optional backward policy and learnable Z
- `DETAILED_BALANCE` ✅ - Fully implemented with joint backward policy representation

**Not Yet Implemented**:
- `FLOW_MATCHING` - All prerequisites now available, ready to implement
- `SUB_TRAJECTORY_BALANCE` - Can be implemented with current infrastructure
- `COMBINED_OBJECTIVES` - Requires design decisions

**Workaround**: Use TRAJECTORY_BALANCE or DETAILED_BALANCE for now.

**Next Steps**: FLOW_MATCHING is the logical next implementation.

### 3. Multiple Initial States

**Status**: Foundation ready, implementation pending

**Current State**: 
- Single initial state with learnable Z implemented ✅
- LEARNABLE_ESTIMATION learns partition function as trainable parameter ✅
- Flow computation infrastructure ready ✅

**Still Limited**: 
- Cannot handle multiple different starting points
- Cannot learn separate Z(s₀) for each initial state
- No initial state distribution learning P(s₀)

**Why Ready Now**: 
- Flow functions implemented: F(s₀) = Z(s₀)
- Can compute per-state partition functions
- See [flow_functions_multistart.md](flow_functions_multistart.md) for details

**Workaround**: Train separate models for each initial state.

**Future Plan**: Implement multi-start with learned P(s₀) ∝ Z(s₀).

## Performance Limitations

### 4. GPU Acceleration

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

### 5. Distributed Training

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

### 6. Continuous State Spaces

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

### 7. Non-Markovian Trajectories

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

### 8. Variance Reduction

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

### 9. Exploration Strategies

**Status**: Limited options

**Current State**:
- ε-greedy via temperature
- No curiosity-driven exploration
- No diversity bonuses

**Impact**:
- May miss rare high-reward states
- Slower discovery of modes
- Limited diversity

**Future Plan**:
- Intrinsic motivation
- Diversity regularization
- Adaptive exploration

## Quality of Life Limitations

### 10. Debugging Tools

**Status**: Basic logging only

**Missing Features**:
- Trajectory visualization
- Gradient flow analysis
- Convergence diagnostics
- Profiling tools

**Impact**:
- Harder to debug issues
- Less insight into training
- Manual analysis required

**Future Plan**:
- TensorBoard integration
- Built-in visualizations
- Automatic diagnostics

### 11. Hyperparameter Tuning

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

### 12. Model Persistence

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

## Theoretical Limitations

### 13. Convergence Guarantees

**Status**: Limited theory

**Current State**:
- Empirical convergence only
- No formal guarantees
- Limited understanding

**Impact**:
- Uncertain when to stop
- No quality bounds
- Trial and error

**Future Research**:
- Convergence proofs
- Sample complexity
- Quality guarantees

### 14. Off-Policy Learning

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

## Compatibility Limitations

### 15. Julia Version Support

**Status**: Julia 1.9+ required

**Why**: Depends on modern Julia features.

**Impact**: Cannot use with older Julia versions.

### 16. Package Ecosystem

**Status**: Limited integrations

**Missing**:
- MLFlow tracking
- Weights & Biases
- Standard benchmarks

**Future Plan**: Add common ML tool integrations.

## Summary Table

| Feature | Status | Priority | Workaround Available |
|---------|---------|----------|---------------------|
| Flow Functions | ✅ Implemented | - | - |
| TRAJECTORY_BALANCE | ✅ Implemented | - | - |
| DETAILED_BALANCE | ✅ Implemented | - | - |
| FLOW_MATCHING | ❌ Not Implemented | High | ✅ Yes (use TB/DB) |
| Multiple Initial States | ⚠️ Ready to implement | High | ✅ Yes (separate models) |
| GPU Sampling | ⚠️ Partial | High | ❌ No |
| Continuous Spaces | ⚠️ Experimental | Medium | ✅ Yes (discretize) |
| SUB_TRAJECTORY_BALANCE | ❌ Not Implemented | Medium | ✅ Yes (use TB/DB) |
| Distributed Training | ❌ Not Implemented | Low | ❌ No |
| Debugging Tools | ⚠️ Basic | Medium | ✅ Yes (manual) |

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
- [Design Decisions](design_decisions.md) - Why things are this way
- [Contributing Guide](../contributing.md) - How to help
- [Roadmap](../roadmap.md) - Future plans