# Known Limitations

This document lists current limitations in GFlowNet.jl and planned improvements.

## Core Functionality Limitations

### 1. Flow Computation Functions

**Status**: ✅ Implemented (August 2025)

**Implemented Functions**:
- `flow(model, state)` - Unified interface for flow computation
- `compute_recursive_flow(model, state)` - Recursive flow using F(s) = Σ P_F(s'|s) * F(s')
- `partition_function(model)` - Computes Z = F(s₀)
- `edge_flow(model, source, target)` - Edge flow F(s→s') = P_F(s'|s) * F(s)

**Implementation Details**:
- Uses on-demand computation (no explicit DAG needed)
- Includes memoization for efficiency
- Maintains Zygote compatibility
- Properly handles terminal states: F(s) = R(s)

**Current Limitations**:
- Performance may degrade for very deep state spaces without caching
- DIRECT_FLOW method requires flow estimator network (not yet implemented)

### 2. Advanced Training Objectives

**Status**: Partially implemented

**Affected Objectives**:
- `DETAILED_BALANCE` - Requires backward policy and flow functions
- `FLOW_MATCHING` - Requires flow network
- `GENERAL_TRAJECTORY_BALANCE` - Not implemented

**Why**: Dependencies on flow computation and state enumeration.

**Workaround**: Use `TRAJECTORY_BALANCE` with optional backward policy.

**Future Plan**: Implement once flow networks are available.

### 3. Multiple Initial States

**Status**: Not supported

**Limitation**: Current implementation assumes single fixed initial state.

**Impact**: 
- Cannot handle problems with multiple starting points
- Cannot learn Z(s₀) for different initial states
- Transfer learning is limited

**Workaround**: Train separate models for each initial state.

**Future Plan**: Add multi-start support with state-dependent Z.

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
| Flow Functions | ❌ Not Implemented | High | ✅ Yes (Z=1) |
| Multiple Initial States | ❌ Not Implemented | Medium | ✅ Yes (separate models) |
| GPU Sampling | ⚠️ Partial | High | ❌ No |
| Continuous Spaces | ⚠️ Experimental | Medium | ✅ Yes (discretize) |
| Advanced Objectives | ⚠️ Partial | Medium | ✅ Yes (trajectory balance) |
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