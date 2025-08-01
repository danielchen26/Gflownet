# Design Decisions

This document explains key architectural and implementation decisions in GFlowNet.jl.

## Core Architecture Decisions

### 1. On-Demand Computation vs Explicit DAG

**Decision**: Use on-demand state space exploration instead of pre-building DAGs.

**Rationale**:
- **Memory Efficiency**: Avoids storing potentially exponential state spaces
- **Flexibility**: Works with infinite or very large state spaces
- **Simplicity**: No need for complex DAG construction logic
- **Performance**: Often faster for sparse exploration

**Trade-offs**:
- Cannot pre-compute all state properties
- Some algorithms requiring full enumeration not available
- Must recompute some information

### 2. ComponentArrays + Lux.jl

**Decision**: Use Lux.jl with ComponentArrays for neural networks.

**Rationale**:
- **Explicit State**: Lux separates parameters from model state
- **AD Performance**: Better compatibility with Zygote
- **Functional**: Immutable patterns work well with AD
- **Composability**: Easy to combine multiple networks

**Implementation**:
```julia
# Parameters stored as ComponentArray
parameters = ComponentArray(
    forward = forward_params,
    backward = backward_params,
    flow = flow_params
)
```

### 3. Partition Function Z = 1

**Decision**: Assume Z = 1 for trajectory balance training.

**Rationale**:
- **Mathematical Validity**: Correct for fixed initial state
- **Simplicity**: No need for complex Z estimation
- **Stability**: Avoids numerical issues with Z learning
- **Sufficient**: Works well for single-start problems

**Future**: Can add Z learning when needed for multi-start problems.

## Interface Design

### 4. Five Required Functions

**Decision**: Domains must implement exactly 5 functions.

**Functions**:
1. `state_to_features` - Neural network input
2. `is_terminal_state` - Termination check
3. `reward` - Terminal rewards
4. `is_applicable` - Action validity
5. `apply_action` - State transitions

**Rationale**:
- **Minimal**: Smallest set that works
- **Clear**: Unambiguous requirements
- **Flexible**: Supports diverse domains
- **Testable**: Easy to verify compliance

### 5. Immutable States

**Decision**: States must be immutable with pure transitions.

**Rationale**:
- **AD Compatibility**: Mutations break Zygote
- **Correctness**: Prevents subtle bugs
- **Parallelism**: Safe for concurrent access
- **Debugging**: Easier to reason about

**Pattern**:
```julia
# Always create new states
new_state = MyState(
    updated_data,
    is_terminal
)
```

## Training System Design

### 6. Configuration-Based Training

**Decision**: Use TrainingConfig for all parameters.

**Rationale**:
- **Clarity**: All settings in one place
- **Validation**: Can check consistency
- **Extensibility**: Easy to add options
- **Reproducibility**: Config fully specifies training

### 7. Single Training Function

**Decision**: One `train_gflownet()` function for all cases.

**Rationale**:
- **Simplicity**: One interface to learn
- **Consistency**: Same behavior everywhere
- **Flexibility**: Config handles variations
- **Maintenance**: Single point of updates

## Performance Decisions

### 8. Float32 for Features

**Decision**: Use Float32 for all neural network operations.

**Rationale**:
- **GPU Performance**: 2x faster than Float64
- **Memory**: Half the memory usage
- **Sufficient Precision**: Good enough for NNs
- **Compatibility**: Standard in deep learning

### 9. Batch Operations

**Decision**: Process trajectories in batches.

**Rationale**:
- **GPU Utilization**: Better parallelism
- **Stability**: Averaged gradients
- **Efficiency**: Amortized overhead
- **Standard**: Expected pattern

## Error Handling

### 10. Validation with @ignore

**Decision**: Wrap validation in `Zygote.@ignore`.

**Rationale**:
- **AD Safety**: Validation isn't differentiable
- **Performance**: Skip validation in gradients
- **Debugging**: Keep helpful error messages
- **Correctness**: Catch issues early

**Pattern**:
```julia
Zygote.@ignore begin
    @assert all(features .>= 0) "Features must be non-negative"
end
```

## Extensibility Decisions

### 11. Optional Backward Policy

**Decision**: Make backward policy optional with flag.

**Rationale**:
- **Backward Compatibility**: Existing code works
- **Performance**: Skip when not needed
- **Flexibility**: User choice
- **Gradual Adoption**: Can upgrade later

### 12. Domain-Specific Creators

**Decision**: Each domain has `create_*_gflownet()`.

**Rationale**:
- **Convenience**: One function does everything
- **Defaults**: Domain-appropriate settings
- **Discovery**: Easy to find entry points
- **Examples**: Shows best practices

## Future-Proofing

### 13. Disabled But Not Removed

**Decision**: Keep unimplemented functions as commented exports.

**Rationale**:
- **Roadmap**: Shows planned features
- **API Stability**: Reserve names
- **Documentation**: Explains limitations
- **Migration Path**: Clear upgrade route

### 14. Hierarchical Organization

**Decision**: Organize code by abstraction level.

**Structure**:
```
core/     - Mathematical foundations
training/ - Training infrastructure  
applications/ - Domain implementations
utils/    - Supporting utilities
```

**Rationale**:
- **Clarity**: Easy to navigate
- **Dependency**: Clear layering
- **Modularity**: Independent components
- **Testing**: Test each layer

## Alternative Approaches Considered

### Explicit DAG Construction
**Rejected Because**:
- Memory intensive for large spaces
- Doesn't work with infinite spaces
- Complex implementation
- Often unnecessary

### Pure Flux.jl
**Rejected Because**:
- Implicit parameters harder with Zygote
- Less functional style
- Mutation issues
- Being phased out

### Learned Partition Function
**Rejected Because**:
- Added complexity
- Numerical instability
- Not needed for fixed start
- Can add later if needed

### Single Mega-Function
**Rejected Because**:
- Too much coupling
- Hard to test
- Difficult to extend
- Poor separation of concerns

## Lessons Learned

1. **Start Simple**: Z=1 works fine for most cases
2. **Pure Functions**: Mutations cause subtle bugs
3. **Clear Interfaces**: 5 functions is perfect
4. **Config Objects**: Better than many parameters
5. **Optional Complexity**: Backward policy when needed

## Future Considerations

### Near Term
- Add Z learning for multi-start
- Implement flow networks
- GPU trajectory sampling
- Distributed training

### Long Term
- Continuous state spaces
- Non-Markovian extensions
- Meta-learning support
- AutoML for architectures

## See Also
- [Architecture Analysis](architecture.md) - Current state
- [Known Limitations](known_limitations.md) - What's missing
- [Contributing](../contributing.md) - How to help