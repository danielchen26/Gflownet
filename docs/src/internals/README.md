# GFlowNet.jl Internals Documentation

This directory contains internal documentation about the GFlowNet.jl implementation, architecture, and development guidelines.

## Directory Structure

### 📐 [architecture/](architecture/)
Core system design and architectural documentation:
- **[architecture.md](architecture/architecture.md)** - System architecture overview
- **[design_decisions.md](architecture/design_decisions.md)** - Key design choices and rationale
- **[web_visualization_architecture.md](architecture/web_visualization_architecture.md)** - Visualization system design

### 📝 [implementation_notes/](implementation_notes/)
Detailed implementation documentation for specific features:
- **[detailed_balance_implementation.md](implementation_notes/detailed_balance_implementation.md)** - DETAILED_BALANCE objective implementation
- **[flow_matching_implementation.md](implementation_notes/flow_matching_implementation.md)** - FLOW_MATCHING objective details
- **[flow_functions_multistart.md](implementation_notes/flow_functions_multistart.md)** - Multi-start flow functions
- **[multi_start_gflownets.md](implementation_notes/multi_start_gflownets.md)** - Multiple initial states support

### 🛠️ [development_guides/](development_guides/)
Development guidelines and best practices:
- **[comprehensive_gflownet_rules.md](development_guides/comprehensive_gflownet_rules.md)** - Critical development rules (Zygote compatibility, etc.)
- **[training_reorganization_plan.md](development_guides/training_reorganization_plan.md)** - Training system refactoring guide
- **[visualization_fixes_plan.md](development_guides/visualization_fixes_plan.md)** - Visualization improvement plan

### 🚧 [known_limitations.md](known_limitations.md)
Current limitations and planned improvements

## Quick Navigation

### For Contributors
1. Start with [comprehensive_gflownet_rules.md](development_guides/comprehensive_gflownet_rules.md) for critical rules
2. Review [architecture.md](architecture/architecture.md) for system overview
3. Check [known_limitations.md](known_limitations.md) for areas needing work

### For Implementers
1. See implementation_notes/ for feature-specific guides
2. Follow patterns in existing implementations
3. Ensure Zygote compatibility (see development rules)

### For Architects
1. Review [design_decisions.md](architecture/design_decisions.md) for context
2. Understand architectural patterns before proposing changes
3. Document new design decisions

## Key Principles

1. **Zygote Compatibility**: No mutations in differentiable code
2. **Type Stability**: Maintain Julia performance best practices
3. **Modularity**: Clean separation of concerns
4. **Documentation**: Implementation notes for complex features
5. **Testing**: Comprehensive test coverage for all features

---
*Last Updated: January 2025*