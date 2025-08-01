# GFlowNet.jl Specialized Agents

This directory contains specialized Claude agents designed to assist with different aspects of GFlowNet.jl development. Each agent has deep expertise in specific areas and can be invoked to help with targeted tasks.

## Available Agents

### 1. 🔍 gflownet-architecture-analyzer
**Expertise**: Architectural design analysis and system understanding
- Analyzes GFlowNet core architecture patterns
- Explains component interactions and design decisions
- Identifies architectural improvements
- Reviews system organization and module structure

**When to use**: Understanding how GFlowNet components work together, reviewing design choices, planning architectural changes

---

### 2. 🐛 gflownet-debugger
**Expertise**: Debugging and issue resolution
- Diagnoses training failures and convergence issues
- Traces trajectory sampling problems
- Identifies gradient flow issues
- Fixes Zygote/AD compatibility problems

**When to use**: When encountering errors, training problems, or unexpected behavior

---

### 3. 📐 gflownet-mathematician
**Expertise**: Mathematical theory and physics applications
- Validates mathematical correctness
- Analyzes flow conservation properties
- Designs physics-inspired reward functions
- Ensures numerical stability

**When to use**: Mathematical proofs, theoretical analysis, physics applications, numerical issues

---

### 4. 📝 gflownet-documentation
**Expertise**: Documentation creation and maintenance
- Writes comprehensive API documentation
- Creates tutorials and guides
- Maintains mathematical notation consistency
- Ensures documentation stays current with code

**When to use**: Creating or updating documentation, writing examples, explaining concepts

---

### 5. 🏗️ gflownet-domain-implementer
**Expertise**: Implementing new GFlowNet domains
- Designs state and action representations
- Implements required interface functions
- Creates domain-specific reward functions
- Ensures implementation correctness

**When to use**: Adding new applications/domains, implementing custom GFlowNet environments

---

### 6. ⚡ gflownet-performance-optimizer
**Expertise**: Performance optimization and scalability
- Profiles and identifies bottlenecks
- Implements GPU acceleration
- Optimizes memory usage
- Designs efficient algorithms

**When to use**: Improving training speed, reducing memory usage, scaling to larger problems

---

### 7. 🎯 gflownet-training-expert
**Expertise**: Training optimization and hyperparameter tuning
- Designs training configurations
- Diagnoses convergence issues
- Implements advanced training techniques
- Tunes hyperparameters

**When to use**: Setting up training, improving convergence, handling difficult optimization landscapes

---

### 8. ✅ gflownet-testing-validator
**Expertise**: Testing and validation strategies
- Designs comprehensive test suites
- Validates mathematical properties
- Creates property-based tests
- Ensures implementation correctness

**When to use**: Writing tests, validating implementations, ensuring code quality

## Usage Examples

### Example 1: Debugging a training issue
```
User: "My GFlowNet training loss is stuck at a high value and not decreasing"
Assistant: I'll use the gflownet-debugger agent to help diagnose this issue.
[Agent analyzes the problem and provides specific solutions]
```

### Example 2: Implementing a new domain
```
User: "I want to implement a GFlowNet for protein folding"
Assistant: I'll use the gflownet-domain-implementer agent to guide you through this.
[Agent provides complete implementation template and guidance]
```

### Example 3: Optimizing performance
```
User: "Training is too slow for my large-scale problem"
Assistant: Let me use the gflownet-performance-optimizer agent to analyze and improve performance.
[Agent profiles code and suggests optimizations]
```

## Agent Coordination

These agents can work together on complex tasks:

1. **Domain Implementation + Testing**: Domain implementer creates the code, testing validator ensures correctness
2. **Debugging + Mathematics**: Debugger identifies issues, mathematician validates theoretical properties
3. **Performance + Training**: Performance optimizer speeds up code, training expert optimizes convergence
4. **Documentation + Architecture**: Architecture analyzer explains design, documentation agent writes it up

## Best Practices

1. **Choose the right agent**: Select the agent whose expertise best matches your task
2. **Provide context**: Give agents relevant code, error messages, and background
3. **Iterate**: Agents can refine their solutions based on feedback
4. **Combine expertise**: Use multiple agents for complex multi-faceted problems

## Future Enhancements

- Cross-agent communication protocols
- Shared knowledge base between agents
- Automated agent selection based on query analysis
- Learning from user feedback to improve responses

## Contributing

To add a new specialized agent:
1. Create a new `.md` file in this directory
2. Define the agent's expertise and competencies
3. Provide examples and templates
4. Update this README with the new agent

Each agent should follow the established format and provide concrete, actionable guidance for GFlowNet.jl development.