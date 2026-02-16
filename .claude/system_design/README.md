# Agent System Design Documentation

This directory contains the comprehensive design documentation for the GFlowNet.jl agent coordination system.

## 📚 Core Documents

### 1. [AGENT_SYSTEM_DESIGN.md](AGENT_SYSTEM_DESIGN.md)
The complete technical design document covering:
- **Vision & Goals**: What we're building and why
- **Current System Analysis**: How agents work today
- **Immediate Improvements**: What can be done now without infrastructure changes
- **Future Architecture**: The master agent system design
- **Implementation Roadmap**: Phased approach with timelines
- **Success Metrics**: How we measure improvement

### 2. [AGENT_SYSTEM_EXAMPLES.md](AGENT_SYSTEM_EXAMPLES.md)
Concrete examples and patterns showing:
- **Current Coordination Patterns**: How to coordinate agents today
- **Master Agent Example**: Complete walkthrough of continuous state spaces implementation
- **Common Query Patterns**: Patterns for performance, bugs, features, etc.
- **Agent Communication Examples**: How agents share information
- **Best Practices**: Guidelines for effective agent use

## 🏗️ System Architecture Overview

```
Current State:                     Future State:
User → Claude → Agents            User → Master Agent → Orchestrated Agents
     (manual)                            (automatic)
```

## 🚀 Quick Start

### For Users
1. Read the current coordination patterns in [AGENT_SYSTEM_EXAMPLES.md](AGENT_SYSTEM_EXAMPLES.md#current-coordination-patterns)
2. Understand which agents to use for your query type
3. Follow the patterns for best results

### For Developers
1. Review the complete design in [AGENT_SYSTEM_DESIGN.md](AGENT_SYSTEM_DESIGN.md)
2. Check the implementation roadmap for current status
3. See examples of agent communication protocols

## 📁 Related Files

### Agent Instructions
The actual agent definitions are in:
```
../ agents/
├── gflownet-master-orchestrator.md  # Orchestrates other agents
├── gflownet-architecture-analyzer.md
├── gflownet-debugger.md
├── gflownet-mathematician.md
├── gflownet-performance-optimizer.md
├── gflownet-domain-implementer.md
├── gflownet-testing-validator.md
├── gflownet-training-expert.md
└── gflownet-documentation.md
```

## 🎯 Key Concepts

### Agent Types
1. **Master Orchestrator**: Analyzes queries and coordinates other agents
2. **Specialist Agents**: Domain experts for specific tasks

### Coordination Patterns
1. **Sequential**: Tasks with dependencies (design → implement → test)
2. **Parallel**: Independent analysis (debug + math + performance)
3. **Hub & Spoke**: One coordinator with multiple inputs

### Communication Types
- **QUERY**: Agent asking another agent
- **ALERT**: Critical findings broadcast
- **HANDOFF**: Task transfer between agents
- **RESPONSE**: Reply to query

## 📈 Current Status

### ✅ Completed
- Master orchestrator agent created
- System design documented
- Examples and patterns defined

### 🚧 In Progress
- Enhanced agent instructions
- Shared context system
- Coordination patterns library

### 📋 Planned
- Query analyzer implementation
- Parallel execution engine
- Learning system
- Production deployment

## 🔄 Evolution Path

1. **Phase 1**: Manual coordination with patterns (current)
2. **Phase 2**: Enhanced instructions and shared context
3. **Phase 3**: Basic automation with master agent
4. **Phase 4**: Parallel execution and intelligence
5. **Phase 5**: Self-improving system

## 📝 Contributing

To improve the agent system:
1. Test current patterns and provide feedback
2. Suggest new coordination patterns
3. Help implement shared context system
4. Contribute to agent instruction improvements

## 🤔 Common Questions

**Q: Can I use the master agent today?**
A: Yes! The master orchestrator agent is available but requires manual pattern selection.

**Q: How do agents share information?**
A: Currently through Claude. Future: shared context files and message passing.

**Q: Which patterns work best?**
A: See success rates in [AGENT_SYSTEM_EXAMPLES.md](AGENT_SYSTEM_EXAMPLES.md#pattern-evolution)

---

*Last Updated: January 2025*