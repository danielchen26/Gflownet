# Current Context - GFlowNet.jl

*Last Updated: August 3, 2025 - Evening*
*Last Session: session_2025-08-03_complete.md*

## Current State

### Repository Organization
- ✅ Documentation reorganized into clear structure
- ✅ Master agent system designed and ready for implementation
- ✅ Root directory cleaned (only README, RELEASE_NOTES, CLAUDE)
- ✅ Conversation persistence system created

### Recent Major Changes
1. **Documentation Structure**:
   - `docs/src/internals/` organized into architecture/, implementation_notes/, development_guides/
   - All scattered markdown files moved to appropriate locations
   - Created comprehensive README for navigation

2. **Master Agent System**:
   - Master orchestrator agent created
   - System design documents in `.claude/system_design/`
   - Git integration module included
   - Ready for implementation testing

3. **Session Persistence**:
   - `.claude/sessions/` directory for conversation logs
   - CONVERSATION_PERSISTENCE.md guide created
   - Session logging system established

## Active Development

### Current Branch: core-development
- Recent commits: Documentation reorganization and master agent system
- No uncommitted changes currently

### Implemented Features
- ✅ All training objectives (TB, DB, FM, STB, DIRECT_FLOW)
- ✅ Flow computation with memoization
- ✅ Backward policy validation
- ✅ Web visualization system
- ✅ Multi-start GFlowNets

### In Progress
- 🔄 GPU acceleration
- 🔄 Continuous state spaces
- 🔄 Master agent system implementation

## Next TODOs

### Immediate:
1. Test master orchestrator agent with real tasks
2. Implement shared context system for agents
3. Create agent coordination pattern library

### Short-term:
1. GPU kernel implementation for trajectory sampling
2. Continuous domain support
3. Performance profiling tools

### Long-term:
1. Distributed training
2. Advanced molecular design domains
3. Cloud deployment templates

## Key Files to Remember

### Documentation:
- `CLAUDE.md` - Main AI instructions
- `.claude/sessions/` - Conversation history
- `docs/src/internals/` - Internal documentation

### Agent System:
- `.claude/agents/` - All agent definitions
- `.claude/system_design/` - System architecture

### Core Code:
- `src/core/` - Mathematical engine
- `src/training/` - Training implementation
- `src/utils/visualization/` - Web UI

## How to Continue

When reopening this project:
1. Read this current_context.md
2. Check latest session log in `.claude/sessions/`
3. Look for any new commits or changes
4. Continue from the Next TODOs section

## Important Context

- User prefers clean organization
- Documentation should be in internals/ subfolders
- Master agent should include git integration
- Conversation persistence is important
- No Co-Authored-By messages in commits