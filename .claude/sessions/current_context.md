# Current Context - GFlowNet.jl

*Last Updated: January 28, 2026*
*Last Session: session_2025-08-03_complete.md (historical)*
*Latest Commits: 724ecac7 (DAG docs), 39697005 (viz plan revision)*

## Current State

### Repository Organization
- ✅ Documentation reorganized into clear structure
- ✅ Master agent system designed (template created)
- ✅ Root directory cleaned (only README, RELEASE_NOTES, CLAUDE)
- ✅ Conversation persistence system created
- ✅ Real training visualization plan revised and validated (Jan 2026)

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
- ✅ All core training objectives (TB, DB, FM, STB, DIRECT_FLOW, COMBINED)
- ✅ Flow computation with memoization (on-demand DAG)
- ✅ Backward policy validation and learning
- ✅ Web visualization system (React + Three.js frontend, mock backend)
- ✅ Multi-start GFlowNets with per-state partition functions
- ✅ Learnable partition function Z (LEARNABLE_ESTIMATION)
- ✅ Training code reorganization (src/training/ module)

### Planned (Roadmap)
- ⏳ GPU acceleration (Phase 1: Q1 2025)
- ⏳ Continuous state spaces (Phase 2: Q1-Q2 2025)
- ⏳ Real training visualization integration (plan revised Jan 2026)
- ⏳ Master agent coordination logic (template exists, logic pending)

## Next TODOs

### Immediate (January 2026):
1. ✅ Fix outdated documentation in `.claude/` and `.cursor/rules/` (in progress)
2. Implement real training visualization (domain adapter + server integration)
3. Test real GFlowNet training with visualization UI

### Short-term (Q1 2026):
1. GPU kernel implementation for trajectory sampling (Phase 1 roadmap)
2. Performance profiling suite
3. Distributed training infrastructure
4. Master agent coordination logic implementation

### Long-term (Q2-Q4 2026):
1. Continuous domain support (Phase 2)
2. Advanced molecular design domains (Phase 2)
3. PyTorch/JAX bridges (Phase 5)
4. Cloud deployment templates (Phase 5)

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