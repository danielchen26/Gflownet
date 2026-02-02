# Current Context - GFlowNet.jl

*Last Updated: February 2, 2026*
*Latest Session: Migration completion and .claude restructuring*
*Latest Commits: 93a1cac5 (critical context migration), ce36156e (cleanup), e7f00e91 (verification)*

## Current State (February 2026)

### Recent Completion (February 2, 2026)
**Complete .cursor → .claude Migration**:
- ✅ Created `.claude/critical_context/` for always-available development rules
  - `zygote_compatibility.md` - Zygote/AD compatibility (fixed contradictions)
  - `high_level_api.md` - API usage requirements (verified all references)
- ✅ Deleted 8 migrated workflow files from `.cursor/rules/`
- ✅ Deleted 3 temporary verification documents
- ✅ Comprehensive invoking mechanism added to CLAUDE.md
- ✅ All content verified with 2.17x enhancement ratio
- ✅ Zero content loss confirmed

**Commits**:
- `93a1cac5` - Critical context structure and invoking mechanism
- `ce36156e` - Initial migration cleanup (8 files deleted)
- `e7f00e91` - Verification report update

### Repository Organization
- ✅ Documentation organized into clear structure
- ✅ Master agent system designed (template created, August 2025)
- ✅ Root directory cleaned (only README, RELEASE_NOTES, CLAUDE.md)
- ✅ Conversation persistence system established
- ✅ Real training visualization plan revised and validated (January 2026)
- ✅ `.claude/` structure finalized with critical_context/, skills/, agents/, etc.

### Major Structural Changes (Chronological)

#### February 2, 2026: Migration Finalization
1. **Critical Context Extraction**:
   - Moved `julia-coding-standards.mdc` → `.claude/critical_context/zygote_compatibility.md`
   - Moved `gflownet-high-level-interface.mdc` → `.claude/critical_context/high_level_api.md`
   - Fixed accuracy issues (removed contradictory performance advice)
   - All API references verified against codebase

2. **Invoking Mechanism Design**:
   - Added comprehensive section to CLAUDE.md explaining when each component is used
   - Clear distinction: ALWAYS (critical context) vs ACTIVE (skills) vs PASSIVE (docs)
   - Workflow examples showing complete development cycle

3. **Structure Cleanup**:
   - `.cursor/rules/` now contains only README.md (legacy notice)
   - All verification documents removed (migration complete)
   - Final structure documented and committed

#### January 28, 2026: Visualization Plan Revision
- Real training visualization plan updated (Revision 2)
- API signatures validated against actual codebase
- Implementation plan ready but NOT YET IMPLEMENTED

#### January 2026: Documentation Reorganization
- `docs/src/internals/` organized into architecture/, implementation_notes/, development_guides/
- All scattered markdown files moved to appropriate locations
- Comprehensive navigation README created

#### August 2025: Master Agent System Design
- Master orchestrator agent created
- System design documents in `.claude/system_design/`
- Git integration module included
- Ready for implementation testing (not yet implemented)

#### January 2025: Core Training Objectives Completion
- All 6 training objectives implemented (TB, DB, FM, STB, DFO, COMBINED)
- Training code reorganized into `src/training/` module
- Backward policy validation added
- Multi-start GFlowNets implemented

## Active Development

### Current Branch: core-development
- Latest commits: Migration finalization and critical context structure (Feb 2, 2026)
- Clean working directory

### Fully Implemented Features
1. **Core Training System** ✅
   - All 6 training objectives (TB, DB, FM, STB, DIRECT_FLOW, COMBINED)
   - Flow computation with memoization (on-demand DAG)
   - Backward policy validation and learning
   - Multi-start GFlowNets with per-initial-state partition functions
   - Learnable partition function Z (LEARNABLE_ESTIMATION)
   - Training code in modular `src/training/` structure

2. **Visualization System** (MOCK BACKEND ONLY)
   - ✅ **Frontend**: Production-ready React + Three.js visualization
     - Training Monitor with real-time metrics display
     - 3D Distribution View with smooth density surfaces
     - Policy Flow Field visualization
     - Training Dashboard with synchronized history
     - Problem Setup interface
   - ✅ **Mock Backend**: Simulated training for demos (`gflownet_server.jl`)
   - ❌ **Real Training Integration**: NOT implemented yet
     - Plan exists: `docs/src/internals/development_guides/real_training_visualization_plan.md`
     - Architecture designed: Domain adapters, unified server, TrainingSession manager
     - Status: Phase 1 ready to implement but not started
     - Missing: `core/`, `domains/`, `unified_server.jl`

3. **Development Infrastructure** ✅
   - `.claude/critical_context/` - Always-available development rules (2 files)
   - `.claude/skills/` - Workflow procedures (4 skills)
   - `.claude/agents/` - Specialized agent definitions (11 agents)
   - `.claude/system_design/` - Agent coordination protocols
   - Comprehensive invoking mechanism in CLAUDE.md

### In Progress / Planned

#### Near-Term (Not Started)
- ⏳ **Real Training Visualization** (Plan exists, Phase 1 ready)
  - Domain adapter interface (`core/adapters.jl`)
  - Grid World adapter implementation (`domains/grid_world.jl`)
  - Unified server with real training loop (`api/unified_server.jl`)
  - Frontend integration with v2 API endpoints

#### Mid-Term Roadmap (2026)
- ⏳ **GPU Acceleration** (Phase 1: Q1-Q2 2026)
  - CUDA.jl integration
  - GPU-accelerated training loop
  - Batch processing optimization

- ⏳ **Continuous State Spaces** (Phase 2: Q2-Q3 2026)
  - Continuous action spaces
  - Normalizing flow policies
  - Molecular generation domains

#### Long-Term Roadmap
- Advanced domain applications (molecular design, protein engineering)
- AutoML integration and hyperparameter optimization
- PyTorch/JAX bridges
- Production deployment tooling

## Technical Accuracy Notes

### Visualization Status (CRITICAL CLARITY)
**Current Reality**:
- Frontend visualization UI: **Fully implemented and production-ready** ✅
- Training backend: **Mock/simulated data ONLY** ⚠️
- Real GFlowNet integration: **Plan exists but NOT implemented** ❌

**Accurate Statements**:
- ✅ "Visualization system with production-ready frontend"
- ✅ "Mock backend for demonstration and UI development"
- ✅ "Real training integration plan validated and ready for Phase 1"

**Inaccurate Statements to AVOID**:
- ❌ "Real-time GFlowNet training visualization"
- ❌ "Live training monitoring system"
- ❌ "Production training visualization" (without "mock" qualifier)

### Critical Files Always Available
Located in `.claude/critical_context/` (always loaded before code generation):
1. **zygote_compatibility.md** - Zygote/AD rules (NEVER violate)
2. **high_level_api.md** - API usage requirements (ALWAYS follow)

### Active Workflows
Located in `.claude/skills/` (invoke via Skill tool):
1. **systematic-debugging.md** - Evidence-first debugging methodology
2. **domain-implementation.md** - New domain creation workflow
3. **code-review.md** - Pre-commit quality checklist
4. **testing-strategy.md** - Comprehensive test writing

## Next TODOs

### Immediate (February 2026):
1. ✅ Complete .cursor → .claude migration
2. ✅ Fix outdated documentation accuracy issues
3. ⏳ **Comprehensive .claude folder audit** (in progress now)
4. ⏳ Implement real training visualization Phase 1 (domain adapter + server)

### Short-term (Q1 2026):
1. Real training visualization integration (Grid World)
2. GPU kernel implementation for trajectory sampling
3. Performance profiling suite
4. Master agent coordination logic testing

### Long-term (Q2-Q4 2026):
1. Continuous domain support (Phase 2)
2. Advanced molecular design domains (Phase 2)
3. PyTorch/JAX bridges (Phase 5)
4. Cloud deployment templates (Phase 5)

## Known Limitations (February 2026)

1. **Visualization**: Mock backend only; real training not integrated
2. **GPU Support**: Not implemented; CPU-only training
3. **Continuous Domains**: Not implemented; discrete only
4. **Agent Coordination**: Master orchestrator designed but not tested in practice
5. **Real Training Viz**: Architecture designed, implementation not started

## Session Continuity

For session persistence, see:
- `.claude/docs/CONVERSATION_PERSISTENCE.md` - How to save/load sessions
- `.claude/sessions/` - Historical session logs
- Latest active session: This file (current_context.md)

## How to Continue

When reopening this project:
1. Read this current_context.md for latest state
2. Check latest session logs in `.claude/sessions/`
3. Review recent commits (git log -5)
4. Continue from Next TODOs section
5. Consult critical_context/ before any code changes

## Important Context

- User prefers clean, well-organized structure
- Documentation should be in `docs/src/internals/` subfolders
- Master agent should include git integration
- Conversation persistence is important
- No Co-Authored-By messages in commits
- Always distinguish implemented vs planned features
- Academic rigor: precise, verifiable, professional

## References

**Critical Commits (Recent)**:
- `93a1cac5` - Critical context migration (Feb 2, 2026)
- `ce36156e` - Migration cleanup (Feb 2, 2026)
- `e7f00e91` - Verification report (Feb 2, 2026)
- `39697005` - Visualization plan revision (Jan 28, 2026)
- `724ecac7` - DAG documentation (Jan 2026)

**Key Documents**:
- `CLAUDE.md` - Main development guidance with invoking mechanism
- `docs/src/reference/quick_reference.md` - Quick lookup reference
- `docs/src/internals/development_guides/roadmap.md` - Detailed roadmap
- `docs/src/internals/development_guides/real_training_visualization_plan.md` - Viz implementation plan (NOT implemented)
- `.claude/critical_context/` - Always-available development rules
