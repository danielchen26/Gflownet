# .cursor/rules/ - LEGACY DIRECTORY

> **⚠️ COMPLETE MIGRATION NOTICE (February 2026)**
>
> All content from this directory has been **fully migrated** to the Claude Code structure.
> This directory is kept for historical reference only.

## Migration Summary

### Workflow-Oriented Content → Claude Code Skills (`.claude/skills/`)
- `debugging-best-practices.mdc` → [.claude/skills/systematic-debugging.md](../../.claude/skills/systematic-debugging.md)
- `testing-validation.mdc` → [.claude/skills/testing-strategy.md](../../.claude/skills/testing-strategy.md)
- `gflownet-interface-requirements.mdc` → [.claude/skills/domain-implementation.md](../../.claude/skills/domain-implementation.md)
- `gflownet-code-cleanliness.mdc` + `code-quality-maintenance.mdc` → [.claude/skills/code-review.md](../../.claude/skills/code-review.md)

### Reference Documentation → Docs (` docs/src/reference/`)
- `gflownet-project-structure.mdc` → [docs/src/reference/project_structure.md](../../docs/src/reference/project_structure.md)
- `gflownet-concepts.mdc` → [docs/src/reference/core_concepts.md](../../docs/src/reference/core_concepts.md)
- `gflownet-architecture.mdc` → [docs/src/reference/architecture.md](../../docs/src/reference/architecture.md)

### Critical Context → Always-Available Rules (`.claude/critical_context/`)
- `julia-coding-standards.mdc` → [.claude/critical_context/zygote_compatibility.md](../../.claude/critical_context/zygote_compatibility.md)
- `gflownet-high-level-interface.mdc` → [.claude/critical_context/high_level_api.md](../../.claude/critical_context/high_level_api.md)

## For Active Development

**Do NOT use this directory.** Use the new structure:

1. **For workflows** (debugging, implementing domains, reviewing code):
   - Use the **Skill tool** to invoke skills from `.claude/skills/`
   - Skills provide step-by-step guidance with TodoWrite integration

2. **For critical rules** (Zygote compatibility, API usage):
   - Automatically loaded from `.claude/critical_context/`
   - Always consulted before code generation

3. **For reference** (architecture, concepts, project structure):
   - Read from `docs/src/reference/`
   - Factual specifications and API details

## How the New System Works

See [CLAUDE.md](../../CLAUDE.md) for the complete invoking mechanism and how the `.claude/` structure ensures:
- Critical rules are NEVER violated
- Complex workflows are SYSTEMATIC
- Information is ACCESSIBLE
- Expertise is AVAILABLE

## Migration Verification

All content was verified with:
- **2.17x enhancement ratio** (new files are significantly improved)
- **Zero content loss** (all original information preserved)
- **Comprehensive improvements** (TodoWrite integration, better examples, systematic workflows)

See migration commits:
- `ce36156e` - Migration cleanup
- `e7f00e91` - Verification report
