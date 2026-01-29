# .claude/ Directory Reorganization Proposal

## Current Issues

The `.claude/` root directory mixes active context files with documentation, making it unclear what's essential vs. reference material.

## Proposed Structure

```
.claude/
├── development_session.md          # ✅ KEEP: Active dev state
├── agent_update_summary.md         # ✅ KEEP: Agent evolution tracking
│
├── agents/                          # ✅ KEEP: Agent definitions
├── sessions/                        # ✅ KEEP: Session logs
├── skills/                          # ✅ KEEP: Claude Code skills
├── system_design/                   # ✅ KEEP: Agent architecture
│
├── docs/                            # ✨ NEW: Documentation
│   ├── CONVERSATION_PERSISTENCE.md  # MOVE: How persistence works
│   └── session_quickstart.md        # MERGE: Quick session commands
│
└── settings.local.json              # ✅ KEEP: Local settings
```

## Proposed Moves

### 1. Create `.claude/docs/` subdirectory
```bash
mkdir -p .claude/docs
```

### 2. Move Documentation Files
```bash
# Move persistence documentation
mv .claude/CONVERSATION_PERSISTENCE.md .claude/docs/

# Merge save_session.md into persistence doc as "Quick Start" section
# Then delete save_session.md
```

### 3. Move Developer Reference
```bash
# Move to main docs (not AI-specific)
mv .claude/quick_reference.md docs/src/reference/quick_reference.md

# Update CLAUDE.md to reference it:
# "For quick reference, see docs/src/reference/quick_reference.md"
```

## Rationale

### Active Context (Stay in Root)
Files that:
- Change frequently
- Track current state
- Directly referenced by CLAUDE.md
- Need immediate visibility

**Keep**: `development_session.md`, `agent_update_summary.md`

### Documentation (Move to `.claude/docs/`)
Files that:
- Explain how systems work
- Rarely change
- Reference material
- Not actively consulted during development

**Move**: `CONVERSATION_PERSISTENCE.md`, `save_session.md`

### Developer Reference (Move to `docs/`)
Files that:
- Useful for human developers
- Not AI-specific
- Part of project documentation

**Move**: `quick_reference.md`

## Benefits

1. **Clearer Organization**: Active context vs. documentation
2. **Less Root Clutter**: Only essential files in `.claude/` root
3. **Better Discoverability**: Docs grouped together
4. **Maintainability**: Clear where to add new content

## Implementation

Would you like me to implement this reorganization?

**Option A**: Full reorganization now
**Option B**: Leave as-is (working, just not optimal)
**Option C**: Minimal cleanup (just move quick_reference.md to docs/)
