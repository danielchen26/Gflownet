# GFlowNet Claude Code Skills

This directory contains Claude Code skills specifically designed for GFlowNet.jl development. These skills provide workflow-driven guidance that complements the reference documentation.

## Available Skills

### 1. `gflownet:systematic-debugging`
**When to use**: Encountering bugs, training failures, or Zygote errors

**What it does**:
- Evidence-first debugging methodology
- Checks previous successful runs before making changes
- Identifies root causes (Zygote mutations, type issues, numerical instability)
- Creates systematic debugging workflow with TodoWrite

**Replaces**: `.cursor/rules/debugging-best-practices.mdc`

### 2. `gflownet:domain-implementation`
**When to use**: Implementing a new GFlowNet domain or application

**What it does**:
- Step-by-step domain creation workflow
- Checklist of required interface functions
- High-level API usage patterns
- Testing and validation integration

**Replaces**: `.cursor/rules/gflownet-interface-requirements.mdc` + parts of `gflownet-high-level-interface.mdc`

### 3. `gflownet:code-review`
**When to use**: Reviewing code before commits or after implementation

**What it does**:
- Automated code quality checklist
- Zygote compatibility verification
- Comment style validation
- Example directory structure compliance

**Replaces**: `.cursor/rules/gflownet-code-cleanliness.mdc` + `code-quality-maintenance.mdc`

### 4. `gflownet:testing-strategy`
**When to use**: Writing tests for new features or domains

**What it does**:
- Property-based testing templates
- Mathematical property validation
- Performance benchmarking patterns
- Test organization guidelines

**Replaces**: `.cursor/rules/testing-validation.mdc`

## Skills vs Reference Documentation

**Skills** (`.skills/` directory):
- Workflow-oriented (procedures with steps)
- Invoked actively via Skill tool
- Create TodoWrite checklists
- Guide HOW to do something

**Reference Docs** (`docs/src/reference/`):
- Fact-oriented (specifications and concepts)
- Read passively for information
- Provide context and API details
- Explain WHAT something is

## Migration from Cursor Rules

The `.cursor/rules/` folder has been migrated as follows:

**Converted to Skills**:
- `debugging-best-practices.mdc` → `gflownet:systematic-debugging`
- `testing-validation.mdc` → `gflownet:testing-strategy`
- `gflownet-interface-requirements.mdc` → `gflownet:domain-implementation`
- `gflownet-code-cleanliness.mdc` + `code-quality-maintenance.mdc` → `gflownet:code-review`

**Moved to Reference Docs** (`docs/src/reference/`):
- `gflownet-project-structure.mdc` → `project_structure.md`
- `gflownet-concepts.mdc` → `core_concepts.md`
- `gflownet-architecture.mdc` → `architecture.md`

**Kept as Critical Context** (in `CLAUDE.md`):
- `julia-coding-standards.mdc` - Critical Zygote/AD compatibility rules
- `gflownet-high-level-interface.mdc` - Core API patterns

## Using Skills

Skills are invoked via the Skill tool:

```julia
# Example: User encounters a bug
user: "Grid world training fails with Zygote mutation error"

# Claude automatically invokes the debugging skill
assistant: [Uses Skill tool: gflownet:systematic-debugging]
# Creates checklist:
# - [ ] Check for previous successful runs
# - [ ] Compare what changed
# - [ ] Look for mutations (+=, push!, etc.)
# - [ ] Fix root cause only
```

## Installation

Skills in this directory are automatically available when using Claude Code in this repository. No additional installation required.

For global GFlowNet skills, these can be registered with the MCP skills server (future enhancement).
