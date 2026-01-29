# GFlowNet Cursor Rules Summary

> **⚠️ MIGRATION NOTICE (January 2026)**: This directory is now **LEGACY**.
>
> **Workflow-oriented content has been migrated to Claude Code skills** (`.skills/` directory):
> - `debugging-best-practices.mdc` → `.skills/systematic-debugging.md`
> - `testing-validation.mdc` → `.skills/testing-strategy.md`
> - `gflownet-interface-requirements.mdc` → `.skills/domain-implementation.md`
> - `gflownet-code-cleanliness.mdc` + `code-quality-maintenance.mdc` → `.skills/code-review.md`
>
> **Reference documentation has been moved to `docs/src/reference/`**:
> - `gflownet-project-structure.mdc` → `docs/src/reference/project_structure.md`
> - `gflownet-concepts.mdc` → `docs/src/reference/core_concepts.md`
> - `gflownet-architecture.mdc` → `docs/src/reference/architecture.md`
>
> **Critical context remains here for backwards compatibility**:
> - `julia-coding-standards.mdc` - Zygote/AD compatibility rules (always loaded)
> - `gflownet-high-level-interface.mdc` - Core API patterns (always loaded)
>
> **For active development, use the new skills via the Skill tool** - they provide workflow-driven guidance with TodoWrite integration.

---

This directory contains Cursor rules that guide AI assistance for GFlowNet development. These rules were consolidated and updated based on real debugging experiences and best practices.

## 🚨 Critical Rules (Always Applied)

### `julia-coding-standards.mdc`
**Most Important**: Contains the **Zygote/AD compatibility** guidelines that prevent the most common and difficult debugging issues.

**Key Topics**:
- **🚨 CRITICAL**: Automatic Differentiation compatibility (NO mutations!)
- Type system best practices
- Numerical stability patterns
- Neural network integration
- Error handling standards

**When to Use**: All Julia files (`.jl`)

## 📋 Core Development Rules

### `gflownet-high-level-interface.mdc`
Enforces using GFlowNet's built-in functions instead of manual implementations.

**Key Topics**:
- ❌ Never manually define neural networks with `Chain()`/`Dense()`
- ✅ Always use `create_forward_policy()`, `create_flow_estimator()`
- Complete working example patterns
- Required high-level function signatures

### `debugging-best-practices.mdc`
Evidence-based debugging methodology learned from the grid world debugging session.

**Key Topics**:
- 🎯 "If it worked before, what changed?"
- Evidence-first vs symptom-based debugging
- How to check previous successful runs
- Root cause analysis patterns
- Real success story from grid world fix

### `gflownet-concepts.mdc`
Core mathematical concepts and implementation guidelines.

**Key Topics**:
- GFlowNet mathematical foundations
- Trajectory balance objective
- Partition function concepts
- Flow computation principles

## 🏗️ Architecture & Organization Rules

### `gflownet-project-structure.mdc`
Overview of the GFlowNet codebase architecture.

**Key Topics**:
- Core module structure (`src/core/`)
- Training infrastructure (`src/training/`)
- Applications (`src/applications/`)
- Interface requirements

### `gflownet-interface-requirements.mdc`
Essential interface functions that new GFlowNet domains must implement.

**Key Topics**:
- Required interface methods
- `state_to_features()`, `is_applicable()`, `apply_action()`, `reward()`
- Concrete implementation examples

### `code-quality-maintenance.mdc`
Professional code standards and maintenance practices.

**Key Topics**:
- Clean comment style (no development tags)
- Function design principles
- Error message standards
- Performance considerations

## 🧪 Testing & Validation

### `testing-validation.mdc`
Testing practices and validation requirements.

**Key Topics**:
- Built-in validation requirements
- Performance monitoring
- Numerical stability checks
- Test organization patterns

### `gflownet-code-cleanliness.mdc`
Standards for clean example directories and code organization.

**Key Topics**:
- Example directory structure
- File naming conventions
- Separation of concerns
- Removal of temporary/development files

## 📊 Most Important Lessons Learned

1. **🚨 Zygote Mutations**: The #1 cause of difficult debugging issues
   - Replace `x += 1` with `x = condition ? new_value : old_value`
   - Rule: `julia-coding-standards.mdc`

2. **🔍 Evidence-Based Debugging**: Check what worked before making changes
   - Look for previous successful runs first
   - Compare what changed from working version
   - Rule: `debugging-best-practices.mdc`

3. **⚡ High-Level Interface**: Use built-in functions, not manual implementations
   - Never manually define neural networks
   - Use `create_forward_policy()`, `train_gflownet()`, etc.
   - Rule: `gflownet-high-level-interface.mdc`

## 📝 How These Rules Help

- **Prevent common issues** before they happen
- **Provide debugging guidance** when things go wrong  
- **Enforce best practices** across all development
- **Capture institutional knowledge** from real debugging sessions
- **Guide AI assistance** to give better, more context-aware help

## 🎯 Quick Reference

**Having Zygote/AD issues?** → Check `julia-coding-standards.mdc`
**Training failing?** → Check `debugging-best-practices.mdc`  
**Writing new domain?** → Check `gflownet-interface-requirements.mdc`
**Setting up examples?** → Check `gflownet-high-level-interface.mdc` 