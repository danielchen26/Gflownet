# Agent Instruction Files Update Summary

## Date: January 2025

All agent instruction files in `.claude/agents/` have been comprehensively updated to reflect the current state of the GFlowNet.jl codebase after the training reorganization.

## Key Updates Made

### 1. Architecture Understanding
All agents now know:
- Training code has been reorganized from `src/core/interface.jl` to `src/training/` folder
- Clean separation: `interface.jl` only contains model creation and sampling
- Training logic is in: `training.jl`, `losses.jl`, `objectives.jl`, `configuration.jl`
- All three objectives (TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING) are fully implemented

### 2. Recent Fixes and Improvements
All agents are aware of:
- Zygote mutation fixes (push! → array comprehensions)
- Flow caching with proper Zygote.@ignore wrapping
- Backward policy validation functions
- Multi-start GFlowNets with per-initial-state Z values
- Import requirements after reorganization

### 3. Function Location Updates
All agents know the correct imports:
- `compute_trajectory_loss` is in `src/training/losses.jl`
- Must use: `using GFlowNet: compute_trajectory_loss`
- Training functions moved to `training/` folder
- Type names standardized (e.g., GridState not GridWorldState)

### 4. Updated Agents

#### gflownet-architecture-analyzer.md
- Updated to reflect training reorganization
- Removed references to old issues
- Added current architecture state section

#### gflownet-debugger.md
- Added current architecture understanding section
- Updated error patterns to include import errors
- Added fixes for reorganization-related issues

#### gflownet-documentation.md
- Updated file organization to match current structure
- Added notes about training in `training/` folder
- Emphasized current implementation status

#### gflownet-domain-implementer.md
- Added current architecture understanding section
- Updated examples to include backward policy for DB/FM
- Added notes about training objectives

#### gflownet-mathematician.md
- Added current implementation status section
- Updated mathematical guarantees
- Fixed code examples to use proper imports

#### gflownet-performance-optimizer.md
- Added performance-critical areas section
- Updated training loop location
- Fixed benchmarking code to use correct imports

#### gflownet-testing-validator.md
- Added current testing status section
- Updated test examples with proper imports
- Added notes about reorganization testing

#### gflownet-training-expert.md
- Already updated in previous session
- Reflects all new features and configurations

## Impact

These updates ensure that all specialized agents have accurate, up-to-date knowledge about:
1. The current code organization
2. Recent bug fixes and improvements
3. Proper import paths and function locations
4. Working examples with correct syntax

This allows the agents to provide accurate assistance without referencing outdated patterns or broken features.