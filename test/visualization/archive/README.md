# Archived Visualization Tests

This directory contains development/debugging test files that were created during the exploration of training configurations and epsilon exploration strategies.

## Why Archived

These files are not part of the main test suite (`runtests.jl`) and represent:
- Development exploration of different epsilon values
- Debugging investigations of mode discovery
- Temporary comparison tests between objectives

## Files

### Epsilon Exploration Variants
- `test_epsilon_aggressive.jl` - ε=0.2, no decay
- `test_epsilon_balanced.jl` - Balanced epsilon settings
- `test_epsilon_exploration.jl` - Exploration investigation
- `test_epsilon_extended.jl` - Extended training with epsilon
- `test_high_epsilon.jl` - High epsilon (0.3-0.5) comparison

### Objective Comparison Tests
- `test_db_vs_tb_exploration.jl` - Detailed Balance vs Trajectory Balance
- `test_tb_comprehensive.jl` - Comprehensive TB testing
- `test_tb_pure_peaks.jl` - TB with pure peak configurations

### Development/Debug Files
- `test_comprehensive_visualization.jl` - Old visualization test (Aug 2025)
- `manual_integration_test.jl` - Manual integration testing
- `quick_subtb_test.jl` - Quick SubTB validation

### Test Outputs
- `results/` - Training curves and output files

## Status (measured 2026-08-28)

The claim that all of these "can still be run individually" was not true. Each was
invoked as
`julia --project=. -e 'using Test, GFlowNet; include("test/visualization/archive/<file>")'`
with a 240 s cap, on Julia 1.11.6 / M1 Pro:

| File | Has assertions | Status |
|---|---|---|
| `test_comprehensive_visualization.jl` | 16 `@test`, none reachable | **DEAD.** Errors on its first statement; targets the removed Makie API (`GFlowNetVisualizer`, `TrainingMonitor`, theme/export helpers — none exist under `src/`). 0 passed, 1 errored, 1.4 s. See the file header. |
| `manual_integration_test.jl` | none (script) | **Was broken by this very archive move** — its `../../src/...` includes resolved to `test/src/...`. Fixed to `../../../src/...`; now loads. |
| `quick_subtb_test.jl` | none (script) | Exceeded 240 s (2 × 500-iteration trainings). |
| `test_high_epsilon.jl` | none (script) | Not re-measured; 2 × 1000-iteration trainings. |
| `test_epsilon_aggressive.jl` | 3 | Exceeded 240 s (1000-iteration trainings). |
| `test_epsilon_balanced.jl` | 6 | Not re-measured; 500 + 1000 + 1500 iterations. Testset 1 was **empty** (guaranteed silent pass) and now asserts. |
| `test_epsilon_exploration.jl` | 12 | Not re-measured; config assertions are fast, testset 3 trains. |
| `test_epsilon_extended.jl` | 6 | Not re-measured; 2000 iterations. |
| `test_db_vs_tb_exploration.jl` | 1 | Not re-measured; two trainings. |
| `test_tb_comprehensive.jl` | 19 | Not re-measured; testsets 1–2 are fast, 3–4 train. |
| `test_tb_pure_peaks.jl` | 3 | Not re-measured; long training. |

`manual_integration_test.jl`, `quick_subtb_test.jl` and `test_high_epsilon.jl` contain
no `using Test`, no `@testset` and no `@test`. They are exploration scripts that
happen to be named `test_*`; they cannot report a pass or a failure, only print.

None of these belong in `runtests.jl`: the ones that still work are multi-minute
stochastic training runs asserting research outcomes, not invariants. The real
visualization package test is `../test_real_training_viz.jl` (193 assertions, ~55 s),
wired into `runtests.jl` under the "Visualization" group.

## Usage

These files can be run individually for debugging, subject to the status table above:
```bash
julia --project=. test/visualization/archive/test_epsilon_aggressive.jl
```

## Date Archived
February 3, 2026
