# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Fragment-based molecular generation pipeline (`src/applications/molecular_generation.jl`)
- Reaction-constrained molecular domain (`src/utils/visualization/domains/reaction_molecular.jl`)
- RDKit bridge via PythonCall (`src/utils/visualization/python/rdkit_bridge.jl`)
- TDC oracle bridge with budget tracking (`src/utils/visualization/python/oracle_bridge.jl`, `core/oracle_manager.jl`)
- PMO 23-task benchmark runner (`src/utils/visualization/core/pmo_benchmark.jl`)
- SQLite persistence for generated molecules (`src/utils/visualization/core/database.jl`)
- Training checkpointing (`src/training/checkpoint.jl`)
- Domain-agnostic dashboard redesign with molecular UI (`src/utils/visualization/web/`)
- `reaction_state_dim(; n_reactions, fp_dim, n_scalar_features)`, the single source of truth for the reaction feature width
- `hooks/useBackendHealth.ts`, real connection status derived from `GET /health`
- `test/fixtures/molecular.jl` with the opt-in `GFLOWNET_TEST_RDKIT` gate, plus `GFLOWNET_TEST_TDC` for network-dependent oracle tests
- `test/core/test_exports.jl`, which fails if any exported name is undefined
- CI: frontend typecheck and build (`.github/workflows/frontend.yml`)
- `CONTRIBUTING.md`

### Fixed

- **The project could not load at all.** Tracked `Manifest.toml` had been resolved on Julia 1.12.4 while development ran 1.11.6, so `Pkg` itself failed to precompile (`ArgumentError: Package MbedTLS_jll ... is required but does not seem to be installed`). The manifest is now untracked, matching `.gitignore` and the `Dockerfile` comment, and `[compat] julia` is `1.11` instead of an unsatisfiable `1.6`.
- **The backend could never start.** `CondaPkg.toml` mixed conda `[deps]` with `[pip.deps]`, which forces pixi to fetch a conda→PyPI mapping over the network; that fetch failed, `using PythonCall` threw, and `rdkit_bridge.jl` is included *outside* the try/catch in `unified_server.jl`, so the failure was fatal rather than degraded.
- **The test suite aborted in group 10 of 15 and reported nothing.** `runtests.jl` rethrew on the first erroring file, making its own failure-collection and summary code unreachable. It now collects and exits nonzero.
- Deleted `test/applications/supply_chain/test_supply_chain.jl`, which constructed `Drug()`/`ProduceAction()` — types no longer present in `src/`.
- 35 exported names were defined nowhere in `src/`, including the entire `Causal*` and `ActiveLearning*` vocabularies (the files define `DAGState`/`ExperimentState` instead) and the whole logging API surface.
- `GFlowNetLogger` and `ReportData` were each compiled into two distinct types with the same name, because `utils/utils.jl` and `GFlowNet.jl` both included `logging.jl` and `report.jl`.
- Reaction state width `1049` was a bare literal in two places while `n_reactions` and `fp_dim` were already keyword arguments, so `create_reaction_gflownet(n_reactions=20)` built a network sized for 17. `scripts/validate_all_gaps.jl` hit exactly this by passing both together.
- `molecular_generation.jl` used `JSON3` without importing it, working only inside the server's scope and throwing for its two other callers.
- `unified_server.jl` registered `/api/v2/molecular/molecules/:id/objectives` with Oxygen's non-parameter `:id` syntax, so the frontend call could never match.
- The dashboard reported "Connected" from a hardcoded `useState(true)` regardless of backend state; there is no WebSocket route at all.
- `src/training/configuration.jl` called `Optimisers.RMSprop`, but the exported name is `RMSProp`, so `create_optimizer(RMSPROP, lr)` threw for every caller.
- `test/integration/test_training.jl` hit `UndefVarError: ADAM` because `using Optimisers` made the name ambiguous with GFlowNet's own `OptimizationMethod` enum.
- Flow-matching and `flow_estimate` tests built models without `include_flow_estimator=true`, so their assertions errored instead of running.
- Docker image never copied `CondaPkg.toml`, leaving the deployed molecular pipeline non-functional, and its `CMD` re-implemented `start_server.jl` inline.
- `docs/make.jl` listed four `internals/*.md` pages at paths that no longer existed, so the documentation build had been failing entirely.
- Ten references across seven files advertised the dev frontend on port 3000; Vite serves 5173, and `show_visualization.jl` actually executed `open http://localhost:3000`.
- `data/` did not exist even though the server, fragment library and TDC cache all resolve into it.
- Three `@test true  # Placeholder` assertions and a `try/catch` that printed failures let files pass without testing anything.

### Changed

- `start_server.jl` reads `PORT` and `HOST` from the environment and is now the single launch path.
- `Manifest.toml` and `docs/Manifest.toml` are untracked by policy; pin through `[compat]`.
- Two previously orphaned test files are wired into the suite (+44 assertions).
- `RELEASE_NOTES.md` replaced by this file.

### Removed

- Editor tooling (`LanguageServer`, `SymbolServer`, `IJulia`, `BenchmarkTools`) from runtime `[deps]`; direct dependencies dropped from 30 to 26.
- Four orphan source files (863 lines) that redeclared the core type system with contradictory fields: `src/{types,rewards,flow_networks,directed_acyclic_graph}.jl`.
- The `/ws` Vite proxy entry, which pointed at a route that does not exist.

### Known issues

- `test/objectives/flow_matching/test_flow_matching_comprehensive.jl` has two behavioural failures: average final loss is 23.50 against an asserted `< 1.0`, and `log Z` never moves from `0.0`, suggesting `LEARNABLE_ESTIMATION` is not wired for the `FLOW_MATCHING` objective.
- `test/objectives/direct_flow/test_direct_flow.jl` (17 pass, 2 fail) and `test/objectives/sub_trajectory_balance/test_sub_trajectory_balance.jl` (constructs an invalid `Trajectory`) remain unwired. The latter is the only coverage of the exported `sub_trajectory_balance_loss`.
- `reward(::GridState)` has no method, and one reduction over an empty collection lacks an `init`.
- Four test files still contain zero assertions and pass unconditionally: `test_backward_policy.jl`, `test_detailed_balance_comprehensive.jl`, `test_detailed_balance_summary.jl`, `test_grid_world_versions.jl`.
- `docker build` is unverified; the Dockerfile fixes were validated statically plus by exercising the `PORT`/`HOST` contract locally.

## [1.0.0] - 2025-07-28

First major release: the transition from research prototype to a
publication-ready package.

### Added

- Professional dark-theme plotting: grid-world trajectory analysis with heat-map
  endpoints, reward-zone annotations, and 300+ DPI output
- Training progress plots with multi-scale moving averages, performance
  milestones and convergence analysis
- Statistical reward-distribution analysis and position heatmaps
- CSV export suite: `trajectories_*`, `rewards_*`, `training_*`, `positions_*`
- Responsive HTML reports with embedded visualizations, plus structured text
  summaries
- Professional grid-world example demonstrating acyclic control, multiple
  configurations and proportional-sampling validation

### Changed

- Examples use exclusively the high-level interface, with no manual neural
  network definitions
- Consistent `Float32` usage for type stability; all functions Zygote-safe
- Results written into the correct directory structure

### Fixed

- Method-overwriting warnings during module precompilation

Reported metrics for the grid-world example: 100/100 valid trajectories,
21% optimal-reward rate (theoretically correct), mean reward 15.2, max 50.0,
21 unique endpoints, convergence in 50 iterations.
