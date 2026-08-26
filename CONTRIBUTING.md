# Contributing

## Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Requires Julia 1.11 or newer (`Project.toml` `[compat] julia = "1.11"`).

`Manifest.toml` is untracked by policy, so `Pkg.instantiate()` is required, not
optional. It also builds the conda environment declared in `CondaPkg.toml`
(Python, RDKit, NumPy, scikit-learn, umap-learn) — roughly 1 GB, once. Verify:

```julia
using PythonCall
pyimport("rdkit").__version__
```

**Never add a `[pip.deps]` section to `CondaPkg.toml`.** Mixing pip and conda
dependencies forces pixi to fetch a conda→PyPI name mapping over the network.
When that fetch fails, `using PythonCall` throws, and because
`unified_server.jl` includes `python/rdkit_bridge.jl` *outside* its try/catch,
the entire backend becomes unloadable rather than degrading gracefully. All
Python dependencies belong in `[deps]` with `channel = "conda-forge"`.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The default suite requires **no Python, no conda environment and no network**.
Chemistry- and oracle-dependent assertions are opt-in:

```bash
GFLOWNET_TEST_RDKIT=true julia --project=. -e 'using Pkg; Pkg.test()'
GFLOWNET_TEST_TDC=true   julia --project=. -e 'using Pkg; Pkg.test()'   # downloads oracles
```

Without `GFLOWNET_TEST_RDKIT`, molecular tests log *why* they skipped. That
warning is deliberate: these tests used to skip silently, so
`test_reward_function.jl` and `test_diversity.jl` reported green while running
zero assertions.

The full suite takes 40–48 minutes. While iterating, run one file:

```bash
julia --project=. -e 'using Test; using GFlowNet; include("test/core/test_core_functions.jl")'
```

### Rules for tests

- **Register every new test file in the `test_groups` table in
  `test/runtests.jl`.** A file that is not listed does not run — 30 files were
  orphaned that way, including the only coverage of
  `sub_trajectory_balance_loss`.
- **Verify a file passes standalone before wiring it in.** A single erroring
  file used to abort the whole suite.
- Shared molecular preconditions and expected values belong in
  `test/fixtures/molecular.jl`, not duplicated per file.
- A test that cannot fail is worse than no test. Use `@test_broken`, not
  `@test true`, and never `try/catch` around the assertion you care about.
- Models exercising flow functions need
  `create_grid_world_gflownet(include_flow_estimator = true)`.

## Running the app

Backend (port 8080; honors `PORT` and `HOST`):

```bash
julia --project=. start_server.jl
```

Frontend (port 5173):

```bash
cd src/utils/visualization/web
npm install
npm run dev
```

Open `http://localhost:5173`. On macOS Vite binds IPv6, so use `localhost`,
not `127.0.0.1`. The sidebar connection indicator reflects a real `GET /health`
poll — if it says Disconnected, the backend is genuinely down.

## Docs

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Output goes to `docbuild/` (gitignored). Every page must appear in the `pages=`
structure of `docs/make.jl`; `makedocs` errors on a listed page that does not
exist, which is how the docs build stayed broken unnoticed.

## Conventions

- Conventional Commits: `fix:`, `feat:`, `chore:`, `test:`, `docs:`, `ci:`
- Export only names that exist — `test/core/test_exports.jl` enforces this
- Never hardcode a state dimension. Use `compute_state_dim()` for fragment
  states and `reaction_state_dim()` for reaction states
- The dev frontend is `5173` and the API is `8080`. Do not reintroduce `3000`
- Pin dependency versions through `[compat]`, never by committing a manifest
