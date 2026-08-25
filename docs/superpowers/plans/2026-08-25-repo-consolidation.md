# GFlowNet Repo Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the repo from "cannot load, cannot test, cannot deploy" to a state where `Pkg.instantiate()`, `Pkg.test()`, the Oxygen backend, the Vite frontend, and the Documenter build all run green, with CI enforcing it.

**Architecture:** Six sequential phases, each ending with the repo in a working state. Phase 0 repairs the environment (nothing else can run until it does). Phase 1 makes the test suite reach the code merged from `core-development`. Phase 2 removes duplicate/phantom source declarations. Phase 3 fixes the server/frontend/deployment contract. Phase 4 adds CI to lock in phases 0–3. Phase 5 makes the docs true. Phase 6 is hygiene.

**Tech Stack:** Julia 1.11/1.12 (package `GFlowNet`, uuid `2d7ca041-c8ad-46e9-a25d-4e8f55c0c8f5`), Oxygen.jl 1.7.5 HTTP server, SQLite.jl, PythonCall + CondaPkg (RDKit, PyTDC), Vite 5 + React 18 + TypeScript frontend, Documenter.jl, GitHub Actions.

## Global Constraints

- Package name `GFlowNet`, uuid `2d7ca041-c8ad-46e9-a25d-4e8f55c0c8f5`. Never change either; five example environments already carry fabricated UUIDs because someone did.
- Julia floor is `1.11`. These are the only two versions this code has ever been resolved against: local dev is 1.11.6, and both `Manifest.toml:3` and `Dockerfile:4` say 1.12.
- Backend port is `8080`. Dev frontend port is `5173` (`vite.config.ts:14`). The string `3000` must not appear as a frontend URL anywhere when this plan is done.
- Dimensions are derived, never literal. The fragment-state width comes from `compute_state_dim()` (`src/applications/molecular_generation.jl:115`) — already true. The reaction-state width comes from `reaction_state_dim()` (added in Task 12). `1042` and `1049` must not appear as literals.
- `Pkg.test()` must pass with **no** Python, no conda env, no network, and no RDKit. RDKit-dependent assertions run only when `ENV["GFLOWNET_TEST_RDKIT"] == "true"`. Network/TDC-dependent assertions run only when `ENV["GFLOWNET_TEST_TDC"] == "true"`.
- Commit after every task, then **push**. Use Conventional Commits (`fix:`, `feat:`, `chore:`, `test:`, `docs:`, `ci:`). Owner authorized push-per-task on 2026-08-25; this supersedes the original "never push" constraint. Pushing `main` also publishes the 9 pending `core-development` merge commits, which is intended.
- Still never run `git filter-repo`, `bfg`, `git gc --aggressive`, or any force-push. History rewrite remains an owner decision (see Deferred).
- Do not delete anything under `.claude/`, `examples/feature_acquisition/archive/`, or `test/visualization/archive/`. Those are Deferred owner decisions.

## Decisions already made (do not relitigate)

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | **Untrack `Manifest.toml` and `docs/Manifest.toml`**; library policy | `.gitignore:6` and `Dockerfile:18` already assert this policy; the tracked file is what breaks loading. Reproducibility moves to `[compat]` + the CI matrix. |
| D2 | **PythonCall/RDKit stay hard `[deps]`**, but tests never require them | The merged pipeline genuinely needs RDKit at runtime; making `Pkg.test()` depend on a conda build would make CI unrunnable. |
| D3 | **No git history rewrite** in this plan | Recovering the 304 MB of `node_modules` history requires force-push, invalidating every published SHA and the `pre-merge-backup-20260825` tag. Owner decision. |
| D4 | **Julia floor `1.11`, CI matrix `['1.11','1.12']`** | `[compat] julia = "1.6"` is unsatisfiable — Lux 1.16.0 needs ≥1.10 and the Manifest was cut on 1.12.4. |

---

## File Structure

Files created by this plan:

- `.github/workflows/ci.yml` — Julia test matrix + Documenter build. One job per Julia version.
- `.github/workflows/frontend.yml` — Node job: `npm ci`, `tsc --noEmit`, `vite build`. Separate from Julia because they share no toolchain.
- `test/fixtures/molecular.jl` — the single home for the molecular test harness: RDKit availability flag, fragment-library expectations, state-dim helpers. Replaces per-file duplicated guards and magic constants.
- `test/fixtures/simple_domain.jl` — moved from `test/core/test_utilities.jl`, which is a fixture with zero `@test` currently masquerading as a test.
- `CHANGELOG.md` — Keep-a-Changelog, replaces `RELEASE_NOTES.md`.
- `CONTRIBUTING.md` — how to instantiate, test, run the server, run the frontend.
- `data/.gitkeep` — makes the runtime data directory exist; `data/` contents stay gitignored.

Files deleted by this plan:

- `src/types.jl`, `src/rewards.jl`, `src/flow_networks.jl`, `src/directed_acyclic_graph.jl` — orphans (in no load graph) that redeclare live core types with contradictory fields.
- `test/applications/supply_chain/test_supply_chain.jl` — references `Drug`/`Facility`/`PatientRegion`/`TransportRoute`, removed from `src` at `src/GFlowNet.jl:92`.
- `announce_v1.jl` — println-only marketing script advertising a `results/` directory that does not exist.
- `examples/core_features/visualization/fixthis.png`, `examples/core_features/visualization/WechatIMG7635.jpg` — accidental screenshot commits, 1.2 MB, no generator.
- `RELEASE_NOTES.md` — content moves to `CHANGELOG.md`.
- `src/utils/visualization/web/VISUALIZATION_FIXES.md` — stale changelog with wrong paths, superseded by git history.

Files whose responsibility changes:

- `test/runtests.jl` — becomes collect-and-continue instead of abort-on-first-error, and gains the previously orphaned test files.
- `Project.toml` — runtime deps only; editor tooling moves out; `[targets] test` gains its real deps.
- `Dockerfile` — must copy `CondaPkg.toml` and call `start_server.jl` instead of re-implementing it inline.

---

## Phase 0 — Make the repo loadable

Nothing in the repo can run right now. `julia --project=. -e 'using Pkg; Pkg.status()'` fails with `ArgumentError: Package MbedTLS_jll [c8ffd9c3-330d-5841-b78e-0817d7145fa1] is required but does not seem to be installed`, cascading into failed precompilation of `LibGit2_jll`, `LibGit2`, and `Pkg` itself. Cause: `Manifest.toml:3` says `julia_version = "1.12.4"` and the installed Julia is 1.11.6, whose stdlib set differs.

### Task 1: Untrack the Manifest and fix the Julia floor

**Files:**
- Modify: `Project.toml:62` (`julia = "1.6"`)
- Modify: `.gitignore:5-8`
- Delete from index (keep on disk): `Manifest.toml`, `docs/Manifest.toml`

**Interfaces:**
- Produces: a loadable project environment. Every later task depends on `julia --project=. -e 'using GFlowNet'` working.

- [ ] **Step 1: Confirm the failure before changing anything**

Run: `julia --project=. -e 'using Pkg; Pkg.status()' 2>&1 | head -5`
Expected: FAIL with `ArgumentError: Package MbedTLS_jll ... is required but does not seem to be installed`.

- [ ] **Step 2: Untrack both manifests**

```bash
git rm --cached Manifest.toml docs/Manifest.toml
```

The files stay on disk. Do not delete them yet — step 4 regenerates them.

- [ ] **Step 3: Make `.gitignore` state the policy exactly once**

Replace `.gitignore` lines 5-8, which currently read:

```
.DS_Store
/Manifest.toml
/docs/Manifest.toml
!/docs/Manifest.toml
```

with:

```
.DS_Store
/Manifest.toml
/docs/Manifest.toml
```

Lines 7 and 8 were a self-cancelling pair. Also delete the duplicate `.DS_Store` at line 15 (under `# System files`); keep the one at line 5.

- [ ] **Step 4: Set the Julia floor and regenerate the manifest**

In `Project.toml`, change line 62 from `julia = "1.6"` to:

```toml
julia = "1.11"
```

Then:

```bash
rm Manifest.toml
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

This downloads packages and takes several minutes on a cold depot. It will also trigger CondaPkg to build a conda environment with RDKit because `PythonCall` is a dep — that is expected and is a one-time ~1 GB cost.

- [ ] **Step 5: Verify the environment loads**

Run: `julia --project=. -e 'using Pkg; Pkg.status(); using GFlowNet; println("loaded OK")'`
Expected: the status table prints, then `loaded OK`. If `using GFlowNet` fails, stop and report the error — do not proceed to Task 2, because every later verification depends on this.

- [ ] **Step 6: Commit**

```bash
git add Project.toml .gitignore
git commit -m "fix: untrack Manifest.toml and set Julia floor to 1.11

Tracked Manifest.toml was resolved on Julia 1.12.4 while dev runs 1.11.6,
so Pkg could not even precompile (missing MbedTLS_jll). .gitignore:6 and
Dockerfile:18 already asserted the untracked policy; make it true.
[compat] julia = 1.6 was unsatisfiable — Lux 1.16.0 requires >= 1.10."
```

### Task 2: Declare the missing `UUIDs` dependency

`src/utils/visualization/api/unified_server.jl:7` does `using UUIDs`, `src/utils/visualization/core/database.jl:7` and `src/utils/visualization/core/training_session.jl:9` do `using UUIDs: uuid4`. `UUIDs` is not in `Project.toml [deps]`.

**CORRECTION, established during execution 2026-08-25.** This plan originally claimed the undeclared `UUIDs` was why the backend had never booted. That was wrong. `Base.load_path()` ends in `.../share/julia/stdlib/v1.11`, so `@stdlib` is always searchable and an undeclared stdlib still resolves: `julia --project=. -e 'using UUIDs'` succeeds. Declaring it is still correct — an undeclared dependency is a real defect, and package-internal code (inside `module GFlowNet`) would not get the `@stdlib` fallback these Main-scope scripts rely on — but it is hygiene, not a blocker.

**The actual reason the backend had never booted** was `CondaPkg.toml` mixing conda `[deps]` with a `[pip.deps]` section. That mix forces pixi to fetch a conda→PyPI name mapping over the network, which failed reproducibly (`prefix.dev` returned HTTP 200, so the host is reachable; the mapping fetch itself failed after 3 retries). `using PythonCall` therefore threw, which made `src/utils/visualization/python/rdkit_bridge.jl:9` unloadable, and `unified_server.jl:20` includes that file **outside** the `try`/`catch` at `:35-42` — so the failure was fatal, not degraded. Fixed by moving `umap-learn`, `requests`, and `fuzzywuzzy` from `[pip.deps]` to conda-forge `[deps]` entries; all three are available there, so the mix bought nothing. Verified: Python 3.11.16, RDKit 2026.03.5, UMAP 0.5.12, scikit-learn 1.2.2 (honoring the `<1.3` TDC pin), and the server prints `RDKitBridge initialized successfully` / `All 50 fragments validated successfully` / three registered domains.

**Files:**
- Modify: `Project.toml` `[deps]` block
- Modify: `CondaPkg.toml` — remove `[pip.deps]`, add conda-forge entries for `umap-learn`, `requests`, `fuzzywuzzy`

**Interfaces:**
- Produces: a loadable `include("src/utils/visualization/api/unified_server.jl")`. Tasks 14, 17, and the server-verification steps of 12 and 13 depend on it.

- [ ] **Step 1: Reproduce the failure**

Run: `julia --project=. -e 'using UUIDs' 2>&1 | tail -3`
Expected: FAIL — `ArgumentError: Package UUIDs not found in current path`.

- [ ] **Step 2: Add the dep**

In `Project.toml [deps]`, insert in alphabetical position (between `SymbolServer` and `Zygote`, i.e. after the `Statistics`/`StatsBase` entries and before `Zygote`):

```toml
UUIDs = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
```

`UUIDs` is a stdlib, so it needs no `[compat]` entry.

- [ ] **Step 3: Verify the server file loads**

```bash
julia --project=. -e 'using Pkg; Pkg.resolve()'
julia --project=. -e 'using UUIDs; println("UUIDs OK")'
```
Expected: `UUIDs OK`.

- [ ] **Step 4: Verify the whole server load graph**

Run:
```bash
julia --project=. -e 'push!(LOAD_PATH, joinpath(pwd(), "src")); include("src/utils/visualization/api/unified_server.jl"); println("server loaded, routes registered")'
```
Expected: it prints RDKit init info (or a warning that RDKit is unavailable, which is fine), then `server loaded, routes registered`. Any `UndefVarError` here is a real defect — record the symbol and which file it came from, because Task 13 covers the known one (`JSON3`/`RDKitBridge` in `molecular_generation.jl`).

- [ ] **Step 5: Commit**

```bash
git add Project.toml
git commit -m "fix: declare UUIDs dependency used by the visualization server

unified_server.jl:7, core/database.jl:7 and core/training_session.jl:9
import UUIDs, but it was absent from [deps]. Those files load as
top-level scripts (not via src/GFlowNet.jl), so the backend could never
boot — including under Dockerfile:38."
```

### Task 3: Move editor tooling out of runtime deps and give the test target real deps

`Project.toml` lists `BenchmarkTools`, `IJulia`, `LanguageServer`, and `SymbolServer` as runtime `[deps]`. They pull `CSTParser`, `StaticLint`, `JuliaFormatter`, `JSONRPC`, `Tokenize`, `ZMQ`, `Conda`, `MbedTLS` into every install — a direct contributor to the resolver fragility in Task 1. Verified: no file under `src/` imports any of the four. The only reference anywhere is `test/visualization/archive/test_comprehensive_visualization.jl:304` (`using BenchmarkTools`), and that file is orphaned — never included by `runtests.jl`.

Separately, `[targets] test = ["Test"]` is a lie: the suite also uses `Zygote` (`test/core/detailed_balance/test_detailed_balance_comprehensive.jl:23`), `Lux` (`test/applications/molecular/test_mogfn.jl:8`), `Random`, and `Statistics`. They resolve today only because they are runtime deps.

**Files:**
- Modify: `Project.toml` `[deps]`, `[compat]`, `[extras]`, `[targets]`

- [ ] **Step 1: Confirm nothing in `src/` uses the four tools**

Run: `grep -rn "using BenchmarkTools\|using IJulia\|using LanguageServer\|using SymbolServer\|@benchmark" src/`
Expected: no output.

- [ ] **Step 2: Delete the four `[deps]` entries**

Remove these lines from `[deps]`:

```toml
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
LanguageServer = "2b0e0bc5-e4fd-59b4-8912-456d1b03d8d7"
SymbolServer = "cf896787-08d5-524d-9de7-132aaa0cb996"
```

And their `[compat]` entries:

```toml
BenchmarkTools = "1"
IJulia = "1.29.0"
LanguageServer = "4.5.1"
SymbolServer = "7.4.0"
```

- [ ] **Step 3: Tighten the four impossible compat bounds**

The lower branches span breaking API changes and cannot work against what actually resolves (Lux 1.16.0 postdates the `LuxCore` split; Lux 0.4 has no `Lux.setup`). In `[compat]`, replace:

```toml
Lux = "0.4, 1.6"
NNlib = "0.8, 0.9"
Optimisers = "0.2, 0.3, 0.4"
Plots = "1.38, 1.40"
StatsBase = "0.33, 0.34"
```

with:

```toml
Lux = "1.6"
NNlib = "0.9"
Optimisers = "0.4"
Plots = "1.40"
StatsBase = "0.34"
```

(`Plots = "1.38"` already covered `1.40` under caret semantics; the pair was redundant.)

- [ ] **Step 4: Declare the real test deps**

Replace the `[extras]` and `[targets]` blocks:

```toml
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"

[targets]
test = ["Test", "BenchmarkTools", "Lux", "Random", "Statistics", "Zygote"]
```

`Lux`, `Random`, `Statistics`, and `Zygote` remain runtime `[deps]`, so they need no `[extras]` entry — listing them in `[targets]` documents the test contract. `BenchmarkTools` needs the `[extras]` entry because it is no longer a runtime dep.

- [ ] **Step 5: Verify resolve and load still work**

```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --project=. -e 'using GFlowNet; println("loaded OK")'
```
Expected: `loaded OK`, and the resolve should now install noticeably fewer packages.

- [ ] **Step 6: Commit**

```bash
git add Project.toml
git commit -m "chore: remove editor tooling from runtime deps, fix test target

LanguageServer/SymbolServer/IJulia/BenchmarkTools were runtime [deps],
dragging CSTParser/StaticLint/JuliaFormatter/ZMQ/Conda into every
install. No file under src/ imports them. Also tightened Lux/NNlib/
Optimisers/StatsBase compat, whose lower branches predate breaking API
changes, and declared the test target's real deps."
```

---

## Phase 1 — Make the test suite honest

`Pkg.test()` currently aborts partway: `test/applications/supply_chain/test_supply_chain.jl:11` constructs `Drug(...)`, and `Drug`/`Facility`/`PatientRegion`/`TransportRoute` exist nowhere in `src/` (`src/GFlowNet.jl:92` has the include commented out). That throws `UndefVarError` inside the try at `runtests.jl:101-108`, which **rethrows at line 106**, terminating the whole `@testset`. Consequence: the entire `"Molecular Generation"` group at `runtests.jl:63-74` — every test file the merge added — never runs.

### Task 4: Delete the supply-chain test and unblock the molecular group

**Files:**
- Delete: `test/applications/supply_chain/test_supply_chain.jl`
- Modify: `test/runtests.jl:60-62`

**Interfaces:**
- Produces: `Pkg.test()` reaches `runtests.jl:63-74`. Tasks 6, 7, and 8 all assume the suite runs to completion.

- [ ] **Step 1: Confirm the types are really gone**

Run: `grep -rn "struct Drug\|struct Facility\|struct PatientRegion\|struct TransportRoute" src/`
Expected: no output. Also confirm `src/GFlowNet.jl:92` reads:
`# include("applications/supply_chain_optimization.jl")  # Removed in core-fixes branch`

- [ ] **Step 2: Reproduce the abort**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -c "Molecular Generation"`
Expected: `0` — the molecular group never prints, proving it never runs.

- [ ] **Step 3: Delete the test file and its group**

```bash
git rm test/applications/supply_chain/test_supply_chain.jl
```

In `test/runtests.jl`, delete lines 60-62:

```julia
    ("Supply Chain Application", [
        "applications/supply_chain/test_supply_chain.jl"
    ]),
```

- [ ] **Step 4: Verify the molecular group now runs**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep "Molecular Generation"`
Expected: the line `📦 Testing: Molecular Generation` appears. Individual molecular tests may still report skips — Task 7 addresses that. What matters here is reachability.

- [ ] **Step 5: Commit**

```bash
git add -A test/ 
git commit -m "test: delete supply-chain test that aborted the whole suite

test_supply_chain.jl:11 constructed Drug(), but Drug/Facility/
PatientRegion/TransportRoute were removed from src (GFlowNet.jl:92).
runtests.jl:106 rethrows, so this single UndefVarError prevented all 10
merged molecular test files from ever running."
```

### Task 5: Make `runtests.jl` collect failures instead of aborting

`runtests.jl:104` already pushes to `failed_groups` and lines 126-137 already print a summary — but line 106 `rethrow(e)` makes that machinery dead code. One broken file hides every downstream result.

**Files:**
- Modify: `test/runtests.jl:101-110`, `test/runtests.jl:126-137`

**Interfaces:**
- Consumes: nothing.
- Produces: a suite that reports every failing file in one run. Task 8 relies on being able to see all zero-assertion files at once.

- [ ] **Step 1: Write the failing test — a deliberately broken test file**

```bash
mkdir -p test/tmp_probe
cat > test/tmp_probe/test_probe_a.jl <<'EOF'
error("deliberate probe failure")
EOF
cat > test/tmp_probe/test_probe_b.jl <<'EOF'
using Test
@testset "probe b runs after probe a fails" begin
    @test 1 + 1 == 2
end
EOF
```

Add a temporary group at the end of the `test_groups` array in `test/runtests.jl` (after the `"Molecular Generation"` entry, before the closing `]` at line 75):

```julia
    ("Probe", [
        "tmp_probe/test_probe_a.jl",
        "tmp_probe/test_probe_b.jl"
    ])
```

- [ ] **Step 2: Run it to confirm probe B never executes**

Run: `julia --project=. test/runtests.jl 2>&1 | grep -c "test_probe_b.jl"`
Expected: `0` — the rethrow at line 106 kills the run at probe A.

- [ ] **Step 3: Replace the rethrow with collect-and-continue**

In `test/runtests.jl`, replace lines 101-107:

```julia
                    try
                        include(test_path)
                    catch e
                        push!(failed_groups, "$group_name - $test_file")
                        @error "Test failed" file=test_file exception=e
                        rethrow(e)
                    end
```

with:

```julia
                    try
                        include(test_path)
                    catch e
                        push!(failed_groups, "$group_name - $test_file")
                        @error "Test file errored" file=test_file exception=(e, catch_backtrace())
                        @test false  # surface the error as a test failure, not a silent note
                    end
```

`@test false` is what keeps `Pkg.test()` exiting nonzero. Without it, a collected error would make the suite pass.

- [ ] **Step 4: Make the summary exit nonzero**

At the end of `test/runtests.jl`, after line 143 (`println("\nRun them individually when debugging specific issues.")`) and before line 144 (`println("\nCompleted at: $(now())")`), insert:

```julia
if !isempty(failed_groups)
    error("$(length(failed_groups)) test file(s) errored — see the list above")
end
```

- [ ] **Step 5: Verify both probes now run and the suite still fails**

Run: `julia --project=. test/runtests.jl 2>&1 | grep -c "test_probe_b.jl"`
Expected: `1` — probe B ran despite probe A erroring.

Run: `julia --project=. test/runtests.jl > /dev/null 2>&1; echo "exit=$?"`
Expected: `exit=1` — a collected error still fails the suite.

- [ ] **Step 6: Remove the probe**

```bash
rm -rf test/tmp_probe
```
Delete the temporary `"Probe"` group from `test_groups`.

- [ ] **Step 7: Verify the real suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -25`
Expected: the suite runs every group to completion. Record the full list of failures — that list is the input to Tasks 6-8.

- [ ] **Step 8: Commit**

```bash
git add test/runtests.jl
git commit -m "test: collect test-file errors instead of aborting the suite

runtests.jl:106 rethrew on the first erroring file, making the
failed_groups machinery at :104 and the summary at :126-137 dead code.
Now every file runs, errors surface as @test false, and the runner exits
nonzero if anything errored."
```

### Task 6: Wire the four orphaned, well-formed test files

30 test files exist on disk that `runtests.jl` never includes. Four of them are well-formed `@testset` suites with real assertions and no external dependencies — they are simply unwired:

- `test/exploration/test_exploration_improvements.jl` — ~8 nested testsets covering `entropy_weight`, `z_learning_rate_multiplier`, `ReplayBuffer`, importance sampling (lines 13-133)
- `test/exploration/test_z_learning_rate_multiplier.jl` — 5 testsets on `scale_z_gradient` and Z convergence (lines 24-221)
- `test/objectives/direct_flow/test_direct_flow.jl` — 5 testsets on `DIRECT_FLOW_OBJECTIVE` (lines 55-224)
- `test/objectives/sub_trajectory_balance/test_sub_trajectory_balance.jl` — the **only** coverage of the exported `sub_trajectory_balance_loss` (lines 70, 83, 182, 207, 213)

The sub-TB file needs a fix first: a top-level `try/catch` at lines 169-172 filters NaN/Inf losses out of the sample before asserting, which hides divergence.

**Files:**
- Modify: `test/runtests.jl` `test_groups`
- Modify: `test/objectives/sub_trajectory_balance/test_sub_trajectory_balance.jl:169-172`

- [ ] **Step 1: Confirm they are orphaned and currently pass standalone**

Run: `grep -c "test_direct_flow\|test_exploration_improvements\|test_z_learning_rate_multiplier\|test_sub_trajectory_balance" test/runtests.jl`
Expected: `0`.

Then run each standalone to see its true state:
```bash
for f in test/exploration/test_exploration_improvements.jl \
         test/exploration/test_z_learning_rate_multiplier.jl \
         test/objectives/direct_flow/test_direct_flow.jl \
         test/objectives/sub_trajectory_balance/test_sub_trajectory_balance.jl; do
  echo "=== $f"; julia --project=. "$f" > /dev/null 2>&1; echo "exit=$?"
done
```
Record which pass. A file that fails standalone must be fixed or reported before wiring — wiring a red file is what caused Task 4.

- [ ] **Step 2: Remove the NaN-swallowing try/catch in the sub-TB test**

Read `test/objectives/sub_trajectory_balance/test_sub_trajectory_balance.jl:160-180` first to get the exact surrounding context, then delete the top-level `try`/`catch` wrapper at lines 169-172 so that a NaN or Inf loss fails the test instead of being filtered out of the sample. Keep the assertions themselves unchanged.

- [ ] **Step 3: Add the groups**

In `test/runtests.jl`, insert after the `"Learnable Z"` group (which ends at line 52) and before `("Multi-Start GFlowNets", [`:

```julia
    ("Sub-Trajectory Balance", [
        "objectives/sub_trajectory_balance/test_sub_trajectory_balance.jl"
    ]),
    ("Direct Flow", [
        "objectives/direct_flow/test_direct_flow.jl"
    ]),
    ("Exploration", [
        "exploration/test_exploration_improvements.jl",
        "exploration/test_z_learning_rate_multiplier.jl"
    ]),
```

- [ ] **Step 4: Verify they run inside the suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "Sub-Trajectory Balance|Direct Flow|Exploration"`
Expected: all three group headers appear.

Run: `julia --project=. -e 'using Pkg; Pkg.test()' > /dev/null 2>&1; echo "exit=$?"`
Expected: `exit=0`. If nonzero, the newly wired files revealed a real defect — report it with the failing assertion rather than unwiring the file.

- [ ] **Step 5: Commit**

```bash
git add test/runtests.jl test/objectives/sub_trajectory_balance/test_sub_trajectory_balance.jl
git commit -m "test: wire four orphaned test files into runtests.jl

test_exploration_improvements.jl, test_z_learning_rate_multiplier.jl,
test_direct_flow.jl and test_sub_trajectory_balance.jl were well-formed
suites that runtests.jl never included; the sub-TB file is the only
coverage of the exported sub_trajectory_balance_loss. Also removed its
top-level try/catch, which filtered NaN/Inf losses before asserting."
```

### Task 7: One explicit RDKit gate and one shared molecular fixture

Every RDKit-dependent molecular assertion is permanently skipped, and nobody can tell from the output. `test/applications/molecular/test_setup.jl:16` includes `src/applications/molecular_generation.jl` but never `src/utils/visualization/python/rdkit_bridge.jl`, so `RDKitBridge` is never defined under `Pkg.test()`. Consequences:

- `test_reward_function.jl:14` — the entire file's assertions are one `@test_skip`
- `test_diversity.jl` — all 6 testsets skipped (`@test_skip` at 20, 43, 61, 77, 86)
- `test_docking.jl` — all testsets skipped (21, 54, 70)
- `test_fragment_joining.jl:59`, `test_integration.jl:39`, `test_mogfn.jl:182` — skipped

Three different guard idioms are in use for the same condition (`isdefined(Main, :RDKitBridge)` at `test_diversity.jl:10`, `@isdefined(RDKitBridge)` at `test_fragment_joining.jl:8`, inline `@isdefined` at `test_integration.jl:27`), and `const _rdkit_available` is defined twice into `Main` (`test_diversity.jl:10`, `test_docking.jl:11`) — a const collision across files in one session. Magic constants are duplicated too: state dims `1042/1058/1106/1074/1122` at `test_state_features.jl:12-28`, `length(FRAGMENT_LIBRARY) == 50` at `test_fragment_joining.jl:63`, `51` at `test_integration.jl:15`.

**Files:**
- Create: `test/fixtures/molecular.jl`
- Modify: `test/applications/molecular/test_setup.jl`
- Modify: `test/applications/molecular/test_diversity.jl:10`, `test_docking.jl:11`, `test_fragment_joining.jl:8`, `test_reward_function.jl:8`, `test_integration.jl:27`, `test_mogfn.jl:131`

**Interfaces:**
- Produces: `RDKIT_AVAILABLE::Bool` and `EXPECTED_FRAGMENT_COUNT::Int` in the `MolecularFixture` module. Consumed by all six molecular test files.

- [ ] **Step 1: Prove the skips are silent**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -i "broken\|skip" | head`
Expected: skip counts appear with no indication that RDKit is the reason.

- [ ] **Step 2: Write the fixture**

Create `test/fixtures/molecular.jl`:

```julia
# test/fixtures/molecular.jl
# Single source of truth for molecular test preconditions.
#
# Pkg.test() must pass with no Python, no conda env and no network, so
# RDKit is opt-in: set GFLOWNET_TEST_RDKIT=true to load the bridge and
# run the assertions that need real chemistry.
module MolecularFixture

export RDKIT_AVAILABLE, EXPECTED_FRAGMENT_COUNT, rdkit_reason

const _REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

"""
Number of fragments the checked-in BRICS library is expected to contain.
Asserted by test_fragment_joining.jl and test_integration.jl; keep it here
so the two cannot drift apart.
"""
const EXPECTED_FRAGMENT_COUNT = 50

function _load_rdkit()
    get(ENV, "GFLOWNET_TEST_RDKIT", "false") == "true" || return (false, "GFLOWNET_TEST_RDKIT is not \"true\"")
    try
        Base.include(Main, joinpath(_REPO_ROOT, "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
        Main.RDKitBridge.init_rdkit!()
        return (true, "loaded")
    catch e
        return (false, "rdkit_bridge failed to load: $(sprint(showerror, e))")
    end
end

const _state = _load_rdkit()

"""Whether RDKit-backed assertions can run in this session."""
const RDKIT_AVAILABLE = _state[1]

"""Human-readable explanation of why RDKit is or is not available."""
rdkit_reason() = _state[2]

end # module
```

- [ ] **Step 3: Load the fixture from the molecular harness and report the reason**

In `test/applications/molecular/test_setup.jl`, after the existing `include` at line 16, append:

```julia
include(joinpath(@__DIR__, "..", "..", "fixtures", "molecular.jl"))
using .MolecularFixture: RDKIT_AVAILABLE, EXPECTED_FRAGMENT_COUNT, rdkit_reason

if RDKIT_AVAILABLE
    @info "Molecular tests: RDKit available — chemistry assertions will run"
else
    @warn "Molecular tests: RDKit assertions SKIPPED" reason=rdkit_reason()
end
```

The `@warn` is the point: the skip is now visible in the log instead of silent.

- [ ] **Step 4: Replace the three guard idioms with the one flag**

In each of `test_diversity.jl`, `test_docking.jl`, `test_fragment_joining.jl`, `test_reward_function.jl`, `test_integration.jl`, `test_mogfn.jl`: delete the local availability const or inline `@isdefined(RDKitBridge)` check and use `RDKIT_AVAILABLE` from the fixture instead. Concretely, `test_diversity.jl:10` currently reads:

```julia
const _rdkit_available = isdefined(Main, :RDKitBridge)
```

Delete that line; the file already includes `test_setup.jl`, which now exports `RDKIT_AVAILABLE`. Then change each guard site from `_rdkit_available` / `@isdefined(RDKitBridge)` to `RDKIT_AVAILABLE`. Do the same in `test_docking.jl:11`. This also removes the duplicate-const collision between those two files.

- [ ] **Step 5: De-duplicate the fragment-count constants**

In `test_fragment_joining.jl:63`, change:

```julia
    @test length(FRAGMENT_LIBRARY) == 50
```

to:

```julia
    @test length(FRAGMENT_LIBRARY) == EXPECTED_FRAGMENT_COUNT
```

In `test_integration.jl:15`, replace the hardcoded `51` with `EXPECTED_FRAGMENT_COUNT + 1` and add a trailing comment naming what the `+1` is (the terminate action). Read the surrounding lines first to confirm that is what `51` means; if it is not, leave the literal and note the discrepancy in the commit message.

- [ ] **Step 6: Verify the RDKit-free path still passes and now explains itself**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A2 "RDKit assertions SKIPPED"
julia --project=. -e 'using Pkg; Pkg.test()' > /dev/null 2>&1; echo "exit=$?"
```
Expected: the warning names `GFLOWNET_TEST_RDKIT is not "true"`, and `exit=0`.

- [ ] **Step 7: Verify the RDKit path actually runs assertions**

```bash
GFLOWNET_TEST_RDKIT=true julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "RDKit available|Test Summary"
```
Expected: `RDKit available` appears and the molecular testsets report real pass counts rather than skips. If RDKit fails to load, the warning will name the underlying error — that is the correct behavior, and the failure is a Phase 0 environment issue, not a test defect.

- [ ] **Step 8: Commit**

```bash
git add test/fixtures/molecular.jl test/applications/molecular/
git commit -m "test: one explicit RDKit gate for molecular tests

test_setup.jl never loaded rdkit_bridge.jl, so every RDKit assertion was
permanently @test_skip and nothing said why — test_reward_function.jl and
test_diversity.jl had zero real assertions. Adds test/fixtures/
molecular.jl with an opt-in GFLOWNET_TEST_RDKIT gate, collapses three
guard idioms into one flag, removes the duplicate _rdkit_available const
shared by test_diversity/test_docking, and centralises the fragment-count
constant."
```

### Task 8: Reclassify the six zero-assertion test files

Six of the executed files contain no `@test` at all — they are `println` scripts that pass unconditionally and only detect hard exceptions:

- `test/core/test_utilities.jl` (wired at `runtests.jl:16`) — actually a fixture defining the state/action interface at lines 5-6
- `test/core/policies/test_backward_policy.jl` (`runtests.jl:31`)
- `test/core/detailed_balance/test_detailed_balance_comprehensive.jl` (`runtests.jl:41`)
- `test/core/detailed_balance/test_detailed_balance_summary.jl` (`runtests.jl:42`)
- `test/applications/grid_world/test_grid_world_versions.jl` (`runtests.jl:58`)
- `test/objectives/detailed_balance/test_training.jl` (`runtests.jl:43`) — worse: lines 33-38 wrap the call in `try ... catch e; println("✗ Error in single step: $e") end`, so a failure prints and the file still passes

Also `test/core/test_core_interface.jl:52` and `:121` are `@test true  # Placeholder to keep test structure`, with comments at 43-45 and 119-120 explaining the real tests were deleted.

This task only moves the fixture and makes the silent-pass explicit. Converting the five scripts into real assertions requires knowing the intended behavior and is a Deferred owner decision.

**Files:**
- Create: `test/fixtures/simple_domain.jl` (moved from `test/core/test_utilities.jl`)
- Modify: `test/runtests.jl:15-17`
- Modify: `test/objectives/detailed_balance/test_training.jl:33-38`
- Modify: `test/core/test_core_interface.jl:52,121`

- [ ] **Step 1: Confirm the six files have no assertions**

```bash
for f in test/core/test_utilities.jl \
         test/core/policies/test_backward_policy.jl \
         test/core/detailed_balance/test_detailed_balance_comprehensive.jl \
         test/core/detailed_balance/test_detailed_balance_summary.jl \
         test/applications/grid_world/test_grid_world_versions.jl \
         test/objectives/detailed_balance/test_training.jl; do
  printf "%s @test=%s\n" "$f" "$(grep -c '@test' "$f")"
done
```
Expected: `@test=0` for all six.

- [ ] **Step 2: Move the mislabeled fixture**

```bash
mkdir -p test/fixtures
git mv test/core/test_utilities.jl test/fixtures/simple_domain.jl
```

Then find every file that relied on it being included by the runner:

```bash
grep -rn "test_utilities" test/
```

Each hit must be repointed to `include(joinpath(@__DIR__, "..", "fixtures", "simple_domain.jl"))` with the correct number of `".."` segments for that file's depth. If there are zero hits outside `runtests.jl`, the fixture was only ever loaded as a side effect of group ordering — note that in the commit message, because it means the files that depend on those definitions were relying on include order.

- [ ] **Step 3: Drop the fixture from `test_groups`**

In `test/runtests.jl`, delete lines 15-17:

```julia
    ("Core Utilities", [
        "core/test_utilities.jl"
    ]),
```

- [ ] **Step 4: Make the swallowed failure fail**

In `test/objectives/detailed_balance/test_training.jl`, read lines 25-45 first, then replace the try/catch at 33-38 so the error propagates. The file has no `using Test` today, so add it at the top, and assert on the loss:

```julia
using Test

@testset "detailed balance single step" begin
    loss = compute_trajectory_loss(model, trajectory, DETAILED_BALANCE)
    @test isfinite(loss)
    @test loss >= 0
end
```

Use the exact variable names already present in the file — read it before editing; do not invent a `model`/`trajectory` that the file does not construct.

- [ ] **Step 5: Turn the placeholders into explicit broken markers**

In `test/core/test_core_interface.jl`, replace the two `@test true  # Placeholder to keep test structure` lines (52 and 121) with:

```julia
    @test_broken false  # real assertions were deleted; see the comment above
```

`@test_broken` reports as "Broken" in the summary rather than a false green, and will flip to a failure if someone restores the behavior without updating the test.

- [ ] **Step 6: Verify**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20
julia --project=. -e 'using Pkg; Pkg.test()' > /dev/null 2>&1; echo "exit=$?"
```
Expected: `exit=0`, the summary shows 2 broken, and the detailed-balance testset shows real assertions. If the new assertions in step 4 fail, that is a genuine bug the try/catch was hiding — report it with the loss value; do not restore the catch.

- [ ] **Step 7: Commit**

```bash
git add -A test/
git commit -m "test: stop five files from passing without asserting anything

test/core/test_utilities.jl was a fixture with zero @test wired in as a
test group -> moved to test/fixtures/simple_domain.jl. objectives/
detailed_balance/test_training.jl:33-38 caught and printed failures so
the file always passed -> now asserts the loss is finite and
non-negative. test_core_interface.jl:52,121 '@test true # Placeholder'
-> @test_broken, so they report as broken instead of green."
```

---

## Phase 2 — Source graph coherence

`src/` (excluding the frontend) holds 48 `.jl` files in **two disjoint load graphs**: the package `module GFlowNet` (27 files via `src/GFlowNet.jl:33-102`) and a second Main-scope graph loaded by `src/utils/visualization/api/unified_server.jl:13-29` (13 files, including the entire merged molecular pipeline). Four root files belong to neither.

### Task 9: Delete the four orphan root source files

`src/types.jl` (5.8 KB), `src/rewards.jl` (4.9 KB), `src/flow_networks.jl` (7.5 KB), `src/directed_acyclic_graph.jl` (6.7 KB) are included by nothing and each contradicts the live implementation:

- `src/types.jl:211` — `GFlowNetModel` with a `dag` field vs the live `src/core/types.jl:154` with `initial_state`/`all_actions`. Same clash for `Trajectory` (26 vs 61), `ForwardPolicy` (166 vs 94), `BackwardPolicy` (179 vs 106), `FlowEstimator` (191 vs 118), `AbstractState`/`AbstractAction` (10/18 vs 23/35). Also duplicates `MoleculeData` (66), `DAGData` (81), `ExperimentData` (96).
- `src/flow_networks.jl` — competing `flow` (27 vs `core/flows.jl:304`), `edge_flow` (73 vs 366), `estimate_partition_function` (172 vs `partition_function` 418), `safe_model_call` (186 vs `core/policies.jl:773`), `sample_trajectory` (212 vs `core/interface.jl:380`); line 174 iterates `model.dag.terminal_states`, a field the live model no longer has.
- `src/directed_acyclic_graph.jl` — `create_dag` (35) builds the deleted `DirectedAcyclicGraph`; `get_next_states` (94) / `get_previous_states` (117) contradict `core/graphs.jl:173/216`.
- `src/rewards.jl` — a parallel reward framework whose `reward(state, env_data=Dict())` at 159 would collide with `core/interface.jl:644`; `_domain_specific_reward` (178) returns a hardcoded `1.0`.

One caveat: `docs/src/guide/examples.md:267` still tells users to `include("types.jl")`. Task 22 fixes the docs.

**Files:**
- Delete: `src/types.jl`, `src/rewards.jl`, `src/flow_networks.jl`, `src/directed_acyclic_graph.jl`

- [ ] **Step 1: Prove they are in no load graph**

```bash
for f in types rewards flow_networks directed_acyclic_graph; do
  printf "%s: %s include references\n" "$f" "$(grep -rn "include(\"$f.jl\")\|include(\"src/$f.jl\")" src/ test/ scripts/ experiments/ examples/ | wc -l | tr -d ' ')"
done
```
Expected: `0` for all four. If any is nonzero, stop and report — that file is load-bearing and this task's premise is wrong.

- [ ] **Step 2: Record the baseline**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/before.txt 2>&1; echo "exit=$?"`
Expected: `exit=0`. Keep `/tmp/before.txt` for comparison.

- [ ] **Step 3: Delete them**

```bash
git rm src/types.jl src/rewards.jl src/flow_networks.jl src/directed_acyclic_graph.jl
```

- [ ] **Step 4: Verify nothing changed**

```bash
julia --project=. -e 'using GFlowNet; println("loaded OK")'
julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/after.txt 2>&1; echo "exit=$?"
diff <(grep -E "Test Summary|Pass|Fail|Error|Broken" /tmp/before.txt) \
     <(grep -E "Test Summary|Pass|Fail|Error|Broken" /tmp/after.txt) && echo "IDENTICAL"
```
Expected: `loaded OK`, `exit=0`, and `IDENTICAL`. Deleting dead code must not move a single test count.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: delete four orphan source files that shadowed core types

src/{types,rewards,flow_networks,directed_acyclic_graph}.jl were included
by nothing and each redeclared live core types with contradictory fields
— types.jl:211 GFlowNetModel had a 'dag' field the live model at
core/types.jl:154 does not, and flow_networks.jl:174 iterated
model.dag.terminal_states. Test counts identical before and after."
```

### Task 10: Remove the 13 phantom exports

`src/GFlowNet.jl` exports 13 names that are defined nowhere in `src/`. The extension files define entirely different names:

| Exported at | Phantom name | What actually exists |
|---|---|---|
| `:317` | `ContinuousGFlowNet` | `ContinuousState:8`, `ContinuousAction:15`, `GaussianPolicy:20`, `create_gaussian_policy:30` in `extensions/continuous.jl` |
| `:318` | `continuous_sampling`, `continuous_flow_estimation` | `sample_continuous_action:73`, `continuous_action_log_prob:106` |
| `:321` | `mutual_information_reward`, `entropy_regularized_sampling` | `entropy_estimator:9`, `kl_divergence:34`, `mutual_information:62` in `extensions/information.jl` |
| `:322` | `information_bottleneck_objective` | — |
| `:325` | `NonAcyclicGFlowNet`, `cycle_breaking_sampling` | `CyclicFlowNetwork:10`, `create_cyclic_network:29`, `cyclic_trajectory_balance_loss:76`, `sample_cyclic_trajectory:124` in `extensions/non_acyclic.jl` |
| `:264` | `plot_dag_structure`, `plot_trajectory_analysis` | only `plot_training_progress:111` in `utils/visualization.jl` |
| `:265` | `generate_training_report`, `save_training_artifacts` | — |

Julia does not error on exporting an undefined name, so these are invisible until a user writes `GFlowNet.plot_dag_structure` and gets `UndefVarError`.

**Files:**
- Modify: `src/GFlowNet.jl:263-265`, `src/GFlowNet.jl:316-325`

- [ ] **Step 1: Write the failing test**

Create `test/core/test_exports.jl`:

```julia
using Test
using GFlowNet

@testset "every exported name is defined" begin
    undefined = Symbol[]
    for name in names(GFlowNet)
        name === :GFlowNet && continue
        isdefined(GFlowNet, name) || push!(undefined, name)
    end
    @test isempty(undefined)
    isempty(undefined) || @info "undefined exports" undefined
end
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `julia --project=. test/core/test_exports.jl`
Expected: FAIL, and the `@info` lists the 13 names above. If the list differs from the table, use the actual list — the table was derived by static grep and the runtime answer wins.

- [ ] **Step 3: Fix the visualization exports**

In `src/GFlowNet.jl`, replace lines 263-265:

```julia
# Visualization and reporting
export plot_training_progress, plot_dag_structure, plot_trajectory_analysis
export generate_training_report, save_training_artifacts
```

with:

```julia
# Visualization and reporting
export plot_training_progress
```

- [ ] **Step 4: Fix the extension exports**

Replace lines 316-325:

```julia
# Continuous state spaces
export ContinuousState, ContinuousGFlowNet
export continuous_sampling, continuous_flow_estimation

# Information-theoretic extensions
export mutual_information_reward, entropy_regularized_sampling
export information_bottleneck_objective

# Non-acyclic extensions (experimental)
export NonAcyclicGFlowNet, cycle_breaking_sampling
```

with the names the extension files actually define:

```julia
# Continuous state spaces (experimental — extensions/continuous.jl)
export ContinuousState, ContinuousAction, GaussianPolicy
export create_gaussian_policy, sample_continuous_action, continuous_action_log_prob

# Information-theoretic extensions (experimental — extensions/information.jl)
export entropy_estimator, kl_divergence, mutual_information

# Non-acyclic extensions (experimental — extensions/non_acyclic.jl)
export CyclicFlowNetwork, create_cyclic_network
export cyclic_trajectory_balance_loss, sample_cyclic_trajectory
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `julia --project=. test/core/test_exports.jl`
Expected: PASS.

- [ ] **Step 6: Wire the test into the suite**

In `test/runtests.jl`, add to the `"Core Interface"` group (currently lines 21-23) so it becomes:

```julia
    ("Core Interface", [
        "core/test_core_interface.jl",
        "core/test_exports.jl"
    ]),
```

- [ ] **Step 7: Verify the suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' > /dev/null 2>&1; echo "exit=$?"`
Expected: `exit=0`.

- [ ] **Step 8: Commit**

```bash
git add src/GFlowNet.jl test/core/test_exports.jl test/runtests.jl
git commit -m "fix: remove 13 exports with no definition, export what exists

GFlowNet.jl exported ContinuousGFlowNet, continuous_sampling,
continuous_flow_estimation, mutual_information_reward,
entropy_regularized_sampling, information_bottleneck_objective,
NonAcyclicGFlowNet, cycle_breaking_sampling, plot_dag_structure,
plot_trajectory_analysis, generate_training_report and
save_training_artifacts — none defined anywhere in src/. Julia does not
error on exporting an undefined name, so this only surfaced as
UndefVarError at the call site. Adds test_exports.jl to keep it fixed."
```

### Task 11: Stop double-compiling the logging and reporting modules

`src/utils/utils.jl:1` opens `module GFlowNetUtils` and includes `logging.jl:17`, `visualization/visualization.jl:20`, and `report.jl:23`. But `src/GFlowNet.jl:86` and `:88` include `utils/logging.jl` and `utils/report.jl` **again**, at the `GFlowNet` level. So `GFlowNetLogger` (`utils/logging.jl:9`) and `ReportData` (`utils/report.jl:17`) are each compiled into two modules, and `GFlowNetLogger`/`benchmark_sampling` are exported from both (`utils/utils.jl:6,11` and `GFlowNet.jl:255,196`). `utils/utils.jl:41` additionally defines its own `softmax`, shadowing the `NNlib.softmax` brought in at `GFlowNet.jl:20`.

**Files:**
- Modify: `src/utils/utils.jl:17,23`

- [ ] **Step 1: Prove the double compile**

Run:
```bash
julia --project=. -e 'using GFlowNet; println(GFlowNet.GFlowNetLogger); println(GFlowNet.GFlowNetUtils.GFlowNetLogger); println(GFlowNet.GFlowNetLogger === GFlowNet.GFlowNetUtils.GFlowNetLogger)'
```
Expected: two type names print and the comparison prints `false` — two distinct types with the same name.

- [ ] **Step 2: Remove the inner includes**

In `src/utils/utils.jl`, delete the `include("logging.jl")` at line 17 and the `include("report.jl")` at line 23. Keep the `include("visualization/visualization.jl")` at line 20 — `GFlowNet.jl:87` includes `utils/visualization.jl`, a *different* file, so that one is not a duplicate. Verify this before deleting:

```bash
grep -n "include" src/utils/utils.jl
grep -n "utils/logging.jl\|utils/report.jl\|utils/visualization.jl\|utils/utils.jl" src/GFlowNet.jl
```

- [ ] **Step 3: Remove the now-dangling re-exports**

`src/utils/utils.jl:6` exports `GFlowNetLogger` and `:11` exports `benchmark_sampling`, both of which this module no longer defines. Delete those two names from the `export` list at lines 5-13, leaving the names it still defines.

- [ ] **Step 4: Verify a single definition survives**

```bash
julia --project=. -e 'using GFlowNet; println(GFlowNet.GFlowNetLogger); println(isdefined(GFlowNet.GFlowNetUtils, :GFlowNetLogger))'
```
Expected: the type prints, then `false`.

- [ ] **Step 5: Verify the suite**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' > /dev/null 2>&1; echo "exit=$?"
julia --project=. test/core/test_exports.jl
```
Expected: `exit=0` and the export test passes.

- [ ] **Step 6: Commit**

```bash
git add src/utils/utils.jl
git commit -m "fix: stop compiling logging.jl and report.jl into two modules

utils/utils.jl:17,23 included logging.jl and report.jl inside module
GFlowNetUtils while GFlowNet.jl:86,88 included the same files at package
level, so GFlowNetLogger and ReportData each existed as two distinct
types with the same name, exported from both modules."
```

### Task 12: One source of truth for the reaction state dimension

`src/applications/molecular_generation.jl:127` is already correct — it reads `const STATE_DIM = compute_state_dim()  # 1042`, deriving from the function at line 115. The defect is the *reaction* dimension, which is a bare literal in two places on opposite sides of the module boundary:

- `src/core/interface.jl:866` — `state_dim::Int = 1049` as a keyword default of `create_reaction_gflownet` (line 864)
- `src/utils/visualization/domains/reaction_molecular.jl:21` — `const REACTION_STATE_DIM = 1049  # 1024 FP + 17 rxn one-hot + 8 scalars`

That comment is the whole problem: 1049 is not a magic number, it is `fp_dim + n_reactions + 8`. And `create_reaction_gflownet` **already takes both components as keyword arguments** — `n_reactions::Int = 17` at line 865 and `fp_dim::Int = 1024` at line 868. So today, passing `n_reactions = 20` silently produces a network whose input layer is sized for 17 reactions: `state_dim` stays 1049 while the real feature vector is 1052. It is a latent dimension-mismatch bug, not just a duplication.

A fourth, unrelated encoding of configuration exists at `src/training/checkpoint.jl:18-20` (a version code where `1=baseline, 2=BRICS, 3=MOGFN, 4=BRICS+MOGFN`). Leave it alone; it encodes feature flags, not a dimension.

**Files:**
- Modify: `src/core/interface.jl:864-872`
- Modify: `src/GFlowNet.jl` (export the new helper)
- Modify: `src/utils/visualization/domains/reaction_molecular.jl:21`
- Create: `test/applications/molecular/test_state_dim_consistency.jl`

**Interfaces:**
- Produces: `reaction_state_dim(; n_reactions = 17, fp_dim = 1024, n_scalar_features = 8) -> Int`, exported from `GFlowNet`. Consumed by `create_reaction_gflownet`'s keyword default and by `reaction_molecular.jl:21`.

- [ ] **Step 1: Confirm the current state of all four sites**

```bash
sed -n '115p;127p' src/applications/molecular_generation.jl
sed -n '864,872p' src/core/interface.jl
sed -n '21p' src/utils/visualization/domains/reaction_molecular.jl
sed -n '14,22p' src/training/checkpoint.jl
```
Expected: `STATE_DIM` is already a `compute_state_dim()` call; `interface.jl` has `n_reactions = 17`, `state_dim = 1049`, `fp_dim = 1024` as three independent keyword defaults; `reaction_molecular.jl:21` is the literal `1049` with the arithmetic in a comment.

- [ ] **Step 2: Write the failing test**

Create `test/applications/molecular/test_state_dim_consistency.jl`:

```julia
using Test
using GFlowNet
include(joinpath(@__DIR__, "test_setup.jl"))

@testset "state dimensions derive from their components" begin
    # The fragment dimension already has one source of truth.
    @test STATE_DIM == compute_state_dim()
    @test STATE_DIM > 0

    # The reaction dimension must track its components, not a literal.
    @test reaction_state_dim() == 1024 + 17 + 8
    @test reaction_state_dim(n_reactions = 20) == 1024 + 20 + 8
    @test reaction_state_dim(fp_dim = 2048) == 2048 + 17 + 8

    # The model must size its input layer from the same formula, so that
    # changing n_reactions cannot silently desync the network.
    model = create_reaction_gflownet(n_reactions = 20)
    first_layer = model.forward_policy.network.layers[1]
    @test first_layer.in_dims == reaction_state_dim(n_reactions = 20)
end
```

The last assertion reaches into the Lux chain. Confirm the accessor path against the real struct before relying on it:
`julia --project=. -e 'using GFlowNet; m = create_reaction_gflownet(); dump(typeof(m.forward_policy))'`
and adjust `.network.layers[1].in_dims` to whatever the actual field path is. If the input dimension is not reachable, assert on `reaction_state_dim` alone and say so in the commit message — do not delete the intent.

- [ ] **Step 3: Run it to confirm it fails**

Run: `julia --project=. test/applications/molecular/test_state_dim_consistency.jl`
Expected: FAIL — `UndefVarError: reaction_state_dim not defined`.

- [ ] **Step 4: Add the helper and derive the keyword default**

In `src/core/interface.jl`, immediately above `create_reaction_gflownet` (line 864), add:

```julia
"""
    reaction_state_dim(; n_reactions = 17, fp_dim = 1024, n_scalar_features = 8) → Int

Feature-vector width for reaction-based molecular states: a Morgan fingerprint,
a one-hot over reaction templates, and a block of scalar descriptors.

This is the only place the reaction state width is computed. `create_reaction_gflownet`
defaults to it, and the server-side `REACTION_STATE_DIM` calls it, so changing
`n_reactions` cannot desync the network's input layer from the real feature width.
"""
reaction_state_dim(; n_reactions::Int = 17, fp_dim::Int = 1024, n_scalar_features::Int = 8) =
    fp_dim + n_reactions + n_scalar_features
```

Then rewrite the keyword block at lines 864-872 so `state_dim` derives from the components. Julia evaluates keyword defaults left to right, so `n_reactions` and `fp_dim` must be declared before `state_dim`:

```julia
function create_reaction_gflownet(;
    n_reactions::Int = 17,
    fp_dim::Int = 1024,
    n_scalar_features::Int = 8,
    state_dim::Int = reaction_state_dim(; n_reactions, fp_dim, n_scalar_features),
    hidden_dim::Int = 256,
    learning_rate::Float64 = 0.001,
    initial_state::Union{Nothing, AbstractState} = nothing,
    all_actions::Union{Nothing, Vector{<:AbstractAction}} = nothing,
    rng = Random.default_rng()
)
```

Note `fp_dim` moved up from line 868 and `n_scalar_features` is new. The body at line 876 (`output_dim = n_reactions + 1 + fp_dim`) is unaffected.

- [ ] **Step 5: Export the helper**

In `src/GFlowNet.jl`, add `reaction_state_dim` to the export beside `create_reaction_gflownet` (line 284), so the line reads:

```julia
export create_reaction_gflownet, reaction_state_dim
```

- [ ] **Step 6: Derive the server-side constant**

In `src/utils/visualization/domains/reaction_molecular.jl`, replace line 21:

```julia
const REACTION_STATE_DIM = 1049  # 1024 FP + 17 rxn one-hot + 8 scalars
```

with:

```julia
# Derived from GFlowNet.reaction_state_dim so this cannot drift from the
# network input width in create_reaction_gflownet.
const REACTION_STATE_DIM = GFlowNet.reaction_state_dim()
```

Confirm `reaction_molecular.jl` can see `GFlowNet` — check its imports at the top of the file. It is loaded at Main scope by `unified_server.jl:26`, which has `using GFlowNet` at line 10, so the qualified call resolves; if the file has its own `using GFlowNet: ...` list, add `reaction_state_dim` to it and drop the qualification.

- [ ] **Step 7: Verify the test passes**

Run: `julia --project=. test/applications/molecular/test_state_dim_consistency.jl`
Expected: PASS, including the `n_reactions = 20` case that silently produced a mis-sized network before.

- [ ] **Step 8: Verify nothing else broke**

```bash
grep -rn "create_reaction_gflownet\|REACTION_STATE_DIM" src/ test/ scripts/ experiments/ examples/
julia --project=. -e 'using Pkg; Pkg.test()' > /dev/null 2>&1; echo "exit=$?"
julia --project=. -e 'push!(LOAD_PATH, joinpath(pwd(), "src")); include("src/utils/visualization/api/unified_server.jl"); println("server OK")'
```
Expected: `exit=0` and `server OK`. Any caller passing `state_dim` explicitly still works — the keyword was not made required, so this change is backward compatible.

- [ ] **Step 9: Wire the new test in**

Add `"applications/molecular/test_state_dim_consistency.jl"` to the `"Molecular Generation"` group in `test/runtests.jl` (the list at lines 63-74).

- [ ] **Step 10: Commit**

```bash
git add src/core/interface.jl src/GFlowNet.jl \
        src/utils/visualization/domains/reaction_molecular.jl \
        test/applications/molecular/test_state_dim_consistency.jl test/runtests.jl
git commit -m "fix: derive reaction state dim from its components

1049 appeared as a bare literal at core/interface.jl:866 and
reaction_molecular.jl:21, with the arithmetic (1024 FP + 17 rxn one-hot +
8 scalars) only in a comment — while create_reaction_gflownet already
took n_reactions and fp_dim as keywords. Passing n_reactions=20 therefore
built a network sized for 17, a silent dimension mismatch. Both now call
the new exported reaction_state_dim() helper."
```

### Task 13: Fix the missing imports in `molecular_generation.jl`

`src/applications/molecular_generation.jl:11-13` imports only `Random` and `GFlowNet`, but the file references `JSON3.read` at line 256 and `RDKitBridge` at line 201. It works only because `unified_server.jl:5` happens to have `using JSON3` at Main scope. Its two other consumers — `scripts/validate_all_gaps.jl:26` and `experiments/objective_comparison_drd2.jl:31` — do not, so `load_fragment_library` throws `UndefVarError` there.

**Files:**
- Modify: `src/applications/molecular_generation.jl:11-13`

- [ ] **Step 1: Reproduce the latent failure**

Run:
```bash
julia --project=. -e 'include("src/applications/molecular_generation.jl")' 2>&1 | tail -5
```
Expected: either an `UndefVarError: JSON3 not defined` at load time, or a clean load. If clean, the reference is inside a function body, so provoke it:
```bash
julia --project=. -e 'include("src/applications/molecular_generation.jl"); load_fragment_library()' 2>&1 | tail -5
```
Expected: `UndefVarError: JSON3 not defined`.

- [ ] **Step 2: Add the import**

In `src/applications/molecular_generation.jl`, add after the existing imports at lines 11-13:

```julia
using JSON3
```

`RDKitBridge` is a different case: it is a module defined by `src/utils/visualization/python/rdkit_bridge.jl`, which this file must not include (that would create a second copy alongside the server's). Leave the `RDKitBridge` reference at line 201 as-is, but make the dependency explicit and the failure legible. Replace the bare reference with a guarded lookup — read lines 195-210 first, then wrap the call so an absent bridge produces a named error:

```julia
    isdefined(Main, :RDKitBridge) || error(
        "molecular_generation.jl requires RDKitBridge to be loaded first: " *
        "include(\"src/utils/visualization/python/rdkit_bridge.jl\") before this file"
    )
```

- [ ] **Step 3: Verify the standalone path works**

```bash
julia --project=. -e 'include("src/applications/molecular_generation.jl"); println(length(load_fragment_library()))'
```
Expected: a fragment count prints (or a clear named error about the missing `data/fragment_libraries/` directory, which Task 16 addresses — not an `UndefVarError`).

- [ ] **Step 4: Verify both real consumers**

```bash
julia --project=. -e 'push!(LOAD_PATH, joinpath(pwd(), "src")); include("src/utils/visualization/api/unified_server.jl"); println("server OK")'
```
Expected: `server OK`. Do not run `scripts/validate_all_gaps.jl` or `experiments/objective_comparison_drd2.jl` — they need RDKit, the TDC network download, and the absent `data/` tree.

- [ ] **Step 5: Commit**

```bash
git add src/applications/molecular_generation.jl
git commit -m "fix: import JSON3 in molecular_generation.jl

Line 256 calls JSON3.read but :11-13 imported only Random and GFlowNet;
it resolved only because unified_server.jl:5 has 'using JSON3' at Main.
The file's other two consumers (scripts/validate_all_gaps.jl:26,
experiments/objective_comparison_drd2.jl:31) lack it, so
load_fragment_library threw UndefVarError there. Also made the
RDKitBridge prerequisite an explicit named error."
```

---

## Phase 3 — Server, frontend, and deployment contract

The backend registers 50 Oxygen routes. The frontend calls them through `src/utils/visualization/web/src/services/api.ts`. Three concrete breaks exist.

### Task 14: Fix the malformed Oxygen route parameter

`src/utils/visualization/api/unified_server.jl:1343` registers the path with a literal `:id` segment while all eight sibling routes use `{id}`. Oxygen extracts path params only from `{}`, so `id` falls through to the query-param branch and the real path becomes `/api/v2/molecular/molecules/:id/objectives?id=…`. The frontend calls the `{}` form at `api.ts:293`, so `molecularApi.getObjectives` 404s.

**Files:**
- Modify: `src/utils/visualization/api/unified_server.jl:1343`

- [ ] **Step 1: Confirm the inconsistency**

```bash
grep -n "molecules/" src/utils/visualization/api/unified_server.jl | grep -n "objectives\|admet\|attribution\|synthesis"
```
Expected: the `objectives` route uses `:id`; `admet` (line 877), `attribution` (890), `reward-decomposition` (914), `generation-dag` (946) and `synthesis` (1596) use `{id}`.

- [ ] **Step 2: Fix the route**

At `src/utils/visualization/api/unified_server.jl:1343`, change the path from the `:id` form to `{id}`, matching the siblings. Read the line first to preserve the exact macro and handler signature; only the path string changes.

- [ ] **Step 3: Verify by starting the server and calling the route**

```bash
julia --project=. start_server.jl &
SERVER_PID=$!
sleep 25
curl -s -o /dev/null -w "health=%{http_code}\n" http://localhost:8080/health
curl -s -o /dev/null -w "objectives=%{http_code}\n" http://localhost:8080/api/v2/molecular/molecules/1/objectives
kill $SERVER_PID
```
Expected: `health=200`, and `objectives` returns 200 or 404-with-JSON-body for an unknown molecule id — but **not** a routing 404 for an unmatched path. If the server fails to start, the error belongs to Phase 0/Task 2 — report it rather than working around it.

- [ ] **Step 4: Commit**

```bash
git add src/utils/visualization/api/unified_server.jl
git commit -m "fix: use {id} not :id in the molecule objectives route

unified_server.jl:1343 was the only route using Oxygen's non-parameter
:id syntax, so the path parameter was never extracted and the frontend
call at api.ts:293 could not match. All eight sibling molecule routes
already used {id}."
```

### Task 15: Report the real connection status instead of a hardcoded `true`

`src/utils/visualization/web/src/hooks/useWebSocket.ts:9-12` hardcodes `ws://localhost:8080/ws`, comments that HTTP polling is used instead, and sets `const [isConnected, setIsConnected] = useState(true) // Set to true to not block UI`. `connect()` is commented out at lines 71-73. No `@websocket` route exists in the backend at all. That fake `true` reaches the UI at `App.tsx:30` → `Sidebar.tsx:132,141` (green pulsing dot, "Connected") and `App.tsx:176` → `StatusBar.tsx:78-83` (Wifi icon, "Connected"). Both lie whenever the backend is down — which is the current state on `main`, and is exactly why the dashboard says "Connected" while every data panel is empty.

**Files:**
- Modify: `src/utils/visualization/web/src/hooks/useWebSocket.ts`
- Modify: `src/utils/visualization/web/src/App.tsx:30,176`

**Interfaces:**
- Produces: a `useBackendHealth()` hook returning `{ isConnected: boolean }` derived from a real `GET /health` poll. Consumed by `Sidebar.tsx` and `StatusBar.tsx` through the same props they use today, so their internals need no change.

- [ ] **Step 1: Observe the lie**

With no backend running, start the frontend and confirm the sidebar reads "Connected":

```bash
cd src/utils/visualization/web && npm run dev
```
Open `http://localhost:5173/` (not `127.0.0.1` — Vite binds IPv6 here) and confirm the sidebar shows a green "Connected" indicator while panels show "Failed to compute diversity".

- [ ] **Step 2: Replace the stub with a real health poll**

Replace the whole body of `src/utils/visualization/web/src/hooks/useWebSocket.ts` with a health-derived hook. Keep the file name so imports do not churn in this task, but rename the export:

```typescript
import { useQuery } from '@tanstack/react-query'
import axios from '../lib/axios'

export interface BackendHealth {
  isConnected: boolean
  lastCheckedAt: number | null
}

/**
 * Real connection status, derived from GET /health (unified_server.jl:342).
 * There is no WebSocket route in the backend; freshness comes from polling.
 */
export function useBackendHealth(intervalMs = 5000): BackendHealth {
  const { isSuccess, dataUpdatedAt } = useQuery({
    queryKey: ['backend-health'],
    queryFn: async () => (await axios.get('/health')).data,
    refetchInterval: intervalMs,
    retry: false,
  })

  return {
    isConnected: isSuccess,
    lastCheckedAt: dataUpdatedAt || null,
  }
}
```

Note the path is `/health`, not `/api/health`. `/api/health` has no backend route and no source anywhere in the repo — it is a stale request; `unified_server.jl:342` serves `/health` and `:350` serves `/api/v2/health`. The Vite proxy at `vite.config.ts:16` only forwards `/api`, so add `/health` to the proxy config:

```typescript
      '/health': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
```

- [ ] **Step 3: Update the two consumers**

In `src/utils/visualization/web/src/App.tsx`, replace the `useWebSocket` import and its use at lines 30 and 176 with `useBackendHealth`. Read those lines first; pass the resulting `isConnected` into `Sidebar` and `StatusBar` through the identical prop they already accept, so `Sidebar.tsx:132,141` and `StatusBar.tsx:78-83` need no edits.

- [ ] **Step 4: Verify it now tells the truth — backend down**

With no backend running, reload `http://localhost:5173/`.
Expected: the sidebar and status bar show a disconnected state, not "Connected".

- [ ] **Step 5: Verify it tells the truth — backend up**

```bash
julia --project=. start_server.jl &
sleep 25
```
Reload the page.
Expected: within one poll interval the indicator flips to "Connected". Stop the server again and confirm it flips back. Take a screenshot of both states for the commit.

- [ ] **Step 6: Verify the build**

```bash
cd src/utils/visualization/web && npx tsc --noEmit && npm run build
```
Expected: both succeed.

- [ ] **Step 7: Commit**

```bash
git add src/utils/visualization/web/src/hooks/useWebSocket.ts \
        src/utils/visualization/web/src/App.tsx \
        src/utils/visualization/web/vite.config.ts
git commit -m "fix: derive connection status from GET /health, not a constant

useWebSocket.ts:11 hardcoded isConnected=true 'to not block UI' with
connect() commented out at :71-73, and there is no @websocket route in
the backend at all. The fake value reached Sidebar.tsx:132,141 and
StatusBar.tsx:78-83, so the dashboard reported 'Connected' while every
data panel failed. Replaced with a real /health poll and proxied /health
in vite.config.ts."
```

### Task 16: Make the dev frontend URL 5173 everywhere, and create the runtime data directory

`3000` is advertised in six places while Vite serves 5173:

- `examples/core_features/visualization/show_visualization.jl:88,92,94,100,104` (line 88 actually runs `open http://localhost:3000`)
- `src/utils/visualization/api/unified_server.jl:1821` (startup banner)
- `src/utils/visualization/visualization.jl:56` (docstring)
- `docs/src/guide/visualization/README.md:80`
- `docs/src/guide/visualization/REAL_TRAINING_IMPLEMENTATION.md:166`
- `examples/core_features/visualization/README.md:22`
- `.claude/agents/gflownet-training-expert.md:655`

Separately, no `data/` directory exists, yet `unified_server.jl:1811` (`data/gflownet_molecules.db`), `:1167`/`:1207` (`data/fragment_libraries`), `:198`/`:1743` and `core/pmo_benchmark.jl:80` (`data/tdc_cache`) all resolve there. `core/database.jl:28` does `mkpath`, so the DB self-heals, but `GET /api/v2/molecular/fragments` has an `isdir` guard at line 1170 and always returns an empty library.

**Files:**
- Modify: the seven files listed above
- Create: `data/.gitkeep`
- Modify: `.gitignore` (`/data/` is already ignored at line 32 — add the `.gitkeep` exception)

- [ ] **Step 1: Enumerate every occurrence**

```bash
grep -rn "localhost:3000\|:3000" --include=*.jl --include=*.md --include=*.ts --include=*.tsx . | grep -v node_modules
```
Use the actual output as the work list; the list above was gathered by scouts and may be incomplete.

- [ ] **Step 2: Replace them all**

Change every `http://localhost:3000` to `http://localhost:5173`. In `examples/core_features/visualization/show_visualization.jl:88`, that is a live `run(\`open http://localhost:3000\`)` call — the port must change there or the script opens a dead page.

- [ ] **Step 3: Create the runtime data directory**

```bash
mkdir -p data
touch data/.gitkeep
```

In `.gitignore`, immediately after the existing `/data/` line (line 32), add:

```
!/data/.gitkeep
```

- [ ] **Step 4: Verify the directory is tracked but its contents are not**

```bash
git add data/.gitkeep && git status --short data/
git check-ignore -v data/gflownet_molecules.db
```
Expected: `A  data/.gitkeep`, and the `.db` path is reported as ignored.

- [ ] **Step 5: Verify no stale port remains**

```bash
grep -rn "localhost:3000" --include=*.jl --include=*.md --include=*.ts --include=*.tsx . | grep -v node_modules | wc -l
```
Expected: `0`.

- [ ] **Step 6: Verify the launcher opens the right page**

```bash
julia --project=. -e 'println(read("examples/core_features/visualization/show_visualization.jl", String) |> s -> count("5173", s))'
```
Expected: a nonzero count. Do not execute `show_visualization.jl` — it runs `pkill -f "node.*vite"` at lines 22-23 and would kill any dev server you have running.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "fix: point every dev-frontend URL at 5173 and create data/

vite.config.ts:14 serves 5173, but six files advertised 3000 —
show_visualization.jl:88 actually ran 'open http://localhost:3000'.
Also creates the data/ runtime directory that unified_server.jl:1811,
:1167, :1207, :198, :1743 and pmo_benchmark.jl:80 all resolve into; the
fragments route at :1170 has an isdir guard and silently returned an
empty library without it."
```

### Task 17: Make the Docker image match reality

`Dockerfile` has four defects, each independently fatal to the deployed molecular pipeline:

- Line 18 comment claims "Manifest.toml is gitignored, so resolve from scratch" — true only after Task 1; before that it was a false statement about a tracked file.
- **`CondaPkg.toml` is never copied.** Only `Project.toml`, `src/`, and `start_server.jl` are. PythonCall therefore resolves an empty conda env: no RDKit, numpy, scikit-learn, or PyTDC. `RDKitBridge.init_rdkit!()` at `unified_server.jl:36` fails, `RDKIT_AVAILABLE[]` becomes `false`, and every molecular route degrades. The image can only serve grid_world.
- Line 38's `CMD` re-implements `start_server.jl` inline (adding `host="0.0.0.0"` and the `PORT` env var), so two launch paths must be kept in sync by hand.
- `COPY src/ src/` at line 25 drags the whole frontend into the image, including the 6.7 MB `web/public/RDKit_minimal.wasm` and ~5.6 MB of `web/templates/theme_examples/*.png`. `.dockerignore` only excludes `node_modules/` and `dist/`.

**Files:**
- Modify: `Dockerfile:18-19`, `:24-26`, `:38`
- Modify: `.dockerignore`
- Modify: `start_server.jl`

- [ ] **Step 1: Confirm the conda env is empty in the current image**

```bash
docker build -t gflownet-check . && \
docker run --rm gflownet-check julia --project=. -e 'using PythonCall; println(pyimport("sys").version)'
```
Expected: failure or a bare Python with no `rdkit`. Confirm with:
```bash
docker run --rm gflownet-check julia --project=. -e 'using PythonCall; pyimport("rdkit")'
```
Expected: `ModuleNotFoundError: No module named 'rdkit'`.

If Docker is unavailable on this machine, skip to step 2 and note in the commit that the before-state was established by static reading of `Dockerfile:19` (no `COPY CondaPkg.toml`).

- [ ] **Step 2: Make `start_server.jl` the single launch path**

Replace `start_server.jl` with a version that honors the environment, so the Dockerfile can just call it:

```julia
# start_server.jl — the single entry point for the Oxygen backend.
# Honors PORT and HOST so the container CMD does not need to re-implement it.
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
include(joinpath(@__DIR__, "src", "utils", "visualization", "api", "unified_server.jl"))

port = parse(Int, get(ENV, "PORT", "8080"))
host = get(ENV, "HOST", "127.0.0.1")
start_real_training_server(port = port, host = host)
```

The default host stays `127.0.0.1` for local use; the container sets `HOST=0.0.0.0`.

- [ ] **Step 3: Fix the Dockerfile**

Replace lines 18-19:

```dockerfile
# Copy Project.toml (Manifest.toml is gitignored, so resolve from scratch)
COPY Project.toml ./
```

with:

```dockerfile
# Manifest.toml is untracked by policy (.gitignore), so resolve from scratch.
# CondaPkg.toml is REQUIRED: without it PythonCall builds an empty conda env
# and every molecular route degrades to unavailable.
COPY Project.toml CondaPkg.toml ./
```

Replace line 38's inline `CMD` with:

```dockerfile
ENV HOST=0.0.0.0
CMD ["julia", "--project=.", "start_server.jl"]
```

- [ ] **Step 4: Keep the frontend out of the image**

Add to `.dockerignore`:

```
src/utils/visualization/web/
```

The image serves only the Julia API — `railway.toml:5` healthchecks `/health` and nothing in the image serves static files. If Task 2 of the Deferred section (frontend deployment) later changes that, this line comes back out.

- [ ] **Step 5: Verify the image builds and RDKit is present**

```bash
docker build -t gflownet-check . && \
docker run --rm gflownet-check julia --project=. -e 'using PythonCall; println(pyimport("rdkit").__version__)'
```
Expected: an RDKit version prints. This build is slow — CondaPkg downloads roughly 1 GB.

- [ ] **Step 6: Verify the container serves health**

```bash
docker run -d -p 8081:8080 --name gflownet-smoke gflownet-check
sleep 40
curl -s -o /dev/null -w "health=%{http_code}\n" http://localhost:8081/health
docker logs gflownet-smoke | tail -20
docker rm -f gflownet-smoke
```
Expected: `health=200`, and the logs show RDKit initializing successfully rather than the `RDKitBridge initialization failed` warning from `unified_server.jl:41`.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore start_server.jl
git commit -m "fix: copy CondaPkg.toml into the image and unify the launch path

The image never copied CondaPkg.toml, so PythonCall built an empty conda
env: no RDKit/numpy/scikit-learn/PyTDC, init_rdkit! failed at
unified_server.jl:36, RDKIT_AVAILABLE stayed false and the whole merged
molecular pipeline was silently dead in deployment. Also replaced the
inline CMD that re-implemented start_server.jl, and excluded the 12 MB
frontend asset tree from the image."
```

---

## Phase 4 — CI

Nothing enforces phases 0-3. Add the frontend job first: it is independently greenable today, so it proves the workflow plumbing before the harder Julia job.

### Task 18: Frontend CI job

**Files:**
- Create: `.github/workflows/frontend.yml`

- [ ] **Step 1: Verify the commands pass locally**

```bash
cd src/utils/visualization/web
npm ci
npx tsc --noEmit
npm run build
```
Expected: all three succeed. `tsconfig.json:19-20` sets `noUnusedLocals`/`noUnusedParameters` to `false`, so unused imports will not fail the build.

- [ ] **Step 2: Write the workflow**

Create `.github/workflows/frontend.yml`:

```yaml
name: Frontend

on:
  push:
    branches: [main]
  pull_request:

defaults:
  run:
    working-directory: src/utils/visualization/web

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1  # the pack is ~91 MB; never fetch full history
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: npm
          cache-dependency-path: src/utils/visualization/web/package-lock.json
      - run: npm ci
      - name: Typecheck
        run: npx tsc --noEmit
      - name: Build
        run: npm run build
```

- [ ] **Step 3: Verify the workflow is valid YAML and references real paths**

```bash
python3 -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/frontend.yml')); print('jobs:', list(d['jobs'])); print('wd:', d['defaults']['run']['working-directory'])"
test -f src/utils/visualization/web/package-lock.json && echo "lockfile present"
```
Expected: `jobs: ['build']`, the working directory prints, `lockfile present`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/frontend.yml
git commit -m "ci: add frontend typecheck and build job

Vite 5 + React 18, 44 .tsx / 16k LOC with no CI. Separate job from Julia
because they share no toolchain. fetch-depth 1 because the pack is 91 MB."
```

### Task 19: Julia CI job

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Verify the test suite is green locally first**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' > /dev/null 2>&1; echo "exit=$?"`
Expected: `exit=0`. Do not add CI over a red suite — that is what Tasks 4-8 were for.

- [ ] **Step 2: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    name: Julia ${{ matrix.julia-version }} / ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        julia-version: ['1.11', '1.12']
        os: [ubuntu-latest]
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/cache@v2
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
        env:
          # RDKit and TDC are opt-in: the conda build costs minutes and the
          # TDC oracle download needs network egress. See test/fixtures/molecular.jl.
          GFLOWNET_TEST_RDKIT: 'false'
          GFLOWNET_TEST_TDC: 'false'

  docs:
    name: Documenter
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.11'
      - uses: julia-actions/cache@v2
      - name: Install docs environment
        run: |
          julia --project=docs -e '
            using Pkg
            Pkg.develop(PackageSpec(path = pwd()))
            Pkg.instantiate()'
      - name: Build docs
        run: julia --project=docs docs/make.jl
```

The `docs` job will fail until Task 20 fixes `docs/make.jl`. That is intentional and correct ordering: land the workflow, then make it green.

- [ ] **Step 3: Verify the docs job command reproduces locally**

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl 2>&1 | tail -20
```
Expected: it fails on the missing `internals/*.md` pages. Record the exact error — Task 20 fixes precisely that.

- [ ] **Step 4: Verify the YAML**

```bash
python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/ci.yml')); print(list(d['jobs'])); print(d['jobs']['test']['strategy']['matrix'])"
```
Expected: `['test', 'docs']` and the matrix prints with both Julia versions.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add Julia test matrix and Documenter build

Matrix is 1.11/1.12 — the only versions this code has been resolved
against (local 1.11.6, Manifest/Dockerfile 1.12). RDKit and TDC gates
default to false so the suite needs no conda env and no network. The docs
job is red until docs/make.jl's four missing page paths are fixed."
```

---

## Phase 5 — Make the documentation true

The Documenter build is broken and has been for a long time; with no CI, nobody noticed. Separately, `README.md`, `CLAUDE.md`, and the entire `docs/` tree contain **zero** mentions of PMO, RDKit, PythonCall, CondaPkg, SQLite, or TDC — the merged feature set is invisible to anyone following the documented entry points.

### Task 20: Fix the four broken page paths in `docs/make.jl`

`docs/make.jl:54-57` references four pages that do not exist. The real files were reorganized into subdirectories:

| `make.jl` line | Referenced | Actual path |
|---|---|---|
| 54 | `internals/architecture.md` | `internals/architecture/architecture.md` |
| 55 | `internals/design_decisions.md` | `internals/architecture/design_decisions.md` |
| 56 | `internals/known_limitations.md` | `internals/development_guides/known_limitations.md` |
| 57 | `internals/flow_functions_multistart.md` | `internals/implementation_notes/flow_functions_multistart.md` |

**Files:**
- Modify: `docs/make.jl:53-58`

- [ ] **Step 1: Reproduce the build failure**

```bash
julia --project=docs docs/make.jl 2>&1 | tail -15
```
Expected: an error naming the missing `internals/` pages.

- [ ] **Step 2: Confirm the real paths**

```bash
ls docs/src/internals/architecture/ docs/src/internals/development_guides/ docs/src/internals/implementation_notes/
```
Expected: the four files appear at the paths in the table. Use the actual listing if it differs.

- [ ] **Step 3: Repoint the four entries**

In `docs/make.jl`, replace lines 53-58:

```julia
        "Internals" => [
            "internals/architecture.md",
            "internals/design_decisions.md", 
            "internals/known_limitations.md",
            "internals/flow_functions_multistart.md"
        ],
```

with:

```julia
        "Internals" => [
            "internals/architecture/architecture.md",
            "internals/architecture/design_decisions.md",
            "internals/development_guides/known_limitations.md",
            "internals/implementation_notes/flow_functions_multistart.md"
        ],
```

- [ ] **Step 4: Verify the build completes**

```bash
julia --project=docs docs/make.jl 2>&1 | tail -15
```
Expected: the build completes. Warnings about missing docstrings and cross-references are expected — `make.jl:14` sets `warnonly = [:missing_docs, :cross_references]`. Errors are not.

- [ ] **Step 5: Verify output landed**

```bash
test -f docbuild/index.html && echo "docs built"
```
Expected: `docs built`. Note `make.jl:11` sets `build = "../docbuild"`, writing outside `docs/`; `docbuild/` is already gitignored at `.gitignore:11`.

- [ ] **Step 6: Commit**

```bash
git add docs/make.jl
git commit -m "docs: repoint four internals pages at their real paths

docs/make.jl:54-57 referenced internals/{architecture,design_decisions,
known_limitations,flow_functions_multistart}.md, but those files were
reorganized into internals/architecture/, internals/development_guides/
and internals/implementation_notes/. makedocs errored on the missing
pages, so the docs build has been broken with no CI to catch it."
```

### Task 21: Fix the false and stale claims in the top-level docs

Concrete falsehoods, each verified:

- `README.md:257-260` — install says only `Pkg.add(url=...)`, omitting `Pkg.instantiate()` and the CondaPkg/RDKit prerequisite now implied by `Project.toml`. Following it cannot produce a working environment.
- `README.md:67,188` — lists `applications/feature_acquisition.jl`, which does not exist; `:55,:174` show `training/` as one file when there are eight.
- `README.md:331` — `cd examples/molecular_design`; the directory is `examples/molecule_design`.
- `README.md:6` — claims Julia 1.9+; `Project.toml` now says 1.11 (Task 1).
- `README.md:5,78,233,339,415,438` — six hardcoded `1.0.0`; `:259,:393,:417` say `yourusername`; `:414` says `author={Your Name}`.
- `CLAUDE.md:77-80` — "**Backend**: Mock simulation… Real GFlowNet training integration NOT yet implemented", contradicted by `CLAUDE.md:733` in the same file. `:80` links `docs/src/internals/development_guides/real_training_visualization_plan.md`, which does not exist.
- `CLAUDE.md:788` — points at `src/utils/visualization/CHANGELOG.md`; the real file is `docs/src/guide/visualization/CHANGELOG.md`.
- `CLAUDE.md:730-753` — the file-structure block omits `core/{pmo_benchmark,oracle_manager,database,metrics}.jl`, `domains/{molecular,reaction_molecular}.jl`, and all of `python/`.
- `CLAUDE.md:8` — tells the reader to check `.claude/sessions/`, which is gitignored and absent from a fresh clone.
- `docs/src/guide/examples.md:267` — tells users to `include("types.jl")`, deleted in Task 9.
- `docs/make.jl:8` and `docs/src/index.md:19` — `yourusername` placeholder canonical URL.

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/src/index.md`, `docs/src/guide/examples.md`, `docs/make.jl:8`

- [ ] **Step 1: Verify each claim is still false**

```bash
grep -n "feature_acquisition.jl" README.md
grep -n "examples/molecular_design" README.md
grep -rn "yourusername\|Your Name" README.md docs/make.jl docs/src/index.md
grep -n "NOT yet implemented" CLAUDE.md
test -f docs/src/internals/development_guides/real_training_visualization_plan.md || echo "CLAUDE.md:80 target missing (confirmed)"
test -f src/utils/visualization/CHANGELOG.md || echo "CLAUDE.md:788 target missing (confirmed)"
grep -n 'include("types.jl")' docs/src/guide/examples.md
```

- [ ] **Step 2: Fix the install instructions**

Replace the install block at `README.md:255-260` with something that actually works:

````markdown
## Installation

```julia
using Pkg
Pkg.develop(url = "https://github.com/danielchen26/Gflownet.git")
Pkg.instantiate()
```

Requires Julia 1.11 or newer.

The molecular-design pipeline additionally needs RDKit, which arrives
through `PythonCall` + `CondaPkg`. `Pkg.instantiate()` builds that conda
environment automatically from `CondaPkg.toml` (roughly 1 GB, one time).
To verify:

```julia
using PythonCall
pyimport("rdkit").__version__
```
````

Replace `danielchen26/Gflownet` with the real canonical URL if it differs — that is the remote this repo is cloned from.

- [ ] **Step 3: Fix the source-tree listings**

In `README.md`, correct the two module trees at `:46-71` and `:164-193`: delete `applications/feature_acquisition.jl`, list the eight real files under `training/`, and add the merged files — `src/applications/molecular_generation.jl`, `src/training/checkpoint.jl`, and the `src/utils/visualization/{core,domains,python}/` sets. Get the real listing from:

```bash
find src -name '*.jl' -not -path '*/web/*' | sort
```

Do the same for the `CLAUDE.md:730-753` block.

- [ ] **Step 4: Fix the outright contradictions**

In `CLAUDE.md`, delete the "Mock simulation / Real GFlowNet training integration NOT yet implemented" claim at `:77-80` — `:733` in the same file already describes `unified_server.jl` as doing real training, and Phase 3 verified the server boots and serves 50 routes. Repoint `:788` to `docs/src/guide/visualization/CHANGELOG.md`. Remove the `:80` link to the nonexistent plan document. At `:8`, replace the instruction to read the gitignored `.claude/sessions/` with a pointer to `CONTRIBUTING.md` (Task 22).

- [ ] **Step 5: Fix the remaining path and placeholder errors**

- `README.md:331` — `examples/molecular_design` → `examples/molecule_design`
- `README.md:6` — Julia 1.9+ → Julia 1.11+
- `docs/src/guide/examples.md:267` — remove the `include("types.jl")` instruction; the file is gone as of Task 9
- `docs/make.jl:8`, `docs/src/index.md:19`, `README.md:259,393,417,414` — replace `yourusername` / `Your Name` with the real GitHub org/user and author. If the canonical public URL is not known, leave a single `TODO(owner):` marker in **one** place rather than eight scattered placeholders, and say so in the commit message.

- [ ] **Step 6: Document the merged feature set**

`README.md`, `CLAUDE.md`, and all of `docs/` currently contain zero mentions of PMO, RDKit, PythonCall, CondaPkg, SQLite, or TDC. Add a section to `README.md` covering: fragment-based molecular generation (`src/applications/molecular_generation.jl`), the PMO 23-task benchmark (`src/utils/visualization/core/pmo_benchmark.jl`), TDC oracles (`src/utils/visualization/python/oracle_bridge.jl`), and SQLite persistence (`src/utils/visualization/core/database.jl`). Link `reports/2026-03-01_molecular_generation_benchmark_report.md`, which is currently the only document describing any of it.

- [ ] **Step 7: Verify**

```bash
grep -c "yourusername\|Your Name" README.md docs/make.jl docs/src/index.md
grep -c "feature_acquisition.jl\|examples/molecular_design\|NOT yet implemented" README.md CLAUDE.md
grep -rn "RDKit\|PythonCall\|PMO" README.md | head -5
julia --project=docs docs/make.jl 2>&1 | tail -5
```
Expected: zero placeholder and zero stale-claim hits, RDKit/PythonCall/PMO now appear in the README, and the docs build still completes.

- [ ] **Step 8: Commit**

```bash
git add README.md CLAUDE.md docs/
git commit -m "docs: correct install steps, dead paths and false backend claims

README's install path could not produce a working environment (no
Pkg.instantiate, no CondaPkg/RDKit note). Fixed dead paths
(applications/feature_acquisition.jl, examples/molecular_design,
include(types.jl)), the Julia version claim, and CLAUDE.md:77-80 which
said the viz backend was mock while :733 said the opposite. Documented
the merged molecular/PMO/TDC/SQLite pipeline, previously mentioned
nowhere outside reports/."
```

### Task 22: Add `CHANGELOG.md` and `CONTRIBUTING.md`

`RELEASE_NOTES.md` has a single entry — v1.0.0 from 2025-07-28 — describing itself at `:5` as the "first major release", with a roadmap at `:113-120` still promising the molecular design work that has since shipped. There is no `CONTRIBUTING.md`; the only contributor guidance is a short section at `README.md:381-399`.

**Files:**
- Create: `CHANGELOG.md`
- Create: `CONTRIBUTING.md`
- Delete: `RELEASE_NOTES.md`
- Modify: `README.md` (link the two new files, drop the inlined contributor section)

- [ ] **Step 1: Write the changelog**

Create `CHANGELOG.md` in Keep-a-Changelog form, carrying the v1.0.0 content forward from `RELEASE_NOTES.md` and adding the merge:

```markdown
# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Fragment-based molecular generation pipeline (`src/applications/molecular_generation.jl`)
- PMO 23-task benchmark runner (`src/utils/visualization/core/pmo_benchmark.jl`)
- TDC oracle bridge with budget tracking (`src/utils/visualization/python/oracle_bridge.jl`)
- RDKit bridge via PythonCall (`src/utils/visualization/python/rdkit_bridge.jl`)
- SQLite persistence for generated molecules (`src/utils/visualization/core/database.jl`)
- Reaction-constrained molecular domain (`src/utils/visualization/domains/reaction_molecular.jl`)
- Training checkpointing (`src/training/checkpoint.jl`)
- Domain-agnostic dashboard redesign with molecular UI (`src/utils/visualization/web/`)
- CI: Julia test matrix, Documenter build, frontend typecheck and build

### Fixed
- `Manifest.toml` was tracked while resolved on a different Julia version, making the project unloadable
- `UUIDs` was used by the visualization server but never declared as a dependency
- `Pkg.test()` aborted on a supply-chain test whose types had been deleted, so no molecular test ever ran
- 13 exported names had no definition anywhere in `src/`
- Docker image never copied `CondaPkg.toml`, leaving the deployed molecular pipeline non-functional
- Frontend reported "Connected" from a hardcoded constant regardless of backend state

### Removed
- Editor tooling (`LanguageServer`, `SymbolServer`, `IJulia`, `BenchmarkTools`) from runtime dependencies
- Four orphan source files that redeclared live core types with contradictory fields

## [1.0.0] - 2025-07-28

First tagged release. See git history for detail; this entry was migrated
from the former `RELEASE_NOTES.md`.
```

Migrate the substantive v1.0.0 bullets from `RELEASE_NOTES.md` into that section rather than leaving the placeholder sentence.

- [ ] **Step 2: Write the contributor guide**

Create `CONTRIBUTING.md` documenting the workflow this plan established:

````markdown
# Contributing

## Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Julia 1.11 or newer. `Pkg.instantiate()` also builds the conda environment
declared in `CondaPkg.toml` (RDKit, numpy, scikit-learn) — roughly 1 GB, once.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The default suite requires no Python, no conda environment, and no network.
Chemistry- and oracle-dependent assertions are opt-in:

```bash
GFLOWNET_TEST_RDKIT=true julia --project=. -e 'using Pkg; Pkg.test()'
GFLOWNET_TEST_TDC=true  julia --project=. -e 'using Pkg; Pkg.test()'   # downloads TDC oracles
```

New test files must be registered in the `test_groups` table in
`test/runtests.jl`. A file that is not listed there does not run — 30 files
were orphaned that way. Shared molecular preconditions and constants belong
in `test/fixtures/molecular.jl`, not duplicated per file.

## Running the app

Backend (port 8080):

```bash
julia --project=. start_server.jl
```

Frontend (port 5173):

```bash
cd src/utils/visualization/web && npm install && npm run dev
```

Open `http://localhost:5173`. Vite binds IPv6 on macOS, so use `localhost`,
not `127.0.0.1`.

## Docs

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Every page must be listed in the `pages=` structure of `docs/make.jl`;
`makedocs` errors on a listed page that does not exist.

## Conventions

- Conventional Commits: `fix:`, `feat:`, `chore:`, `test:`, `docs:`, `ci:`
- `Manifest.toml` is untracked by policy; pin versions through `[compat]`
- Never hardcode a state dimension — derive it from `compute_state_dim()` (fragment) or `reaction_state_dim()` (reaction)
- Export only names that exist; `test/core/test_exports.jl` enforces this
````

- [ ] **Step 3: Retire `RELEASE_NOTES.md`**

```bash
git rm RELEASE_NOTES.md
```

In `README.md`, replace the inlined contributor section at `:381-399` with a link to `CONTRIBUTING.md`, and link `CHANGELOG.md` wherever `RELEASE_NOTES.md` was referenced. Find those references:

```bash
grep -rn "RELEASE_NOTES" --include=*.md . | grep -v node_modules
```

- [ ] **Step 4: Verify no dangling references**

```bash
grep -rn "RELEASE_NOTES" --include=*.md . | grep -v node_modules | wc -l
test -f CHANGELOG.md && test -f CONTRIBUTING.md && echo "both present"
```
Expected: `0` and `both present`.

- [ ] **Step 5: Verify the documented commands actually work**

Run each command block in `CONTRIBUTING.md` and confirm it does what the document claims. A contributor guide with a command that fails is worse than none.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md CONTRIBUTING.md README.md
git commit -m "docs: add CHANGELOG and CONTRIBUTING, retire RELEASE_NOTES

RELEASE_NOTES.md had one entry (v1.0.0, 2025-07-28) whose roadmap still
listed molecular design as future work, and there was no contributor
guide at all. CONTRIBUTING documents the instantiate/test/serve/docs
commands verified in this plan, including the GFLOWNET_TEST_RDKIT and
GFLOWNET_TEST_TDC gates and the runtests.jl registration requirement."
```

---

## Phase 6 — Hygiene

### Task 23: Fix the fabricated GFlowNet UUIDs in the example environments

Five example environments declare a GFlowNet UUID that is not this package's. The real one is `2d7ca041-c8ad-46e9-a25d-4e8f55c0c8f5`. These environments can never instantiate:

| File | Line | Bogus UUID |
|---|---|---|
| `examples/core_features/direct_flow/Project.toml` | 2 | `3a72fb2e-1f81-4654-998d-6d0e9e1eeb04` |
| `examples/core_features/flow_matching/Project.toml` | 2 | `c1b459ff-ab72-4622-9809-60a0a5e9c1c7` |
| `examples/core_features/multi_start/Project.toml` | 2 | `c1b459ff-ab72-4622-9809-60a0a5e9c1c7` |
| `examples/core_features/objective_comparison/Project.toml` | 4 | `713c75ef-2927-5cde-a4cc-cb30f9955ef3` |
| `examples/core_features/sub_trajectory_balance/Project.toml` | 2 | `29eb5d72-a7cd-416f-93a8-881ea6daebce` |

A sixth, `examples/core_features/visualization/Project.toml`, lists only `Oxygen`/`HTTP`/`JSON3` and no GFlowNet at all — yet `show_visualization.jl:31` and all three `demo_*.jl` files do `using GFlowNet` under it.

`examples/setup_examples.jl:8-14` omits all seven `core_features` environments, so it never repairs any of this.

**Files:**
- Modify: the six `Project.toml` files above
- Modify: `examples/setup_examples.jl:8-14,82`

- [ ] **Step 1: Confirm the real UUID and the bogus ones**

```bash
grep '^uuid' Project.toml
grep -rn 'GFlowNet = "' examples/*/Project.toml examples/core_features/*/Project.toml
```
Expected: root uuid is `2d7ca041-c8ad-46e9-a25d-4e8f55c0c8f5`; five example files disagree.

- [ ] **Step 2: Reproduce one failure**

```bash
cd examples/core_features/direct_flow && julia --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tail -5; cd -
```
Expected: a resolver error about an unsatisfiable or unknown `GFlowNet`.

- [ ] **Step 3: Correct all five UUIDs**

Set the value to `2d7ca041-c8ad-46e9-a25d-4e8f55c0c8f5` in each of the five files.

- [ ] **Step 4: Add the missing dep to the visualization example**

In `examples/core_features/visualization/Project.toml`, add to `[deps]`:

```toml
GFlowNet = "2d7ca041-c8ad-46e9-a25d-4e8f55c0c8f5"
```

- [ ] **Step 5: Make `setup_examples.jl` cover every environment**

`examples/setup_examples.jl:8-14` hardcodes a partial directory list, which is why the bogus UUIDs survived. Replace that list with a discovery walk over every directory under `examples/` containing a `Project.toml`. Also fix line 82, which maps `feature_acquisition` to `feature_acquisition.jl`; the real entry point is `examples/feature_acquisition/main.jl`.

- [ ] **Step 6: Verify each environment instantiates**

```bash
for d in examples/core_features/direct_flow examples/core_features/flow_matching \
         examples/core_features/multi_start examples/core_features/objective_comparison \
         examples/core_features/sub_trajectory_balance examples/core_features/visualization; do
  printf "=== %s " "$d"
  (cd "$d" && julia --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path=joinpath(pwd(),"..","..",".."))); Pkg.instantiate()' > /dev/null 2>&1 && echo OK || echo FAIL)
done
```
Expected: `OK` for all six. Each needs the root package `develop`ed by path since it is unregistered — `setup_examples.jl:53` already does this, which is why step 5 matters.

- [ ] **Step 7: Commit**

```bash
git add examples/
git commit -m "fix: correct fabricated GFlowNet UUIDs in five example envs

direct_flow, flow_matching, multi_start, objective_comparison and
sub_trajectory_balance each declared a GFlowNet UUID that is not this
package's (real: 2d7ca041-c8ad-46e9-a25d-4e8f55c0c8f5), so none could
ever instantiate; visualization/Project.toml omitted GFlowNet entirely
while its four scripts do 'using GFlowNet'. setup_examples.jl:8-14
hardcoded a partial directory list and never touched core_features,
which is how this survived — it now discovers every env."
```

### Task 24: Delete accidental binaries and stop tracking regenerable output

Committed artifacts with no generator, or fully regenerable:

- `examples/core_features/visualization/fixthis.png` — 824 KB screenshot, no generator anywhere in the repo
- `examples/core_features/visualization/WechatIMG7635.jpg` — 408 KB screenshot, no generator
- `examples/grid_world/results/` — 10 files, 965 KB, timestamped 2025-07-28, fully regenerable by `examples/grid_world/grid_world.jl:258,276,279`
- `src/utils/visualization/web/VISUALIZATION_FIXES.md` — stale changelog whose "Files Modified" list uses paths that do not exist

**Files:**
- Delete: the four items above
- Modify: `.gitignore`

- [ ] **Step 1: Confirm no generator exists for the two screenshots**

```bash
grep -rn "fixthis\|WechatIMG" --include=*.jl --include=*.md --include=*.ts --include=*.tsx . | grep -v node_modules
```
Expected: no output (nothing references them).

- [ ] **Step 2: Confirm the grid-world results are regenerable**

```bash
sed -n '250,285p' examples/grid_world/grid_world.jl
```
Expected: `savefig`/`CSV.write` calls writing into a `results/` directory — that is the generator.

- [ ] **Step 3: Delete them**

```bash
git rm examples/core_features/visualization/fixthis.png \
       examples/core_features/visualization/WechatIMG7635.jpg \
       src/utils/visualization/web/VISUALIZATION_FIXES.md
git rm -r examples/grid_world/results/
```

- [ ] **Step 4: Stop tracking regenerated output**

`.gitignore:40` already has `/results/`, which only matches the repo root. Add anchored patterns for the example output directories:

```
# Regenerable example output
examples/*/results/
examples/core_features/*/results/
```

Do **not** add `examples/feature_acquisition/archive/` — those figures are not reproducible (their generators are broken by dead `include` paths) and their fate is a Deferred owner decision.

- [ ] **Step 5: Verify**

```bash
git check-ignore -v examples/grid_world/results/foo.png
git status --short
```
Expected: the path is reported as ignored, and `git status` shows only the intended deletions plus the `.gitignore` change.

- [ ] **Step 6: Verify the example still regenerates its output**

```bash
cd examples/grid_world && julia --project=. grid_world.jl > /dev/null 2>&1; echo "exit=$?"; ls results/ | head; cd -
```
Expected: `exit=0` and the `results/` directory is recreated with fresh files — proving the deletion lost nothing. If the example fails, report it: that is a real defect the committed output was masking.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: delete accidental screenshots and untrack example output

fixthis.png (824 KB) and WechatIMG7635.jpg (408 KB) are referenced by
nothing and have no generator. examples/grid_world/results/ (10 files,
965 KB, dated 2025-07-28) is fully regenerated by grid_world.jl:258-279
— verified by re-running it. VISUALIZATION_FIXES.md was a stale
changelog with paths that no longer exist."
```

---

## Deferred — needs an owner decision before it can be planned

Each of these was found during reconnaissance and deliberately left out of the tasks above, because the right answer changes the work substantially.

1. **Git history rewrite.** `.git` is 90.96 MiB for a 452-file repo. `src/utils/visualization/web/node_modules/` accounts for **304 MB** of blob history (removed at HEAD in `b867295e`), plus 6.1 MB of a committed `docbuild/`. `git count-objects -vH` reports `garbage 0` and `prune-packable 0`, so `git gc` recovers nothing — only `git filter-repo`/BFG plus a force-push will. That invalidates every published SHA, including `origin/core-development` and the `pre-merge-backup-20260825` tag. **Decision needed:** rewrite, or accept the bloat and rely on `fetch-depth: 1`?

2. **`web/public/RDKit_minimal.wasm` (6.7 MB).** `@rdkit/rdkit@^2025.3.4` in `package.json:14` ships the identical artifact at `node_modules/@rdkit/rdkit/dist/`. But the committed copy is currently load-bearing: `MoleculeViewer2D.tsx:25,34` loads `/RDKit_minimal.js` with `locateFile: file => \`/${file}\`` from `public/`. Removing it requires a copy step (`vite-plugin-static-copy` or a `prebuild` script). **Decision needed:** add the copy step, or keep the vendored file?

3. **`src/applications/molecular_design.jl` (atom-level).** Its create path installs a `ForwardPolicy(identity)` placeholder at `:327` and then throws "outdated and needs to be migrated" at `:329-330`, yet all five of its types are still exported at `src/GFlowNet.jl:280`. The merged fragment-level `MolState` pipeline supersedes it. **Decision needed:** delete the file and its exports, or finish the migration?

4. **Frontend deployment.** `.dockerignore:1` asserts the frontend is "deployed separately on Vercel", but there is no `vercel.json`, no build script, and nothing setting `VITE_API_URL` (`api.ts:8`). **Decision needed:** add the Vercel config, or have the Dockerfile build the frontend and let Oxygen serve `web/dist`?

5. **Five zero-assertion test scripts.** `test_backward_policy.jl`, `test_detailed_balance_comprehensive.jl`, `test_detailed_balance_summary.jl`, `test_grid_world_versions.jl` and (after Task 8) the remainder encode intended behavior as `println` output. Converting them to assertions requires knowing what the correct values are. **Decision needed:** convert them (needs the expected values), or move them to `scripts/` as demos?

6. **Frontend component consolidation.** `DiversityStats` and `ParetoFrontExplorer` are mounted from three places each (`MolecularToolkit.tsx:8-15`, `MonitoringDashboard.tsx:11-12`, `ResultsHub.tsx:13-14`), each with its own un-shared `useEffect` fetch, so the same `POST /diversity` fires 2-3× per navigation. Two data-fetching conventions coexist (react-query vs raw `useState`+`useEffect` in all seven "Gap" panels). 14 files exceed 400 lines (`ProblemSetup.tsx` is 1368). `problemConfig: any` is prop-drilled through six pages. This is a substantial refactor deserving its own plan.

7. **Dead API surface.** Six backend routes have no caller (`/api/v2/sessions` 575, `/api/v2/sessions/{id}/molecules` 585, `/api/v2/molecules/all` 596, `/api/v2/domain/info` 408, `/api/v2/molecular/retrain` 1064 which is a `not_implemented` stub), and 14 `api.ts` client wrappers have no call site. **Decision needed:** wire to the UI, or delete?

8. **`.claude/` in version control.** 23 tracked files, ~260 KB of agent prompts and skills, including a tracked session log (`development_session.md`, 13.4 KB). `.claude/skills/` is tracked *despite* `.gitignore` listing `skills/`, because tracked files bypass gitignore. Also: the generic unanchored patterns `config.json`, `events.jsonl`, `labels/`, `sessions/`, `skills/`, `sources/`, `statuses/`, `views.json` match at any depth — they shadow nothing today (verified: `git status --ignored` returns zero matches), but any future `src/**/config.json` will be silently unaddable. **Decision needed:** keep `.claude/` tracked, and re-anchor the patterns to `/.claude/`?

9. **`examples/feature_acquisition/archive/` and `test/visualization/archive/`.** The former is ~200 KB of `.jl` plus 1.8 MB of figures across a v1/v2/v3 lineage whose every runner is broken by dead `include("visualization.jl")` and `cleanup.jl` paths. The latter is 11 orphaned epsilon-sweep and TB/DB exploration scripts. Both encode past experimental results. **Decision needed:** delete, or move to `reports/`?

10. **`data/` provisioning.** `scripts/generate_brics_library.py` produces `data/fragment_libraries/brics_drug_fragments.json`, which `unified_server.jl:1167` and `molecular_generation.jl:254` expect; `scripts/validate_all_gaps.jl:31` needs `data/reactions/reaction_templates.json`, which nothing in the repo generates. **Decision needed:** commit these as fixtures, generate them in a setup script, or document them as user-supplied?

---

## Self-Review

**Spec coverage.** Every blocker found in reconnaissance maps to a task: unloadable environment → Task 1; missing `UUIDs` → Task 2; dev tools as runtime deps → Task 3; aborting test suite → Tasks 4-5; orphaned tests → Task 6; silent RDKit skips → Task 7; zero-assertion tests → Task 8; orphan source files → Task 9; phantom exports → Task 10; double-compiled modules → Task 11; duplicated reaction state dim → Task 12; missing imports → Task 13; broken route → Task 14; fake connection status → Task 15; port 3000 and missing `data/` → Task 16; Docker/CondaPkg → Task 17; no CI → Tasks 18-19; broken docs build → Task 20; false docs → Task 21; no changelog/contributing → Task 22; bogus example UUIDs → Task 23; junk binaries → Task 24. The ten items with no task are listed under Deferred with the specific decision each needs.

**Placeholder scan.** Two tasks intentionally require reading before editing rather than supplying final code, because the correct code depends on values only the file can tell you: Task 8 step 4 (the existing variable names in `test_training.jl`) and Task 21 step 5 (the canonical repo URL). Each says exactly what to read and what to do with the answer. Task 21 step 5 also caps the unknown at a single `TODO(owner):` marker instead of leaving eight placeholders. Task 12 step 2 asks the engineer to confirm one Lux field-accessor path (`forward_policy.network.layers[1].in_dims`) with a `dump()` call before relying on it, and states the fallback assertion if it is unreachable.

**Type consistency.** `RDKIT_AVAILABLE` and `EXPECTED_FRAGMENT_COUNT` are defined in `test/fixtures/molecular.jl` (Task 7 step 2) and consumed by name in steps 4-5 and in the `CONTRIBUTING.md` conventions (Task 22). `GFLOWNET_TEST_RDKIT` / `GFLOWNET_TEST_TDC` are introduced in Task 7 and referenced identically in the CI workflow (Task 19) and `CONTRIBUTING.md` (Task 22). `useBackendHealth` is defined in Task 15 step 2 and consumed in step 3. `reaction_state_dim(; n_reactions, fp_dim, n_scalar_features)` is defined in Task 12 step 4, exported in step 5, consumed in step 6, asserted in step 2, and named in Global Constraints and the `CONTRIBUTING.md` conventions with the same signature throughout. `start_server.jl`'s `HOST`/`PORT` contract is defined in Task 17 step 2 and consumed by the `CMD` in step 3.

**Ordering check.** Task 19's `docs` job is knowingly red until Task 20 lands — stated in the task. Task 9 deletes `src/types.jl` while `docs/src/guide/examples.md:267` still references it; Task 21 step 5 fixes that reference. Task 16 creates `data/`, which Task 13 step 3's verification depends on for a clean error message — Task 13 runs first and its expected output accounts for that.

**Corrections applied after verifying every cited line number against the tree.** Four claims from reconnaissance did not survive checking and were rewritten rather than shipped: (1) `Project.toml`'s `julia = "1.6"` is at line **62**, not 39; (2) `molecular_generation.jl:127` is **already** `const STATE_DIM = compute_state_dim()`, so Task 12 was rewritten around the two real `1049` literals — and that rewrite surfaced a latent bug the original framing missed, namely that `create_reaction_gflownet` takes `n_reactions` and `fp_dim` as keywords while `state_dim` stays a literal, so `n_reactions = 20` builds a network sized for 17; (3) `/data/` is at `.gitignore:32`, not 33; (4) `/results/` is at `.gitignore:40`, not 37. Every other line reference in this plan was confirmed by reading the file.
