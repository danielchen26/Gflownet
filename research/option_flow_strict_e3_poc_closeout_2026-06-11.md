# Option-Flow E3 Strict Same-Snapshot POC Closeout

**Authoritative time:** Thursday, 2026-06-11 17:17 EDT  
**Workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Runner:** `test/smiles_gflownet/run_option_flow_strict_e3_poc.jl`  
**Evidence level:** E3 strict same-snapshot generated catalogs  

---

## 1. Plain-English conclusion

This is the first real direct test of the philosophical object itself:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

The result is **positive**.

In real molecular HE runs, for the same frozen frontier state `S_t`, different bounded option schemas had very different realized frontier utilities. A small Option-Flow model learned to assign more probability to the high-utility options on held-out strict catalogs.

Headline verdict:

```text
E3_STRICT_OBJECT_SIGNAL_PRESENT
```

This means:

> The object itself is now supported by a real strict same-snapshot POC.

But it does **not** yet mean:

> We beat PMO / GFlowNet SOTA.

The correct next step after this is online-lite PMO, where the learned option selector actually chooses options during search and is compared against uniform / heuristic / greedy / Genetic GFN-like baselines.

---

## 2. What made this test stricter than E1/E2

Earlier evidence was proxy-state only:

- E1 summary catalogs grouped similar but not identical states.
- E2 typed-path catalogs used real raw decision logs but still proxy grouping.

E3 is different:

```text
same frozen frontier S_t
  -> clone frontier K times
  -> run K bounded HE options from the exact same state
  -> score each option by realized frontier utility U_i
  -> train/evaluate Pθ(ω_i | S_t) ∝ U_i
```

So each catalog is a real same-state option distribution.

---

## 3. Runtime issue resolved non-destructively

Earlier full GFlowNet/RDKit load was blocked by `.CondaPkg/lock`.

I did **not** delete the lock.

Instead, I used the existing pixi Python directly:

```bash
JULIA_CONDAPKG_BACKEND=Null
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python
```

This allowed:

- PythonCall to load;
- RDKit to import;
- OracleBridge to run;
- full `using GFlowNet` to succeed;
- real HE runtime to execute.

---

## 4. Implementation

Added:

| File | Purpose |
|---|---|
| `test/smiles_gflownet/run_option_flow_strict_e3_poc.jl` | strict same-snapshot generated catalog runner |

Used existing / prior modules:

- `src/training/option_flow_dataset.jl`
- `src/training/option_flow_model.jl`
- `src/training/option_flow_loss.jl`
- `src/training/option_flow_training.jl`
- `src/training/option_flow_real_catalog.jl`
- HE runtime: `run_hierarchical_edit_episode!`
- Oracle runtime: `OracleBridge`

---

## 5. Smoke run

Command shape:

```bash
OPTION_FLOW_E3_MODE=smoke \
OPTION_FLOW_E3_TASKS=qed \
OPTION_FLOW_E3_SNAPSHOTS=2 \
OPTION_FLOW_E3_ORACLE_BUDGET=250 \
OPTION_FLOW_E3_EPOCHS=80 \
julia --project=. test/smiles_gflownet/run_option_flow_strict_e3_poc.jl
```

Result:

```text
E3_STRICT_OBJECT_SIGNAL_PRESENT
```

QED smoke metrics:

| Metric | Value |
|---|---:|
| strict catalogs | 2 |
| options | 12 |
| CE gain vs uniform | +0.0753 |
| expected utility lift | +0.2621 |
| top-quartile lift | +0.1233 |
| rank correlation | 0.6000 |
| gate | pass |

This validated the runtime and end-to-end pipeline.

---

## 6. Full E3 run

Command shape:

```bash
OPTION_FLOW_E3_MODE=full_e3 \
OPTION_FLOW_E3_TASKS=qed,drd2,celecoxib_rediscovery \
OPTION_FLOW_E3_SNAPSHOTS=5 \
OPTION_FLOW_E3_ORACLE_BUDGET=900 \
OPTION_FLOW_E3_EPOCHS=240 \
OPTION_FLOW_E3_TRAIN_SEEDS=17,23,31 \
julia --project=. test/smiles_gflownet/run_option_flow_strict_e3_poc.jl
```

Result bundle:

- `checkpoints/option_flow_strict_e3_poc/option_flow_strict_e3_poc_full_e3_results.jls`

Log:

- `checkpoints/option_flow_strict_e3_poc/run_full_e3.log`

Verdict:

```text
E3_STRICT_OBJECT_SIGNAL_PRESENT
```

---

## 7. Full E3 headline metrics

| Metric | Mean | Std | Interpretation |
|---|---:|---:|---|
| CE gain vs uniform | +0.2333 | 0.0908 | strong held-out distribution learning |
| Expected utility lift | +0.3066 | 0.0769 | model samples better options than uniform |
| Expected utility lift fraction | +46.38% | 5.43% | large practical lift over uniform option choice |
| Top-quartile mass lift | +0.1873 | 0.0356 | model puts more mass on best options |
| Rank correlation | 0.6286 | 0.2000 | learned logits align with realized utility |
| Entropy | 1.5627 | 0.0338 | no greedy collapse; uniform entropy is 1.7918 |
| Model greedy expected utility | 1.8698 | 0.1930 | near oracle-greedy upper bound |
| Oracle-greedy expected utility | 1.9033 | 0.2099 | upper bound |

Seed-level results:

| Seed | CE gain | Utility lift | Utility lift fraction | Top-Q lift | Rank corr | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 17 | +0.1286 | +0.3213 | +40.53% | +0.1802 | 0.5429 | pass |
| 23 | +0.2805 | +0.2234 | +47.37% | +0.1558 | 0.4857 | pass |
| 31 | +0.2907 | +0.3751 | +51.26% | +0.2260 | 0.8571 | pass |

Gate result:

```text
3 / 3 seeds pass
```

---

## 8. Catalog statistics

| Item | Value |
|---|---:|
| tasks | 3 |
| strict same-snapshot catalogs | 15 |
| option candidates | 90 |
| candidates per catalog | 6 |
| informative catalogs | 15 / 15 |
| positive utility-spread catalogs | 15 / 15 |
| mean utility spread | 1.8081 |
| max utility | 3.4113 |
| min utility | 0.0 |

Per task:

| Task | Catalogs | Candidates | Mean utility spread | Max utility | Oracle calls |
|---|---:|---:|---:|---:|---:|
| `qed` | 5 | 30 | 2.3446 | 3.4113 | 141 |
| `drd2` | 5 | 30 | 1.6915 | 2.8025 | 154 |
| `celecoxib_rediscovery` | 5 | 30 | 1.3884 | 2.7569 | 155 |

This is important: every task produced nontrivial same-state utility differences across options.

---

## 9. Example strict catalogs

### QED snapshot 1

```text
utilities = [1.9451, 3.3699, 0.4251, 0.4251, 0.7559, 1.1795]
target probs = [0.240, 0.416, 0.052, 0.052, 0.093, 0.146]
```

### DRD2 snapshot 1

```text
utilities = [1.9653, 2.8025, 0.0, 0.0, 0.8510, 0.7504]
target probs = [0.309, 0.440, 0.0, 0.0, 0.134, 0.118]
```

### Celecoxib snapshot 1

```text
utilities = [1.9525, 2.7569, 0.3663, 0.3663, 0.5026, 1.8215]
target probs = [0.251, 0.355, 0.047, 0.047, 0.065, 0.235]
```

These are exactly the kind of distributions the philosophy predicts should exist.

---

## 10. Done vs remaining

| Done | Remaining |
|---|---|
| Resolved runtime path without deleting lock | Online-lite PMO deployment |
| Generated strict same-snapshot catalogs | Compare against heuristic HE selector online |
| Used real RDKit/TDC oracles | Compare against greedy ranker online |
| Used real HE option rollouts | Compare against TB-only / Genetic GFN baselines |
| Trained/evaluated Option-Flow on held-out strict catalogs | Scale to 10K / 23-task PMO |
| Passed 3/3 seeds | Formal stochastic edit-TB objective |
| Included structural task `celecoxib_rediscovery` | SOTA claim |

---

## 11. Theory verdict

**Positive object-level proof.**

This E3 POC shows that the philosophical object is real enough to learn:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

We now have direct evidence that:

1. same frontier state `S_t` has multiple bounded options with different realized frontier utilities;
2. the utility distribution is nontrivial and task-dependent;
3. a learned scorer can recover this distribution on held-out strict catalogs;
4. the learned distribution improves expected utility over uniform by about **46%**;
5. the model does not collapse greedily; entropy remains high.

So the theory is no longer merely philosophical or synthetic. It has passed the first realistic object-level proof.

---

## 12. SOTA verdict

**Not proven yet.**

This result says the object is correct enough to pursue seriously. It does **not** yet prove we beat GFlowNet SOTA because:

- no online PMO loop used the learned selector to spend oracle budget;
- no 10K / 23-task run was performed;
- no direct benchmark against Genetic GFN / TB-only / Augmented Memory was performed.

But this is the strongest evidence so far that Option-Flow could be a better route than terminal-molecule-only GFlowNet for PMO, because it learns over the actual search operation that changes the frontier.

---

## 13. Testing / automation verdict

**E3 runner is automated and produced real artifacts.**

Artifacts:

- `checkpoints/option_flow_strict_e3_poc/option_flow_strict_e3_poc_smoke_results.jls`
- `checkpoints/option_flow_strict_e3_poc/option_flow_strict_e3_poc_full_e3_results.jls`
- `checkpoints/option_flow_strict_e3_poc/run_smoke_qed.log`
- `checkpoints/option_flow_strict_e3_poc/run_full_e3.log`

Prior tests still pass:

```text
Option-Flow real artifact catalog | 25 pass / 25 total
```

E3 itself is a runner, not yet a unit test, because it invokes real RDKit/TDC oracles and performs real HE rollouts.

---

## 14. Next step

The next proof step is **online-lite Option-Flow PMO**:

```text
for each frontier state:
  generate K candidate HE options
  score them with learned Option-Flow
  sample option proportional to predicted flow
  execute selected option
  compare PMO AUC/top-k/diversity vs:
    - uniform option choice
    - heuristic HE
    - greedy ranker
    - TB-only / Genetic-style baseline
```

If online-lite improves PMO AUC, especially on structural tasks, then we can start making a realistic SOTA-facing claim.
