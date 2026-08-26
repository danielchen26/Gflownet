# Option-Flow v0 POC Closeout — 2026-06-11

## What was built

A lightweight finite-catalog Option-Flow prototype that validates the new philosophical object in the smallest clean setting:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

Instead of immediately claiming full TB over stochastic edit environments, v0 treats each realized bounded option as a terminal candidate in a finite catalog for a frozen search state.

## Implemented files

| File | Purpose |
|---|---|
| `research/option_flow_v0_poc_design_2026-06-11.md` | POC design spec |
| `src/training/option_flow_dataset.jl` | finite catalog data structures and synthetic catalogs |
| `src/training/option_flow_model.jl` | small MLP option scorer |
| `src/training/option_flow_loss.jl` | catalog CE/KL, entropy, flow residual, top-utility diagnostics |
| `src/training/option_flow_training.jl` | lightweight manual-backprop training loop |
| `test/smiles_gflownet/test_option_flow_poc.jl` | unit tests |
| `test/smiles_gflownet/run_option_flow_poc.jl` | smoke/offline POC runner |

## Test results

```text
julia --project=. test/smiles_gflownet/test_option_flow_poc.jl
```

Result:

```text
Option-Flow v0 POC | 25 pass / 25 total
```

## POC runs

### Smoke run

Config:

- mode: `smoke`
- seed: `11`
- catalogs: `48`
- candidates/catalog: `6`
- epochs: `120`

Validation metrics:

| Metric | Value |
|---|---:|
| CE | 1.4981 |
| Uniform CE | 1.7918 |
| CE gain vs uniform | +0.2937 |
| Top-quartile mass | 0.5820 |
| Uniform top-quartile mass | 0.3333 |
| Top-quartile lift | +0.2487 |
| Rank correlation | 0.8190 |
| Entropy | 1.5109 |
| Verdict | `POC_SIGNAL_PRESENT` |

### Offline-style synthetic run

Config:

- mode: `offline`
- seed: `23`
- catalogs: `96`
- candidates/catalog: `6`
- epochs: `250`

Validation metrics:

| Metric | Value |
|---|---:|
| CE | 1.5164 |
| Uniform CE | 1.7918 |
| CE gain vs uniform | +0.2753 |
| Top-quartile mass | 0.5972 |
| Uniform top-quartile mass | 0.3333 |
| Top-quartile lift | +0.2639 |
| Rank correlation | 0.7952 |
| Entropy | 1.4798 |
| Verdict | `POC_SIGNAL_PRESENT` |

## Artifact paths

| Artifact | Path |
|---|---|
| smoke log | `checkpoints/option_flow_v0_poc/run_smoke_seed11.log` |
| smoke result bundle | `checkpoints/option_flow_v0_poc/option_flow_v0_poc_smoke_seed11_results.jls` |
| offline log | `checkpoints/option_flow_v0_poc/run_offline_seed23.log` |
| offline result bundle | `checkpoints/option_flow_v0_poc/option_flow_v0_poc_offline_seed23_results.jls` |
| latest result alias | `checkpoints/option_flow_v0_poc/option_flow_v0_poc_latest_results.jls` |

## Theory verdict

**POC signal is present.**

This validates the finite-catalog version of the Option-Flow object: a small model can learn a distribution over bounded options that beats uniform, assigns more mass to high-utility options, preserves nonzero entropy, and has positive utility rank correlation on held-out catalogs.

This gives a believable reason to continue because the target object is learnable in the cleanest controlled setting.

## Important limitation

This is still synthetic finite-catalog validation, not yet real HE/PMO validation.

It does **not** prove:

- full TB over stochastic edit trajectories;
- online improvement over heuristic HE;
- PMO AUC gains;
- SOTA competitiveness.

## Testing / automation verdict

A first automatic POC test suite now exists and passes.

Covered:

- utility normalization;
- all-zero utility fallback;
- catalog construction / validation;
- softmax probabilities;
- CE loss and flow residual diagnostics;
- grouped split;
- synthetic training improvement.

Not yet covered:

- real HE artifact catalog extraction;
- online option catalog generation;
- stochastic proposal/selected-child semantics;
- backward policy for true TB over options;
- PMO-level AUC impact.

## Next step

Move from synthetic finite catalogs to real logged HE option catalogs.

The next implementation stage should build:

1. HE artifact loader -> `OptionFlowCatalog`s grouped by snapshot/task;
2. decision-time vs post-outcome feature audit;
3. real-catalog offline validation;
4. greedy/ranker ablation;
5. only then online-lite catalog sampling.
