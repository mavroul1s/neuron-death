# Beyond Exact Silence: Neuronal Inactivity and Adaptive Recycling in Continual Learning

[![Status](https://img.shields.io/badge/Status-Submitted_to_Machine_Learning-blue.svg)](https://github.com/mavroul1s/neuron-death)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official experimental infrastructure and code for the research paper:
> **Beyond Exact Silence: Neuronal Inactivity and Adaptive Recycling in Continual Learning**  
> *Nikolaos Mavros and Kostas Kolomvatsos*  
> Submitted to **Machine Learning (Springer Nature)** on 9 November 2026.  
> Preprint & submission package: `kk_paper/` | Code repository: [https://github.com/mavroul1s/neuron-death](https://github.com/mavroul1s/neuron-death)

---

### Project Overview & Evolution

Originally initiated under the working title *"Recycling the Living"* as an empirical measurement study into neuronal dormancy and mortality (335 runs, ~62 GPU-hours, 153 tests, 9 figures), this project has expanded into an integrated investigation combining **longitudinal inactivity diagnostics** with **Fuzzy Recovery-Accelerated Self-Normalized Resets (Fuzzy-RA-SNR)**—an adaptive online reset-and-recovery procedure evaluated against SNR, ReDo, and ReGraMa across multiple activations, datasets, and architectures.

- **Manuscript Status:** Submitted as an original research article to *Machine Learning* (Springer Nature) on 9 November 2026.
- **Benchmark & Extension Runs:** Comprises 335 initial measurement runs, 15 ReDo/SNR/ReGraMa baseline comparisons, 40-run multi-activation study (seeds 28–32), 70-run prospective component/replay extension (seeds 33–37), Fashion-MNIST/GELU, and CIFAR-10/CNN runs.
- **Artifacts & Archives:** Complete Git-LFS result archives, session ledgers, and C4 per-neuron extracts are organized under [`results/neuron_methods/`](results/neuron_methods/) and [`runs/`](runs/).

---

## The question

When a network trained on non-stationary data loses its ability to learn, is the
accumulation of inactive neurons a **cause** of that loss, or merely a **correlate**?

The literature assumes cause. Roughly fifteen mitigation methods have been built on
that assumption — most prominently ReDo (Sokar et al., 2023), which every ~1000 steps
re-initialises neurons whose activation magnitude falls below a threshold τ.

| | Claim | Verdict |
|---|---|---|
| **C1** | ReDo's benefit does not come from reviving *dead* neurons | **Supported** — strong form refuted |
| **C2** | The published definitions of "dead" disagree materially | **Established**, six independent ways |
| **C3** | L2 helps plasticity while *increasing* dead units; online norm *increases* the units it was designed to prevent | **Confirmed**, both halves |
| **C4** | Neuron mortality follows characterisable dynamics | **Analysed** — death is reversible |
| **C5** | Adam-induced death is a distinct mechanism | **Half** — real, but the predicted timing is wrong |

The design that makes C1 testable is a **size-matched random control**: at every
recycling event we compute the τ-dormant set, take `k = |dormant set|`, and recycle
`k` neurons chosen *uniformly at random* from the same layer. `k` is recomputed per
layer, per event. Sokar et al.'s own random baseline used a fixed percentage on a
cosine schedule, which confounds *which* neurons are recycled with *how many*.

---

## What we found

### C1 — ReDo is not a resurrection method

The full τ-sweep: 4 arms × 6 τ × 10 seeds, 190 runs. Late-window online accuracy,
baseline `none` = 87.683%.

| arm | what it recycles | dead share of it | Δ vs baseline |
|---|---|---:|---:|
| **ReDo** | the τ-dormant set | 12–31% | **+4.76 … +4.92 pp** |
| **Random-matched** | k units at random | 8–13% | +4.32 … +4.39 pp |
| **Inverse-matched** | the k *highest*-scoring | ~4% | +3.01 … +3.31 pp |

Random selection recovers **88–92%** of ReDo's benefit. Recycling the units *least*
likely to be dead still buys +3 pp. In both controls the dose confound runs the wrong
way — they recycle **more** units and do **worse** — so the ordering is not explained
by perturbation volume.

The pre-registered C1 prediction fired exactly: accuracy rises with τ while the
truly-dead fraction of the recycled set falls (31.4% → 12.3%).

**The strong form is refuted.** Targeting contributes ~10% (vs random) to ~35% (vs
inverse) of the effect. Report that, not "targeting is irrelevant".

Two findings about the published method:

- Even at **τ=0** — supposedly "only provably dead units" — ReDo recycles 20% of the
  network and **69% of that is alive**, because the score uses Sokar et al.'s
  64-example default while `dead_exact` is measured on 2048.
- **Sokar et al. Fig. 15 did not replicate.** Inverse-matched was pre-registered to
  collapse. It doesn't — the arm designed to be maximally destructive still captures
  two thirds of the benefit. Reported as a domain difference (supervised continual
  learning vs. their RL setting), not a bug.

### C2 — the definitions disagree, and one of them vanishes

Six independent demonstrations, the strongest result in the project:

| setting | result |
|---|---|
| permuted MNIST, lr=0.1 | `dead_exact` 20.5% vs `dormant τ=0.1` **64.8%** — 3.2× on identical activations |
| GELU / SiLU / LeakyReLU | `dead_exact` **exactly 0.00%** while the others flag 1–20%. GELU has the **largest** plasticity drop (4.84 pp) |
| **tanh (lr=0.03)** | **13.0 pp** plasticity loss — the largest of any arm — with `dead_exact`, `dormant τ`, **and** `dead_absolute` all at **exactly 0.00%**. Only `saturated`, undefined for every activation the source papers use, sees anything (25.8%) |
| conv channels (CIFAR CNN) | `dead_exact` **0.00%** in conv layers while 81–88% of spatial positions are silent; the fc layer in the same net says 26% |
| reference vs current batch | `dead_exact` is *higher* on the fixed reference batch, localised to the last hidden layer (20.8% vs 37.4%) — death is partly distribution-relative |
| numerics | whether a GELU unit counts as dead depends on **float32 vs float64** |

The reference-batch measurement is one none of the four source papers can make,
because none of them probes a second, fixed distribution.

### C3 — helping plasticity while killing more neurons

35 runs, lr=0.1, late window:

| arm | accuracy | vs backprop | `dead_exact` | effective rank |
|---|---:|---:|---:|---:|
| plain backprop | 87.71% | — | 25.45% | 126.3 |
| **L2 λ=1e-4** | **89.22%** | **+1.50** [1.41, 1.59] | **49.22%** | **43.6** |
| L2 λ=1e-3 | 85.34% | −2.38 | 72.20% | 9.5 |
| L2 λ=1e-2 | 75.85% | −11.86 | 67.45% | 4.9 |
| shrink & perturb | 89.66% | +1.94 | 6.69% | 35.2 |
| dropout 0.1 | 83.31% | −4.40 | 32.32% | 125.2 |
| **online norm** | **90.87%** | **+3.16** [3.08, 3.23] | **39.52%** | 136.8 |

Both pre-registered target observations fire. L2 at its *beneficial* dose nearly
doubles dead units and cuts effective rank by two thirds. Online Normalization —
designed to prevent dead units — is the **best** arm on accuracy while carrying **55%
more** dead units than plain backprop.

Shrink-and-perturb is the counterexample that keeps this honest: it helps *and*
reduces dead units. The claim is a dissociation, not a law.

### C4 — death is reversible

Run on the 25 gate runs (no intervention, so nothing is an artefact of recycling).
On the **fixed reference batch**, a unit that is `dead_exact` at one task boundary is
alive again at the next with probability **21.71% [21.37, 22.21]** — with no
intervention of any kind:

| layer | P(dead → alive), one task later |
|---|---:|
| 0 | 4.36% [4.22, 4.47] |
| 1 | 39.68% [38.27, 42.93] |
| 2 | 29.82% [29.51, 31.71] |
| **pooled** | **21.71%** |

**86.7% of units die at least once**, against a late-window prevalence of ~25%.
Median death episode is 2 tasks; 46% last exactly one; only 3.2% are still open at
task 199. Permanent death is real but it is a **first-layer** phenomenon.

This matters because every method the paper argues against is premised on dead units
being lost capacity that must be restored. If a fifth of them restore themselves
within one task, the premise fails before any intervention is applied.

> Do not quote the current-task-batch version of this number (59.94%). The permutation
> changes at every boundary, so a unit silent on task *t*'s inputs may fire on task
> *t+1*'s without changing at all. The gap between 60% and 22% is itself a C2 result.

**Recycling recurrence.** Per-unit recycle counts against a null that holds `k` fixed
per task and layer and redraws only *which* units:

| arm | Gini | null | enrichment | % alive when chosen |
|---|---:|---:|---:|---:|
| random-matched | 0.139–0.221 | 0.138–0.222 | **1.001 / 1.005 / 0.999** | 86.8–91.2% |
| inverse-matched | 0.291–0.361 | 0.067 | 1.26–1.93× | 95.1–95.3% |

Random-matched is the **negative control** and it passes exactly: enrichment lands on
1.000 to three decimals in every layer, Gini matches its null to the third decimal.
The measure captures targeting and nothing else — which is what makes
inverse-matched's 5× excess concentration meaningful.

### C5 — Adam-induced death is real; its predicted timing is not

| optimizer | accuracy | drop | `dead_exact` |
|---|---:|---:|---:|
| SGD (lr 0.1) | 87.69% | 4.56 pp | 20.5% |
| Adam (default) | 90.27% | 3.18 pp | **58.8%** |
| Adam (Lyle-tuned ε=1e-3, β₂=0.9) | 90.45% | 2.27 pp | **24.6%** |
| AdamW | 91.17% | 2.28 pp | 58.0% |

Lyle et al.'s two-hyperparameter fix cuts dead units 2.4× at identical learning rate.
But the predicted post-task-switch death spike **is not there** — SGD has one (+2.44 pp
at step 25), Adam declines monotonically from the switch onward. Adam's death
accumulates *across* tasks (25.5% → 64.9%), not within them. Reported as a negative
result, with two limitations stated: a 25-step probe grid could hide a faster spike,
and the reference batch was disabled for those runs.

### The thread running through all of it

Four independent dissociations between dead units and plasticity:

1. At lr=0.001, dead units nearly tripled while accuracy **improved**
2. Adam carries 3× SGD's dead units and a **smaller** plasticity drop
3. GELU has the largest plasticity drop and **zero** dead units by the standard metric
4. L2 at its beneficial dose doubles dead units while *improving* accuracy

Plus the one that undercuts the premise directly: **dead units come back on their own**.

---

## The Proposed Method: Fuzzy-RA-SNR

To address when a hidden neuron should be replaced and how its replacement should resume contributing, the paper introduces **Fuzzy Recovery-Accelerated Self-Normalized Resets (Fuzzy-RA-SNR)**:

1. **Neuron-Specific Temporal Inactivity Thresholds:** Extends Self-Normalized Resets (SNR; Farias & Jozefiak, 2025), which tracks consecutive minibatches without positive firing ($a_i$) against an adaptive inter-firing history threshold ($T_i$).
2. **Bounded Takagi–Sugeno Fuzzy Reset Control:** Combines normalized inactivity age $r_i = a_i / T_i$ with recent incoming-gradient retention $g_i = S_i / L_i$ (short- vs. long-term moving averages) mapped to membership functions:
   $$u_i = \text{clip}(r_i - 0.5, 0, 1), \quad v_i = \text{clip}\left(\frac{g_i - 0.1}{0.9}, 0, 1\right)$$
   A four-rule zero-order Sugeno controller applies threshold reduction multiplier $m_i = 1 - \frac{1}{4} u_i (1 - v_i)$ (bounded within $[0.75, 1.0]$) so that acceleration strictly requires both accumulated inactivity and declining recent gradients. The controller introduces **zero trainable parameters**.
3. **Temporary Post-Reset Gradient Amplification (Recovery):** Following reset (resampling incoming weights, zeroing outgoing weights, clearing momentum/optimizer state), outgoing-weight gradients are multiplied by $\beta = 2$ for $H = 100$ updates to accelerate functional reconnection to the network.

### Prospective Component & Replay Evaluation

Prospective 7-arm ReLU component study on 200 permuted-MNIST tasks using 5 independent, paired seeds (33–37) with 11 Bonferroni-adjusted bootstrap intervals:

| Method / Control | Late-task IQM Accuracy (%) | Δ vs Full (pp) | Adjusted 95% Bootstrap Interval | Notes |
|---|---:|---:|---:|---|
| **Fuzzy-RA-SNR (Full)** | **92.756%** | — | [92.743, 92.776] | Integrated fuzzy threshold + post-reset recovery |
| **SNR Baseline** | 92.641% | **+0.115** | [+0.065, +0.147] | Base inter-firing threshold detector |
| **RA-SNR (Recovery only)** | 92.718% | **+0.037** | [+0.007, +0.072] | Unmodified SNR threshold + $\beta=2, H=100$ recovery |
| **Fuzzy-only (No recovery)** | 92.621% | **+0.135** | [+0.081, +0.178] | Fuzzy threshold adaptation without post-reset recovery |
| **Age-only + recovery** | 92.753% | +0.003 | [−0.049, +0.077] | Fuzzy rule ignoring gradient retention ($v_i = 0$) |
| **Fixed 0.75 + recovery** | 92.799% | −0.044 | [−0.088, −0.011] | Applies max acceleration uniformly; ~21.3% more resets |
| **Replayed RA-SNR** | 92.754% | +0.002 | [−0.028, +0.037] | Replays exact Full reset schedule & counts; ranking by SNR |

#### Key Empirical Findings

- **Statistical Significance on MNIST:** Full Fuzzy-RA-SNR achieves statistically verified improvements over base SNR (+0.115 pp), RA-SNR (+0.037 pp), and fuzzy-only (+0.135 pp); all three Bonferroni-adjusted intervals exclude zero.
- **Transferable Schedule via Replay:** The exact replay control matches all 281,361 reset interventions across 263,583 event-layer records. The minute difference (+0.002 pp, interval [−0.028, +0.037]) indicates benefits are transferable through the reset schedule itself.
- **Activation & Dataset Generalization:**
  - *Multi-activation Permuted MNIST (seeds 28–32):* Consistent gains over SNR across ReLU (+0.125 pp), GELU (+0.036 pp), LeakyReLU (+0.099 pp), and SiLU (+0.054 pp).
  - *Fashion-MNIST / GELU MLP:* Full reaches 81.280% (+0.070 pp vs SNR [−0.030, +0.234]).
  - *CIFAR-10 / 3-layer CNN (label permutations):* Full reaches 59.211% (+0.455 pp vs SNR [−0.656, +1.776]; No reset: 58.116%).
- **Interventions Beyond Exact Silence:** In the prospective MNIST full arm, 64.12% of the 281,361 reset interventions replace units that remain active (nonzero) on the current probe. In Fashion-MNIST/GELU, **0.00%** of selected units are classified as exactly silent on recorded probes.

---

## Repository

```
neuron-death/
├── CLAUDE.md                    project context; §11 is the live status block
├── protocol_weeks_1_2_v2.md     authoritative experimental specification
├── README.md                    this file
│
├── configs/
│   ├── analysis_plan.json       FROZEN pre-registration, 2026-08-05T14:09:03Z
│   ├── DEVIATIONS.md            every calibration change, dated, with reasons
│   ├── gate/ (15) gate_hi/ (10)         reproduction gate, lr ladder
│   ├── tau_sweep/ (190)                 C1: none · redo · random · inverse × 6τ × 10 seeds
│   ├── c3/ (35)                         C3: backprop · L2×3 · S&P · dropout · online norm
│   ├── c5/ (20)                         C5: sgd · adam · adam-Lyle · adamw
│   ├── setting3/ (25)                   C2: relu · leaky · gelu · silu · tanh
│   ├── setting3_tanh_gate/ (15)         tanh lr calibration
│   ├── setting2/ (15) setting2_gate/ (10)   CIFAR CNN — dropped, see DEVIATIONS
│   └── eps/ (25)                        not run; cut per protocol §B.4
│
├── src/
│   ├── probes.py                ALL metric definitions. Nothing else computes a metric.
│   ├── models.py                MLP + CNN; activation and norm are config fields
│   ├── data.py                  permuted MNIST, label-shuffled CIFAR-10
│   ├── interventions.py         ReDo, random-matched, inverse-matched, L2, shrink&perturb
│   ├── online_norm.py           Online Normalization (Chiley et al. 2019), eqs 8a–12b
│   ├── temporal_fuzzy.py        Takagi–Sugeno fuzzy reset controller & recovery
│   ├── learning_degree.py       gradient retention tracking (Si/Li moving averages)
│   ├── train.py                 task loop, sharded parquet logging, checkpoint/resume
│   ├── config.py  logs.py       config hashing, parquet schemas
│   └── analysis/                post-hoc only; never imported by training code
│       ├── stats.py             IQM + stratified bootstrap (Agarwal et al. 2021)
│       ├── gate.py              frozen gate criterion + collapse health check
│       ├── load.py              reading runs back off disk
│       ├── summarize.py         in-session sweep summary
│       ├── survival.py          C4 mortality, episodes, recycling recurrence
│       └── figures.py           the paper's figures
│
├── scripts/
│   ├── make_configs.py          generates every config set
│   ├── prepare_data.py          caches MNIST / CIFAR-10 locally
│   ├── launch_pair.py           two configs in parallel, one per GPU
│   ├── make_analysis_extract.py runs/ tree -> small analysis extract (+ C4)
│   ├── make_kaggle_notebooks.py one notebook per independently runnable job
│   ├── make_c4_notebook.py      CPU-only C4 recovery from an archival runs.zip
│   ├── package_for_kaggle.py    builds the code Dataset zip
│   └── update_ledger.py         regenerates runs/LEDGER.md
│
├── tests/                       153 tests, synthetic data, ~24 s
├── results/
│   └── neuron_methods/
│       ├── extract/            analysis-ready task/metric/recycling tables
│       ├── c4/                 slim per-neuron C4 extract plus analysis tables
│       ├── LEDGER.md           the 15-run session ledger
│       └── full_results.zip    all runs and checkpoints (Git LFS)
├── runs/
│   ├── LEDGER.md                run ledger / compute appendix
│   └── _extracts/               per-session analysis extracts
└── figures/                     9 figures, pdf + png
```

Four parquet tables per run: `tasks` (per task), `metrics` (per task × layer),
`neurons` (per task × layer × neuron), `recycling` (per event × layer), plus
`intra_task` when sub-task probing is enabled. Every metric is logged on **two** probe
batches — the current task's distribution and a fixed reference distribution that never
changes — which is what separates "this neuron is dead" from "the input distribution
moved".

### The figures

| file | shows |
|---|---|
| `fig1_tau_sweep` | C1: accuracy rises with τ while the dead share of the recycled set falls |
| `fig2_c2_definitions` | C2: three published definitions on one set of activations, per layer |
| `fig3_c4_survival` | C4: time to first death, recovery on both probes, episode lengths |
| `fig4_c4_recurrence` | C4/C1: which units recycling keeps choosing, and their state |
| `fig5_gate_dose_response` | the gate as a dose–response in step size |
| `fig6_reference_asymmetry` | `dead_exact` current vs reference batch — the layer-2 gap |
| `fig7_c3_anomaly` | C3: accuracy against dead units, one point per arm |
| `fig8_c5_optimizers` | C5: where Adam's death actually accumulates |
| `fig9_setting3_activations` | C2: four definitions across five activations |

No number in any figure is typed in by hand — all are regenerated from `runs/`.

---

## Design decisions worth knowing

- **`configs/analysis_plan.json` is frozen.** Outcome measure, task windows, decision
  thresholds (`|Δ| < 1 pp` for "≈"; `Δ > 2 pp` with CI excluding zero for "≫"), seed
  counts and the gate criterion were committed before any data existed. If a result is
  ambiguous the pre-specified response is *add seeds*, never adjust a threshold. It was
  never edited — including where it returned `inconclusive` on the C1 control and where
  its gate criterion proved unable to detect a collapsed network.
- **Calibration is separated from pre-registration.** Learning rate, batch size, number
  of tasks and dataset are *calibrated*; every change is recorded in
  `configs/DEVIATIONS.md` with a date and a reason, before the run.
- **The gate has a health check outside the frozen criterion.** A network that dies
  completely satisfies "accuracy dropped ≥3 pp" and "dead units rose" *maximally*, so
  collapse produces the most emphatic possible PASS. `gate.collapse_diagnosis` reports
  it alongside — accuracy at chance, loss at ln(n_classes), gradient norm zero — without
  touching the frozen verdict. This is what caught the Setting 2 calibration failure.
- **Metrics are implemented exactly as published, including their flaws.** The Sokar
  score normalises by the layer mean, so it is blind to a layer whose activations all
  shrink uniformly. That blind spot is the point of C2 and must not be "fixed" —
  `tests/test_probes.py` asserts that `dormant_tau` *misses* a uniformly shrunken layer
  while `dead_absolute` catches it.
- **ReDo zeroes outgoing weights.** Re-initialising incoming weights without zeroing
  outgoing ones turns ReDo into a far more destructive intervention while still
  producing plausible curves. A test asserts the network's function is unchanged after
  a τ=0 recycling event.
- **Online Normalization's backward pass is not the derivative of its forward pass.**
  It is a separate control process with its own state. A version that writes the
  forward and lets autograd differentiate it is a *different algorithm* that trains
  perfectly well — `test_online_norm.py` asserts the two disagree.
- **Statistics are IQM with stratified bootstrap 95% CIs** (Agarwal et al., 2021),
  never bare means over seeds.
- **Determinism is required and demonstrated.** Seeded torch/numpy/python,
  `cudnn.deterministic`, weight resampling on CPU so recycling is identical across
  hardware. 15 gate runs were accidentally executed twice in different Kaggle sessions
  and reproduced **bit-identically** — every `online_accuracy` equal at `atol=0`.

---

## Running it

```bash
python -m pytest tests/
```

```bash
python scripts/prepare_data.py --dataset mnist --root data
```

```bash
python -m src.train --config configs/gate/gate_pmnist_w500_sgd_lr0p01_s0.json
```

Two configs in parallel, one per GPU:

```bash
python scripts/launch_pair.py configs/c3/*.json --data-root data --budget-hours 10.5
```

Evaluate the reproduction gate:

```bash
python -m src.analysis.gate --runs-root runs --pattern "gate_*"
```

Rebuild the C4 analysis and every figure (no GPU, ~2 min):

```bash
python -m src.analysis.survival <c4-dir-or-parquet> --out dist/surv_<arm> --null-draws 50
```

```bash
python -m src.analysis.figures --extracts runs/_extracts --survival dist/surv_gate dist/surv_random dist/surv_inverse --out figures --tanh-lr 0.03
```

On Kaggle: `python scripts/package_for_kaggle.py` stages a code Dataset zip plus one
pre-filled notebook per job. There is no download-at-runtime path anywhere in `src/` —
Kaggle sessions have no internet, and a silent fallback would be worse than a crash.

## Scope and Attribution

- **Evolution from Measurement to Method:** The project began as a diagnostic audit of neuron mortality ("Recycling the Living"), demonstrating that neuronal inactivity is a time-dependent state rather than a permanent binary label of death, and that units spontaneously reactivate. It expanded into proposing and prospectively evaluating **Fuzzy-RA-SNR**, combining neuron-specific temporal monitoring with fuzzy threshold adaptation and outgoing-weight gradient recovery.
- **Attribution and Limitations:** Paired comparisons and replay controls delimit the exact attribution of gains: the gradient-specific increment remains unresolved relative to age-only adaptation, uniform fixed acceleration attains slightly higher MNIST accuracy with ~21.3% more resets, and transfer contrasts on Fashion-MNIST and CIFAR-10 carry statistical uncertainty. We emphasize these boundaries explicitly rather than claiming universal superiority.
- **Deliberately Excluded:** Deep RL environments, Mixture-of-Experts, transformers, and large foundation models (>50M parameters).

## Citation

```bibtex
@article{mavros2026beyond,
  title={Beyond Exact Silence: Neuronal Inactivity and Adaptive Recycling in Continual Learning},
  author={Mavros, Nikolaos and Kolomvatsos, Kostas},
  journal={Machine Learning},
  year={2026},
  note={Submitted}
}
```

## References

Sokar et al. 2023, *The Dormant Neuron Phenomenon in Deep RL* ·
Dohare et al. 2024, *Loss of plasticity in deep continual learning* (Nature) ·
Farias & Jozefiak 2025, *Self-normalized resets for plasticity in continual learning* (ICLR) ·
Liu et al. 2025, *Measure gradients, not activations!* (NeurIPS) ·
Lyle et al. 2023/2025 · Chiley et al. 2019 (Online Normalization) ·
Roy & Vetterli 2007 (effective rank) · Agarwal et al. 2021 (IQM, stratified bootstrap) ·
Ash & Adams 2020 (shrink-and-perturb)
