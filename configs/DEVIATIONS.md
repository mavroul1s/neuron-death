# Setting changes

CLAUDE.md §3 splits the experiment in two. **`analysis_plan.json` is frozen**:
outcome measure, task windows, decision thresholds, seed counts, the gate
criterion. **The setting is calibrated**: batch size, width, depth, number of
tasks, learning rate, dataset. Calibrating so that the phenomenon under study is
present at all is a manipulation check, not p-hacking — the published work being
replicated calibrated its own setting the same way.

The rule that keeps that honest is this file. **Every setting change is recorded
here before the run, with a date and a reason** — never by editing a frozen
field, and never after seeing which setting produces the preferred answer.

> **Entries dated 2026-08-08 marked `[reconstructed]` were written after the
> fact.** This file should have existed from the first session and did not; it
> was created on 2026-08-08. Those entries are reconstructed from
> `runs/LEDGER.md`, the run configs and CLAUDE.md §11, all of which are
> contemporaneous with the runs. They are recorded as reconstructions rather
> than backdated, because a deviations log that cannot be told apart from one
> kept properly is worth less than no log at all. Everything from
> `2026-08-08 setting 2` onward was written before its run.

---

## 2026-08-05 — batch size stays at 128, against the protocol's own arithmetic
`[reconstructed]`

**Change:** none. Recorded because the *absence* of a change is the deviation.

Protocol §A.4 specifies `batch_size: 128`. Dohare et al.'s Online Permuted MNIST
is genuinely online — batch size 1, one example at a time, single pass. Their
setup performs 800 × 60,000 = 48M updates; ours performs 200 × (60,000/128)
≈ 94k, a factor of **512** fewer, and plasticity loss accumulates per update
rather than per task boundary.

**Reason for not changing it:** batch size 1 at 200 tasks does not fit the
11-hour job ceiling. The learning-rate ladder below was the affordable response.
CLAUDE.md §3 records the standing instruction that **batch size should be
reduced before any other knob is escalated**, so if the phenomenon ever needs
strengthening again, that is the first move and not a further lr increase.

## 2026-08-05 → 2026-08-06 — learning rate 0.001 → 0.1 (Setting 1)
`[reconstructed]`

**Change:** `optim.lr` 0.001 → 0.003 → 0.01 (`configs/gate/`), then 0.03 → 0.1
(`configs/gate_hi/`). Every subsequent Setting 1 experiment uses **lr = 0.1**.

**Reason:** the reproduction gate failed its accuracy criterion at all three of
the originally planned learning rates (drop of +1.33 pp at lr=0.01 against a
required 3 pp), while the dead-unit criterion passed at every one. Protocol
§A.4's failure response step 1 is "raise the learning rate", which was followed
in order. The drop is perfectly monotonic in step size across two decades
(−1.18 → +4.54 pp), which is Dohare et al.'s own finding — largest step size,
strongest effect — so the phenomenon was **present and under-driven, not
absent**, and escalating the step size was the indicated move rather than a
search for a congenial number.

**Checked before accepting the pass**, so that a diverged run could not be
mistaken for a strong effect: no NaN, max mean-task-loss 0.472, minimum online
accuracy 0.8505, peak 0.9304 at task 1, held-out probe accuracy falling 95.88%
→ 93.77% independently of the online measure, and no wholly dead layer
(max 27.9%). Recorded in CLAUDE.md §11.

## 2026-08-05 — Setting 3 (activation sweep) inherits lr = 0.1
`[reconstructed]`

**Change:** `configs/setting3/` generated at lr = 0.1 for all five activations.

**Reason:** it is the calibrated Setting 1 value and the sweep is meant to vary
one field. **This was wrong for tanh** and is the deviation worth recording:
tanh reached 10.05%, chance for ten classes, so it diverged rather than trained
and its death metrics are meaningless. `configs/setting3_tanh_gate/` (lr ∈
{0.003, 0.01, 0.03}) calibrates it separately. The failed row is excluded from
every table and figure until then — `fig_setting3_activations` drops any arm
within 5 pp of chance and says so, rather than trusting anyone to remember.

## 2026-08-05 — Setting 2 (CIFAR-10 + CNN) at lr = 0.01, 50 tasks
`[reconstructed]`

**Change:** `configs/setting2/` generated at lr = 0.01, `n_tasks = 50`.

**Reason:** a guess. It is recorded because it was a guess and because it
failed: baseline online accuracy **rises** 45.10% (tasks 0–4) → 56.63% (tasks
40–49), so there is no plasticity loss, nothing for recycling to fix, and the
C1 comparison is void. The gate calibrated 0.1 for an MLP on permuted MNIST and
that does not transfer to a CNN on CIFAR-10. The C2-on-channels measurement from
those runs is still valid, because it is a statement about definitions rather
than about plasticity loss.

---

## 2026-08-08 — Setting 2 gate result: BOTH rungs collapse. Notebook 10 is blocked.

**Result, not a change.** `configs/setting2_gate/` (lr ∈ {0.03, 0.1}, 200 tasks,
baseline only, 5 seeds each) ran as notebook 9. **The frozen criterion returns
PASS at both learning rates, and both passes are spurious.**

| lr | early acc | late acc | drop | frozen verdict | health check |
|---|---:|---:|---:|---|---|
| 0.03 | 46.68% | **9.91%** | 36.75 pp | PASS | **COLLAPSED** |
| 0.1 | 29.22% | **9.95%** | 20.11 pp | PASS | **COLLAPSED** |

Late accuracy is chance for ten classes at both. The supporting evidence is
unambiguous and consistent across every seed:

- **mean loss = 2.303 = ln(10)** exactly — the loss of a uniform predictor;
- **gradient norm = 0.0 in every layer** — no parameter is moving any more;
- **weight L2 frozen** from task ~30 (lr=0.03) and task ~20 (lr=0.1);
- the **fully-connected layer is 99.8% / 100% `dead_exact`**, effective rank
  0.20 / 0.00.

The FC layer died completely, so no gradient reaches anything upstream and
training stops permanently. That is a dying-ReLU collapse of the whole network,
**not** loss of plasticity: accuracy fell because training stopped, not because
the network became unable to fit new labels.

**Why the frozen criterion cannot see it.** Its two conditions are "accuracy
fell ≥ 3 pp with CI excluding zero" and "`dead_exact` rose". A network that dies
completely satisfies both *maximally* — accuracy falls as far as it possibly
can, and every unit ends up dead — so a total collapse produces the most
emphatic possible PASS. This is a real limitation of the frozen gate. **It is
being reported, not fixed**: the plan is not amended, `GateResult.passed` still
reports the criterion verbatim, and the health check is a separate advisory
(`gate.collapse_diagnosis`, added 2026-08-08) that sets `usable` alongside it.
The same check was already performed by hand before the Setting 1 gate was
accepted (CLAUDE.md §11, "lr=0.1 is a healthy regime, not a diverged one"); it
is now written down so it cannot be skipped.

**Consequence: do NOT run notebook 10 at either learning rate.** Both would
produce 15 runs of a dead network.

**What we now know bounds the answer.** lr=0.01 over 50 tasks gave *rising*
accuracy (45.10% → 56.63%, no plasticity loss); lr=0.03 over 200 tasks gives
total collapse by task 30. The usable setting, if it exists, is between them —
and note lr=0.01 has **never been run at 200 tasks**, so its 50-task rise may
simply be the pre-decline phase. That is the cheapest thing to test.

**Proposed next rung — PENDING, needs a decision:** extend
`configs/setting2_gate/` with lr ∈ {0.01, 0.02} at 200 tasks, 5 seeds, ~1.1
GPU-h. CLAUDE.md §3 also ranks *reducing batch size* ahead of any further
learning-rate move, which applies here too and is untried in Setting 2.

## 2026-08-08 — Setting 2 horizon 50 → 200 tasks

**Change:** `scripts/make_configs.py:setting2_cifar_cnn` hardcodes
`n_tasks = 50`; proposed change to 200.

**Reason:** not cosmetic. The frozen plan's late window is tasks **151–200**, so
at 50 tasks Setting 2 has no late window and **cannot be evaluated with the
frozen estimator at all** — the first Setting 2 run was read with ad-hoc windows
(tasks 0–4 vs 40–49), which is exactly the freedom the frozen plan exists to
remove. `configs/setting2_gate/` already runs 200 tasks, so the gate and the arm
it calibrates currently disagree about the horizon. At 200 tasks the arm lands
inside the frozen windows and needs no bespoke analysis.

**Cost:** ~0.4 → ~1.6 GPU-h. `n_tasks` is a calibrated field under §3.

**Status: PENDING — moot until a learning rate survives the gate.**
# 2026-09-07 — Temporal fuzzy V2 development arm

At the researcher and supervisor's explicit request, a separate exploratory
method-development experiment was added after the original analysis study. It
does not modify `configs/analysis_plan.json` or any completed baseline. Its
settings were frozen in `configs/fuzzy_v2_dev_plan.json` before remote launch.


## 2026-09-08 — Temporal fuzzy V2.1 dose sweep, frozen before launch

**Authorization and scope.** The researcher explicitly relayed the supervisor's
request for a new multi-parameter method and a V2.1 dose sweep. This overrides
the earlier no-new-method scope of AGENTS.md and protocol Part D for this
separate exploratory extension. The original `configs/analysis_plan.json`, its
outcomes, thresholds, seed counts, and completed experiments remain untouched.

**Change.** Twelve new configs in `configs/fuzzy_v21_dev/`: targeted temporal V2
with degree threshold 0.30 or 0.50, plus exact-yoked random for each threshold,
seeds 15–17. Degree threshold is the only algorithm parameter changed from V2
0.20. Patience 2, EWMA beta 0.9, cooldown 1000, task grace 100, the score,
saliency guard, monitoring, warmup, 7.5% cap, optimizer, data and reset operation
all retain their V2 values. New identities and yoke references prevent reuse of
old run directories. The six targeted runs finish before the six paired random
runs begin; two independent experiments run concurrently, one per T4.

**Reason recorded before the run.** Completed V2 improved over no reset by
+2.496 pp and exact-yoked random by +0.441 pp, but reached 90.015% against SNR
92.624% and ReDo 92.346%. Mean reset count was only about 2113 and cap saturation
was zero. Prior eligible observations numbered 14685 at degree <=0.20, 48856 at
<=0.30, and 118795 at <=0.50. This motivates a dose test without redesigning the
function. Those prior counts are not forecasts of actual new trajectories.

**Pre-run plan.** `configs/fuzzy_v21_dev_plan.json` pins all twelve resolved
config hashes, the prior V2 analysis and extract hashes, the 18 completed V2
reference-run hashes, and current core source hashes. Primary readout: IQM of
online accuracy over tasks 150–199 (zero indexed). CIs resample whole paired seed
trajectories, 10000 replicates, with no within-task resampling. Both thresholds,
all planned comparisons, exact event/per-layer dose equality, candidate/cap
statistics and reset composition must be reported. Seeds 15–17 are development
seeds: this sweep does not establish SOTA or alter the original frozen verdicts.

**Reuse and cost.** The existing none, ReDo, ReGraMa and SNR runs on seeds 15–17,
and V2 threshold 0.20 with its paired random control, are reused from
`remote_runs/neuron-death-fuzzy-v1-v2dev0907-9c5b295b/analysis_unpacked/`.
No baseline is rerun. Approximate new cost is 2.8 GPU-hours based on observed V2
runtime, with a shared 10-hour session budget, checkpointing every 10 tasks and
preservation of outputs before 11 hours. Use the existing private
`nmavros/neuron-death-code` Dataset and the private Kaggle workflow.

## 2026-09-08 — Temporal fuzzy V2.2 controller sweep, frozen before launch

**Authorization and scope.** The researcher clarified that the target baseline
is the strongest published comparator (SNR/ReDo/ReGraMa), not the no-reset arm,
and explicitly asked us to find a setting in which the proposed method is best.
This remains a separate exploratory method-development extension. The original
`configs/analysis_plan.json` is unchanged.

**Evidence recorded before the run.** V2.1 threshold 0.50 reached 90.588%,
beating no reset by 3.068 pp and its exact-yoked random control by 0.503 pp
[0.385, 0.550], but remained 2.036 pp below SNR. It used only 3673 resets.
Controller logs show a score-distribution cliff immediately above 0.50 in the
middle hidden layer: threshold 0.50 selected only 136–207 layer-1 units per run,
while 21k–83k eligible layer-1 observations scored at most 0.60. Cap saturation
was zero.

**Change.** `configs/fuzzy_v22_dev/` contains four targeted settings and an
exact-yoked random control for every setting, seeds 15–17: `(threshold,
patience,cooldown) = (0.55,2,1000), (0.60,2,1000), (0.60,2,500),
(0.60,1,500)`. EWMA beta 0.9, task grace 100, monitoring, warmup, score,
saliency guard, reset cap and reset operation are unchanged. Fixed-trajectory
simulation on the supplied V2.1 logs projects approximately 9.3k, 10.6k, 13.6k
and 18.5k resets respectively; these are range estimates, not outcome
predictions, because new resets change later scores.

**Pre-run plan.** `configs/fuzzy_v22_dev_plan.json` freezes 24 config files,
their resolved hashes, the evidence hashes, source hashes, estimator and all
comparisons. All twelve targeted runs complete before any yoked control starts.
The development winner is the highest late-window IQM, with fewer resets as the
tie-break. Beating SNR's 92.624% point estimate here triggers a new held-out,
paired-seed confirmation; the three reused development seeds cannot establish
SOTA by themselves.
