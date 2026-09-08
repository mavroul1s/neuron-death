"""Analyze the frozen V2.1 dose sweep with the existing paired V2 comparators.

Usage: python scripts/analyze_fuzzy_v21_extract.py EXTRACT --out analysis.json
EXTRACT is the unpacked directory containing runs.json and the parquet tables.
This is an exploratory three-seed development comparison, separate from the
original analysis plan. The declared development ranking never changes configs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.stats import Estimate, iqm

SEEDS = (15, 16, 17)
THRESHOLDS = (0.3, 0.5)
REFERENCE = ROOT / "remote_runs/neuron-death-fuzzy-v1-v2dev0907-9c5b295b/analysis_unpacked"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stored_config_hash(config: dict) -> str:
    """Use src.config's persisted-config hash format without importing torch.

    Extracts already contain resolved configs; only data.root is environmental.
    The expected hash comes from the separate pre-run plan, not current defaults.
    """
    config = deepcopy(config)
    config.get("data", {}).pop("root", None)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def arm_label(config: dict, configs: dict[str, dict]) -> str:
    recycling = config["recycling"]
    kind = recycling["kind"]
    if kind in ("fuzzy_v2", "fuzzy_v2_yoked_random"):
        target = config
        if kind == "fuzzy_v2_yoked_random":
            target = configs[recycling["learning_degree"]["yoked_from_run_id"]]
            if target["seed"] != config["seed"] or target["recycling"]["kind"] != "fuzzy_v2":
                raise ValueError("yoke source is not the same-seed targeted run")
        threshold = float(target["recycling"]["learning_degree"]["degree_threshold"])
        suffix = "_yoked_random" if kind == "fuzzy_v2_yoked_random" else ""
        return f"fuzzy_v2_d{threshold:g}{suffix}"
    return kind


def bootstrap_contrast(matrices, weights=None, *, n_bootstrap=10_000,
                       confidence=0.95, seed=0, scale=1.0) -> dict:
    """Resample paired whole seed trajectories for a linear contrast of IQMs.

    For two arms this is exactly the V2 seed_block_bootstrap estimand and RNG:
    IQM(target) - IQM(control), never IQM(elementwise differences). Four arms
    allow the change in selection advantage with shared seed indices throughout.
    """
    arrays = [np.asarray(matrix, dtype=float) for matrix in matrices]
    if not arrays or any(a.ndim != 2 or a.shape != arrays[0].shape or
                         0 in a.shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("contrast needs matching finite seed-by-task matrices")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    weights = [1.0] if weights is None else weights
    if len(weights) != len(arrays):
        raise ValueError("one contrast weight is required for every matrix")
    # Use the existing V2 implementation where its optional training dependency
    # can be imported. The fallback below is tested against that exact function
    # and only avoids requiring PyTorch for an extract-only local analysis.
    if list(weights) in ([1], [1, -1]):
        try:
            from src.analysis.fuzzy_comparison import seed_block_bootstrap
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
        else:
            result = seed_block_bootstrap(*arrays, n_bootstrap=n_bootstrap,
                                          confidence=confidence, seed=seed).as_dict()
            for key in ("point", "ci_lo", "ci_hi"):
                result[key] *= scale
            return result
    rng = np.random.default_rng(seed)
    # With three seeds there are only ten distinct bootstrap multisets. Caching
    # avoids recomputing identical IQMs without changing the sampled indices.
    cache = {}
    values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = tuple(sorted(rng.integers(0, arrays[0].shape[0], arrays[0].shape[0])))
        if indices not in cache:
            cache[indices] = sum(w * iqm(a[list(indices)]) for w, a in zip(weights, arrays))
        values[i] = cache[indices]
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(values, [alpha, 1 - alpha])
    point = sum(w * iqm(a) for w, a in zip(weights, arrays))
    estimate = Estimate(point, float(lo), float(hi), *arrays[0].shape,
                        n_bootstrap, confidence).as_dict()
    for key in ("point", "ci_lo", "ci_hi"):
        estimate[key] *= scale
    return estimate


def validate_runs(root: Path, manifest: list[dict]) -> dict[str, dict]:
    runs = read_json(root / "runs.json")
    actual = {run["run_id"]: run for run in runs}
    expected = {run["run_id"]: run for run in manifest}
    if len(actual) != len(runs) or len(expected) != len(manifest) or set(actual) != set(expected):
        raise ValueError(f"{root}: completed run IDs differ from the frozen manifest")
    for rid, run in actual.items():
        summary = run.get("summary") or {}
        config = run["config"]
        if summary.get("status") != "complete":
            raise ValueError(f"{rid}: run is incomplete")
        hashed = stored_config_hash(config)
        if hashed != expected[rid]["config_hash"] or hashed != summary.get("config_hash"):
            raise ValueError(f"{rid}: stored config differs from its frozen hash")
        if config["run_id"] != rid or config["seed"] != expected[rid]["seed"]:
            raise ValueError(f"{rid}: inconsistent identity/seed")
    return actual


def schedule(table: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = table.loc[table.run_id.eq(run_id), ["step", "layer_idx", "k"]]
    if rows.duplicated(["step", "layer_idx"]).any():
        raise ValueError(f"{run_id}: duplicate recycling step/layer")
    return rows.sort_values(["step", "layer_idx"]).reset_index(drop=True)


def exact_yoke_check(table: pd.DataFrame, target_id: str, random_id: str) -> dict:
    target, random = schedule(table, target_id), schedule(table, random_id)
    if not target.equals(random):
        raise ValueError(f"{random_id}: step/layer/cardinality schedule differs from target")
    return {"target_run_id": target_id, "random_run_id": random_id,
            "exact_schedule_match": True, "schedule_rows": len(target),
            "target_resets": int(target.k.sum()), "random_resets": int(random.k.sum())}


def analyze(extract: Path, reference: Path, plan: dict) -> dict:
    new = validate_runs(extract, plan["run_manifest"])
    old = validate_runs(reference, plan["reused_completed_runs"])
    for name, expected in plan["prior_evidence"]["extract_sha256"].items():
        if hashlib.sha256((reference / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"reused {name} differs from the frozen source extract")
    if set(new) & set(old):
        raise ValueError("new and reused runs overlap")
    runs = {**new, **old}
    configs = {rid: item["config"] for rid, item in runs.items()}
    labels = {rid: arm_label(config, configs) for rid, config in configs.items()}
    seeds = {rid: int(config["seed"]) for rid, config in configs.items()}
    expected_new = {(f"fuzzy_v2_d{d:g}{suffix}", s) for d in THRESHOLDS
                    for suffix in ("", "_yoked_random") for s in SEEDS}
    expected_old = {(arm, s) for arm in ("none", "redo", "regrama", "snr",
                    "fuzzy_v2_d0.2", "fuzzy_v2_d0.2_yoked_random") for s in SEEDS}
    for group, expected in ((new, expected_new), (old, expected_old)):
        pairs = [(labels[rid], seeds[rid]) for rid in group]
        if len(pairs) != len(expected) or set(pairs) != expected:
            raise ValueError("threshold/arm/seed grid is incomplete or duplicated")

    def table(name, columns):
        return pd.concat([pq.read_table(root / f"{name}.parquet", columns=columns).to_pandas()
                          for root in (extract, reference)], ignore_index=True)

    tasks = table("tasks", ["run_id", "task_idx", "probe_point", "online_accuracy", "n_recycled"])
    tasks = tasks.loc[tasks.probe_point.eq("task_end")]
    if set(tasks.run_id) != set(runs):
        raise ValueError("task log contains missing or unexpected run IDs")
    for rid, rows in tasks.groupby("run_id"):
        if sorted(rows.task_idx.tolist()) != list(range(200)):
            raise ValueError(f"{rid}: task log must contain exactly tasks 0..199")
    window = plan["primary_outcome"]["task_window_zero_indexed_inclusive"]
    if window != [150, 199]:
        raise ValueError("unexpected V2.1 late window")
    ids = {arm: sorted((rid for rid in runs if labels[rid] == arm), key=seeds.get)
           for arm in sorted(set(labels.values()))}
    matrices = {arm: np.stack([tasks.loc[tasks.run_id.eq(rid) &
                tasks.task_idx.between(*window)].sort_values("task_idx").online_accuracy
                .to_numpy(dtype=float) for rid in arm_ids]) for arm, arm_ids in ids.items()}
    if any(a.shape != (3, 50) or not np.isfinite(a).all() or
           ((a < 0) | (a > 1)).any() for a in matrices.values()):
        raise ValueError("invalid late accuracy matrix")

    metrics = table("metrics", ["run_id", "task_idx", "probe_point", "layer_idx",
                    "n_neurons", "dead_exact_count", "dead_exact_count_ref",
                    "dormant_frac_tau_0p1", "dormant_frac_tau_0p1_ref"])
    metrics = metrics.loc[metrics.probe_point.eq("task_end")]
    metric_values = metrics[["n_neurons", "dead_exact_count", "dead_exact_count_ref",
                             "dormant_frac_tau_0p1", "dormant_frac_tau_0p1_ref"]]
    if not np.isfinite(metric_values.to_numpy(dtype=float)).all():
        raise ValueError("nonfinite required death/dormancy metrics")
    for rid, rows in metrics.groupby("run_id"):
        expected_pairs = {(task, layer) for task in range(200)
                          for layer in range(len(configs[rid]["model"]["hidden_dims"]))}
        actual_pairs = list(zip(rows.task_idx, rows.layer_idx))
        if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
            raise ValueError(f"{rid}: missing or duplicate task/layer metrics")
    final = metrics.loc[metrics.probe_point.eq("task_end") & metrics.task_idx.eq(199)]
    if set(final.run_id) != set(runs) or final.duplicated(["run_id", "layer_idx"]).any():
        raise ValueError("final metric grid has missing runs or duplicate layers")
    final_dead = {}
    for rid, rows in final.groupby("run_id"):
        rows = rows.sort_values("layer_idx")
        widths = configs[rid]["model"]["hidden_dims"]
        if rows.layer_idx.tolist() != list(range(len(widths))) or rows.n_neurons.tolist() != widths:
            raise ValueError(f"{rid}: final metric layers disagree with config")
        final_dead[rid] = float(rows.dead_exact_count.sum() / sum(widths))

    recycling = table("recycling", ["run_id", "step", "layer_idx", "k",
                                    "n_dead_exact", "n_alive_but_quiet", "n_dead_exact_ref"])
    counts = recycling[["k", "n_dead_exact", "n_alive_but_quiet"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("invalid recycled composition counts")
    if not recycling.k.eq(recycling.n_dead_exact + recycling.n_alive_but_quiet).all():
        raise ValueError("recycled composition does not partition reset units")
    reset_counts = recycling.groupby("run_id").k.sum().to_dict()
    dead_resets = recycling.groupby("run_id").n_dead_exact.sum().to_dict()
    dead_resets_ref = recycling.groupby("run_id").n_dead_exact_ref.sum().to_dict()
    per_layer_resets = recycling.groupby(["run_id", "layer_idx"]).k.sum().to_dict()
    for rid in runs:
        if reset_counts.get(rid, 0) != tasks.loc[tasks.run_id.eq(rid), "n_recycled"].sum():
            raise ValueError(f"{rid}: recycling and task reset totals disagree")
    yokes = {}
    for rid, config in configs.items():
        if config["recycling"]["kind"] == "fuzzy_v2_yoked_random":
            target = config["recycling"]["learning_degree"]["yoked_from_run_id"]
            yokes[rid] = exact_yoke_check(recycling, target, rid)

    neuron_rows = Counter()
    for root in (extract, reference):
        path = root / "neurons_c4.parquet"
        if not path.exists():
            raise ValueError(f"{root}: required per-neuron extract is missing")
        for batch in pq.ParquetFile(path).iter_batches(columns=["run_id"]):
            neuron_rows.update(batch.column(0).to_pylist())
    for rid in runs:
        if neuron_rows[rid] < 200 * sum(configs[rid]["model"]["hidden_dims"]):
            raise ValueError(f"{rid}: required per-neuron log is incomplete")

    boot = plan["bootstrap"]
    kwargs = {key: boot[key] for key in ("n_bootstrap", "confidence")}
    def estimate(values, *, scale=1, seed=0):
        return bootstrap_contrast([np.asarray(values).reshape(3, -1)],
                                  **kwargs, seed=seed, scale=scale)
    report = {"experiment": "fuzzy_v21_dev", "stage": "exploratory three-seed development sweep",
              "new_complete_runs": len(new), "reused_complete_runs": len(old),
              "late_window": window, "seeds": list(SEEDS), "arms": {}, "comparisons": {},
              "selection_advantage_change_vs_v2": {}, "yoked_validation": yokes,
              "diagnostic_probe_definition": "Unsuffixed dead/dormant values use the current task's fixed 2048-example probe; _ref uses the fixed reference probe.",
              "limitations": ["Whole seed trajectories are resampled together; tasks are not independent replicates.",
                  "Nominal 95% percentile intervals from only three development seeds are exploratory and not multiplicity-adjusted.",
                  "Exceeding a reused comparator here is a development result, not a general SOTA claim.",
                  "Reset dose is measured after the run; raising a threshold need not increase realised resets monotonically."]}
    for arm, matrix in matrices.items():
        arm_ids = ids[arm]
        resets = [int(reset_counts.get(rid, 0)) for rid in arm_ids]
        shares = [dead_resets.get(rid, 0) / k if k else None for rid, k in zip(arm_ids, resets)]
        ref_shares = [dead_resets_ref.get(rid, 0) / k if k else None
                      for rid, k in zip(arm_ids, resets)]
        report["arms"][arm] = {
            "source": "new" if arm_ids[0] in new else "reused_v2",
            "late_accuracy_pct": estimate(matrix, scale=100, seed=boot["accuracy_seed"]),
            "resets": estimate(resets, seed=3),
            "final_dead_pct": estimate([final_dead[rid] for rid in arm_ids], scale=100, seed=1),
            "dead_share_of_resets_pct": None if any(v is None for v in shares)
                                        else estimate(shares, scale=100, seed=4),
            "dead_share_of_resets_pct_ref": None if any(v is None for v in ref_shares)
                                            else estimate(ref_shares, scale=100, seed=4),
            "runtime_minutes": estimate([runs[rid]["summary"]["wall_time_s"] / 60
                                         for rid in arm_ids], seed=5),
            "per_seed": {str(seeds[rid]): {"run_id": rid, "late_accuracy_pct": 100 * iqm(row),
                          "resets": k, "final_dead_pct": 100 * final_dead[rid],
                          "completed_tasks": 200, "per_neuron_rows": neuron_rows[rid],
                          "runtime_minutes": runs[rid]["summary"]["wall_time_s"] / 60,
                          "resets_by_layer": {str(layer): int(per_layer_resets.get((rid, layer), 0))
                                              for layer in range(len(configs[rid]["model"]["hidden_dims"]))},
                          "dead_share_of_resets_pct": None if share is None else 100 * share}
                         for rid, row, k, share in zip(arm_ids, matrix, resets, shares)},
        }

    for threshold in THRESHOLDS:
        arm = f"fuzzy_v2_d{threshold:g}"
        comparators = (arm + "_yoked_random", "fuzzy_v2_d0.2", "none", "redo", "regrama", "snr")
        report["comparisons"][arm] = {other: bootstrap_contrast(
            [matrices[arm], matrices[other]], [1, -1], **kwargs,
            seed=boot["comparison_seed"], scale=100) for other in comparators}
        report["selection_advantage_change_vs_v2"][arm] = bootstrap_contrast(
            [matrices[arm], matrices[arm + "_yoked_random"], matrices["fuzzy_v2_d0.2"],
             matrices["fuzzy_v2_d0.2_yoked_random"]], [1, -1, -1, 1],
            **kwargs, seed=boot["comparison_seed"], scale=100)
    report["target_0p5_minus_0p3_pp"] = bootstrap_contrast(
        [matrices["fuzzy_v2_d0.5"], matrices["fuzzy_v2_d0.3"]], [1, -1],
        **kwargs, seed=boot["comparison_seed"], scale=100)

    for arm, item in report["arms"].items():
        # Ratios/efficiency are descriptive ratios of IQM estimates, not a new
        # accuracy objective or an assertion about marginal reset causality.
        dose = item["resets"]["point"]
        item["dose_ratio_to"] = {
            other: dose / report["arms"][other]["resets"]["point"]
            if report["arms"][other]["resets"]["point"] else None
            for other in ("fuzzy_v2_d0.2", "none", "redo", "regrama", "snr")}
        gain = item["late_accuracy_pct"]["point"] - report["arms"]["none"]["late_accuracy_pct"]["point"]
        item["descriptive_gain_pp_per_1000_resets"] = gain * 1000 / dose if dose else None
        item["death_dormancy_pct"] = {}
        for window_name, bounds in (("late", (150, 199)), ("final", (199, 199))):
            by_window = item["death_dormancy_pct"][window_name] = {}
            for probe, suffix in (("current", ""), ("reference", "_ref")):
                by_probe = by_window[probe] = {}
                for measure, column in (("dead_exact", "dead_exact_count" + suffix),
                                        ("dormant_tau0p1", "dormant_frac_tau_0p1" + suffix)):
                    by_measure = by_probe[measure] = {}
                    for layer in ("pooled", *range(len(configs[ids[arm][0]]["model"]["hidden_dims"]))):
                        values = []
                        for rid in ids[arm]:
                            rows = metrics.loc[metrics.run_id.eq(rid) & metrics.task_idx.between(*bounds)]
                            if layer != "pooled":
                                rows = rows.loc[rows.layer_idx.eq(layer)]
                            count = rows[column] if measure == "dead_exact" else rows[column] * rows.n_neurons
                            counts_by_task = count.groupby(rows.task_idx).sum()
                            widths_by_task = rows.n_neurons.groupby(rows.task_idx).sum()
                            values.append((counts_by_task / widths_by_task).sort_index().to_numpy())
                        by_measure[str(layer)] = estimate(values, scale=100, seed=1)

    # Read one target's monitor at a time: the large per-neuron tables need not
    # coexist in memory. The cap is taken from each actual config, never fixed at 37.
    for rid, config in configs.items():
        if config["recycling"]["kind"] != "fuzzy_v2":
            continue
        root = extract if rid in new else reference
        learning = pq.read_table(root / "learning_degree.parquet", filters=[("run_id", "=", rid)],
                                 columns=["step", "layer_idx", "candidate", "selected"]).to_pandas()
        if learning.empty:
            raise ValueError(f"{rid}: required learning-degree monitor is missing")
        decisions = learning.groupby(["step", "layer_idx"]).agg(
            candidates=("candidate", "sum"), selected=("selected", "sum"))
        fraction = config["recycling"]["learning_degree"]["max_reset_fraction"]
        widths = config["model"]["hidden_dims"]
        caps = np.asarray([int(np.floor(widths[layer] * fraction))
                           for _, layer in decisions.index])
        if (decisions.selected.to_numpy() > caps).any():
            raise ValueError(f"{rid}: monitor reports selection above the cap")
        if int(decisions.selected.sum()) != reset_counts.get(rid, 0):
            raise ValueError(f"{rid}: monitor and recycling reset totals disagree")
        diagnostic = {"candidate_rows": int(decisions.candidates.sum()),
                      "selected_rows": int(decisions.selected.sum()),
                      "decision_layer_rows": len(decisions),
                      "cap_saturation_pct": float((decisions.selected.to_numpy() == caps).mean() * 100),
                      "cap_binding_pct": float((decisions.candidates.to_numpy() > caps).mean() * 100),
                      "events": [dict(step=int(step), layer_idx=int(layer), candidates=int(row.candidates),
                                      selected=int(row.selected), cap=int(cap), cap_saturated=bool(row.selected == cap))
                                 for ((step, layer), row), cap in zip(decisions.iterrows(), caps)]}
        report["arms"][labels[rid]]["per_seed"][str(seeds[rid])]["monitor"] = diagnostic
    for arm, item in report["arms"].items():
        if all("monitor" in record for record in item["per_seed"].values()):
            item["cap_saturation_pct"] = estimate([
                record["monitor"]["cap_saturation_pct"] for record in item["per_seed"].values()], seed=6)
    ranked = sorted(THRESHOLDS, key=lambda d: (
        -report["arms"][f"fuzzy_v2_d{d:g}"]["late_accuracy_pct"]["point"],
        report["arms"][f"fuzzy_v2_d{d:g}"]["resets"]["point"], d))
    report["development_ranking"] = {
        "degree_thresholds_best_first": ranked,
        "rule": "Descending late accuracy IQM; exact ties use fewer IQM resets, then the lower threshold.",
        "interpretation": "Ranking of these two development settings only; inspect the separate selection-advantage CIs and validate any chosen setting independently.",
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path)
    parser.add_argument("--reference-extract", type=Path, default=REFERENCE)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/fuzzy_v21_dev_plan.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.extract, args.reference_extract, read_json(args.plan))
    report["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    report["extract"] = str(args.extract.resolve())
    report["reference_extract"] = str(args.reference_extract.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
