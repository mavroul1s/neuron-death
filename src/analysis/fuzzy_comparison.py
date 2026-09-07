"""Post-hoc audit of the first fuzzy-reset experiment (never imported by training).

Keep the original task-stratified IQM confidence interval unchanged. Tasks in
continual learning share a trajectory, however, so also resample complete seed
trajectories: every task in a replicate gets the same sampled seed indices.
The latter is a sensitivity analysis, not a change to the frozen old plan.
Two candidate methods versus five baselines give ten planned comparisons.
Nominal Bonferroni percentile intervals are reported alongside unadjusted CIs;
neither a positive interval nor this one-dataset pilot establishes SOTA.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import config_hash, resolve_config
from .gate import PLAN_PATH, _window, load_plan
from .stats import Estimate, iqm, stratified_bootstrap, stratified_bootstrap_difference


CANDIDATES = ("fuzzy", "fuzzy_trend")
BASELINES = ("none", "redo_tau0.1", "redo_tau0.25", "regrama", "snr")
ARMS = BASELINES + CANDIDATES
SEEDS = (10, 11, 12, 13, 14)
FAMILY_SIZE = len(CANDIDATES) * len(BASELINES)


def arm_label(config: dict) -> str:
    """Identify an algorithm from the actual config, not a run-id substring."""
    recycling = config.get("recycling") or {}
    kind = recycling.get("kind", "none")
    if kind == "redo":
        return f"redo_tau{float(recycling['tau']):g}"
    return str(kind)


def _matrix(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or 0 in scores.shape or not np.isfinite(scores).all():
        raise ValueError("bootstrap requires a nonempty finite seed-by-task matrix")
    return scores


def seed_block_bootstrap(
    scores: np.ndarray,
    other: np.ndarray | None = None,
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """IQM or paired IQM difference, resampling whole seed trajectories.

    ``iqm(a) - iqm(b)`` is intentional: IQM of elementwise differences is a
    different estimand. Paired arms must be sorted by the actual shared seed.
    """
    a = _matrix(scores)
    b = None if other is None else _matrix(other)
    if b is not None and a.shape != b.shape:
        raise ValueError("paired seed-block bootstrap requires matching shapes")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap count or confidence")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        indices = rng.integers(0, a.shape[0], size=a.shape[0])
        estimates[i] = iqm(a[indices]) - (0.0 if b is None else iqm(b[indices]))
    alpha = 1.0 - confidence
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    point = iqm(a) - (0.0 if b is None else iqm(b))
    return Estimate(point, float(lo), float(hi), *a.shape, n_bootstrap, confidence)


def _estimate_dict(estimate: Estimate, scale: float = 1.0) -> dict:
    result = estimate.as_dict()
    for name in ("point", "ci_lo", "ci_hi"):
        result[name] *= scale
    return result


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_table(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _task_end(table: pd.DataFrame) -> pd.DataFrame:
    return table.loc[table["probe_point"].eq("task_end")].copy()


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns) - set(table)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _finite(table: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    _require_columns(table, columns, name)
    if not np.isfinite(table[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains nonfinite values in required columns")


def _pooled_fraction(metrics: pd.DataFrame, column: str) -> float | None:
    if column not in metrics:
        return None
    values = metrics[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"metrics contains nonfinite {column}")
    weights = metrics["n_neurons"].to_numpy(dtype=float)
    if column.startswith("dead_exact_count"):
        return float(values.sum() / weights.sum())
    return float(np.average(values, weights=weights))


def inspect_run(run_dir: Path, expected: dict, plan: dict) -> dict:
    """Audit a single run without dropping invalid/unfinished observations."""
    result = {
        "run_id": expected["run_id"], "arm": arm_label(expected),
        "seed": int(expected["seed"]), "path": str(run_dir),
        "status": "missing", "errors": [], "warnings": [],
        "health": {}, "diagnostics": {},
    }
    if not (run_dir / "config.json").exists():
        return result
    try:
        config = _read_json(run_dir / "config.json")
        expected_hash = config_hash(resolve_config(expected))
        actual_hash = config_hash(resolve_config(config))
        result["config_hash"] = actual_hash
        if actual_hash != expected_hash:
            raise ValueError("stored config differs from the declared experiment config")
        summary = _read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
        if summary.get("status") != "complete":
            result["status"] = "incomplete"
            result["recorded_status"] = summary.get("status", "no summary")
            return result
        if summary.get("config_hash") != actual_hash:
            raise ValueError("summary config_hash disagrees with stored config")
        tasks = _read_table(run_dir, "tasks")
        _require_columns(tasks, ("task_idx", "probe_point", "online_accuracy", "mean_loss"), "tasks")
        tasks = _task_end(tasks).sort_values("task_idx")
        n_tasks = int(expected["data"]["n_tasks"])
        if tasks["task_idx"].duplicated().any():
            raise ValueError("duplicate task_end rows (possible resume/log duplication)")
        if tasks["task_idx"].tolist() != list(range(n_tasks)):
            raise ValueError("task_end log does not contain exactly all expected tasks")
        _finite(tasks, ("online_accuracy", "mean_loss"), "tasks")
        if not tasks["online_accuracy"].between(0, 1).all() or (tasks["mean_loss"] < 0).any():
            raise ValueError("invalid accuracy/loss range")
        metrics = _read_table(run_dir, "metrics")
        _require_columns(metrics, ("task_idx", "probe_point", "layer_idx", "n_neurons", "dead_exact_count", "grad_norm_layer"), "metrics")
        metrics = _task_end(metrics)
        _finite(metrics, ("n_neurons", "dead_exact_count", "grad_norm_layer"), "metrics")
        if (metrics["n_neurons"] <= 0).any() or (metrics["dead_exact_count"] < 0).any() or (metrics["dead_exact_count"] > metrics["n_neurons"]).any():
            raise ValueError("invalid layer neuron/death count")
        widths = resolve_config(config)["model"]["hidden_dims"]
        expected_pairs = {(t, l) for t in range(n_tasks) for l in range(len(widths))}
        actual_pairs = list(zip(metrics["task_idx"], metrics["layer_idx"]))
        if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
            raise ValueError("metrics task/layer grid is incomplete or duplicated")
        for layer, width in enumerate(widths):
            if not metrics.loc[metrics["layer_idx"].eq(layer), "n_neurons"].eq(width).all():
                raise ValueError("logged layer widths disagree with config")
        neurons_path = run_dir / "neurons.parquet"
        if not neurons_path.exists() or pq.read_metadata(neurons_path).num_rows < n_tasks * sum(widths):
            raise ValueError("missing or short per-neuron log; this run cannot support its measurements")
        late = _window(plan, "late")
        if not set(late).issubset(set(tasks["task_idx"])):
            raise ValueError("run is too short for the frozen late window")
        late_tasks = tasks.set_index("task_idx").loc[late]
        scores = late_tasks[plan["outcome_measure"]["primary"]].to_numpy(dtype=float)
        late_metrics = metrics.loc[metrics["task_idx"].isin(late)]
        final = metrics.loc[metrics["task_idx"].eq(n_tasks - 1)]
        accuracy = iqm(scores)
        loss = iqm(late_tasks["mean_loss"].to_numpy())
        dead = late_metrics["dead_exact_count"] / late_metrics["n_neurons"]
        at_chance = accuracy <= 1.1 / int(resolve_config(config)["model"]["out_features"])
        uniform_loss = abs(loss - np.log(int(resolve_config(config)["model"]["out_features"]))) < 0.05
        zero_gradient = bool(late_metrics["grad_norm_layer"].abs().max() == 0.0)
        dead_layer = bool((dead >= 0.99).any())
        collapse = bool(at_chance and (uniform_loss or zero_gradient or dead_layer))
        result["health"] = {
            "finite_required_values": True, "late_at_chance": bool(at_chance),
            "late_uniform_loss": bool(uniform_loss), "late_zero_hidden_gradients": zero_gradient,
            "late_any_layer_at_least_99pct_dead": dead_layer, "suspected_collapse": collapse,
        }
        if collapse:
            result["warnings"].append("Suspected collapse: retain the failed run in estimates; inspect before interpreting a method advantage.")
        elif dead_layer:
            result["warnings"].append("A layer reached >=99% dead in the late window; inspect layer diagnostics.")
        runtime = float(summary.get("wall_time_s", np.nan))
        if not np.isfinite(runtime) or runtime <= 0:
            raise ValueError("missing/nonfinite/nonpositive recorded runtime")
        diagnostics = {
            "runtime_minutes": runtime / 60,
            "late_online_accuracy": accuracy,
            "late_loss": loss,
        }
        for source, key in (
            ("dead_exact_count", "final_dead_fraction"),
            ("dead_exact_count_ref", "final_dead_fraction_ref"),
            ("dormant_frac_tau_0p1", "final_dormant_tau0p1"),
            ("dormant_frac_tau_0p1_ref", "final_dormant_tau0p1_ref"),
        ):
            diagnostics[key] = _pooled_fraction(final, source)
        if "probe_accuracy_ref" in late_tasks:
            _finite(late_tasks, ("probe_accuracy_ref",), "tasks")
            diagnostics["late_reference_accuracy"] = iqm(late_tasks["probe_accuracy_ref"].to_numpy())
        recycling = _read_table(run_dir, "recycling")
        if recycling.empty:
            recorded = int(tasks["n_recycled"].sum()) if "n_recycled" in tasks else None
            if result["arm"] != "none" and recorded != 0:
                raise ValueError("missing recycling composition despite unknown/nonzero reset count")
            diagnostics["reset_count"] = 0
            diagnostics["dead_share_of_resets"] = None
            diagnostics["alive_share_of_resets"] = None
        else:
            _finite(recycling, ("k", "n_dead_exact", "n_alive_but_quiet"), "recycling")
            if (recycling[["k", "n_dead_exact", "n_alive_but_quiet"]] < 0).any().any() or not np.array_equal(recycling["k"].to_numpy(), (recycling["n_dead_exact"] + recycling["n_alive_but_quiet"]).to_numpy()):
                raise ValueError("recycled composition does not partition selected units")
            total = int(recycling["k"].sum())
            if "n_recycled" in tasks and total != int(tasks["n_recycled"].sum()):
                raise ValueError("recycling and task reset totals disagree")
            diagnostics["reset_count"] = total
            diagnostics["dead_share_of_resets"] = float(recycling["n_dead_exact"].sum() / total) if total else None
            diagnostics["alive_share_of_resets"] = float(recycling["n_alive_but_quiet"].sum() / total) if total else None
            for suffix in ("_ref",):
                col = f"n_dead_exact{suffix}"
                diagnostics[f"dead_share_of_resets{suffix}"] = float(recycling[col].sum() / total) if total and col in recycling else None
        learning = _read_table(run_dir, "learning_degree")
        diagnostics["learning_degree_log_present"] = not learning.empty
        diagnostics["learning_degree_rows"] = len(learning)
        if result["arm"] in CANDIDATES and learning.empty:
            result["warnings"].append("No learning_degree monitor log: prospective-trigger mechanism cannot be evaluated.")
        result["diagnostics"] = diagnostics
        result["_scores"] = scores.tolist()
        result["status"] = "complete"
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["status"] = "invalid"
        result["errors"].append(str(error))
    return result


def analyze(runs_root: Path, expected_configs: Path, plan: dict) -> dict:
    """Require the declared 7 x 5 matrix; never quietly discard absent seeds."""
    configs = [_read_json(path) for path in sorted(Path(expected_configs).glob("*.json"))]
    if not configs:
        raise ValueError(f"no expected run configs in {expected_configs}")
    pairs = [(arm_label(config), int(config["seed"])) for config in configs]
    run_ids = [config["run_id"] for config in configs]
    if len(set(run_ids)) != len(run_ids) or len(set(pairs)) != len(pairs):
        raise ValueError("duplicate declared run_id or arm/seed")
    if set(pairs) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("expected configs must declare all seven arms and fresh paired seeds 10..14 (35 runs)")
    if any(int(config["data"]["n_tasks"]) != 200 for config in configs):
        raise ValueError("fuzzy_v1 comparison requires 200 tasks in every declared run")
    runs_root = Path(runs_root)
    runs = [inspect_run(runs_root / config["run_id"], config, plan) for config in configs]
    unexpected = sorted(path.name for path in runs_root.glob("fuzzy_v1_*") if path.is_dir() and path.name not in run_ids)
    stats = plan["statistics"]
    boot = dict(n_bootstrap=int(stats["n_bootstrap"]), confidence=float(stats["confidence"]), seed=int(stats["bootstrap_seed"]))
    family_confidence = 1 - (1 - boot["confidence"]) / FAMILY_SIZE
    report = {
        "experiment": "fuzzy_v1", "stage": "exploratory single-dataset pilot",
        "plan_frozen_at": plan.get("frozen_at"), "task_window": plan["windows"]["late"],
        "outcome": plan["outcome_measure"]["primary"], "expected_seeds": list(SEEDS),
        "expected_runs": 35, "complete_runs": sum(run["status"] == "complete" for run in runs),
        "unexpected_run_directories": unexpected,
        "multiplicity": {
            "family_size": FAMILY_SIZE, "family": "two candidates x five baselines",
            "method": "nominal Bonferroni-adjusted seed-block percentile intervals",
            "family_confidence": boot["confidence"], "per_comparison_confidence": family_confidence,
        },
        "limitations": [
            "Only Permuted-MNIST, one MLP/training configuration and five fresh seeds; not a SOTA or generalisation claim.",
            "Historical task-stratified intervals independently resample seeds per task and do not preserve within-seed temporal covariance.",
            "Seed-block intervals preserve trajectories but remain approximate percentile intervals with only five independent seeds.",
            "Bonferroni adjustments are nominal bootstrap sensitivity intervals, not an exact family-wise significance guarantee.",
            "CI exclusion of zero alone is neither practical superiority nor evidence of equivalence; inspect effect sizes and compute/reset costs.",
            "Runtime includes research probes and monitoring, so it does not isolate intrinsic reset-algorithm overhead.",
            "Current-probe alive status at reset does not prove future death was prevented; a counterfactual/control is needed.",
        ],
        "runs": runs, "arms": {}, "comparisons": [],
    }
    matrices = {}
    for arm in ARMS:
        group = sorted([run for run in runs if run["arm"] == arm], key=lambda run: run["seed"])
        missing = [run["seed"] for run in group if run["status"] != "complete"]
        item = {"seeds": [run["seed"] for run in group], "unusable_or_missing_seeds": missing}
        if not missing:
            scores = _matrix([run["_scores"] for run in group])
            matrices[arm] = scores
            item["late_accuracy_pct"] = {
                "historical_task_stratified": _estimate_dict(stratified_bootstrap(scores, **boot), 100),
                "seed_block_sensitivity": _estimate_dict(seed_block_bootstrap(scores, **boot), 100),
            }
            item["has_health_warning"] = any(run["health"].get("suspected_collapse", False) for run in group)
            item["diagnostics"] = {}
            # Every scalar diagnostic is bootstrapped over complete seed rows.
            # Missing values remain explicitly unavailable; they are never averaged away.
            keys = sorted(set().union(*(run["diagnostics"].keys() for run in group)))
            for key in keys:
                values = [run["diagnostics"].get(key) for run in group]
                if key == "learning_degree_log_present":
                    item["diagnostics"][key] = {"present_seeds": [run["seed"] for run in group if run["diagnostics"].get(key)]}
                elif any(value is None for value in values):
                    item["diagnostics"][key] = {"estimate": None, "missing_seeds": [run["seed"] for run, value in zip(group, values) if value is None]}
                else:
                    item["diagnostics"][key] = _estimate_dict(seed_block_bootstrap(np.asarray(values), **boot))
        report["arms"][arm] = item
    for candidate in CANDIDATES:
        for baseline in BASELINES:
            comparison = {"candidate": candidate, "baseline": baseline, "status": "unavailable_incomplete_arm"}
            if candidate in matrices and baseline in matrices:
                a, b = matrices[candidate], matrices[baseline]
                adjusted = dict(boot, confidence=family_confidence)
                comparison.update({
                    "status": "available", "paired_seeds": list(SEEDS),
                    "delta_accuracy_pp": {
                        "historical_task_stratified": _estimate_dict(stratified_bootstrap_difference(a, b, paired=True, **boot), 100),
                        "seed_block_sensitivity": _estimate_dict(seed_block_bootstrap(a, b, **boot), 100),
                        "seed_block_bonferroni_sensitivity": _estimate_dict(seed_block_bootstrap(a, b, **adjusted), 100),
                    },
                })
            report["comparisons"].append(comparison)
    for run in runs:
        run.pop("_scores", None)
    report["status"] = "complete" if report["complete_runs"] == report["expected_runs"] else "incomplete_or_invalid"
    report["has_health_warnings"] = any(run["health"].get("suspected_collapse", False) for run in runs)
    return report


def _format_estimate(value: dict | None) -> str:
    if not value or "point" not in value:
        return "unavailable"
    return f"{value['point']:.3f} [{value['ci_lo']:.3f}, {value['ci_hi']:.3f}]"


def write_report(report: dict, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Fuzzy-reset pilot comparison", "", report["stage"] + ".",
        f"Expected/completed: {report['expected_runs']}/{report['complete_runs']}; status: {report['status']}.",
        "Fresh paired seeds: 10, 11, 12, 13, 14. Late window: tasks 151-200. Estimates: IQM.", "",
        "| Arm | Historical accuracy % [95% CI] | Seed-block accuracy % [95% CI] |", "|---|---:|---:|",
    ]
    rows = []
    for arm, item in report["arms"].items():
        estimates = item.get("late_accuracy_pct", {})
        historical = estimates.get("historical_task_stratified")
        block = estimates.get("seed_block_sensitivity")
        lines.append(f"| {arm} | {_format_estimate(historical)} | {_format_estimate(block)} |")
        row = {"arm": arm, "missing_seeds": ",".join(map(str, item["unusable_or_missing_seeds"]))}
        for prefix, estimate in (("historical_accuracy_pct", historical), ("seed_block_accuracy_pct", block)):
            for key in ("point", "ci_lo", "ci_hi"):
                row[f"{prefix}_{key}"] = None if estimate is None else estimate[key]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    lines += ["", "| Candidate - baseline | Historical delta pp [95% CI] | Seed-block delta pp [95% CI] | Bonferroni seed-block delta pp [99.5% CI] |", "|---|---:|---:|---:|"]
    comparisons = []
    for item in report["comparisons"]:
        estimates = item.get("delta_accuracy_pp", {})
        row = {key: item[key] for key in ("candidate", "baseline", "status")}
        formatted = []
        for key in ("historical_task_stratified", "seed_block_sensitivity", "seed_block_bonferroni_sensitivity"):
            estimate = estimates.get(key)
            formatted.append(_format_estimate(estimate))
            for component in ("point", "ci_lo", "ci_hi"):
                row[f"{key}_{component}_pp"] = None if estimate is None else estimate[component]
        comparisons.append(row)
        lines.append(f"| {item['candidate']} - {item['baseline']} | " + " | ".join(formatted) + " |")
    pd.DataFrame(comparisons).to_csv(out / "comparisons.csv", index=False)
    run_rows = []
    lines += ["", "## Completeness and health audit", ""]
    for run in report["runs"]:
        run_rows.append({**{key: run[key] for key in ("run_id", "arm", "seed", "status")}, **run["diagnostics"], **run["health"], "errors": "; ".join(run["errors"]), "warnings": "; ".join(run["warnings"])})
        if run["status"] != "complete" or run["errors"] or run["warnings"]:
            lines.append(f"- {run['run_id']}: {run['status']}. " + "; ".join(run["errors"] + run["warnings"]))
    if all(run["status"] == "complete" and not run["errors"] and not run["warnings"] for run in report["runs"]):
        lines.append("All 35 runs have complete required logs and no detected health warnings.")
    if report["unexpected_run_directories"]:
        lines.append("Unexpected directories (not pooled): " + ", ".join(report["unexpected_run_directories"]))
    pd.DataFrame(run_rows).to_csv(out / "per_run.csv", index=False)
    lines += ["", "## Interpretation limits", ""] + [f"- {value}" for value in report["limitations"]]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--expected-configs", type=Path, required=True, help="directory with the declared 35 run configs")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    report = analyze(args.runs_root, args.expected_configs, plan)
    report["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    write_report(report, args.out)
    print(f"Fuzzy comparison: {report['complete_runs']}/35 complete; {report['status']}; outputs: {args.out}")
    return 0 if report["status"] == "complete" and not report["has_health_warnings"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
