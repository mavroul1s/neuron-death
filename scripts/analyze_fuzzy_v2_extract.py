"""Audit and summarize the completed temporal-fuzzy V2 extract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.fuzzy_comparison import seed_block_bootstrap
from src.analysis.stats import iqm, stratified_bootstrap, stratified_bootstrap_difference


def label(config: dict) -> str:
    recycling = config["recycling"]
    kind = recycling["kind"]
    if kind == "redo":
        return "redo"
    if kind == "regrama":
        return "regrama"
    if kind == "snr":
        return "snr"
    return kind


def est(value, scale=1.0) -> dict:
    result = value.as_dict()
    return {key: (number * scale if key in {"point", "ci_lo", "ci_hi"} else number)
            for key, number in result.items()}


def run_level_iqm_ci(values: list[float], seed: int = 0) -> dict:
    array = np.asarray(values, dtype=np.float64)[:, None]
    return est(seed_block_bootstrap(array, n_bootstrap=10_000, seed=seed))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("extract", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.extract
    runs = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    if len(runs) != 18 or any(item["summary"]["status"] != "complete" for item in runs):
        raise RuntimeError("V2 extract does not contain exactly 18 completed runs")
    metadata = {
        item["run_id"]: {
            "arm": label(item["config"]), "seed": int(item["config"]["seed"]),
            "runtime_minutes": float(item["summary"]["wall_time_s"]) / 60,
        }
        for item in runs
    }
    expected = {(arm, seed) for arm in
                ("none", "redo", "regrama", "snr", "fuzzy_v2", "fuzzy_v2_yoked_random")
                for seed in (15, 16, 17)}
    if {(v["arm"], v["seed"]) for v in metadata.values()} != expected:
        raise RuntimeError("V2 arm/seed grid is incomplete or duplicated")

    tasks = pq.read_table(root / "tasks.parquet", columns=[
        "run_id", "task_idx", "probe_point", "online_accuracy", "n_recycled"
    ]).to_pandas()
    tasks = tasks[(tasks.probe_point == "task_end") & tasks.task_idx.between(0, 199)]
    counts = tasks.groupby("run_id").task_idx.nunique()
    if not counts.eq(200).all() or len(counts) != 18:
        raise RuntimeError("task log is incomplete")

    matrices = {}
    run_accuracy = {}
    for arm in sorted({v["arm"] for v in metadata.values()}):
        ids = sorted((rid for rid, v in metadata.items() if v["arm"] == arm),
                     key=lambda rid: metadata[rid]["seed"])
        matrix = np.stack([
            tasks[(tasks.run_id == rid) & tasks.task_idx.between(150, 199)]
            .sort_values("task_idx").online_accuracy.to_numpy(dtype=float)
            for rid in ids
        ])
        if matrix.shape != (3, 50) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid late matrix for {arm}")
        matrices[arm] = matrix
        for rid, row in zip(ids, matrix):
            run_accuracy[rid] = float(iqm(row))

    metrics = pq.read_table(root / "metrics.parquet", columns=[
        "run_id", "task_idx", "probe_point", "n_neurons", "dead_exact_count",
        "dormant_frac_tau_0p1"
    ]).to_pandas()
    final = metrics[(metrics.probe_point == "task_end") & (metrics.task_idx == 199)].copy()
    final["dormant_count"] = final.dormant_frac_tau_0p1 * final.n_neurons
    per_run_final = final.groupby("run_id").agg(
        dead=("dead_exact_count", "sum"), dormant=("dormant_count", "sum"),
        neurons=("n_neurons", "sum"), layers=("n_neurons", "size"),
    )
    if len(per_run_final) != 18 or not per_run_final.layers.eq(3).all():
        raise RuntimeError("final metric grid is incomplete")

    recycling = pq.read_table(root / "recycling.parquet", columns=[
        "run_id", "step", "layer_idx", "k", "n_dead_exact", "n_alive_but_quiet"
    ]).to_pandas()
    reset_count = recycling.groupby("run_id").k.sum().to_dict()
    dead_resets = recycling.groupby("run_id").n_dead_exact.sum().to_dict()

    # The central mechanism check: every paired random run must reproduce the
    # target run's complete step/layer/cardinality schedule byte-for-byte.
    yoke_checks = {}
    for seed in (15, 16, 17):
        target_id = next(r for r, v in metadata.items()
                         if v["arm"] == "fuzzy_v2" and v["seed"] == seed)
        yoke_id = next(r for r, v in metadata.items()
                       if v["arm"] == "fuzzy_v2_yoked_random" and v["seed"] == seed)
        a = recycling[recycling.run_id == target_id][["step", "layer_idx", "k"]]
        b = recycling[recycling.run_id == yoke_id][["step", "layer_idx", "k"]]
        same = a.sort_values(["step", "layer_idx"]).reset_index(drop=True).equals(
            b.sort_values(["step", "layer_idx"]).reset_index(drop=True)
        )
        yoke_checks[str(seed)] = {"exact_schedule_match": bool(same),
                                  "target_resets": int(a.k.sum()),
                                  "random_resets": int(b.k.sum())}
    if not all(item["exact_schedule_match"] for item in yoke_checks.values()):
        raise RuntimeError("a yoked-random schedule differs from its target")

    report = {"complete_runs": 18, "late_window": [150, 199], "arms": {},
              "comparisons_vs_fuzzy_v2": {}, "yoked_validation": yoke_checks}
    for arm, matrix in matrices.items():
        ids = [rid for rid, v in metadata.items() if v["arm"] == arm]
        dead = [(per_run_final.loc[rid, "dead"] / per_run_final.loc[rid, "neurons"])
                for rid in ids]
        dormant = [(per_run_final.loc[rid, "dormant"] / per_run_final.loc[rid, "neurons"])
                   for rid in ids]
        resets = [float(reset_count.get(rid, 0)) for rid in ids]
        dead_share = [(dead_resets.get(rid, 0) / reset_count[rid])
                      for rid in ids if reset_count.get(rid, 0) > 0]
        report["arms"][arm] = {
            "late_accuracy_pct": est(stratified_bootstrap(matrix, n_bootstrap=10_000, seed=0), 100),
            "late_accuracy_seed_block_pct": est(seed_block_bootstrap(matrix, n_bootstrap=10_000, seed=0), 100),
            "final_dead_pct": run_level_iqm_ci(dead, 1),
            "final_dormant_tau0p1_pct": run_level_iqm_ci(dormant, 2),
            "resets": run_level_iqm_ci(resets, 3),
            "dead_share_of_resets_pct": None if not dead_share else run_level_iqm_ci(dead_share, 4),
            "runtime_minutes": run_level_iqm_ci([metadata[r]["runtime_minutes"] for r in ids], 5),
            "per_seed_late_accuracy_pct": {
                str(metadata[r]["seed"]): run_accuracy[r] * 100 for r in ids
            },
        }
        for name in ("final_dead_pct", "final_dormant_tau0p1_pct", "dead_share_of_resets_pct"):
            value = report["arms"][arm].get(name)
            if value:
                for key in ("point", "ci_lo", "ci_hi"):
                    value[key] *= 100

    target = matrices["fuzzy_v2"]
    for arm, matrix in matrices.items():
        if arm == "fuzzy_v2":
            continue
        report["comparisons_vs_fuzzy_v2"][arm] = {
            "target_minus_arm_pp": est(stratified_bootstrap_difference(
                target, matrix, paired=True, n_bootstrap=10_000, seed=10), 100),
            "target_minus_arm_seed_block_pp": est(seed_block_bootstrap(
                target, matrix, n_bootstrap=10_000, seed=10), 100),
        }

    learning = pq.read_table(root / "learning_degree.parquet", columns=[
        "run_id", "step", "layer_idx", "candidate", "selected", "degree",
        "process_health", "saliency_health"
    ]).to_pandas()
    diagnostics = {}
    for rid, group in learning.groupby("run_id"):
        decisions = group.groupby(["step", "layer_idx"]).agg(
            candidates=("candidate", "sum"), selected=("selected", "sum")
        )
        selected = group[group.selected]
        diagnostics[str(metadata[rid]["seed"])] = {
            "candidate_rows": int(group.candidate.sum()),
            "selected_rows": int(group.selected.sum()),
            "cap_saturation_pct": float((decisions.selected == 37).mean() * 100),
            "selected_degree_median": float(selected.degree.median()),
            "selected_process_health_median": float(selected.process_health.median()),
            "selected_saliency_health_median": float(selected.saliency_health.median()),
        }
    report["fuzzy_v2_monitor"] = diagnostics
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
