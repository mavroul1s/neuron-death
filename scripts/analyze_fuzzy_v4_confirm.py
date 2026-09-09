"""Analyze the locked RA-SNR winner on held-out seeds 18--22."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_fuzzy_v21_extract import bootstrap_contrast, stored_config_hash
from src.analysis.stats import iqm


SEEDS = (18, 19, 20, 21, 22)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def task_matrices(path, labels, seeds, expected_seeds=SEEDS):
    tasks = pq.read_table(path / "tasks.parquet", columns=[
        "run_id", "task_idx", "probe_point", "online_accuracy"]).to_pandas()
    tasks = tasks[(tasks.probe_point == "task_end") & tasks.task_idx.between(150, 199)]
    matrices, ids = {}, {}
    for label in sorted(set(labels.values())):
        run_ids = sorted((rid for rid, value in labels.items() if value == label), key=lambda rid: seeds[rid])
        if [seeds[rid] for rid in run_ids] != list(expected_seeds):
            raise ValueError(f"{label}: wrong held-out seed grid")
        matrix = np.stack([tasks[tasks.run_id.eq(rid)].sort_values("task_idx").online_accuracy.to_numpy(float)
                           for rid in run_ids])
        if matrix.shape != (len(expected_seeds), 50) or not np.isfinite(matrix).all():
            raise ValueError(f"{label}: incomplete late-window matrix")
        matrices[label], ids[label] = matrix, run_ids
    return matrices, ids


def analyze(extract: Path, plan_path: Path):
    plan = read_json(plan_path)
    records = read_json(extract / "runs.json")
    expected = {row["run_id"]: row for row in plan["run_manifest"]}
    actual = {row["run_id"]: row for row in records}
    if len(records) != 20 or set(actual) != set(expected):
        raise ValueError("Confirmation extract differs from its frozen 20-run manifest")
    labels, seeds = {}, {}
    for run_id, row in actual.items():
        if row.get("summary", {}).get("status") != "complete":
            raise ValueError(f"{run_id}: incomplete")
        if stored_config_hash(row["config"]) != expected[run_id]["config_hash"]:
            raise ValueError(f"{run_id}: config hash mismatch")
        labels[run_id] = expected[run_id]["arm"]
        seeds[run_id] = int(row["config"]["seed"])
    matrices, ids = task_matrices(extract, labels, seeds)

    recycling = pq.read_table(extract / "recycling.parquet").to_pandas()
    resets = recycling.groupby("run_id").k.sum().to_dict()
    gpu_hours = sum(float(row["summary"]["gpu_hours"]) for row in records)
    boot = {"n_bootstrap": 20_000, "confidence": 0.95}
    report = {
        "experiment": "fuzzy_v4_heldout_confirmation",
        "heldout_seeds": list(SEEDS),
        "complete_runs": 20,
        "summed_gpu_hours": gpu_hours,
        "late_window": [150, 199],
        "arms": {},
        "comparisons": {},
    }
    for arm in ("rasnr", "snr", "redo", "regrama"):
        run_ids, matrix = ids[arm], matrices[arm]
        counts = [int(resets.get(run_id, 0)) for run_id in run_ids]
        report["arms"][arm] = {
            "late_accuracy_pct": bootstrap_contrast([matrix], [1], seed=0, scale=100, **boot),
            "mean_resets": float(np.mean(counts)),
            "per_seed": {str(seeds[run_id]): {"late_accuracy_pct": 100 * iqm(row), "resets": count}
                         for run_id, row, count in zip(run_ids, matrix, counts)},
        }
    for other in ("snr", "redo", "regrama"):
        report["comparisons"][f"rasnr_minus_{other}_pp"] = bootstrap_contrast(
            [matrices["rasnr"], matrices[other]], [1, -1], seed=10, scale=100, **boot)
    primary = report["comparisons"]["rasnr_minus_snr_pp"]
    report["verdict"] = {
        "heldout_gain_over_snr_confirmed": bool(primary["ci_lo"] > 0),
        "rasnr_best_point_estimate": bool(all(
            report["comparisons"][f"rasnr_minus_{other}_pp"]["point"] > 0
            for other in ("snr", "redo", "regrama"))),
        "claim": ("CONFIRMED" if primary["ci_lo"] > 0 else "NOT_CONFIRMED"),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("extract", type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/fuzzy_v4_confirm_plan.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.extract, args.plan)
    report["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
