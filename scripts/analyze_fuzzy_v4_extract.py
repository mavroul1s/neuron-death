"""Analyze RA-SNR V4 against the paired, reused seed-15--17 leaders."""

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
from scripts.analyze_fuzzy_v22_extract import task_matrices
from src.analysis.stats import iqm


SEEDS = (15, 16, 17)
OLD = ROOT / "remote_runs/neuron-death-fuzzy-v1-v2dev0907-9c5b295b/analysis_unpacked"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def variant_label(config):
    recycling = config["recycling"]
    boost = float(recycling["recovery_boost"])
    steps = int(recycling["recovery_steps"])
    return f"rasnr_b{boost:g}_h{steps}"


def analyze(extract: Path, plan_path: Path, old: Path):
    plan = read_json(plan_path)
    records = read_json(extract / "runs.json")
    actual = {row["run_id"]: row for row in records}
    expected = {row["run_id"]: row for row in plan["run_manifest"]}
    if len(records) != 12 or set(actual) != set(expected):
        raise ValueError("V4 extract does not match its frozen 12-run manifest")
    for run_id, row in actual.items():
        if row.get("summary", {}).get("status") != "complete":
            raise ValueError(f"{run_id}: incomplete")
        if stored_config_hash(row["config"]) != expected[run_id]["config_hash"]:
            raise ValueError(f"{run_id}: resolved config hash differs from frozen plan")

    configs = {run_id: row["config"] for run_id, row in actual.items()}
    labels = {run_id: variant_label(config) for run_id, config in configs.items()}
    seeds = {run_id: int(config["seed"]) for run_id, config in configs.items()}
    matrices, ids = task_matrices(extract, labels, seeds)

    old_labels, old_seeds = {}, {}
    for row in read_json(old / "runs.json"):
        kind = row["config"]["recycling"]["kind"]
        if kind in {"none", "redo", "regrama", "snr"}:
            old_labels[row["run_id"]] = kind
            old_seeds[row["run_id"]] = int(row["config"]["seed"])
    old_matrices, _ = task_matrices(old, old_labels, old_seeds)
    matrices.update(old_matrices)

    recycling = pq.read_table(extract / "recycling.parquet").to_pandas()
    resets = recycling.groupby("run_id").k.sum().to_dict()
    dead = recycling.groupby("run_id").n_dead_exact.sum().to_dict()
    event_steps = recycling[recycling.k.gt(0)].groupby("run_id").step.nunique().to_dict()

    boot = {"n_bootstrap": 10_000, "confidence": 0.95}
    target_labels = sorted(set(labels.values()))
    report = {
        "experiment": "fuzzy_v4_dev",
        "method": "Recovery-Accelerated SNR (RA-SNR)",
        "new_complete_runs": 12,
        "late_window": [150, 199],
        "arms": {},
        "comparisons": {},
        "limitations": [
            "Three reused development seeds cannot establish SOTA.",
            "A winning recovery setting requires new held-out paired-seed confirmation.",
        ],
    }
    for label in target_labels:
        matrix, run_ids = matrices[label], ids[label]
        counts = [int(resets[run_id]) for run_id in run_ids]
        dead_shares = [
            float(dead[run_id] / resets[run_id]) if resets[run_id] else float("nan")
            for run_id in run_ids
        ]
        active_events = [int(event_steps.get(run_id, 0)) for run_id in run_ids]
        report["arms"][label] = {
            "late_accuracy_pct": bootstrap_contrast(
                [matrix], [1], seed=0, scale=100, **boot
            ),
            "mean_resets": float(np.mean(counts)),
            "mean_active_event_steps": float(np.mean(active_events)),
            "dead_share_of_resets_pct": float(100 * np.nanmean(dead_shares)),
            "per_seed": {
                str(seeds[run_id]): {
                    "late_accuracy_pct": 100 * iqm(row),
                    "resets": count,
                    "active_event_steps": events,
                }
                for run_id, row, count, events in zip(
                    run_ids, matrix, counts, active_events
                )
            },
        }

    comparators = ["snr", "redo", "regrama", "none"]
    for label in target_labels:
        report["comparisons"][label] = {
            other: bootstrap_contrast(
                [matrices[label], matrices[other]],
                [1, -1],
                seed=10,
                scale=100,
                **boot,
            )
            for other in comparators
        }

    def rank_key(name):
        config = configs[ids[name][0]]["recycling"]
        recovery_dose = float(config["recovery_boost"]) * int(
            config["recovery_steps"]
        )
        return (-iqm(matrices[name]), recovery_dose, report["arms"][name]["mean_resets"])

    ranking = sorted(target_labels, key=rank_key)
    winner = ranking[0]
    report["target_ranking"] = [
        {
            "rank": rank + 1,
            "arm": name,
            "late_accuracy_pct": 100 * iqm(matrices[name]),
            "mean_resets": report["arms"][name]["mean_resets"],
        }
        for rank, name in enumerate(ranking)
    ]
    vs_snr = report["comparisons"][winner]["snr"]
    report["development_winner"] = {
        "arm": winner,
        "late_accuracy_pct": 100 * iqm(matrices[winner]),
        "beats_snr_point": bool(vs_snr["point"] > 0),
        "beats_snr_ci": bool(vs_snr["ci_lo"] > 0),
        "paired_vs_snr_pp": vs_snr,
        "heldout_confirmation_required": bool(vs_snr["point"] > 0),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("extract", type=Path)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "configs/fuzzy_v4_dev_plan.json"
    )
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.extract, args.plan, args.old)
    report["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.out)


if __name__ == "__main__":
    main()
