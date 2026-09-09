"""Analyze V2.3 against exact yokes and reused seed-15–17 comparators."""

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
from scripts.analyze_fuzzy_v22_extract import exact_schedule, task_matrices
from src.analysis.stats import iqm


SEEDS = (15, 16, 17)
OLD = ROOT / "remote_runs/neuron-death-fuzzy-v1-v2dev0907-9c5b295b/analysis_unpacked"
V22 = ROOT / "remote_runs/neuron-death-fuzzy-v1-v22sota0908-78924178/analysis_unpacked"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def variant_label(config, configs):
    recycling = config["recycling"]
    if recycling["kind"] == "fuzzy_v2":
        learning = recycling["learning_degree"]
    else:
        source = recycling["learning_degree"]["yoked_from_run_id"]
        learning = configs[source]["recycling"]["learning_degree"]
    threshold = float(learning["degree_threshold"])
    effective_saliency = threshold * float(learning["saliency_full_ratio"]) / 0.1
    base = f"t{threshold:.2f}_es{effective_saliency:.2f}"
    return base + ("_yoked" if recycling["kind"] == "fuzzy_v2_yoked_random" else "")


def analyze(extract: Path, plan_path: Path, old: Path, v22: Path):
    plan = read_json(plan_path)
    records = read_json(extract / "runs.json")
    actual = {row["run_id"]: row for row in records}
    expected = {row["run_id"]: row for row in plan["run_manifest"]}
    if len(records) != 24 or set(actual) != set(expected):
        raise ValueError("V2.3 extract does not match its frozen 24-run manifest")
    for run_id, row in actual.items():
        if row.get("summary", {}).get("status") != "complete":
            raise ValueError(f"{run_id}: incomplete")
        if stored_config_hash(row["config"]) != expected[run_id]["config_hash"]:
            raise ValueError(f"{run_id}: resolved config hash differs from frozen plan")
    configs = {run_id: row["config"] for run_id, row in actual.items()}
    labels = {run_id: variant_label(config, configs) for run_id, config in configs.items()}
    seeds = {run_id: int(config["seed"]) for run_id, config in configs.items()}
    matrices, ids = task_matrices(extract, labels, seeds)

    old_records = read_json(old / "runs.json")
    old_labels, old_seeds = {}, {}
    for row in old_records:
        kind = row["config"]["recycling"]["kind"]
        if kind in {"none", "redo", "regrama", "snr"}:
            old_labels[row["run_id"]] = kind
            old_seeds[row["run_id"]] = int(row["config"]["seed"])
    old_matrices, _ = task_matrices(old, old_labels, old_seeds)

    v22_records = read_json(v22 / "runs.json")
    v22_labels, v22_seeds = {}, {}
    for row in v22_records:
        config = row["config"]
        learning = config["recycling"].get("learning_degree", {})
        if (config["recycling"]["kind"] == "fuzzy_v2"
                and learning.get("degree_threshold") == 0.6
                and learning.get("patience") == 1
                and learning.get("cooldown_steps") == 500):
            v22_labels[row["run_id"]] = "v22_winner"
            v22_seeds[row["run_id"]] = int(config["seed"])
    v22_matrices, _ = task_matrices(v22, v22_labels, v22_seeds)
    matrices.update(old_matrices)
    matrices.update(v22_matrices)

    recycling = pq.read_table(extract / "recycling.parquet").to_pandas()
    resets = recycling.groupby("run_id").k.sum().to_dict()
    dead = recycling.groupby("run_id").n_dead_exact.sum().to_dict()
    yokes = {}
    for run_id, config in configs.items():
        if config["recycling"]["kind"] != "fuzzy_v2_yoked_random":
            continue
        source = config["recycling"]["learning_degree"]["yoked_from_run_id"]
        target_schedule = exact_schedule(recycling, source)
        random_schedule = exact_schedule(recycling, run_id)
        if not target_schedule.equals(random_schedule):
            raise ValueError(f"{run_id}: exact yoke mismatch")
        yokes[run_id] = {"target": source, "exact_schedule_match": True,
                         "resets": int(target_schedule.k.sum())}

    boot = {"n_bootstrap": 10_000, "confidence": 0.95}
    target_labels = sorted({label for label in labels.values() if not label.endswith("_yoked")})
    report = {"experiment": "fuzzy_v23_dev", "new_complete_runs": 24,
              "late_window": [150, 199], "arms": {}, "comparisons": {},
              "yoked_validation": yokes,
              "limitations": ["Three reused development seeds cannot establish SOTA.",
                              "A winning setting requires a new held-out paired-seed confirmation."]}
    for label in sorted(set(labels.values())):
        matrix = matrices[label]
        run_ids = ids[label]
        counts = [int(resets[run_id]) for run_id in run_ids]
        shares = [dead[run_id] / resets[run_id] for run_id in run_ids]
        report["arms"][label] = {
            "late_accuracy_pct": bootstrap_contrast([matrix], [1], seed=0, scale=100, **boot),
            "mean_resets": float(np.mean(counts)),
            "dead_share_of_resets_pct": float(100 * np.mean(shares)),
            "per_seed": {str(seeds[run_id]): {"late_accuracy_pct": 100 * iqm(row),
                                                "resets": count}
                         for run_id, row, count in zip(run_ids, matrix, counts)},
        }
    comparators = ["v22_winner", "none", "redo", "regrama", "snr"]
    for label in target_labels:
        paired = label + "_yoked"
        report["comparisons"][label] = {
            other: bootstrap_contrast([matrices[label], matrices[other]], [1, -1],
                                      seed=10, scale=100, **boot)
            for other in [paired, *comparators]
        }
    ranking = sorted(target_labels, key=lambda name: (-iqm(matrices[name]), report["arms"][name]["mean_resets"]))
    winner = ranking[0]
    vs_snr = report["comparisons"][winner]["snr"]
    report["target_ranking"] = [
        {"rank": rank + 1, "arm": name, "late_accuracy_pct": 100 * iqm(matrices[name]),
         "mean_resets": report["arms"][name]["mean_resets"]}
        for rank, name in enumerate(ranking)
    ]
    report["development_winner"] = {
        "arm": winner,
        "late_accuracy_pct": 100 * iqm(matrices[winner]),
        "beats_snr_point": bool(vs_snr["point"] > 0),
        "beats_snr_ci": bool(vs_snr["ci_lo"] > 0),
        "selection_advantage_ci": report["comparisons"][winner][winner + "_yoked"],
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("extract", type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/fuzzy_v23_dev_plan.json")
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--v22", type=Path, default=V22)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.extract, args.plan, args.old, args.v22)
    report["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
