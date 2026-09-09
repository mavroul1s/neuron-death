"""Compact post-hoc diagnostics for the completed BT-FR development sweep."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def iqm(values) -> float:
    values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    n = values.size
    lo, hi = n // 4, n - n // 4
    return float(values[lo:hi].mean())


def main(root: Path) -> dict:
    runs = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    configs = {row["run_id"]: row["config"] for row in runs}
    summaries = {row["run_id"]: row["summary"] for row in runs}
    labels = {}
    for run_id, config in configs.items():
        learning = config["recycling"]["learning_degree"]
        rate = float(learning["replacement_rate"])
        labels[run_id] = f"r{rate:.5f}" + (
            "_yoked" if config["recycling"]["kind"].endswith("_yoked_random") else ""
        )

    tasks = pq.read_table(root / "tasks.parquet", columns=[
        "run_id", "task_idx", "probe_point", "online_accuracy"
    ]).to_pandas()
    tasks = tasks[tasks.probe_point.eq("task_end")].copy()
    tasks["label"] = tasks.run_id.map(labels)
    windows = {}
    for label, frame in tasks.groupby("label"):
        windows[label] = {
            f"tasks_{start}_{start + 49}": 100 * iqm(
                frame.loc[frame.task_idx.between(start, start + 49), "online_accuracy"]
            ) for start in (0, 50, 100, 150)
        }

    recycling = pq.read_table(root / "recycling.parquet", columns=[
        "run_id", "layer_idx", "k", "n_dead_exact", "n_alive_but_quiet"
    ]).to_pandas()
    recycling["label"] = recycling.run_id.map(labels)
    composition = {}
    for label, frame in recycling.groupby("label"):
        per_layer = frame.groupby("layer_idx")[["k", "n_dead_exact", "n_alive_but_quiet"]].sum()
        composition[label] = {
            str(int(layer)): {
                "resets": int(row.k),
                "dead_share_pct": 100 * float(row.n_dead_exact / row.k),
                "alive_but_quiet_share_pct": 100 * float(row.n_alive_but_quiet / row.k),
            } for layer, row in per_layer.iterrows()
        }

    metrics = pq.read_table(root / "metrics.parquet", columns=[
        "run_id", "task_idx", "probe_point", "layer_idx", "dead_exact_frac", "erank"
    ]).to_pandas()
    final = metrics[(metrics.probe_point == "task_end") & (metrics.task_idx == 199)].copy()
    final["label"] = final.run_id.map(labels)
    final_metrics = {
        label: {"dead_exact_pct": 100 * float(frame.dead_exact_frac.mean()),
                "erank": float(frame.erank.mean())}
        for label, frame in final.groupby("label")
    }

    selected = pq.read_table(root / "learning_degree.parquet", columns=[
        "run_id", "layer_idx", "neuron_idx", "selected", "degree",
        "process_health", "saliency_health", "activity_health", "gradient_health"
    ], filters=[("selected", "=", True)]).to_pandas()
    selected["label"] = selected.run_id.map(labels)
    concentration = {}
    for label, frame in selected.groupby("label"):
        counts = frame.groupby(["run_id", "layer_idx", "neuron_idx"]).size()
        per_run = []
        for (run_id, layer), layer_counts in counts.groupby(level=[0, 1]):
            values = layer_counts.to_numpy(dtype=float)
            per_run.append({
                "run_id": run_id, "layer": int(layer),
                "unique_units_reset": int(values.size),
                "mean_resets_per_touched_unit": float(values.mean()),
                "max_resets_one_unit": int(values.max()),
            })
        concentration[label] = {
            "mean_unique_units_per_layer": float(np.mean([x["unique_units_reset"] for x in per_run])),
            "mean_resets_per_touched_unit": float(np.mean([x["mean_resets_per_touched_unit"] for x in per_run])),
            "max_resets_one_unit": int(max(x["max_resets_one_unit"] for x in per_run)),
            "mean_selected_degree": float(frame.degree.mean()),
            "mean_selected_process_health": float(frame.process_health.mean()),
            "mean_selected_saliency_health": float(frame.saliency_health.mean()),
        }

    return {
        "complete_runs": sum(s.get("status") == "complete" for s in summaries.values()),
        "gpu_hours": float(sum(s["gpu_hours"] for s in summaries.values())),
        "accuracy_windows_pct": windows,
        "composition_by_layer": composition,
        "final_metrics": final_metrics,
        "target_selection_concentration": concentration,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = main(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)
