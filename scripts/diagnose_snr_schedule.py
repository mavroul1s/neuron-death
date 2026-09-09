"""Describe the completed SNR development runs before designing a hybrid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main(root: Path, output: Path) -> None:
    records = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    snr_ids = {
        row["run_id"] for row in records
        if row["config"]["recycling"]["kind"] == "snr"
    }
    frame = pq.read_table(root / "recycling.parquet").to_pandas()
    frame = frame[frame.run_id.isin(snr_ids)].copy()
    result = {"runs": {}}
    for run_id, run in frame.groupby("run_id"):
        positive = run[run.k > 0]
        event_steps = np.sort(positive.step.unique())
        layers = {}
        for layer, part in run.groupby("layer_idx"):
            k = part.k.to_numpy(dtype=int)
            layers[str(int(layer))] = {
                "resets": int(k.sum()),
                "positive_events": int((k > 0).sum()),
                "mean_k_when_positive": float(k[k > 0].mean()),
                "median_k_when_positive": float(np.median(k[k > 0])),
                "max_k": int(k.max()),
                "dead_share_pct": 100 * float(part.n_dead_exact.sum() / k.sum()),
            }
        result["runs"][run_id] = {
            "resets": int(run.k.sum()),
            "event_steps": int(event_steps.size),
            "median_gap_steps": float(np.median(np.diff(event_steps))),
            "p90_gap_steps": float(np.quantile(np.diff(event_steps), 0.9)),
            "layers": layers,
        }
    result["mean_total_resets"] = float(np.mean([r["resets"] for r in result["runs"].values()]))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.out)
