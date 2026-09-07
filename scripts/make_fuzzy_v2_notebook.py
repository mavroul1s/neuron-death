"""Generate the thin Kaggle entry point for temporal fuzzy V2."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {},
             "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


BODY = r'''
import glob, json, os, shutil, subprocess, sys
from pathlib import Path

REPO = Path(os.environ["NEURON_DEATH_SOURCE"])
DATA = Path(os.environ["NEURON_DEATH_DATA"])
os.chdir(REPO)
assert (REPO / "src/temporal_fuzzy.py").is_file()
assert (DATA / "mnist.npz").is_file()
assert __import__("torch").cuda.device_count() == 2, "This job requires two independent T4 GPUs"

configs = sorted(glob.glob(str(REPO / "configs/fuzzy_v2_dev/*.json")))
assert len(configs) == 18, f"expected 18 frozen configs, found {len(configs)}"
plan = json.loads((REPO / "configs/fuzzy_v2_dev_plan.json").read_text())
assert plan["status"] == "FROZEN_BEFORE_FIRST_RUN"
assert plan["setting"]["seeds"] == [15, 16, 17]
target = [p for p in configs if json.loads(Path(p).read_text())["recycling"]["kind"] == "fuzzy_v2"]
rest = [p for p in configs if p not in target]
assert len(target) == 3 and len(rest) == 15

RUNS = Path("/kaggle/working/fuzzy_v2_dev_runs")
print("FUZZY_V2_TARGETED_PHASE_LAUNCHING", flush=True)
subprocess.run([sys.executable, "scripts/launch_pair.py", *target,
                "--gpus", "0,1", "--runs-root", str(RUNS),
                "--data-root", str(DATA), "--budget-hours", "4"],
               cwd=REPO, check=True)
print("FUZZY_V2_TARGETED_PHASE_COMPLETE", flush=True)

print("FUZZY_V2_CONTROL_PHASE_LAUNCHING", flush=True)
subprocess.run([sys.executable, "scripts/launch_pair.py", *rest,
                "--gpus", "0,1", "--runs-root", str(RUNS),
                "--data-root", str(DATA), "--budget-hours", "7"],
               cwd=REPO, check=True)

summaries = []
for config in configs:
    run_id = json.loads(Path(config).read_text())["run_id"]
    summary = json.loads((RUNS / run_id / "summary.json").read_text())
    assert summary["status"] == "complete", run_id
    summaries.append(summary)
assert len(summaries) == 18

EXTRACT = Path("/kaggle/working/fuzzy_v2_dev_extract")
subprocess.run([sys.executable, "scripts/make_analysis_extract.py",
                "--runs-root", str(RUNS), "--pattern", "fuzzy_v2_dev_*",
                "--out", str(EXTRACT), "--with-c4", "--zip"],
               cwd=REPO, check=True)
shutil.make_archive("/kaggle/working/fuzzy_v2_dev_full_runs", "zip", RUNS)
print("FUZZY_V2_DEVELOPMENT_COMPLETE", flush=True)
'''


def main() -> None:
    nb = {
        "cells": [
            cell("markdown", "# Temporal fuzzy V2 development\n\nThin entry point; all method logic lives in `src/`."),
            cell("code", BODY),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / "notebooks/fuzzy_v2_development.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
