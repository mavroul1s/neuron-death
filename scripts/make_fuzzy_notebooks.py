"""Generate the two thin Kaggle entry points for the fuzzy-reset pilot."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": lines(source)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def notebook(title: str, body: str) -> dict:
    return {
        "cells": [
            cell("markdown", f"# {title}\n\nGenerated thin entry point. All method logic lives in `src/`."),
            cell("code", body),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON = r'''
import glob, os, shutil, subprocess, sys
from pathlib import Path

REPO = Path(os.environ["NEURON_DEATH_SOURCE"])
DATA = Path(os.environ["NEURON_DEATH_DATA"])
assert (REPO / "src/train.py").is_file()
assert (DATA / "mnist.npz").is_file()
os.chdir(REPO)
print("CUDA devices:", __import__("torch").cuda.device_count(), flush=True)
assert __import__("torch").cuda.device_count() == 2, "This job requires two independent T4 GPUs"
subprocess.run([sys.executable, "-m", "pytest", "tests", "-q",
                "-p", "no:cacheprovider"], cwd=REPO, check=True)
'''


SMOKE = COMMON + r'''
configs = sorted(glob.glob(str(REPO / "configs/fuzzy_smoke/*.json")))
assert len(configs) == 2, f"expected 2 smoke configs, found {len(configs)}"
RUNS = Path("/kaggle/working/fuzzy_smoke_runs")
subprocess.run([sys.executable, "scripts/launch_pair.py", *configs,
                "--gpus", "0,1", "--runs-root", str(RUNS),
                "--data-root", str(DATA), "--budget-hours", "1"],
               cwd=REPO, check=True)
for config in configs:
    run_id = __import__("json").loads(Path(config).read_text())["run_id"]
    summary = __import__("json").loads((RUNS / run_id / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert (RUNS / run_id / "learning_degree.parquet").is_file()
print("FUZZY_SMOKE_VALIDATED", flush=True)
'''


MAIN = COMMON + r'''
configs = sorted(glob.glob(str(REPO / "configs/fuzzy_v1/*.json")))
assert len(configs) == 35, f"expected 35 frozen configs, found {len(configs)}"
plan = __import__("json").loads((REPO / "configs/fuzzy_v1_plan.json").read_text())
assert plan["status"] == "FROZEN_BEFORE_FIRST_RUN"
assert plan["setting"]["seeds"] == [10, 11, 12, 13, 14]
RUNS = Path("/kaggle/working/fuzzy_v1_runs")
print("FUZZY_MAIN_LAUNCHING", flush=True)
subprocess.run([sys.executable, "scripts/launch_pair.py", *configs,
                "--gpus", "0,1", "--runs-root", str(RUNS),
                "--data-root", str(DATA), "--budget-hours", "10.5"],
               cwd=REPO, check=True)
EXTRACT = Path("/kaggle/working/fuzzy_v1_extract")
subprocess.run([sys.executable, "scripts/make_analysis_extract.py",
                "--runs-root", str(RUNS), "--pattern", "fuzzy_v1_*",
                "--out", str(EXTRACT), "--with-c4", "--zip"],
               cwd=REPO, check=True)
ANALYSIS = Path("/kaggle/working/fuzzy_v1_analysis")
analysis = subprocess.run([sys.executable, "-m", "src.analysis.fuzzy_comparison",
                "--runs-root", str(RUNS), "--expected-configs",
                str(REPO / "configs/fuzzy_v1"), "--out", str(ANALYSIS),
                "--plan", str(REPO / "configs/analysis_plan.json")], cwd=REPO)
print("ANALYSIS_EXIT", analysis.returncode, flush=True)
shutil.make_archive("/kaggle/working/fuzzy_v1_analysis", "zip", ANALYSIS)
shutil.make_archive("/kaggle/working/fuzzy_v1_full_runs", "zip", RUNS)
print("FUZZY_MAIN_COMPLETE", flush=True)
'''


def main() -> None:
    targets = {
        ROOT / "notebooks/fuzzy_smoke.ipynb": notebook("Fuzzy learning-degree smoke test", SMOKE),
        ROOT / "notebooks/fuzzy_learning.ipynb": notebook("Fuzzy learning-degree comparison", MAIN),
    }
    for path, value in targets.items():
        path.write_text(json.dumps(value, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
