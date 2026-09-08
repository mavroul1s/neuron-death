"""Generate the thin, private Kaggle entry point for the V2.1 dose sweep."""

from __future__ import annotations

import json
from pathlib import Path

from make_fuzzy_v2_notebook import cell

ROOT = Path(__file__).resolve().parents[1]

BODY = r'''
import os, sys
from pathlib import Path

REPO = Path(os.environ["NEURON_DEATH_SOURCE"])
sys.path.insert(0, str(REPO))
from scripts.run_fuzzy_v21_sweep import run_sweep

run_sweep(source=REPO, data=Path(os.environ["NEURON_DEATH_DATA"]))
'''


def main() -> None:
    notebook = {
        "cells": [
            cell("markdown", "# Temporal fuzzy V2.1 dose sweep\n\n"
                 "Thresholds 0.30 and 0.50; development seeds 15–17. "
                 "Six targeted runs finish before six exact-yoked random controls. "
                 "Existing deterministic baselines are reused in analysis.\n\n"
                 "Two T4s run independent experiments. The session has one 10-hour "
                 "budget, including 30 minutes reserved for preserving results. "
                 "The separate V2.1 plan is frozen before execution; method logic remains in `src/`."),
            cell("code", BODY),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / "notebooks/fuzzy_v21_dose_sweep.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
