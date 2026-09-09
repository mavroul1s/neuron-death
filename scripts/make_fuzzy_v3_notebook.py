"""Generate the thin private Kaggle entry point for BT-FR V3."""

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
from scripts.run_fuzzy_v3_sweep import run_sweep

run_sweep(REPO, Path(os.environ["NEURON_DEATH_DATA"]))
'''


def main():
    cells = [
        cell("markdown", "# Budgeted Temporal Fuzzy Recycling (BT-FR) V3\n\n"
             "Three replacement rates matched to the observed ReGraMa, ReDo and SNR dose range, "
             "using seeds 15–17. The temporal fuzzy score ranks units while an independent "
             "continual budget controls dose. All nine targeted runs finish before the nine "
             "exact-yoked random controls. Existing baselines are reused."),
        cell("code", BODY),
    ]
    for cell_id, value in zip(("v3-overview", "v3-run"), cells):
        value["id"] = cell_id
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = ROOT / "notebooks/fuzzy_v3_budget_sweep.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
