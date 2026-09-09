"""Generate the thin private Kaggle entry point for the V2.3 dose sweep."""

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
from scripts.run_fuzzy_v23_sweep import run_sweep

run_sweep(REPO, Path(os.environ["NEURON_DEATH_DATA"]))
'''


def main():
    cells = [
        cell("markdown", "# Temporal fuzzy V2.3 process/saliency dose sweep\n\n"
             "Four settings, seeds 15–17. All twelve targeted runs finish before "
             "the twelve exact-yoked controls. Existing baselines are reused.\n\n"
             "This frozen sweep raises process-health coverage while separately "
             "controlling the equivalent saliency guard, spanning approximately "
             "9k–17k projected resets without changing the V2 reset operation."),
        cell("code", BODY),
    ]
    for cell_id, value in zip(("v23-overview", "v23-run"), cells):
        value["id"] = cell_id
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / "notebooks/fuzzy_v23_saliency_dose_sweep.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
