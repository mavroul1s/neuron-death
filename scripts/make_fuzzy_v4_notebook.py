"""Generate the thin private Kaggle entry point for RA-SNR V4."""

from __future__ import annotations

import json

from make_fuzzy_v2_notebook import ROOT, cell


BODY = r'''
import os, sys
from pathlib import Path

REPO = Path(os.environ["NEURON_DEATH_SOURCE"])
sys.path.insert(0, str(REPO))
from scripts.run_fuzzy_v4_sweep import run_sweep

run_sweep(REPO, Path(os.environ["NEURON_DEATH_DATA"]))
'''


def main():
    cells = [
        cell(
            "markdown",
            "# Recovery-Accelerated SNR (RA-SNR) V4\n\n"
            "Four post-reset recovery settings on seeds 15–17. The SNR eta=0.08 "
            "detector and reset rule are unchanged; only the newly reset unit's "
            "outgoing gradient is temporarily amplified. Existing SNR, ReDo, "
            "ReGraMa and baseline runs are reused.",
        ),
        cell("code", BODY),
    ]
    for cell_id, value in zip(("v4-overview", "v4-run"), cells):
        value["id"] = cell_id
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / "notebooks/fuzzy_v4_recovery_sweep.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
