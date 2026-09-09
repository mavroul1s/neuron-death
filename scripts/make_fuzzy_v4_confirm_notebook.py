"""Generate the thin Kaggle entry point for held-out RA-SNR confirmation."""

from __future__ import annotations

import json

from make_fuzzy_v2_notebook import ROOT, cell


BODY = r'''
import os, sys
from pathlib import Path

REPO = Path(os.environ["NEURON_DEATH_SOURCE"])
sys.path.insert(0, str(REPO))
from scripts.run_fuzzy_v4_confirm import run_sweep

run_sweep(REPO, Path(os.environ["NEURON_DEATH_DATA"]))
'''


def main():
    cells = [
        cell("markdown", "# RA-SNR held-out confirmation\n\nLocked RA-SNR boost 2.0, window 100 against SNR, ReDo and ReGraMa on unseen seeds 18–22."),
        cell("code", BODY),
    ]
    for cell_id, value in zip(("confirm-overview", "confirm-run"), cells):
        value["id"] = cell_id
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/fuzzy_v4_heldout_confirmation.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
