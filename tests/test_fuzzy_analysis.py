from __future__ import annotations

import numpy as np

from src.analysis.fuzzy_comparison import arm_label, seed_block_bootstrap


def test_arm_label_comes_from_config_not_run_name():
    assert arm_label({"run_id": "misleading", "recycling":
                      {"kind": "redo", "tau": 0.25}}) == "redo_tau0.25"
    assert arm_label({"recycling": {"kind": "fuzzy_trend"}}) == "fuzzy_trend"


def test_seed_block_bootstrap_preserves_paired_trajectory_difference():
    a = np.arange(20, dtype=float).reshape(5, 4)
    b = a - 0.03
    result = seed_block_bootstrap(a, b, n_bootstrap=200, seed=7)
    assert np.isclose(result.point, 0.03)
    assert np.isclose(result.lo, 0.03)
    assert np.isclose(result.hi, 0.03)
