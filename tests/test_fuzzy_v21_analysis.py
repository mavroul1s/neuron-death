"""Focused extract-analysis checks; runnable directly without torch or pytest."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_fuzzy_v21_extract import (
    REFERENCE, analyze, arm_label, bootstrap_contrast, exact_yoke_check, read_json,
    validate_runs,
)
from src.analysis.stats import Estimate, iqm


def original_bootstrap():
    """Load the unmodified shared function, excluding unrelated torch imports."""
    path = ROOT / "src/analysis/fuzzy_comparison.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_matrix", "seed_block_bootstrap")]
    namespace = {"np": np, "Estimate": Estimate, "iqm": iqm}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["seed_block_bootstrap"]


class FuzzyV21AnalysisTests(unittest.TestCase):
    def test_thresholds_and_yokes_identified_from_config_not_name(self):
        a = {"seed": 15, "recycling": {"kind": "fuzzy_v2",
             "learning_degree": {"degree_threshold": 0.3}}}
        b = deepcopy(a)
        b["recycling"]["learning_degree"]["degree_threshold"] = 0.5
        yoke = {"seed": 15, "recycling": {"kind": "fuzzy_v2_yoked_random",
                "learning_degree": {"yoked_from_run_id": "unrelated_name"}}}
        configs = {"unrelated_name": b}
        self.assertEqual(arm_label(a, configs), "fuzzy_v2_d0.3")
        self.assertEqual(arm_label(b, configs), "fuzzy_v2_d0.5")
        self.assertEqual(arm_label(yoke, configs), "fuzzy_v2_d0.5_yoked_random")
        yoke["seed"] = 16
        with self.assertRaisesRegex(ValueError, "same-seed"):
            arm_label(yoke, configs)

    def test_general_bootstrap_matches_unmodified_v2_rng_and_estimand(self):
        rng = np.random.default_rng(91)
        target = rng.normal(size=(3, 50))
        other = rng.normal(size=(3, 50))
        shared = original_bootstrap()
        for arrays, weights in (([target], [1]), ([target, other], [1, -1])):
            actual = bootstrap_contrast(arrays, weights, n_bootstrap=401, seed=10)
            expected = shared(*arrays, n_bootstrap=401, seed=10).as_dict()
            for key in ("point", "ci_lo", "ci_hi"):
                self.assertAlmostEqual(actual[key], expected[key], places=14)

    def test_four_arm_selection_contrast_keeps_shared_seed_blocks(self):
        a = np.arange(150, dtype=float).reshape(3, 50)
        result = bootstrap_contrast([a, a - 4, a * 2, a * 2 - 1],
                                    [1, -1, -1, 1], n_bootstrap=101)
        for key in ("point", "ci_lo", "ci_hi"):
            self.assertAlmostEqual(result[key], 3.0)

    def test_equal_total_resets_do_not_pass_wrong_event_schedule(self):
        rows = pd.DataFrame({"run_id": ["target", "target", "random", "random"],
                             "step": [1000, 1100, 1000, 1100],
                             "layer_idx": [0, 1, 0, 1], "k": [2, 0, 2, 0]})
        self.assertTrue(exact_yoke_check(rows, "target", "random")["exact_schedule_match"])
        rows.loc[3, "step"] = 1200  # Also tests preservation of zero-k records.
        with self.assertRaisesRegex(ValueError, "schedule differs"):
            exact_yoke_check(rows, "target", "random")

    @unittest.skipUnless((REFERENCE / "runs.json").exists(), "archived local V2 extract unavailable")
    def test_reused_comparators_reproduce_prior_v2_accuracy_and_pairing(self):
        plan = read_json(ROOT / "configs/fuzzy_v21_dev_plan.json")
        runs = validate_runs(REFERENCE, plan["reused_completed_runs"])
        configs = {rid: run["config"] for rid, run in runs.items()}
        tasks = pq.read_table(REFERENCE / "tasks.parquet", columns=[
            "run_id", "task_idx", "probe_point", "online_accuracy"]).to_pandas()
        tasks = tasks.loc[tasks.probe_point.eq("task_end") & tasks.task_idx.between(150, 199)]
        prior = read_json(REFERENCE.parent / "v2_analysis.json")
        matrices = {}
        for arm in ("none", "redo", "regrama", "snr", "fuzzy_v2_d0.2",
                    "fuzzy_v2_d0.2_yoked_random"):
            ids = sorted((rid for rid in runs if arm_label(configs[rid], configs) == arm),
                         key=lambda rid: configs[rid]["seed"])
            self.assertEqual([configs[rid]["seed"] for rid in ids], [15, 16, 17])
            matrix = np.stack([tasks.loc[tasks.run_id.eq(rid)].sort_values("task_idx")
                               .online_accuracy.to_numpy(dtype=float) for rid in ids])
            matrices[arm] = matrix
            old_arm = arm.replace("_d0.2", "")
            actual = bootstrap_contrast([matrix], seed=0, scale=100)
            expected = prior["arms"][old_arm]["late_accuracy_seed_block_pct"]
            for key in ("point", "ci_lo", "ci_hi"):
                self.assertAlmostEqual(actual[key], expected[key], places=11)
        difference = bootstrap_contrast([matrices["fuzzy_v2_d0.2"],
                matrices["fuzzy_v2_d0.2_yoked_random"]], [1, -1], seed=10, scale=100)
        expected = prior["comparisons_vs_fuzzy_v2"]["fuzzy_v2_yoked_random"]["target_minus_arm_seed_block_pp"]
        for key in ("point", "ci_lo", "ci_hi"):
            self.assertAlmostEqual(difference[key], expected[key], places=11)

    @unittest.skipUnless((REFERENCE / "runs.json").exists(), "archived local V2 extract unavailable")
    def test_full_extract_integration_keeps_two_thresholds_and_all_reused_arms(self):
        # Synthetic new runs deliberately reuse V2 trajectories, so every
        # targeted-vs-V2 contrast must be zero. No synthetic results leave tmp.
        plan = read_json(ROOT / "configs/fuzzy_v21_dev_plan.json")
        plan["bootstrap"]["n_bootstrap"] = 101
        prior = {run["run_id"]: run for run in read_json(REFERENCE / "runs.json")}
        synthetic, source_ids = [], {}
        for expected in plan["run_manifest"]:
            raw = read_json(ROOT / expected["config_path"])
            target = expected["kind"] == "fuzzy_v2"
            source_id = f"fuzzy_v2_dev_{'fuzzy_v2' if target else 'yoked_random'}_lr0p1_s{expected['seed']}"
            item = deepcopy(prior[source_id])
            item["run_id"] = expected["run_id"]
            item["config"]["run_id"] = expected["run_id"]
            item["config"]["notes"] = raw["notes"]
            item["config"]["recycling"]["learning_degree"] = raw["recycling"]["learning_degree"]
            item["summary"]["config_hash"] = expected["config_hash"]
            synthetic.append(item)
            source_ids[expected["run_id"]] = source_id
        with tempfile.TemporaryDirectory(prefix="v21-analysis-test-") as temporary:
            root = Path(temporary)
            (root / "runs.json").write_text(json.dumps(synthetic), encoding="utf-8")
            for name in ("tasks", "metrics", "recycling"):
                original = pq.read_table(REFERENCE / f"{name}.parquet").to_pandas()
                parts = []
                for rid, source_id in source_ids.items():
                    rows = original.loc[original.run_id.eq(source_id)].copy()
                    rows["run_id"] = rid
                    parts.append(rows)
                pq.write_table(pa.Table.from_pandas(pd.concat(parts), preserve_index=False), root / f"{name}.parquet")
            # The analyzer only consumes candidate/selected fields; synthetic
            # rows reproduce each original event's k, including zero-k events.
            recycling = pq.read_table(root / "recycling.parquet").to_pandas()
            rows = []
            for rid in source_ids:
                if "yoked_random" in rid:
                    continue
                for event in recycling.loc[recycling.run_id.eq(rid)].itertuples():
                    for selected in ([True] * int(event.k) if event.k else [False]):
                        rows.append({"run_id": rid, "step": event.step, "layer_idx": event.layer_idx,
                                     "selected": selected, "candidate": selected})
            pq.write_table(pa.Table.from_pylist(rows), root / "learning_degree.parquet")
            schema = pa.schema([("run_id", pa.string())])
            with pq.ParquetWriter(root / "neurons_c4.parquet", schema) as writer:
                for rid in source_ids:
                    writer.write_table(pa.table({"run_id": [rid] * 300_000}, schema=schema))
            report = analyze(root, REFERENCE, plan)
            self.assertEqual(report["new_complete_runs"], 12)
            self.assertEqual(report["reused_complete_runs"], 18)
            self.assertEqual(len(report["arms"]), 10)
            self.assertEqual(len(report["yoked_validation"]), 9)
            for threshold in ("0.3", "0.5"):
                arm = f"fuzzy_v2_d{threshold}"
                self.assertAlmostEqual(report["comparisons"][arm]["fuzzy_v2_d0.2"]["point"], 0)
                self.assertAlmostEqual(report["selection_advantage_change_vs_v2"][arm]["point"], 0)
                self.assertIn("reference", report["arms"][arm]["death_dormancy_pct"]["late"])
                self.assertEqual(report["arms"][arm]["per_seed"]["15"]["resets"], 2417)
            self.assertIsNone(report["arms"]["none"]["descriptive_gain_pp_per_1000_resets"])
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
