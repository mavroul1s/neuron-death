"""Fast orchestration checks; no model or GPU is required."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_fuzzy_v21_sweep as sweep


class V21SweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "mnist.npz").touch()
        self.target = [(Path("target.json"), {"run_id": "target"})] * 6
        self.yoked = [(Path("random.json"), {
            "run_id": "random", "recycling": {"learning_degree": {"yoked_from_run_id": "target"}},
        })] * 6

    def test_all_targets_complete_before_yoke_and_budget_is_shared(self):
        events = []

        def command(cmd, source, deadline):
            if "scripts/launch_pair.py" in cmd:
                events.append(("launch", cmd[2], deadline))

        def complete(configs, runs):
            events.append(("check", configs[0][1]["run_id"]))

        with patch.object(sweep, "load_sweep", return_value=(self.target, self.yoked)), \
                patch.object(sweep, "_run_command", side_effect=command), \
                patch.object(sweep, "check_complete", side_effect=complete), \
                patch.object(sweep, "verify_yokes", return_value=[]), \
                patch.object(sweep, "_archive_results") as archive:
            report = sweep.run_sweep(self.root, self.data, self.root / "output")
        self.assertEqual([e[:2] for e in events], [
            ("launch", "target.json"), ("check", "target"),
            ("launch", "random.json"), ("check", "random"),
        ])
        self.assertEqual(events[0][2], events[2][2])
        self.assertEqual(archive.call_args.args[3] - events[0][2], 1800)
        self.assertEqual(report["complete_runs"], 12)

    def test_incomplete_target_prevents_yoke_but_preserves_partial_results(self):
        commands = []
        with patch.object(sweep, "load_sweep", return_value=(self.target, self.yoked)), \
                patch.object(sweep, "_run_command", side_effect=lambda cmd, *_: commands.append(cmd)), \
                patch.object(sweep, "check_complete", side_effect=RuntimeError("target incomplete")), \
                patch.object(sweep, "_archive_results") as archive:
            with self.assertRaisesRegex(RuntimeError, "target incomplete"):
                sweep.run_sweep(self.root, self.data, self.root / "output")
        self.assertEqual(len([cmd for cmd in commands if "scripts/launch_pair.py" in cmd]), 1)
        archive.assert_called_once()
        status = json.loads((self.root / "output/fuzzy_v21_dev_session_status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["artifacts_preserved"])

    def test_equal_reset_totals_do_not_hide_different_step_or_layer(self):
        for different in ([(200, 0, 3), (200, 1, 2)], [(100, 0, 2), (100, 1, 3)]):
            with self.subTest(different=different), patch.object(
                sweep, "_read_schedule", side_effect=[[(100, 0, 3), (100, 1, 2)], different]
            ):
                with self.assertRaisesRegex(RuntimeError, "step/layer/cardinality mismatch"):
                    sweep.verify_yokes(self.yoked[:1], self.root)

    def test_matching_schedule_preserves_zero_count_rows(self):
        rows = [(100, 0, 3), (100, 1, 0), (100, 2, 2)]
        with patch.object(sweep, "_read_schedule", return_value=rows):
            check = sweep.verify_yokes(self.yoked[:1], self.root)[0]
        self.assertEqual(check["schedule_rows"], 3)
        self.assertEqual(check["target_resets"], check["random_resets"])
        self.assertEqual(check["target_resets"], 5)

    def test_raw_archive_precedes_extract(self):
        runs = self.root / "runs"
        (runs / "one").mkdir(parents=True)
        (runs / "one/config.json").write_text("{}")
        calls = []
        with patch.object(sweep, "_run_command", side_effect=lambda cmd, *_: calls.append(cmd)):
            sweep._archive_results(self.root, runs, self.root, 1234)
        self.assertIn("make_archive", calls[0][2])
        self.assertEqual(calls[1][1], "scripts/make_analysis_extract.py")


if __name__ == "__main__":
    unittest.main()
