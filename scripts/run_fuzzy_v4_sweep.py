"""Run the frozen Recovery-Accelerated SNR development sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from scripts.run_fuzzy_v21_sweep import PREFLIGHT_TESTS, _run_command, check_complete


EXPECTED_VARIANTS = {(1.5, 25), (2.0, 25), (1.5, 100), (2.0, 100)}
V4_PREFLIGHT_TESTS = [
    *PREFLIGHT_TESTS,
    "tests/test_interventions.py::test_snr_recovery_config_requires_a_real_boost_and_window",
    "tests/test_interventions.py::test_snr_recovery_boosts_only_recovering_outgoing_columns",
    "tests/test_interventions.py::test_standard_snr_has_no_recovery_gradient_side_effect",
    "tests/test_interventions.py::test_snr_recovery_state_round_trips",
    "tests/test_fuzzy_v4_sweep.py",
]


def load_sweep(source: Path):
    plan_path = source / "configs/fuzzy_v4_dev_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_FIRST_V4_RUN":
        raise RuntimeError("V4 plan must be frozen before running")
    pinned = dict(plan["integrity"]["core_source_sha256"])
    pinned[plan["integrity"]["original_frozen_plan_path"]] = (
        plan["integrity"]["original_frozen_plan_sha256"]
    )
    pinned["configs/fuzzy_v3_dev_plan.json"] = plan["integrity"]["prior_v3_plan_sha256"]
    pinned.update(
        {row["config_path"]: row["config_file_sha256"] for row in plan["run_manifest"]}
    )
    for name, expected in pinned.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source/config differs from frozen V4 plan: {name}")

    configs = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((source / "configs/fuzzy_v4_dev").glob("*.json"))
    ]
    if len(configs) != 12:
        raise RuntimeError("Expected exactly twelve RA-SNR runs and no baseline reruns")
    if {config["run_id"] for _, config in configs} != {
        row["run_id"] for row in plan["run_manifest"]
    }:
        raise RuntimeError("V4 config identities differ from frozen manifest")
    grid = {
        (
            float(config["recycling"]["recovery_boost"]),
            int(config["recycling"]["recovery_steps"]),
            int(config["seed"]),
        )
        for _, config in configs
    }
    expected = {
        (boost, steps, seed)
        for boost, steps in EXPECTED_VARIANTS
        for seed in (15, 16, 17)
    }
    if grid != expected:
        raise RuntimeError("V4 recovery/seed grid is incomplete or duplicated")
    for _, config in configs:
        recycling = config["recycling"]
        if recycling["kind"] != "snr_recovery" or recycling["snr_eta"] != 0.08:
            raise RuntimeError("V4 must preserve the SNR eta=0.08 detector")
    order = {run_id: i for i, run_id in enumerate(plan["execution"]["run_ids"])}
    configs.sort(key=lambda pair: order[pair[1]["run_id"]])
    return configs


def archive_results(source: Path, runs: Path, output: Path, deadline: float):
    _run_command(
        [
            sys.executable,
            "-c",
            "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])",
            str(output / "fuzzy_v4_dev_full_runs"),
            str(runs),
        ],
        source,
        deadline,
    )
    if any(runs.glob("*/config.json")):
        _run_command(
            [
                sys.executable,
                "scripts/make_analysis_extract.py",
                "--runs-root",
                str(runs),
                "--pattern",
                "fuzzy_v4_dev_*",
                "--out",
                str(output / "fuzzy_v4_dev_extract"),
                "--with-c4",
                "--zip",
            ],
            source,
            deadline,
        )


def run_sweep(source: Path, data: Path, output: Path = Path("/kaggle/working")):
    started = time.monotonic()
    deadline = started + 10 * 3600
    training_deadline = deadline - 30 * 60
    runs = output / "fuzzy_v4_dev_runs"
    runs.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "complete_runs": 0}
    error = None
    try:
        configs = load_sweep(source)
        if not (data / "mnist.npz").is_file():
            raise FileNotFoundError("Attached runtime must contain cached MNIST")
        _run_command(
            [sys.executable, "-c", "import torch; assert torch.cuda.device_count() == 2"],
            source,
            min(training_deadline, time.monotonic() + 60),
        )
        _run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                *V4_PREFLIGHT_TESTS,
            ],
            source,
            min(training_deadline, time.monotonic() + 5 * 60),
        )
        print("FUZZY_V4_PREFLIGHT_PASSED", flush=True)
        _run_command(
            [
                sys.executable,
                "scripts/launch_pair.py",
                *[str(path) for path, _ in configs],
                "--gpus",
                "0,1",
                "--runs-root",
                str(runs),
                "--data-root",
                str(data),
                "--budget-hours",
                str(max(0, training_deadline - time.monotonic()) / 3600),
            ],
            source,
            training_deadline,
        )
        check_complete(configs, runs)
        report.update(status="complete", complete_runs=len(configs))
    except BaseException as exc:
        error = exc
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        plan = source / "configs/fuzzy_v4_dev_plan.json"
        if plan.exists():
            (runs / plan.name).write_bytes(plan.read_bytes())
        report["elapsed_seconds_before_archive"] = time.monotonic() - started
        (runs / "session_status.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        try:
            archive_results(source, runs, output, deadline)
            report["artifacts_preserved"] = True
        except BaseException as archive_error:
            report.update(
                artifacts_preserved=False,
                archive_error=f"{type(archive_error).__name__}: {archive_error}",
            )
            if error is None:
                error = archive_error
        report["status"] = "failed" if error else "complete"
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "fuzzy_v4_dev_session_status.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if error:
        raise error
    print("FUZZY_V4_DEVELOPMENT_COMPLETE", flush=True)
    return report


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1]
    run_sweep(source, Path(sys.argv[1]) if len(sys.argv) > 1 else source / "data")
