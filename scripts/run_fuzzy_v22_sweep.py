"""Run the frozen V2.2 controller sweep in two ordered GPU phases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from scripts.run_fuzzy_v21_sweep import (
    PREFLIGHT_TESTS,
    _run_command,
    check_complete,
    verify_yokes,
)

EXPECTED_VARIANTS = {
    (0.55, 2, 1000),
    (0.60, 2, 1000),
    (0.60, 2, 500),
    (0.60, 1, 500),
}
V22_PREFLIGHT_TESTS = [*PREFLIGHT_TESTS, "tests/test_fuzzy_v22_sweep.py"]


def load_sweep(source: Path):
    plan_path = source / "configs/fuzzy_v22_dev_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_FIRST_V22_RUN":
        raise RuntimeError("V2.2 plan must be frozen before running")
    pinned = dict(plan["integrity"]["core_source_sha256"])
    pinned[plan["integrity"]["original_frozen_plan_path"]] = plan["integrity"]["original_frozen_plan_sha256"]
    pinned["configs/fuzzy_v21_dev_plan.json"] = plan["integrity"]["prior_v21_plan_sha256"]
    pinned.update({row["config_path"]: row["config_file_sha256"] for row in plan["run_manifest"]})
    for name, expected in pinned.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source/config differs from frozen V2.2 plan: {name}")

    configs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(
        (source / "configs/fuzzy_v22_dev").glob("*.json"))]
    target = [pair for pair in configs if pair[1]["recycling"]["kind"] == "fuzzy_v2"]
    yoked = [pair for pair in configs if pair[1]["recycling"]["kind"] == "fuzzy_v2_yoked_random"]
    if len(configs) != 24 or len(target) != 12 or len(yoked) != 12:
        raise RuntimeError("Expected twelve targets and twelve exact-yoked controls")
    if {config["run_id"] for _, config in configs} != {row["run_id"] for row in plan["run_manifest"]}:
        raise RuntimeError("V2.2 config identities differ from frozen manifest")
    actual_grid = {
        (float(config["recycling"]["learning_degree"]["degree_threshold"]),
         int(config["recycling"]["learning_degree"]["patience"]),
         int(config["recycling"]["learning_degree"]["cooldown_steps"]),
         int(config["seed"]))
        for _, config in target
    }
    expected_grid = {(*variant, seed) for variant in EXPECTED_VARIANTS for seed in (15, 16, 17)}
    if actual_grid != expected_grid:
        raise RuntimeError("V2.2 target grid is incomplete or duplicated")
    targets = {config["run_id"]: config for _, config in target}
    sources = []
    for _, config in yoked:
        source_id = config["recycling"]["learning_degree"]["yoked_from_run_id"]
        if source_id not in targets or config["seed"] != targets[source_id]["seed"]:
            raise RuntimeError("Invalid V2.2 target/yoke pairing")
        sources.append(source_id)
    if len(set(sources)) != 12:
        raise RuntimeError("Each V2.2 target must have exactly one yoke")
    order = {run_id: index for index, run_id in enumerate(plan["execution"]["targeted_run_ids"])}
    target.sort(key=lambda pair: order[pair[1]["run_id"]])
    order = {run_id: index for index, run_id in enumerate(plan["execution"]["yoked_run_ids"])}
    yoked.sort(key=lambda pair: order[pair[1]["run_id"]])
    return target, yoked


def archive_results(source: Path, runs: Path, output: Path, deadline: float):
    _run_command([
        sys.executable, "-c",
        "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])",
        str(output / "fuzzy_v22_dev_full_runs"), str(runs),
    ], source, deadline)
    if any(runs.glob("*/config.json")):
        _run_command([
            sys.executable, "scripts/make_analysis_extract.py", "--runs-root", str(runs),
            "--pattern", "fuzzy_v22_dev_*", "--out", str(output / "fuzzy_v22_dev_extract"),
            "--with-c4", "--zip",
        ], source, deadline)


def run_sweep(source: Path, data: Path, output: Path = Path("/kaggle/working")):
    started = time.monotonic()
    deadline = started + 10 * 3600
    training_deadline = deadline - 30 * 60
    runs = output / "fuzzy_v22_dev_runs"
    runs.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "complete_runs": 0, "yoked_validation": []}
    error = None
    try:
        target, yoked = load_sweep(source)
        if not (data / "mnist.npz").is_file():
            raise FileNotFoundError("Attached runtime must contain cached MNIST")
        _run_command([
            sys.executable, "-c", "import torch; assert torch.cuda.device_count() == 2",
        ], source, min(training_deadline, time.monotonic() + 60))
        _run_command([
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            *V22_PREFLIGHT_TESTS,
        ], source, min(training_deadline, time.monotonic() + 5 * 60))
        print("FUZZY_V22_PREFLIGHT_PASSED", flush=True)
        for phase, configs in (("TARGETED", target), ("YOKED_RANDOM", yoked)):
            print(f"FUZZY_V22_{phase}_PHASE_LAUNCHING", flush=True)
            _run_command([
                sys.executable, "scripts/launch_pair.py", *[str(path) for path, _ in configs],
                "--gpus", "0,1", "--runs-root", str(runs), "--data-root", str(data),
                "--budget-hours", str(max(0, training_deadline - time.monotonic()) / 3600),
            ], source, training_deadline)
            check_complete(configs, runs)
            report["complete_runs"] += len(configs)
            print(f"FUZZY_V22_{phase}_PHASE_COMPLETE", flush=True)
        report["yoked_validation"] = verify_yokes(yoked, runs)
        report["status"] = "complete"
    except BaseException as exc:
        error = exc
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        plan = source / "configs/fuzzy_v22_dev_plan.json"
        if plan.exists():
            (runs / plan.name).write_bytes(plan.read_bytes())
        report["elapsed_seconds_before_archive"] = time.monotonic() - started
        (runs / "session_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        try:
            archive_results(source, runs, output, deadline)
            report["artifacts_preserved"] = True
        except BaseException as archive_error:
            report.update(artifacts_preserved=False,
                          archive_error=f"{type(archive_error).__name__}: {archive_error}")
            if error is None:
                error = archive_error
        report["status"] = "failed" if error else "complete"
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "fuzzy_v22_dev_session_status.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if error:
        raise error
    print("FUZZY_V22_DEVELOPMENT_COMPLETE", flush=True)
    return report


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1]
    run_sweep(source, Path(sys.argv[1]) if len(sys.argv) > 1 else source / "data")
