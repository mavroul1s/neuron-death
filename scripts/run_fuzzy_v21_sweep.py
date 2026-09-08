"""Run the frozen V2.1 dose sweep, preserving results even on a failed phase.

This is orchestration only. The unchanged method and exact-yoke replay live in
src/. One launch_pair child uses one T4; all targeted runs finish before controls.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PREFLIGHT_TESTS = [
    "tests/test_temporal_fuzzy.py",
    "tests/test_probes.py::test_dead_exact_detects_forced_dead_units",
    "tests/test_probes.py::test_dormant_tau_is_blind_to_uniform_shrinkage_tensor_level",
    "tests/test_probes.py::test_effective_rank_on_known_spectra",
    "tests/test_interventions.py::test_redo_preserves_function",
    "tests/test_interventions.py::test_every_arm_recycles_exactly_k_equals_dormant_count",
    "tests/test_train.py::test_checkpoint_round_trip_matches_an_uninterrupted_run",
    "tests/test_learning_degree.py::test_joint_layer_reset_keeps_all_selected_outgoing_slices_zero",
]


def load_sweep(source: Path) -> tuple[list[tuple[Path, dict]], list[tuple[Path, dict]]]:
    plan = json.loads((source / "configs/fuzzy_v21_dev_plan.json").read_text())
    if plan["status"] != "FROZEN_BEFORE_FIRST_V21_RUN":
        raise RuntimeError("V2.1 plan must be frozen before running")
    pinned = dict(plan["integrity"]["core_source_sha256"])
    pinned[plan["integrity"]["original_frozen_plan_path"]] = (
        plan["integrity"]["original_frozen_plan_sha256"])
    pinned["configs/fuzzy_v2_dev_plan.json"] = plan["integrity"]["prior_v2_plan_sha256"]
    pinned.update({row["config_path"]: row["config_file_sha256"]
                   for row in plan["run_manifest"]})
    for name, expected_hash in pinned.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Source/config differs from frozen V2.1 plan: {name}")
    configs = [(p, json.loads(p.read_text())) for p in sorted(
        (source / "configs/fuzzy_v21_dev").glob("*.json"))]
    target = [pair for pair in configs if pair[1]["recycling"]["kind"] == "fuzzy_v2"]
    yoked = [pair for pair in configs
             if pair[1]["recycling"]["kind"] == "fuzzy_v2_yoked_random"]
    if len(configs) != 12 or len(target) != 6 or len(yoked) != 6:
        raise RuntimeError("Expected exactly six targets and six yokes; no baseline reruns")
    if {c["run_id"] for _, c in configs} != {r["run_id"] for r in plan["run_manifest"]}:
        raise RuntimeError("V2.1 config identities differ from the frozen manifest")
    expected = {(degree, seed) for degree in (0.3, 0.5) for seed in (15, 16, 17)}
    if {(c["recycling"]["learning_degree"]["degree_threshold"], c["seed"])
            for _, c in target} != expected:
        raise RuntimeError("Target threshold/seed grid is incomplete")
    targets = {c["run_id"]: c for _, c in target}
    sources = []
    for _, config in yoked:
        source_id = config["recycling"]["learning_degree"]["yoked_from_run_id"]
        if source_id not in targets or config["seed"] != targets[source_id]["seed"]:
            raise RuntimeError("Invalid exact-yoked target/seed pairing")
        sources.append(source_id)
    if len(set(sources)) != 6 or len({c["run_id"] for _, c in configs}) != 12:
        raise RuntimeError("Duplicated target/yoke run identity")
    return target, yoked


def _run_command(command: list[str], source: Path, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Shared V2.1 session budget exhausted")
    # Killing only launch_pair would orphan its training children on both GPUs.
    # Kaggle is Linux; a process group makes the session deadline apply to all.
    process = subprocess.Popen(command, cwd=source, start_new_session=os.name != "nt")
    try:
        code = process.wait(timeout=remaining)
    except BaseException:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def check_complete(configs: list[tuple[Path, dict]], runs: Path) -> None:
    for _, config in configs:
        directory = runs / config["run_id"]
        summary = directory / "summary.json"
        value = json.loads(summary.read_text()) if summary.is_file() else {}
        if value.get("status") != "complete" or value.get("n_tasks") != config["data"]["n_tasks"]:
            raise RuntimeError(f"Incomplete run: {config['run_id']}")
        required = ["config.json", "tasks.parquet", "metrics.parquet",
                    "neurons.parquet", "recycling.parquet"]
        if config["recycling"]["kind"] == "fuzzy_v2":
            required.append("learning_degree.parquet")
        for name in required:
            if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
                raise RuntimeError(f"Missing required log: {config['run_id']}/{name}")


def _read_schedule(path: Path) -> list[tuple[int, int, int]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["step", "layer_idx", "k"]).to_pylist()
    result = sorted((int(r["step"]), int(r["layer_idx"]), int(r["k"])) for r in rows)
    if len({(step, layer) for step, layer, _ in result}) != len(result):
        raise RuntimeError(f"Duplicated step/layer in {path}")
    return result


def verify_yokes(yoked: list[tuple[Path, dict]], runs: Path) -> list[dict]:
    checks = []
    for _, config in yoked:
        source_id = config["recycling"]["learning_degree"]["yoked_from_run_id"]
        target = _read_schedule(runs / source_id / "recycling.parquet")
        random = _read_schedule(runs / config["run_id"] / "recycling.parquet")
        if target != random:
            raise RuntimeError(f"Exact step/layer/cardinality mismatch: {config['run_id']}")
        checks.append({"target_run_id": source_id, "random_run_id": config["run_id"],
                       "exact_schedule_match": True, "schedule_rows": len(target),
                       "target_resets": sum(k for _, _, k in target),
                       "random_resets": sum(k for _, _, k in random)})
    return checks


def _archive_results(source: Path, runs: Path, output: Path, deadline: float) -> None:
    # Preserve raw checkpoints/shards first, so even an extract failure leaves a
    # resumable archive. The helper's shared deadline includes both operations.
    _run_command([
        sys.executable, "-c",
        "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])",
        str(output / "fuzzy_v21_dev_full_runs"), str(runs),
    ], source, deadline)
    if any(runs.glob("*/config.json")):
        _run_command([
            sys.executable, "scripts/make_analysis_extract.py", "--runs-root", str(runs),
            "--pattern", "fuzzy_v21_dev_*", "--out", str(output / "fuzzy_v21_dev_extract"),
            "--with-c4", "--zip",
        ], source, deadline)


def run_sweep(source: Path, data: Path, output: Path = Path("/kaggle/working")) -> dict:
    """Run 12 configs with a 10h total ceiling and 30min reserved for artifacts."""
    started = time.monotonic()
    deadline = started + 10 * 3600
    training_deadline = deadline - 30 * 60
    runs = output / "fuzzy_v21_dev_runs"
    runs.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "complete_runs": 0, "yoked_validation": []}
    error = None
    try:
        target, yoked = load_sweep(source)
        if not (data / "mnist.npz").is_file():
            raise FileNotFoundError("The attached runtime must include cached MNIST")
        _run_command([
            sys.executable, "-c", "import torch; assert torch.cuda.device_count() == 2, "
            "'This job requires two independent T4 GPUs'",
        ], source, min(training_deadline, time.monotonic() + 60))
        _run_command([
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *PREFLIGHT_TESTS,
        ], source, min(training_deadline, time.monotonic() + 5 * 60))
        print("FUZZY_V21_PREFLIGHT_PASSED", flush=True)
        for phase, configs in (("TARGETED", target), ("YOKED_RANDOM", yoked)):
            print(f"FUZZY_V21_{phase}_PHASE_LAUNCHING", flush=True)
            _run_command([
                sys.executable, "scripts/launch_pair.py", *[str(p) for p, _ in configs],
                "--gpus", "0,1", "--runs-root", str(runs), "--data-root", str(data),
                "--budget-hours", str(max(0, training_deadline - time.monotonic()) / 3600),
            ], source, training_deadline)
            # launch_pair may exit successfully after leaving a budget-limited
            # queue. Completion, not process exit alone, is the phase barrier.
            check_complete(configs, runs)
            report["complete_runs"] += len(configs)
            print(f"FUZZY_V21_{phase}_PHASE_COMPLETE", flush=True)
        report["yoked_validation"] = verify_yokes(yoked, runs)
        report["status"] = "complete"
    except BaseException as exc:
        error = exc
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        plan = source / "configs/fuzzy_v21_dev_plan.json"
        if plan.exists():
            (runs / "fuzzy_v21_dev_plan.json").write_bytes(plan.read_bytes())
        report["elapsed_seconds_before_archive"] = time.monotonic() - started
        (runs / "session_status.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            _archive_results(source, runs, output, deadline)
            report["artifacts_preserved"] = True
        except BaseException as archive_error:
            report["artifacts_preserved"] = False
            report["archive_error"] = f"{type(archive_error).__name__}: {archive_error}"
            if error is None:
                error = archive_error
        report["status"] = "failed" if error is not None else "complete"
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "fuzzy_v21_dev_session_status.json").write_text(
            json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error
    print("FUZZY_V21_DEVELOPMENT_COMPLETE", flush=True)
    return report
