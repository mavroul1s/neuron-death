"""Generate and freeze the Recovery-Accelerated SNR development sweep."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

from make_fuzzy_v22_configs import ROOT, default_config, resolved_hash, sha, write_json


CONFIG_DIR = ROOT / "configs/fuzzy_v4_dev"
PLAN_PATH = ROOT / "configs/fuzzy_v4_dev_plan.json"
SEEDS = (15, 16, 17)
VARIANTS = (
    {"id": "b1p5_h25", "recovery_boost": 1.5, "recovery_steps": 25},
    {"id": "b2p0_h25", "recovery_boost": 2.0, "recovery_steps": 25},
    {"id": "b1p5_h100", "recovery_boost": 1.5, "recovery_steps": 100},
    {"id": "b2p0_h100", "recovery_boost": 2.0, "recovery_steps": 100},
)


def main() -> None:
    if PLAN_PATH.exists() or CONFIG_DIR.exists():
        raise FileExistsError(
            "V4 plan/config directory already exists; frozen artifacts are never overwritten"
        )
    base = json.loads(
        (ROOT / "configs/fuzzy_v2_dev/fuzzy_v2_dev_snr_eta0p08_lr0p1_s15.json")
        .read_text(encoding="utf-8")
    )
    defaults = default_config()
    CONFIG_DIR.mkdir(parents=True)
    manifest = []
    run_ids = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run_id = f"fuzzy_v4_dev_rasnr_{variant['id']}_lr0p1_s{seed}"
            config = deepcopy(base)
            config.update(
                run_id=run_id,
                seed=seed,
                notes=(
                    "Pre-run Recovery-Accelerated SNR development sweep; "
                    "see configs/fuzzy_v4_dev_plan.json."
                ),
            )
            config["recycling"].update(
                kind="snr_recovery",
                recovery_boost=variant["recovery_boost"],
                recovery_steps=variant["recovery_steps"],
            )
            path = CONFIG_DIR / f"{run_id}.json"
            write_json(path, config)
            manifest.append(
                {
                    "run_id": run_id,
                    "kind": "snr_recovery",
                    "variant": variant["id"],
                    "seed": seed,
                    "recovery_boost": variant["recovery_boost"],
                    "recovery_steps": variant["recovery_steps"],
                    "config_path": path.relative_to(ROOT).as_posix(),
                    "config_hash": resolved_hash(config, defaults),
                    "config_file_sha256": sha(path),
                }
            )
            run_ids.append(run_id)

    v3_root = ROOT / "remote_runs/neuron-death-fuzzy-v1-v3btfr0909-c9919583"
    plan = {
        "status": "FROZEN_BEFORE_FIRST_V4_RUN",
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method_name": "Recovery-Accelerated SNR (RA-SNR)",
        "scope": (
            "Supervisor-authorized method development, separate from "
            "configs/analysis_plan.json."
        ),
        "objective": (
            "Exceed SNR, ReDo and ReGraMa by preserving SNR's adaptive detector "
            "and accelerating post-reset reintegration."
        ),
        "rationale_before_freeze": {
            "v3_analysis_sha256": sha(v3_root / "v3_analysis.json"),
            "v3_diagnostics_sha256": sha(v3_root / "v3_diagnostics.json"),
            "snr_schedule_diagnostics_sha256": sha(
                v3_root / "snr_schedule_diagnostics.json"
            ),
            "v3_best_accuracy_pct": 91.312,
            "v3_target_minus_yoke_pp": -0.698,
            "snr_accuracy_pct": 92.624,
            "snr_mean_resets": 52_286.3,
            "diagnosis": (
                "Replacing SNR's event detector with a high-dose fuzzy ranker was "
                "harmful. SNR's success comes from many small neuron-specific "
                "events and strong dead-unit targeting in deeper layers."
            ),
            "design_change": (
                "Keep SNR eta=0.08, thresholds, event times, reset operation and "
                "optimizer-state reset. After each reset, multiply only that "
                "unit's outgoing-weight gradient for a fixed recovery window."
            ),
            "mechanistic_hypothesis": (
                "Function-preserving outgoing zeroing delays gradient flow back "
                "into the reinitialized incoming feature. Faster outgoing "
                "reconnection should shorten this recovery bottleneck."
            ),
            "reused_comparator_points_pct": {
                "none": 87.520,
                "redo_tau0.1": 92.346,
                "regrama_tau0.01": 92.293,
                "snr_eta0.08": 92.624,
            },
        },
        "setting": {
            "dataset": "online permuted MNIST",
            "n_tasks": 200,
            "batch_size": 128,
            "optimizer": "SGD momentum 0.9",
            "learning_rate": 0.1,
            "hidden_dims": [500, 500, 500],
            "seeds": list(SEEDS),
        },
        "variants": list(VARIANTS),
        "unchanged_snr_fields": {
            "snr_eta": 0.08,
            "snr_tau_max": 20_000,
            "snr_update_every_tasks": 16,
            "snr_expansion_factor": 2.0,
            "snr_min_age": 100,
            "score_batch_size": 64,
            "zero_outgoing_after_event": True,
            "reset_optimizer_state": True,
            "reset": (
                "incoming reinitialization, outgoing zeroing and optimizer-state reset"
            ),
        },
        "new_run_count": 12,
        "run_manifest": manifest,
        "execution": {
            "run_ids": run_ids,
            "gpu_mode": "Two independent runs, one per T4",
            "session_budget_hours": 10.0,
            "hard_limit_hours": 12,
            "checkpoint_every_tasks": 10,
            "reuse_baselines": True,
            "rerun_existing_baselines": False,
            "dataset": "nmavros/neuron-death-code",
            "visibility": "private",
        },
        "primary_outcome": {
            "column": "online_accuracy",
            "probe_point": "task_end",
            "task_window_zero_indexed_inclusive": [150, 199],
            "estimator": "IQM over the 3x50 seed-by-task matrix",
        },
        "bootstrap": {
            "unit": "whole paired seed trajectory",
            "resample_tasks": False,
            "n_bootstrap": 10_000,
            "confidence": 0.95,
        },
        "planned_comparisons": {
            "each_variant": [
                "paired existing SNR eta=0.08",
                "ReDo tau=0.1",
                "ReGraMa tau=0.01",
                "none",
            ],
            "report_all_variants": True,
            "control_note": (
                "The detector and reset rule are identical to the reused SNR arm; "
                "the only added treatment is post-reset recovery."
            ),
        },
        "decision": {
            "development_winner": (
                "Highest late-window IQM; ties broken by the smaller boost-window "
                "product, then fewer resets."
            ),
            "accuracy_goal": (
                "Point estimate above SNR 92.624%, followed by held-out paired-seed "
                "confirmation before a SOTA claim."
            ),
            "mechanism_goal": (
                "Paired RA-SNR-minus-SNR CI lower bound above zero."
            ),
            "no_posthoc_changes": (
                "No variants, seed changes or in-run parameter adjustment after freeze."
            ),
        },
        "integrity": {
            "original_frozen_plan_path": "configs/analysis_plan.json",
            "original_frozen_plan_sha256": sha(ROOT / "configs/analysis_plan.json"),
            "prior_v3_plan_sha256": sha(ROOT / "configs/fuzzy_v3_dev_plan.json"),
            "core_source_sha256": {
                name: sha(ROOT / name)
                for name in (
                    "src/probes.py",
                    "src/interventions.py",
                    "src/train.py",
                    "src/models.py",
                    "src/config.py",
                )
            },
        },
    }
    write_json(PLAN_PATH, plan)
    print(f"wrote {len(manifest)} configs and {PLAN_PATH}")
    print("plan_sha256", sha(PLAN_PATH))


if __name__ == "__main__":
    main()
