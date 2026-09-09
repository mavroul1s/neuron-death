"""Generate and freeze the BT-FR replacement-rate sweep before its first run."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

from make_fuzzy_v22_configs import ROOT, default_config, resolved_hash, sha, write_json


CONFIG_DIR = ROOT / "configs/fuzzy_v3_dev"
PLAN_PATH = ROOT / "configs/fuzzy_v3_dev_plan.json"
SEEDS = (15, 16, 17)
# With 500 units/layer, a 100-step monitor cadence and 732 observations outside
# warm-up/task grace, these rates target the measured ReGraMa/ReDo/SNR doses.
VARIANTS = (
    {"id": "r0p00030", "replacement_rate": 0.00030, "planned_resets": 32_940,
     "dose_reference": "ReGraMa (32,755 mean resets)"},
    {"id": "r0p00040", "replacement_rate": 0.00040, "planned_resets": 43_920,
     "dose_reference": "ReDo tau=0.1 (44,250 mean resets)"},
    {"id": "r0p00048", "replacement_rate": 0.00048, "planned_resets": 52_704,
     "dose_reference": "SNR eta=0.08 (52,286 mean resets)"},
)


def main() -> None:
    if PLAN_PATH.exists() or CONFIG_DIR.exists():
        raise FileExistsError("V3 plan/config directory already exists; frozen artifacts are never overwritten")
    target_base = json.loads((ROOT / "configs/fuzzy_v23_dev/fuzzy_v23_dev_target_t0p99_es0p90_lr0p1_s15.json").read_text())
    yoke_base = json.loads((ROOT / "configs/fuzzy_v23_dev/fuzzy_v23_dev_yoked_t0p99_es0p90_lr0p1_s15.json").read_text())
    defaults = default_config()
    CONFIG_DIR.mkdir(parents=True)
    manifest, targeted_ids, yoked_ids = [], [], []
    for variant in VARIANTS:
        for seed in SEEDS:
            target_id = f"fuzzy_v3_dev_target_{variant['id']}_lr0p1_s{seed}"
            yoke_id = f"fuzzy_v3_dev_yoked_{variant['id']}_lr0p1_s{seed}"
            target = deepcopy(target_base)
            target.update(run_id=target_id, seed=seed,
                          notes="Pre-run BT-FR replacement-rate sweep; see configs/fuzzy_v3_dev_plan.json.")
            target["recycling"]["kind"] = "fuzzy_budget"
            target["recycling"]["learning_degree"]["replacement_rate"] = variant["replacement_rate"]
            yoke = deepcopy(yoke_base)
            yoke.update(run_id=yoke_id, seed=seed,
                        notes="Pre-run exact-yoked random control for BT-FR; see configs/fuzzy_v3_dev_plan.json.")
            yoke["recycling"]["kind"] = "fuzzy_budget_yoked_random"
            yoke["recycling"]["learning_degree"].update(
                replacement_rate=variant["replacement_rate"], yoked_from_run_id=target_id)
            for kind, config, run_id in (("fuzzy_budget", target, target_id),
                                         ("fuzzy_budget_yoked_random", yoke, yoke_id)):
                path = CONFIG_DIR / f"{run_id}.json"
                write_json(path, config)
                manifest.append({
                    "run_id": run_id, "kind": kind, "variant": variant["id"],
                    "seed": seed, "replacement_rate": variant["replacement_rate"],
                    "planned_resets": variant["planned_resets"],
                    "config_path": path.relative_to(ROOT).as_posix(),
                    "config_hash": resolved_hash(config, defaults),
                    "config_file_sha256": sha(path),
                    "yoked_from_run_id": target_id if kind.endswith("_yoked_random") else None,
                })
            targeted_ids.append(target_id)
            yoked_ids.append(yoke_id)

    v23 = ROOT / "remote_runs/neuron-death-fuzzy-v1-v23sota0909-0a440e4e/v23_analysis.json"
    plan = {
        "status": "FROZEN_BEFORE_FIRST_V3_RUN",
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method_name": "Budgeted Temporal Fuzzy Recycling (BT-FR)",
        "scope": "Supervisor-authorized method development, separate from configs/analysis_plan.json.",
        "objective": "Beat ReDo, ReGraMa and SNR by combining their effective reset dose with the measured temporal-fuzzy selection advantage.",
        "rationale_before_freeze": {
            "v23_analysis_sha256": sha(v23),
            "v23_best_accuracy_pct": 91.227,
            "v23_best_mean_resets": 7_917,
            "v23_exact_yoked_selection_advantage_pp": 0.213,
            "diagnosis": "V2.3 still couples dose to a health threshold. Its targeting beats an exact schedule-and-cardinality yoke, but the controller self-limits far below the 32.8k-52.3k reset doses of the leading comparators.",
            "design_change": "A Continual-Backprop-style replacement-rate accumulator determines k. Temporal process health plus conditional loss saliency only ranks eligible neurons. Cooldown acts as maturity and task grace pauses both accrual and replacement.",
            "reused_comparator_points_pct": {"none": 87.520, "redo_tau0.1": 92.346,
                                               "regrama_tau0.01": 92.293, "snr_eta0.08": 92.624},
        },
        "setting": {"dataset": "online permuted MNIST", "n_tasks": 200, "batch_size": 128,
                    "optimizer": "SGD momentum 0.9", "learning_rate": 0.1,
                    "hidden_dims": [500, 500, 500], "seeds": list(SEEDS)},
        "variants": list(VARIANTS),
        "unchanged_fields": {"ewma_beta": 0.9, "cooldown_steps": 500,
                             "task_grace_steps": 100, "monitor_every": 100,
                             "warmup_steps": 1000, "max_reset_fraction": 0.075,
                             "activity_full": 0.1, "saliency_quantile": 0.75,
                             "scale_quantile": 0.75, "update_full_ratio": 0.1,
                             "saliency_full_ratio": 0.0909090909,
                             "rank": "ascending V2 temporal TOPSIS degree with saliency guard",
                             "reset": "incoming reinitialization, outgoing zeroing and optimizer-state reset"},
        "new_run_count": 18,
        "run_manifest": manifest,
        "execution": {"targeted_run_ids": targeted_ids, "yoked_run_ids": yoked_ids,
                      "stage_order": "All nine targeted runs complete before any exact-yoked random control starts.",
                      "gpu_mode": "Two independent runs, one per T4", "session_budget_hours": 10.0,
                      "hard_limit_hours": 12, "checkpoint_every_tasks": 10,
                      "reuse_baselines": True, "rerun_existing_baselines": False,
                      "dataset": "nmavros/neuron-death-code", "visibility": "private"},
        "primary_outcome": {"column": "online_accuracy", "probe_point": "task_end",
                            "task_window_zero_indexed_inclusive": [150, 199],
                            "estimator": "IQM over the 3x50 seed-by-task matrix"},
        "bootstrap": {"unit": "whole paired seed trajectory", "resample_tasks": False,
                      "n_bootstrap": 10_000, "confidence": 0.95},
        "planned_comparisons": {"each_target": ["paired exact-yoked random", "V2.3 winner",
                                                  "none", "ReDo tau=0.1", "ReGraMa tau=0.01", "SNR eta=0.08"],
                                "report_all_variants": True},
        "decision": {"development_winner": "Highest target late-window IQM; ties broken by fewer resets.",
                     "accuracy_goal": "Point estimate above SNR 92.624%, followed by held-out paired-seed confirmation before a SOTA claim.",
                     "selection_goal": "Paired target-minus-yoked CI lower bound above zero.",
                     "no_posthoc_changes": "No variants, seed changes or in-run parameter adjustment after freeze."},
        "integrity": {"original_frozen_plan_path": "configs/analysis_plan.json",
                      "original_frozen_plan_sha256": sha(ROOT / "configs/analysis_plan.json"),
                      "prior_v23_plan_sha256": sha(ROOT / "configs/fuzzy_v23_dev_plan.json"),
                      "core_source_sha256": {name: sha(ROOT / name) for name in (
                          "src/temporal_fuzzy.py", "src/probes.py", "src/interventions.py", "src/train.py", "src/config.py")}},
    }
    write_json(PLAN_PATH, plan)
    print(f"wrote {len(manifest)} configs and {PLAN_PATH}")
    print("plan_sha256", sha(PLAN_PATH))


if __name__ == "__main__":
    main()
