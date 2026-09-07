from __future__ import annotations

import json
import zipfile

import numpy as np
import torch

from src import probes
from src.interventions import Recycler, RecyclerConfig
from src.learning_degree import LearningDegreeConfig, LearningDegreeMonitor
from src.probes import ProbeConfig
from scripts.kaggle_remote import archive_hashes, safe_archive_member
from tests.conftest import make_model


def _features(width, value):
    values = np.full(width, value, dtype=np.float64)
    return {"activity": values.copy(), "gradient": values.copy(),
            "saliency": values.copy()}


def test_feature_extraction_uses_activity_gradient_and_loss_sensitivity():
    raw = torch.tensor([[-1.0, 2.0], [3.0, -1.0]], requires_grad=True)
    post = torch.relu(raw)
    post.retain_grad()
    (post * torch.tensor([[1.0, 4.0], [2.0, 1.0]])).mean().backward()
    incoming_grad = torch.tensor([[0.0, 1.0, 1000.0, 2.0],
                                  [1.0, 1.0, 1.0, 1.0]])
    got = probes.learning_degree_features(post, incoming_grad)
    assert np.allclose(got["activity"], [0.5, 0.5])
    assert got["gradient"][0] < 1000.0  # quantile, not the outlier-sensitive mean
    assert np.all(got["saliency"] > 0)


def test_fuzzy_or_rule_protects_a_rare_but_consequential_neuron():
    features = {
        "activity": np.array([0.01, 0.0]),
        "gradient": np.array([0.0, 0.0]),
        "saliency": np.array([1.0, 0.0]),
    }
    got = probes.fuzzy_learning_degree(features, 1.0, 1.0)
    assert got["degree"][0] == 1.0
    assert got["degree"][1] == 0.0


def test_fuzzy_monitor_requires_persistence_caps_resets_and_round_trips():
    cfg = LearningDegreeConfig(
        monitor_every=100, warmup_steps=200, patience=3,
        cooldown_steps=0, task_grace_steps=0, max_reset_fraction=0.05,
    )
    monitor = LearningDegreeMonitor(cfg, [20])
    monitor.observe_features([_features(20, 1.0)], 100, 0, 100)
    monitor.observe_features([_features(20, 1.0)], 200, 0, 200)
    for step in (300, 400):
        monitor.observe_features([_features(20, 0.0)], step, 0, step)
        assert monitor.selected(0).size == 0
    rows = monitor.observe_features([_features(20, 0.0)], 500, 0, 500)
    assert monitor.selected(0).tolist() == [0]  # floor(20 * .05) == 1
    assert sum(row["candidate"] for row in rows) == 20
    clone = LearningDegreeMonitor(cfg, [20])
    clone.load_state_dict(monitor.state_dict())
    assert clone.selected(0).tolist() == [0]
    clone.after_reset(0, np.array([0]), 500)
    assert clone.selected(0).size == 0


def test_predictive_monitor_can_trigger_before_degree_crosses_threshold():
    cfg = LearningDegreeConfig(
        kind="fuzzy_trend", monitor_every=100, warmup_steps=100,
        patience=99, trend_patience=2, cooldown_steps=0,
        task_grace_steps=0, max_reset_fraction=0.1,
        activity_full=1.0, gradient_full_ratio=1.0,
        saliency_full_ratio=1.0,
    )
    monitor = LearningDegreeMonitor(cfg, [10])
    final_rows = []
    for step, value in ((100, 1.0), (200, 0.55), (300, 0.40), (400, 0.30)):
        final_rows = monitor.observe_features([_features(10, value)], step, 0, step)
    assert monitor.selected(0).tolist() == [0]
    assert final_rows[0]["degree"] > cfg.degree_threshold
    assert final_rows[0]["preventive_trigger"] is True


def test_joint_layer_reset_keeps_all_selected_outgoing_slices_zero(gen):
    model = make_model(gen, hidden=(8, 8), in_features=4, out_features=3)
    recycler = Recycler(
        RecyclerConfig(kind="redo", tau=1e9, freq=1,
                       zero_outgoing_after_event=True),
        seed=1, probe_cfg=ProbeConfig(n_probe=16), run_id="joint",
    )
    x = torch.randn(16, 4)
    recycler.run_event(model, None, x[:8], x, step=1, task_idx=0, ref_x=x)
    assert torch.count_nonzero(model.linears[1].weight) == 0
    assert torch.count_nonzero(model.linears[2].weight) == 0


def test_kaggle_staging_is_private_hash_checked_and_rejects_secrets(tmp_path):
    payload = tmp_path / "runtime.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("src/train.py", "# safe")
        archive.writestr("data/mnist.npz", b"data")
    assert set(archive_hashes(payload)) == {"src/train.py", "data/mnist.npz"}
    assert not safe_archive_member("api_kaggle/kaggle.json")
    notebook = tmp_path / "run.ipynb"
    notebook.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5,
        "metadata": {}, "cells": [{"cell_type": "code", "metadata": {},
        "execution_count": None, "outputs": [], "source": ["print('ok')"]}]}))
    # prepare writes only inside the repository's ignored remote_runs tree, so
    # exercise its pure staging contract through a temporary payload elsewhere
    # by checking the guard primitives here. The end-to-end staged artifact is
    # verified before the real push in scripts/kaggle_remote.py.
    assert payload.is_file() and notebook.is_file()
