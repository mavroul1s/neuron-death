import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.interventions import Recycler, RecyclerConfig
from src.models import MLP
from src.probes import fuzzy_topsis_degree, temporal_learning_features
from src.temporal_fuzzy import TemporalFuzzyConfig, TemporalFuzzyMonitor


def test_temporal_features_measure_real_weight_movement():
    post = torch.tensor([[1.0, 0.0], [2.0, 3.0]], requires_grad=True)
    post.retain_grad()
    post.mean().backward()
    previous = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    current = torch.tensor([[6.0, 8.0], [0.0, 3.0]])
    values = temporal_learning_features(post, current, previous)
    np.testing.assert_allclose(values["activity"], [1.0, 0.5])
    np.testing.assert_allclose(values["update_ratio"], [1.0, 0.5])
    assert np.all(values["saliency"] >= 0)


def test_fuzzy_topsis_ideal_states_and_importance_guard():
    values = fuzzy_topsis_degree(
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.1, 0.1]),
        1.0, 1.0,
    )
    np.testing.assert_allclose(values["process_health"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(values["degree"], [0.0, 1.0, 1.0])


def test_yoked_schedule_is_exact_and_complete(tmp_path):
    path = tmp_path / "recycling.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"step": 1100, "layer_idx": 0, "k": 3},
        {"step": 1100, "layer_idx": 1, "k": 0},
        {"step": 1100, "layer_idx": 2, "k": 2},
    ]), path)
    recycler = Recycler(RecyclerConfig(kind="fuzzy_v2_yoked_random"), seed=15)
    recycler.load_yoked_schedule(path, [5, 5, 5])
    assert recycler.due(1100)
    assert not recycler.due(1000)
    assert recycler._yoked_schedule[1100] == {0: 3, 1: 0, 2: 2}


def test_budget_controller_decouples_dose_from_degree_threshold():
    model = MLP(in_features=4, hidden_dims=(10,), out_features=2)
    monitor = TemporalFuzzyMonitor(TemporalFuzzyConfig(
        kind="fuzzy_budget",
        monitor_every=100,
        warmup_steps=0,
        patience=99,
        cooldown_steps=500,
        task_grace_steps=0,
        max_reset_fraction=0.5,
        degree_threshold=0.0001,
        replacement_rate=0.002,
    ), model)
    feature = {
        "activity": np.linspace(0.0, 0.1, 10),
        "update_ratio": np.linspace(0.0, 0.1, 10),
        "saliency": np.linspace(0.0, 0.1, 10),
    }

    rows = monitor.observe_features([feature], step=100, task_idx=0, step_in_task=100)

    # 10 units * 0.002 replacements/update * 100 updates = exactly two.
    # Both are selected even though patience=99 and the threshold is tiny:
    # those fields no longer control dose in budget mode.
    np.testing.assert_array_equal(monitor.selected(0), [0, 1])
    assert sum(row["selected"] for row in rows) == 2
    assert {row["trigger_reason"] for row in rows if row["selected"]} == {
        "temporal_budget"
    }


def test_budget_credit_and_yoked_kind_survive_checkpoint(tmp_path):
    model = MLP(in_features=4, hidden_dims=(10,), out_features=2)
    cfg = TemporalFuzzyConfig(
        kind="fuzzy_budget", monitor_every=100, warmup_steps=0,
        cooldown_steps=0, task_grace_steps=0, max_reset_fraction=0.5,
        replacement_rate=0.0015,
    )
    monitor = TemporalFuzzyMonitor(cfg, model)
    feature = {name: np.ones(10) * 0.1 for name in (
        "activity", "update_ratio", "saliency"
    )}
    monitor.observe_features([feature], step=100, task_idx=0, step_in_task=100)
    assert monitor.selected(0).size == 1
    assert monitor.state_dict()["budget_credit"][0] == 0.5
    restored = TemporalFuzzyMonitor(cfg, model)
    restored.load_state_dict(monitor.state_dict())
    assert restored.state_dict()["budget_credit"][0] == 0.5

    path = tmp_path / "recycling.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"step": 100, "layer_idx": 0, "k": 1},
    ]), path)
    yoke = Recycler(RecyclerConfig(kind="fuzzy_budget_yoked_random"), seed=15)
    yoke.load_yoked_schedule(path, [10])
    assert yoke.due(100)
