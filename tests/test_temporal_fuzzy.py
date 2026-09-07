import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.interventions import Recycler, RecyclerConfig
from src.probes import fuzzy_topsis_degree, temporal_learning_features


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
