"""Temporal fuzzy learning-degree controller (V2 development experiment).

The controller is causal and training-only.  It replaces V1's instantaneous,
strongly correlated activity/gradient/saliency triplet with exponentially
smoothed firing, realised weight movement and loss saliency.  References are
calibrated during warm-up and then frozen; the published probes are unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from .probes import (
    fuzzy_topsis_degree,
    learning_degree_reference,
    temporal_learning_features,
)


@dataclass(frozen=True)
class TemporalFuzzyConfig:
    kind: str = "fuzzy_v2"
    monitor_every: int = 100
    warmup_steps: int = 1000
    patience: int = 2
    cooldown_steps: int = 1000
    task_grace_steps: int = 100
    max_reset_fraction: float = 0.075
    degree_threshold: float = 0.2
    ewma_beta: float = 0.9
    activity_full: float = 0.1
    saliency_quantile: float = 0.75
    scale_quantile: float = 0.75
    update_full_ratio: float = 0.1
    saliency_full_ratio: float = 0.1

    def __post_init__(self):
        if self.kind != "fuzzy_v2":
            raise ValueError("temporal learning-degree kind must be fuzzy_v2")
        for name in ("monitor_every", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "cooldown_steps", "task_grace_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("degree_threshold", "ewma_beta", "max_reset_fraction",
                     "activity_full", "saliency_quantile", "scale_quantile",
                     "update_full_ratio", "saliency_full_ratio"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")

    @classmethod
    def from_dict(cls, value: dict | None) -> "TemporalFuzzyConfig":
        return cls(**dict(value or {}))

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalFuzzyMonitor:
    STATE_VERSION = 1

    def __init__(self, cfg: TemporalFuzzyConfig, model):
        self.cfg = cfg
        self.widths = tuple(int(w) for w in model.hidden_dims)
        self.last_step = -1
        self.last_task = None
        self.calibrated = False
        self.update_reference = np.zeros(len(self.widths), dtype=np.float64)
        self.saliency_reference = np.zeros(len(self.widths), dtype=np.float64)
        self._update_samples = [[] for _ in self.widths]
        self._saliency_samples = [[] for _ in self.widths]
        self._activity_ema = [np.zeros(w, dtype=np.float64) for w in self.widths]
        self._update_ema = [np.zeros(w, dtype=np.float64) for w in self.widths]
        self._saliency_ema = [np.zeros(w, dtype=np.float64) for w in self.widths]
        self._seen = [False for _ in self.widths]
        self._low_count = [np.zeros(w, dtype=np.int64) for w in self.widths]
        self._last_reset = [np.full(w, -cfg.cooldown_steps - 1, dtype=np.int64)
                            for w in self.widths]
        self._pending = [np.empty(0, dtype=np.int64) for _ in self.widths]
        self._latest_degree = [np.full(w, np.nan, dtype=np.float64) for w in self.widths]
        self._weight_snapshot = [
            model.incoming_linear(i).weight.detach().clone()
            for i in range(len(self.widths))
        ]

    def should_observe(self, step: int) -> bool:
        return step > 0 and step % self.cfg.monitor_every == 0

    def selected(self, layer_idx: int) -> np.ndarray:
        return self._pending[layer_idx].copy()

    def current_degrees(self, layer_idx: int) -> np.ndarray:
        return self._latest_degree[layer_idx].copy()

    def observe(self, model, posts, step: int, task_idx: int,
                step_in_task: int) -> list[dict]:
        if len(posts) != len(self.widths):
            raise ValueError("temporal fuzzy hidden layer count mismatch")
        features = []
        for i, post in enumerate(posts):
            weight = model.incoming_linear(i).weight
            features.append(temporal_learning_features(
                post, weight, self._weight_snapshot[i],
                saliency_quantile=self.cfg.saliency_quantile,
            ))
            self._weight_snapshot[i] = weight.detach().clone()
        return self.observe_features(features, step, task_idx, step_in_task)

    def observe_features(self, features: Sequence[dict], step: int,
                         task_idx: int, step_in_task: int) -> list[dict]:
        if not self.should_observe(step) or step <= self.last_step:
            raise ValueError("invalid temporal fuzzy observation schedule")
        if task_idx < 0 or step_in_task < 0 or (
                self.last_task is not None and task_idx < self.last_task):
            raise ValueError("temporal fuzzy observations must be causal")
        if len(features) != len(self.widths):
            raise ValueError("temporal fuzzy hidden layer count mismatch")
        for values, width in zip(features, self.widths):
            for key in ("activity", "update_ratio", "saliency"):
                value = np.asarray(values[key], dtype=np.float64)
                if value.shape != (width,) or not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f"invalid temporal fuzzy {key} vector")
            if (np.asarray(values["activity"]) > 1).any():
                raise ValueError("activity fractions cannot exceed one")

        if task_idx != self.last_task:
            for count in self._low_count:
                count.fill(0)
        self._pending = [np.empty(0, dtype=np.int64) for _ in self.widths]
        rows: list[dict] = []
        c = self.cfg
        for i, (values, width) in enumerate(zip(features, self.widths)):
            if self._seen[i]:
                for key, state in (("activity", self._activity_ema),
                                   ("update_ratio", self._update_ema),
                                   ("saliency", self._saliency_ema)):
                    state[i] *= c.ewma_beta
                    state[i] += (1.0 - c.ewma_beta) * np.asarray(values[key])
            else:
                self._activity_ema[i] = np.asarray(values["activity"], dtype=np.float64).copy()
                self._update_ema[i] = np.asarray(values["update_ratio"], dtype=np.float64).copy()
                self._saliency_ema[i] = np.asarray(values["saliency"], dtype=np.float64).copy()
                self._seen[i] = True

            if not self.calibrated:
                self._update_samples[i].append(learning_degree_reference(
                    self._update_ema[i], c.scale_quantile))
                self._saliency_samples[i].append(learning_degree_reference(
                    self._saliency_ema[i], c.scale_quantile))
                self.update_reference[i] = float(np.median(self._update_samples[i]))
                self.saliency_reference[i] = float(np.median(self._saliency_samples[i]))
            health = fuzzy_topsis_degree(
                self._activity_ema[i], self._update_ema[i], self._saliency_ema[i],
                self.update_reference[i], self.saliency_reference[i],
                activity_full=c.activity_full, update_full_ratio=c.update_full_ratio,
                saliency_full_ratio=c.saliency_full_ratio,
            )
            degree = health["degree"]
            self._latest_degree[i] = degree.copy()
            eligible = ((step >= c.warmup_steps) &
                        (step_in_task >= c.task_grace_steps) &
                        ((step - self._last_reset[i]) >= c.cooldown_steps))
            low = degree <= c.degree_threshold
            self._low_count[i] = np.where(eligible & low, self._low_count[i] + 1, 0)
            candidates = np.flatnonzero(eligible & (self._low_count[i] >= c.patience))
            order = np.lexsort((candidates, degree[candidates]))
            cap = int(np.floor(width * c.max_reset_fraction))
            selected = candidates[order[:cap]]
            self._pending[i] = selected.copy()
            selected_mask = np.zeros(width, dtype=bool)
            selected_mask[selected] = True
            candidate_mask = np.zeros(width, dtype=bool)
            candidate_mask[candidates] = True
            for neuron in range(width):
                rows.append({
                    "step": int(step), "task_idx": int(task_idx),
                    "step_in_task": int(step_in_task), "layer_idx": i,
                    "neuron_idx": neuron, "method": c.kind,
                    "degree": float(degree[neuron]), "forecast": float("nan"),
                    "slope": float("nan"),
                    "gradient_reference": float(self.update_reference[i]),
                    "saliency_reference": float(self.saliency_reference[i]),
                    "calibrated": bool(step >= c.warmup_steps),
                    "eligible": bool(eligible[neuron]),
                    "low_count": int(self._low_count[i][neuron]),
                    "trend_count": 0, "candidate": bool(candidate_mask[neuron]),
                    "selected": bool(selected_mask[neuron]),
                    "trigger_reason": "temporal_low" if selected_mask[neuron] else "",
                    "alive_on_training_batch": bool(values["activity"][neuron] > 0),
                    "preventive_trigger": False,
                    "activity": float(values["activity"][neuron]),
                    "gradient": float(values["update_ratio"][neuron]),
                    "saliency": float(values["saliency"][neuron]),
                    "activity_ema": float(self._activity_ema[i][neuron]),
                    "update_ema": float(self._update_ema[i][neuron]),
                    "saliency_ema": float(self._saliency_ema[i][neuron]),
                    "activity_health": float(health["activity_health"][neuron]),
                    "gradient_health": float(health["update_health"][neuron]),
                    "saliency_health": float(health["saliency_health"][neuron]),
                    "process_health": float(health["process_health"][neuron]),
                })
        if step >= c.warmup_steps:
            self.calibrated = True
        self.last_step, self.last_task = int(step), int(task_idx)
        return rows

    def after_reset(self, layer_idx: int, indices: np.ndarray, step: int,
                    model=None) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if not np.isin(indices, self._pending[layer_idx]).all() or step != self.last_step:
            raise ValueError("reset acknowledgement does not match V2 selection")
        self._last_reset[layer_idx][indices] = step
        self._low_count[layer_idx][indices] = 0
        self._activity_ema[layer_idx][indices] = self.cfg.activity_full
        self._update_ema[layer_idx][indices] = (
            self.update_reference[layer_idx] * self.cfg.update_full_ratio
        )
        self._saliency_ema[layer_idx][indices] = (
            self.saliency_reference[layer_idx] * self.cfg.saliency_full_ratio
        )
        if model is not None:
            weight = model.incoming_linear(layer_idx).weight.detach()
            idx = torch.as_tensor(indices, dtype=torch.long, device=weight.device)
            self._weight_snapshot[layer_idx].index_copy_(0, idx, weight.index_select(0, idx))
        self._pending[layer_idx] = np.setdiff1d(self._pending[layer_idx], indices)

    def state_dict(self) -> dict:
        return deepcopy({
            "version": self.STATE_VERSION, "config": self.cfg.to_dict(),
            "widths": list(self.widths), "last_step": self.last_step,
            "last_task": self.last_task, "calibrated": self.calibrated,
            "update_reference": self.update_reference,
            "saliency_reference": self.saliency_reference,
            "update_samples": self._update_samples,
            "saliency_samples": self._saliency_samples,
            "activity_ema": self._activity_ema, "update_ema": self._update_ema,
            "saliency_ema": self._saliency_ema, "seen": self._seen,
            "low_count": self._low_count, "last_reset": self._last_reset,
            "pending": self._pending, "latest_degree": self._latest_degree,
            "weight_snapshot": [v.detach().cpu() for v in self._weight_snapshot],
        })

    def load_state_dict(self, state: dict) -> None:
        state = deepcopy(state)
        if (state["version"] != self.STATE_VERSION
                or state["config"] != self.cfg.to_dict()
                or tuple(state["widths"]) != self.widths):
            raise ValueError("incompatible temporal fuzzy checkpoint")
        for key in ("last_step", "last_task", "calibrated", "update_reference",
                    "saliency_reference"):
            setattr(self, key, state[key])
        for key in ("update_samples", "saliency_samples", "activity_ema", "update_ema",
                    "saliency_ema", "seen", "low_count", "last_reset", "pending",
                    "latest_degree"):
            setattr(self, "_" + key, state[key])
        self._weight_snapshot = [
            saved.to(device=current.device, dtype=current.dtype)
            for saved, current in zip(state["weight_snapshot"], self._weight_snapshot)
        ]
