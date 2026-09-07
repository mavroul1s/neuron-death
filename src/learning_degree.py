"""Prospective fuzzy learning-degree reset controllers (September 2026).

This is a new-method pilot explicitly requested by the researcher/supervisor,
not a change to published baselines or the frozen historical analysis plan.
Primitive measurements and the fuzzy/forecast formulae live in probes.py.
Here we implement causal calibration, monitoring state and reset safeguards.
No evaluation batch, future task, test loss or test label enters the decision.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .probes import (
    fuzzy_learning_degree,
    learning_degree_features,
    learning_degree_forecast,
    learning_degree_reference,
)


@dataclass(frozen=True)
class LearningDegreeConfig:
    kind: str = "fuzzy"
    monitor_every: int = 100
    warmup_steps: int = 1000
    patience: int = 3
    cooldown_steps: int = 1000
    task_grace_steps: int = 100
    max_reset_fraction: float = 0.05
    degree_threshold: float = 0.2
    activity_full: float = 0.1
    gradient_quantile: float = 0.75
    saliency_quantile: float = 0.75
    scale_quantile: float = 0.75
    gradient_full_ratio: float = 0.1
    saliency_full_ratio: float = 0.1
    trend_window: int = 5
    trend_min_points: int = 3
    trend_horizon: int = 2
    trend_patience: int = 2
    trend_max_degree: float = 0.5
    trend_min_decline: float = 0.02

    def __post_init__(self):
        if self.kind not in ("fuzzy", "fuzzy_trend"):
            raise ValueError("learning degree kind must be fuzzy or fuzzy_trend")
        for name in ("monitor_every", "patience", "trend_window", "trend_min_points",
                     "trend_horizon", "trend_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "cooldown_steps", "task_grace_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("gradient_quantile", "saliency_quantile", "scale_quantile"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        for name in ("max_reset_fraction", "activity_full", "gradient_full_ratio",
                     "saliency_full_ratio", "trend_max_degree", "trend_min_decline"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in (0,1]")
        if not 0 <= self.degree_threshold < self.trend_max_degree <= 1:
            raise ValueError("degree_threshold < trend_max_degree <= 1 is required")
        if not 2 <= self.trend_min_points <= self.trend_window:
            raise ValueError("2 <= trend_min_points <= trend_window is required")

    @classmethod
    def from_dict(cls, d: dict | None) -> "LearningDegreeConfig":
        # Fail on misspelled options instead of silently running another method.
        return cls(**dict(d or {}))

    def to_dict(self) -> dict:
        return asdict(self)


class LearningDegreeMonitor:
    """CPU numpy state; model/optimizer mutations belong to the recycler.

    Observe AFTER backward of mean cross-entropy and BEFORE optimizer.step on
    ``should_observe(step)`` steps. Call retain_grad on posts before backward.
    Read ``selected(layer)`` and call ``after_reset`` only for executed resets.

    Fuzzy: degree<=threshold for three consecutive eligible observations.
    Trend: additionally allow a decline forecast<=threshold, current degree
    <=0.5, and slope<-0.02, for two consecutive eligible observations. The
    shorter trend persistence allows a within-task decision with ~469 steps
    per task and 100-step sampling. It is explicit and identical across seeds.

    Both kinds share warmup, cooldown, task grace and the exact same reset cap.
    Candidates are ranked by current degree (fuzzy) or forecast (trend), with
    neuron index as deterministic tie-break. The cap is FLOOR(fraction*width),
    including zero for tiny test layers: it is never silently rounded upward.
    """

    STATE_VERSION = 2

    def __init__(self, cfg: LearningDegreeConfig, widths: Sequence[int]):
        self.cfg = cfg
        self.widths = tuple(int(w) for w in widths)
        if not self.widths or any(w <= 0 for w in self.widths):
            raise ValueError("learning-degree needs positive hidden widths")
        self.last_step = -1
        self.last_task = None
        self.calibrated = False
        self.gradient_reference = np.zeros(len(self.widths), dtype=np.float64)
        self.saliency_reference = np.zeros(len(self.widths), dtype=np.float64)
        self._gradient_samples = [[] for _ in self.widths]
        self._saliency_samples = [[] for _ in self.widths]
        self._low_count = [np.zeros(w, dtype=np.int64) for w in self.widths]
        self._trend_count = [np.zeros(w, dtype=np.int64) for w in self.widths]
        self._last_reset = [np.full(w, -cfg.cooldown_steps - 1, dtype=np.int64)
                            for w in self.widths]
        self._pending = [np.empty(0, dtype=np.int64) for _ in self.widths]
        self._latest_degree = [np.full(w, np.nan, dtype=np.float64) for w in self.widths]
        self._history = [[] for _ in self.widths]
        self._history_steps = [[] for _ in self.widths]

    def should_observe(self, step: int) -> bool:
        return step > 0 and step % self.cfg.monitor_every == 0

    def selected(self, layer_idx: int) -> np.ndarray:
        return self._pending[layer_idx].copy()

    def current_degrees(self, layer_idx: int) -> np.ndarray:
        """Return the causal score used at the most recent monitoring event."""
        return self._latest_degree[layer_idx].copy()

    def observe(self, model, posts, step: int, task_idx: int,
                step_in_task: int) -> list[dict]:
        if getattr(model, "activation_name", None) != "relu":
            raise ValueError("learning-degree pilot supports ReLU only")
        if getattr(model, "norm_name", None) not in ("none", None):
            raise ValueError("learning-degree pilot requires norm=none")
        if getattr(model, "dropout_p", 0) != 0:
            raise ValueError("learning-degree pilot requires dropout=0")
        if len(posts) != len(self.widths):
            raise ValueError("learning-degree hidden layer count mismatch")
        features = []
        for i, post in enumerate(posts):
            gradient = model.incoming_linear(i).weight.grad
            if gradient is None:
                raise RuntimeError("learning-degree requires incoming weight gradients")
            features.append(learning_degree_features(
                post, gradient, gradient_quantile=self.cfg.gradient_quantile,
                saliency_quantile=self.cfg.saliency_quantile))
        return self.observe_features(features, step, task_idx, step_in_task)

    def observe_features(self, features: Sequence[dict], step: int, task_idx: int,
                         step_in_task: int) -> list[dict]:
        """Decision-only entry point for instrument tests on synthetic signals."""
        if not self.should_observe(step):
            raise ValueError("observe called outside the configured monitor schedule")
        if step <= self.last_step:
            raise ValueError("monitor observations must be strictly chronological")
        if task_idx < 0 or step_in_task < 0 or (self.last_task is not None and task_idx < self.last_task):
            raise ValueError("task indices/within-task steps must be nonnegative and causal")
        if len(features) != len(self.widths):
            raise ValueError("learning-degree hidden layer count mismatch")
        # Validate every layer BEFORE changing any state.
        for values, width in zip(features, self.widths):
            for key in ("activity", "gradient", "saliency"):
                value = np.asarray(values[key], dtype=np.float64)
                if value.shape != (width,) or not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f"invalid learning-degree {key} vector")
                if key == "activity" and (value > 1).any():
                    raise ValueError("activity fractions cannot exceed one")
        if task_idx != self.last_task:
            # Task switches can create a spurious decline. Clear local temporal
            # evidence; retain the warmup scale and reset cooldown across tasks.
            self._history = [[] for _ in self.widths]
            self._history_steps = [[] for _ in self.widths]
            for count in (*self._low_count, *self._trend_count):
                count.fill(0)
        self._pending = [np.empty(0, dtype=np.int64) for _ in self.widths]
        logs = []
        cfg = self.cfg
        for i, (values, width) in enumerate(zip(features, self.widths)):
            if not self.calibrated:
                self._gradient_samples[i].append(learning_degree_reference(
                    values["gradient"], cfg.scale_quantile))
                self._saliency_samples[i].append(learning_degree_reference(
                    values["saliency"], cfg.scale_quantile))
                self.gradient_reference[i] = np.median(self._gradient_samples[i])
                self.saliency_reference[i] = np.median(self._saliency_samples[i])
            health = fuzzy_learning_degree(
                values, self.gradient_reference[i], self.saliency_reference[i],
                activity_full=cfg.activity_full,
                gradient_full_ratio=cfg.gradient_full_ratio,
                saliency_full_ratio=cfg.saliency_full_ratio)
            degree = health["degree"]
            self._latest_degree[i] = degree.copy()
            self._history[i].append(degree.copy())
            self._history_steps[i].append(step)
            self._history[i] = self._history[i][-cfg.trend_window:]
            self._history_steps[i] = self._history_steps[i][-cfg.trend_window:]
            slope, forecast = learning_degree_forecast(
                self._history_steps[i], np.stack(self._history[i]),
                monitor_every=cfg.monitor_every, horizon=cfg.trend_horizon,
                min_points=cfg.trend_min_points)
            eligible = ((step >= cfg.warmup_steps) &
                        (step_in_task >= cfg.task_grace_steps) &
                        ((step - self._last_reset[i]) >= cfg.cooldown_steps))
            low = degree <= cfg.degree_threshold
            declining = ((cfg.kind == "fuzzy_trend") &
                         (degree <= cfg.trend_max_degree) &
                         (forecast <= cfg.degree_threshold) &
                         (slope < -cfg.trend_min_decline))
            self._low_count[i] = np.where(eligible & low, self._low_count[i] + 1, 0)
            self._trend_count[i] = np.where(eligible & declining, self._trend_count[i] + 1, 0)
            by_low = self._low_count[i] >= cfg.patience
            by_trend = self._trend_count[i] >= cfg.trend_patience
            candidates = np.flatnonzero(eligible & (by_low | by_trend))
            priority = forecast if cfg.kind == "fuzzy_trend" else degree
            order = np.lexsort((candidates, priority[candidates]))
            cap = int(np.floor(width * cfg.max_reset_fraction))
            selected = candidates[order[:cap]]
            self._pending[i] = selected.copy()
            selected_mask = np.zeros(width, dtype=bool)
            selected_mask[selected] = True
            for neuron in range(width):
                row = {
                    "step": int(step), "task_idx": int(task_idx),
                    "step_in_task": int(step_in_task), "layer_idx": i,
                    "neuron_idx": neuron, "method": cfg.kind,
                    "degree": float(degree[neuron]), "forecast": float(forecast[neuron]),
                    "slope": float(slope[neuron]),
                    "gradient_reference": float(self.gradient_reference[i]),
                    "saliency_reference": float(self.saliency_reference[i]),
                    "calibrated": bool(step >= cfg.warmup_steps),
                    "eligible": bool(eligible[neuron]),
                    "low_count": int(self._low_count[i][neuron]),
                    "trend_count": int(self._trend_count[i][neuron]),
                    "candidate": bool(eligible[neuron] and (by_low[neuron] or by_trend[neuron])),
                    "selected": bool(selected_mask[neuron]),
                    "trigger_reason": ("low" if by_low[neuron] else "trend") if selected_mask[neuron] else "",
                    "alive_on_training_batch": bool(values["activity"][neuron] > 0),
                    "preventive_trigger": bool(selected_mask[neuron] and not low[neuron] and by_trend[neuron]),
                }
                row.update({key: float(values[key][neuron]) for key in ("activity", "gradient", "saliency")})
                row.update({key: float(health[key][neuron]) for key in
                            ("activity_health", "gradient_health", "saliency_health")})
                logs.append(row)
        if step >= cfg.warmup_steps:
            self.calibrated = True
        self.last_step, self.last_task = int(step), int(task_idx)
        return logs

    def after_reset(self, layer_idx: int, indices: np.ndarray, step: int,
                    model=None) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.widths[layer_idx]):
            raise ValueError("invalid reset neuron indices")
        if step != self.last_step:
            raise ValueError("reset acknowledgement must match the current observation")
        if not np.isin(indices, self._pending[layer_idx]).all():
            raise ValueError("cannot acknowledge resets not selected by the controller")
        self._last_reset[layer_idx][indices] = step
        self._low_count[layer_idx][indices] = 0
        self._trend_count[layer_idx][indices] = 0
        for past in self._history[layer_idx]:
            past[indices] = np.nan
        self._pending[layer_idx] = np.setdiff1d(self._pending[layer_idx], indices)

    def state_dict(self) -> dict:
        return deepcopy({
            "version": self.STATE_VERSION, "config": self.cfg.to_dict(),
            "widths": list(self.widths), "last_step": self.last_step,
            "last_task": self.last_task, "calibrated": self.calibrated,
            "gradient_reference": self.gradient_reference,
            "saliency_reference": self.saliency_reference,
            "gradient_samples": self._gradient_samples, "saliency_samples": self._saliency_samples,
            "low_count": self._low_count, "trend_count": self._trend_count,
            "last_reset": self._last_reset, "pending": self._pending,
            "latest_degree": self._latest_degree,
            "history": self._history, "history_steps": self._history_steps,
        })

    def load_state_dict(self, state: dict) -> None:
        state = deepcopy(state)
        if (state["version"] != self.STATE_VERSION or state["config"] != self.cfg.to_dict()
                or tuple(state["widths"]) != self.widths):
            raise ValueError("incompatible learning-degree checkpoint config or widths")
        for key in ("last_step", "last_task", "calibrated", "gradient_reference", "saliency_reference"):
            setattr(self, key, state[key])
        for key in ("gradient_samples", "saliency_samples", "low_count", "trend_count",
                    "last_reset", "pending", "latest_degree", "history", "history_steps"):
            setattr(self, "_" + key, state[key])
