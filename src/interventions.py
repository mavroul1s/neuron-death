"""Recycling interventions and regularisers.

Contains ReDo (Sokar et al. 2023), Self-Normalized Resets (SNR; Farias &
Jozefiak 2025), ReGraMa (Liu et al. 2025), our size-matched controls, L2, and
shrink-and-perturb.

Two things in here are load-bearing for the paper:

**Zeroing the outgoing weights.** ReDo re-initialises a recycled neuron's
incoming weights and sets its *outgoing* weights to zero; that is what makes the
event function-preserving for a genuinely dead unit. Applying the mask to
incoming weights only is a known reimplementation bug that turns ReDo into a far
more destructive intervention while still producing plausible curves
(CLAUDE.md §6). ``tests/test_interventions.py::test_redo_preserves_function``
exists solely to catch it.

**Per-event, per-layer cardinality matching.** ``k`` is recomputed at every
event for every layer. Sokar et al.'s own random baseline used a fixed
percentage on a cosine schedule, which confounds *which* neurons are recycled
with *how many*. Removing that confound is the crux of C1, so ``k`` must never
become a schedule or a fixed fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import probes
from .probes import LayerProbe, ProbeConfig

#: Recycling arms of the primary experiment (protocol §B.1).
RECYCLE_KINDS = (
    "none",
    "redo",
    "random_matched",
    "inverse_matched",
    "snr",
    "regrama",
)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_recycle_indices(
    kind: str,
    scores: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose which neurons of one layer to recycle at one event.

    Returns ``(selected_indices, dormant_indices)``; both sorted ascending.
    Every arm recycles exactly ``k = |dormant set|`` neurons, so the arms differ
    only in *which* neurons, never in *how many*.
    """
    scores = np.asarray(scores, dtype=np.float64)
    h = scores.size
    dormant = np.flatnonzero(scores <= tau)  # Sokar: tau-dormant iff s <= tau
    k = int(dormant.size)

    if kind == "none" or k == 0:
        return np.empty(0, dtype=np.int64), dormant
    if kind == "redo":
        return dormant.astype(np.int64), dormant
    if kind == "random_matched":
        # Uniformly at random from the whole layer -- not from the complement of
        # the dormant set. The control asks "does recycling k arbitrary neurons
        # do as well as recycling the k quietest", so the draw must be over all
        # neurons.
        return np.sort(rng.choice(h, size=k, replace=False)).astype(np.int64), dormant
    if kind == "inverse_matched":
        # k highest-scoring neurons. Stable sort so ties break by index and the
        # arm is reproducible.
        order = np.argsort(-scores, kind="stable")
        return np.sort(order[:k]).astype(np.int64), dormant
    raise ValueError(f"unknown recycling kind {kind!r}; known: {RECYCLE_KINDS}")


# ---------------------------------------------------------------------------
# Applying a recycle
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reset_optimizer_slice(
    optimizer: Optional[torch.optim.Optimizer],
    param: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    spatial: int = 1,
) -> None:
    """Zero the optimizer's per-parameter state for the recycled slice.

    Sokar et al.'s Algorithm 1 resets the optimizer state of the recycled
    weights; without it, a freshly re-initialised neuron inherits the momentum
    (or Adam moments) of the dead neuron it replaced and is immediately dragged
    back down.

    Adam's ``step`` counter is a per-parameter scalar and cannot be reset for a
    slice without also resetting it for the untouched weights of the same
    tensor, which would change their bias correction. It is therefore left
    alone -- a deliberate, documented deviation.
    """
    if optimizer is None:
        return
    state = optimizer.state.get(param)
    if not state:
        return
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != param.shape:
            continue  # 'step' scalars and anything else non-conformant
        if dim == 0:
            value[index] = 0
        elif dim == 1:
            if spatial == 1:
                value[:, index] = 0
            else:
                # conv -> flatten -> Linear: mirror the weight slicing exactly,
                # or the moments of the wrong columns get cleared.
                for c in index.tolist():
                    value[:, flattened_channel_columns(int(c), spatial)] = 0
        else:
            raise ValueError(f"unsupported dim {dim}")


def flattened_channel_columns(channel: int, spatial: int) -> slice:
    """Columns of a post-flatten Linear that belong to one conv channel.

    A feature map flattened with ``.reshape(N, -1)`` is channel-major, so
    channel ``c`` owns ``[c*spatial, (c+1)*spatial)``.

    Isolated into its own function because CLAUDE.md §5.5 singles this out:
    "getting that indexing wrong will silently zero the wrong columns". An
    off-by-``spatial`` here recycles one channel and blanks another's outgoing
    weights, and nothing raises.
    """
    return slice(channel * spatial, (channel + 1) * spatial)


@torch.no_grad()
def _zero_outgoing(module, idx_t: torch.Tensor, spatial: int) -> None:
    """Zero the outgoing weights of the given units of the previous layer."""
    if spatial == 1:
        # Linear (H_next, H) or Conv2d (C_next, C, kH, kW): unit is axis 1.
        module.weight[:, idx_t] = 0.0
        return
    # conv -> flatten -> Linear: each unit owns `spatial` contiguous columns.
    for c in idx_t.tolist():
        module.weight[:, flattened_channel_columns(int(c), spatial)] = 0.0


@torch.no_grad()
def recycle_neurons(
    model,
    layer_idx: int,
    indices: np.ndarray,
    weight_generator: Optional[torch.Generator] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    reset_optimizer_state: bool = True,
) -> int:
    """Re-initialise the given units of one hidden layer, in place.

    Sokar et al. 2023, Algorithm 1:
      * incoming weights re-sampled from the layer's *original* init
        distribution, bias reset to its init value;
      * outgoing weights set to **zero**.

    Works for a fully-connected neuron and for a conv channel, where "incoming
    weights" is the whole filter bank ``W[c, :, :, :]`` and the outgoing slice
    may span every spatial position belonging to that channel (CLAUDE.md §5.5).

    Returns the number of units recycled.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return 0

    mod_in = model.incoming_linear(layer_idx)
    mod_out = model.outgoing_linear(layer_idx)
    spatial = getattr(model, "outgoing_spatial", lambda _: 1)(layer_idx)
    device = mod_in.weight.device
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)

    # --- incoming ---------------------------------------------------------
    new_w = model.sample_incoming_weights(layer_idx, int(idx.size), weight_generator)
    mod_in.weight[idx_t] = new_w.to(device=device, dtype=mod_in.weight.dtype)
    if mod_in.bias is not None:
        mod_in.bias[idx_t] = model.init_bias_value(layer_idx)

    # --- outgoing: THE half that gets forgotten ---------------------------
    _zero_outgoing(mod_out, idx_t, spatial)
    # The outgoing layer's *bias* is untouched: it is not a property of this
    # unit, and zeroing it would change the function for every other unit.

    if reset_optimizer_state:
        _reset_optimizer_slice(optimizer, mod_in.weight, idx_t, dim=0)
        if mod_in.bias is not None:
            _reset_optimizer_slice(optimizer, mod_in.bias, idx_t, dim=0)
        _reset_optimizer_slice(
            optimizer, mod_out.weight, idx_t, dim=1, spatial=spatial
        )

    return int(idx.size)


# ---------------------------------------------------------------------------
# The recycler
# ---------------------------------------------------------------------------


@dataclass
class RecyclerConfig:
    kind: str = "none"
    tau: float = 0.0
    freq: int = 1000  # F = 1000 optimizer steps (Sokar et al. 2023)
    score_batch_size: int = 64  # Sokar et al.'s default
    reset_optimizer_state: bool = True
    composition_on_reference: bool = True
    # SNR implementation defaults from the authors' official Permuted-MNIST
    # Colab. eta is the upper-tail rejection probability, so the corresponding
    # inter-firing-time percentile is 1 - eta.
    snr_eta: float = 0.08
    snr_tau_max: int = 20_000
    snr_update_every_tasks: int = 16
    snr_expansion_factor: float = 2.0
    snr_min_age: int = 100

    def __post_init__(self):
        if self.kind not in RECYCLE_KINDS:
            raise ValueError(f"unknown recycling kind {self.kind!r}; known: {RECYCLE_KINDS}")
        if self.freq <= 0:
            raise ValueError("recycling freq must be positive")
        if not 0.0 < self.snr_eta < 1.0:
            raise ValueError("snr_eta must be strictly between 0 and 1")
        if self.snr_tau_max <= 0:
            raise ValueError("snr_tau_max must be positive")
        if self.snr_update_every_tasks <= 0:
            raise ValueError("snr_update_every_tasks must be positive")
        if self.snr_expansion_factor < 1.0:
            raise ValueError("snr_expansion_factor must be at least 1")
        if not 0 < self.snr_min_age <= self.snr_tau_max:
            raise ValueError("snr_min_age must be in [1, snr_tau_max]")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RecyclerConfig":
        d = dict(d or {})
        return cls(
            kind=d.get("kind", "none"),
            tau=float(d.get("tau", 0.0)),
            freq=int(d.get("freq", 1000)),
            score_batch_size=int(d.get("score_batch_size", 64)),
            reset_optimizer_state=bool(d.get("reset_optimizer_state", True)),
            composition_on_reference=bool(d.get("composition_on_reference", True)),
            snr_eta=float(d.get("snr_eta", 0.08)),
            snr_tau_max=int(d.get("snr_tau_max", 20_000)),
            snr_update_every_tasks=int(d.get("snr_update_every_tasks", 16)),
            snr_expansion_factor=float(d.get("snr_expansion_factor", 2.0)),
            snr_min_age=int(d.get("snr_min_age", 100)),
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "tau": self.tau,
            "freq": self.freq,
            "score_batch_size": self.score_batch_size,
            "reset_optimizer_state": self.reset_optimizer_state,
            "composition_on_reference": self.composition_on_reference,
            "snr_eta": self.snr_eta,
            "snr_tau_max": self.snr_tau_max,
            "snr_update_every_tasks": self.snr_update_every_tasks,
            "snr_expansion_factor": self.snr_expansion_factor,
            "snr_min_age": self.snr_min_age,
        }


@dataclass
class EventResult:
    """Outcome of one recycling event, across all hidden layers."""

    event_idx: int
    step: int
    task_idx: int
    rows: List[dict] = field(default_factory=list)
    recycled: Dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def total_recycled(self) -> int:
        return int(sum(v.size for v in self.recycled.values()))


class Recycler:
    """Runs a recycling event every F optimizer steps.

    The intervention and its measurement use deliberately different batches:

    * **selection** uses a fresh 64-example batch from the current task, exactly
      as Sokar et al.'s Algorithm 1 specifies;
    * **composition logging** uses the fixed 2048-example probe batch, because
      "was this neuron genuinely dead" is a claim about the network, not about
      64 draws. Sixty-four samples would systematically over-report
      ``dead_exact`` and inflate exactly the number C1 is about.

    Both random streams (weight resampling, random-matched selection) are
    dedicated generators, checkpointed with the run, so recycling is
    reproducible independently of anything else that consumes randomness.
    """

    def __init__(
        self,
        cfg: RecyclerConfig,
        seed: int,
        probe_cfg: Optional[ProbeConfig] = None,
        run_id: str = "",
    ):
        self.cfg = cfg
        self.run_id = run_id
        self.probe_cfg = probe_cfg or ProbeConfig()
        # CPU generators: recycling must be bit-identical whether a resumed run
        # lands on a T4 or a laptop.
        self._weight_gen = torch.Generator()
        self._weight_gen.manual_seed(int(seed) ^ 0x5EED_1)
        self._select_rng = np.random.default_rng([int(seed), 0x5EED_2])
        self.event_idx = 0
        # SNR state is kept on CPU.  Its histogram is sparse in the age axis:
        # with a mini-batch implementation, observed ages are mostly multiples
        # of batch size, so a dense H x 20,001 array would waste ~120 MB for the
        # default MLP and bloat every checkpoint.
        self._snr_ages: List[np.ndarray] = []
        self._snr_thresholds: List[np.ndarray] = []
        self._snr_hist: List[Dict[int, np.ndarray]] = []
        self._snr_pending: List[np.ndarray] = []

    @property
    def enabled(self) -> bool:
        return self.cfg.kind != "none"

    def due(self, step: int) -> bool:
        """True on steps 1000, 2000, ... (never on step 0: an event before any
        training would recycle the initialisation itself)."""
        if self.cfg.kind == "snr":
            return step > 0 and any(np.any(mask) for mask in self._snr_pending)
        return self.enabled and step > 0 and step % self.cfg.freq == 0

    @property
    def needs_training_activations(self) -> bool:
        """Whether the training forward must expose hidden activations."""
        return self.cfg.kind == "snr"

    def _initialize_snr(self, widths: Sequence[int]) -> None:
        self._snr_ages = [np.zeros(int(h), dtype=np.int64) for h in widths]
        self._snr_thresholds = [
            np.full(int(h), self.cfg.snr_tau_max, dtype=np.int64) for h in widths
        ]
        self._snr_hist = [{} for _ in widths]
        self._snr_pending = [np.zeros(int(h), dtype=bool) for h in widths]

    def _snr_add_intervals(
        self, layer_idx: int, ages: np.ndarray, mask: np.ndarray
    ) -> None:
        """Add completed/censored inter-firing intervals to the sparse histogram."""
        idx = np.flatnonzero(mask & (ages > 0))
        if idx.size == 0:
            return
        hist = self._snr_hist[layer_idx]
        clipped = np.minimum(ages[idx], self.cfg.snr_tau_max)
        for age in np.unique(clipped):
            age_i = int(age)
            counts = hist.setdefault(
                age_i, np.zeros(ages.size, dtype=np.int64)
            )
            counts[idx[clipped == age]] += 1

    @torch.no_grad()
    def observe_activations(self, posts: Sequence[torch.Tensor]) -> None:
        """Update SNR inter-firing ages from one training mini-batch.

        This follows the released implementation: a unit fires if it is
        positive for at least one example (and, for a convolutional channel,
        at least one spatial position) in the mini-batch.  A non-firing unit's
        age increases by the number of examples in that mini-batch.
        """
        if self.cfg.kind != "snr":
            return
        widths = [int(probes.as_unit_matrix(p).shape[1]) for p in posts]
        if not self._snr_ages:
            self._initialize_snr(widths)
        if widths != [int(a.size) for a in self._snr_ages]:
            raise ValueError("SNR state does not match the model's hidden widths")

        for layer_idx, post in enumerate(posts):
            units = probes.as_unit_matrix(post.detach())
            fired = (units > 0).any(dim=0).cpu().numpy().astype(bool, copy=False)
            ages = self._snr_ages[layer_idx]
            self._snr_add_intervals(layer_idx, ages, fired)
            ages[fired] = 0
            ages[~fired] += int(post.shape[0])
            self._snr_pending[layer_idx] = ages >= self._snr_thresholds[layer_idx]

    def end_task(self, task_idx: int) -> None:
        """Update SNR's neuron-specific thresholds at the official cadence."""
        if self.cfg.kind != "snr" or not self._snr_ages:
            return
        if (int(task_idx) + 1) % self.cfg.snr_update_every_tasks != 0:
            return

        percentile = 1.0 - self.cfg.snr_eta
        for layer_idx, ages in enumerate(self._snr_ages):
            hist = self._snr_hist[layer_idx]
            total = np.zeros(ages.size, dtype=np.int64)
            for counts in hist.values():
                total += counts
            target = percentile * total
            cumulative = np.zeros(ages.size, dtype=np.int64)
            quantile = np.ones(ages.size, dtype=np.int64)
            unresolved = total > 0
            for age, counts in sorted(hist.items()):
                cumulative += counts
                hit = unresolved & (cumulative >= target)
                quantile[hit] = int(age)
                unresolved[hit] = False

            old = self._snr_thresholds[layer_idx]
            updated = np.where(
                quantile < old,
                quantile,
                np.ceil(self.cfg.snr_expansion_factor * old).astype(np.int64),
            )
            self._snr_thresholds[layer_idx] = np.clip(
                updated, self.cfg.snr_min_age, self.cfg.snr_tau_max
            ).astype(np.int64)
            self._snr_hist[layer_idx] = {}
            self._snr_pending[layer_idx] = (
                ages >= self._snr_thresholds[layer_idx]
            )

    @staticmethod
    def _grama_scores(model) -> List[np.ndarray]:
        """Eq. 2 / official ReGraMa implementation, one score per hidden unit."""
        out: List[np.ndarray] = []
        for layer_idx in range(model.n_hidden):
            grad = model.incoming_linear(layer_idx).weight.grad
            if grad is None:
                h = int(model.hidden_dims[layer_idx])
                out.append(np.full(h, np.inf, dtype=np.float64))
                continue
            reduce_dims = tuple(range(1, grad.ndim))
            magnitude = grad.detach().abs().mean(dim=reduce_dims)
            normalized = magnitude / (magnitude.mean() + 1e-9)
            out.append(normalized.to(torch.float64).cpu().numpy())
        return out

    @torch.no_grad()
    def run_event(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        score_x: torch.Tensor,
        probe_x: torch.Tensor,
        step: int,
        task_idx: int,
        ref_x: Optional[torch.Tensor] = None,
    ) -> EventResult:
        was_training = model.training
        model.eval()
        try:
            # 1. Scores, from the 64-example batch (Sokar Algorithm 1).
            _, _, score_posts = model.forward_with_activations(score_x)
            # as_unit_matrix folds a conv layer's (N, C, H, W) to (N*H*W, C) so
            # the unit is the channel. Without it a conv layer yields C*H*W
            # "units" and every index downstream is meaningless.
            layer_scores = [
                probes.sokar_scores(
                    probes.mean_abs_activation(probes.as_unit_matrix(p))
                )
                .cpu()
                .numpy()
                for p in score_posts
            ]
            grama_scores = (
                self._grama_scores(model) if self.cfg.kind == "regrama" else None
            )

            # 2. Composition, from the full probe batch, BEFORE any weights move.
            # compute_erank=False: the composition table does not use effective
            # rank, and the SVD dominates probe cost.
            cur_probes = probes.probe_model(
                model, probe_x, self.probe_cfg, compute_erank=False
            )
            ref_probes = (
                probes.probe_model(model, ref_x, self.probe_cfg, compute_erank=False)
                if (ref_x is not None and self.cfg.composition_on_reference)
                else None
            )

            result = EventResult(event_idx=self.event_idx, step=step, task_idx=task_idx)

            for layer_idx, scores in enumerate(layer_scores):
                dormant = np.flatnonzero(scores <= self.cfg.tau).astype(np.int64)
                method_scores = scores
                method_threshold = self.cfg.tau
                selection_metric = "sokar_activation"
                if self.cfg.kind == "regrama":
                    assert grama_scores is not None
                    method_scores = grama_scores[layer_idx]
                    selected = np.flatnonzero(method_scores <= self.cfg.tau).astype(
                        np.int64
                    )
                    selection_metric = "grama_gradient"
                elif self.cfg.kind == "snr":
                    selected = np.flatnonzero(self._snr_pending[layer_idx]).astype(
                        np.int64
                    )
                    method_scores = self._snr_ages[layer_idx].astype(np.float64)
                    method_threshold = float("nan")  # neuron-specific thresholds
                    selection_metric = "snr_inter_firing_age"
                else:
                    selected, dormant = select_recycle_indices(
                        self.cfg.kind, scores, self.cfg.tau, self._select_rng
                    )
                comp = probes.composition(cur_probes[layer_idx], selected)
                row = {
                    "run_id": self.run_id,
                    "event_idx": self.event_idx,
                    "step": step,
                    "task_idx": task_idx,
                    "arm": self.cfg.kind,
                    "layer_idx": layer_idx,
                    "tau": self.cfg.tau,
                    "k": comp.k,
                    "n_neurons": int(scores.size),
                    "n_dormant": int(dormant.size),
                    "score_batch_size": int(score_x.shape[0]),
                    "selection_metric": selection_metric,
                    "selection_threshold": method_threshold,
                    "mean_selection_score_selected": (
                        float(method_scores[selected].mean())
                        if selected.size else float("nan")
                    ),
                    # Mean Sokar score over the *dormant* set on the selection
                    # batch, for reference; the headline number is the mean
                    # score of the neurons actually recycled, below.
                    "mean_sokar_score_dormant_scorebatch": (
                        float(scores[dormant].mean()) if dormant.size else float("nan")
                    ),
                }
                if self.cfg.kind == "snr":
                    thresholds = self._snr_thresholds[layer_idx]
                    row.update(
                        {
                            "snr_eta": self.cfg.snr_eta,
                            "snr_mean_threshold_selected": (
                                float(thresholds[selected].mean())
                                if selected.size else float("nan")
                            ),
                            "snr_min_threshold_layer": int(thresholds.min()),
                            "snr_max_threshold_layer": int(thresholds.max()),
                        }
                    )
                row.update(comp.as_dict())
                if ref_probes is not None:
                    row.update(
                        probes.composition(ref_probes[layer_idx], selected).as_dict("_ref")
                    )
                result.rows.append(row)

                # 3. Apply.
                recycle_neurons(
                    model,
                    layer_idx,
                    selected,
                    weight_generator=self._weight_gen,
                    optimizer=optimizer,
                    reset_optimizer_state=self.cfg.reset_optimizer_state,
                )
                if self.cfg.kind == "snr" and selected.size:
                    ages = self._snr_ages[layer_idx]
                    mask = np.zeros(ages.size, dtype=bool)
                    mask[selected] = True
                    # A reset terminates the current (right-censored) interval,
                    # exactly as in the released implementation.
                    self._snr_add_intervals(layer_idx, ages, mask)
                    ages[selected] = 0
                    self._snr_pending[layer_idx][selected] = False
                result.recycled[layer_idx] = selected

            self.event_idx += 1
            return result
        finally:
            if was_training:
                model.train()

    # -- checkpointing --------------------------------------------------------

    def state_dict(self) -> dict:
        state = {
            "event_idx": self.event_idx,
            "weight_gen": self._weight_gen.get_state(),
            "select_rng": self._select_rng.bit_generator.state,
        }
        if self.cfg.kind == "snr":
            state["snr"] = {
                "ages": self._snr_ages,
                "thresholds": self._snr_thresholds,
                "hist": self._snr_hist,
                "pending": self._snr_pending,
            }
        return state

    def load_state_dict(self, state: dict) -> None:
        self.event_idx = int(state["event_idx"])
        self._weight_gen.set_state(state["weight_gen"])
        self._select_rng.bit_generator.state = state["select_rng"]
        snr = state.get("snr")
        if snr is not None:
            self._snr_ages = [np.asarray(x, dtype=np.int64) for x in snr["ages"]]
            self._snr_thresholds = [
                np.asarray(x, dtype=np.int64) for x in snr["thresholds"]
            ]
            self._snr_hist = [
                {int(age): np.asarray(v, dtype=np.int64) for age, v in h.items()}
                for h in snr["hist"]
            ]
            self._snr_pending = [
                np.asarray(x, dtype=bool) for x in snr["pending"]
            ]


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def l2_penalty(model, include_bias: bool = True) -> torch.Tensor:
    """0.5 * sum(theta^2) over the model's parameters.

    Added to the loss as ``lambda * l2_penalty(model)``, whose gradient is
    ``lambda * theta`` -- identical to passing ``weight_decay=lambda`` to SGD,
    but written out so it is visible in the loss and cannot be confused with
    AdamW's *decoupled* decay. That distinction matters for the C5 AdamW arm
    (protocol §B.3).
    """
    total = None
    for name, p in model.named_parameters():
        if not include_bias and name.endswith("bias"):
            continue
        s = p.pow(2).sum()
        total = s if total is None else total + s
    if total is None:
        raise ValueError("model has no parameters")
    return 0.5 * total


# ---------------------------------------------------------------------------
# Shrink and perturb
# ---------------------------------------------------------------------------


@dataclass
class ShrinkPerturbConfig:
    """Ash & Adams 2020, applied continually (Dohare et al. Fig. 4b arm).

    ``theta <- shrink * theta + perturb * nu``, ``nu`` drawn from the layer's
    original initialisation distribution.
    """

    enabled: bool = False
    shrink: float = 0.5
    perturb: float = 0.01
    every_tasks: int = 1  # applied at task boundaries

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ShrinkPerturbConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", False)),
            shrink=float(d.get("shrink", 0.5)),
            perturb=float(d.get("perturb", 0.01)),
            every_tasks=int(d.get("every_tasks", 1)),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "shrink": self.shrink,
            "perturb": self.perturb,
            "every_tasks": self.every_tasks,
        }


@torch.no_grad()
def shrink_and_perturb(
    model, shrink: float, perturb: float, generator: Optional[torch.Generator] = None
) -> None:
    """Apply shrink-and-perturb to every Linear in the model, in place.

    Biases are shrunk but not perturbed: the bias initialisation distribution is
    the constant 0, so "perturb with noise from the init distribution" adds
    nothing for them.
    """
    for lin, spec in zip(model.linears, model.init_specs):
        noise = spec.sample(tuple(lin.weight.shape), generator).to(
            device=lin.weight.device, dtype=lin.weight.dtype
        )
        lin.weight.mul_(shrink).add_(noise, alpha=perturb)
        if lin.bias is not None:
            lin.bias.mul_(shrink)
