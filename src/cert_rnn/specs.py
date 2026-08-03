"""Property specifications and a single high-level entry point.

A `Spec` is a small object that answers one question: *given the model's
abstract output over an eps-ball, does the property hold?* The model
wrapper (cert_rnn.models) produces the abstract output; the spec judges
it. `certify(model, x, spec, ...)` bisects epsilon (Du et al. Algorithm 1)
to find the largest perturbation under which the spec provably holds.

This decouples the three axes a user used to have to wire by hand:

    model  (what network)   x  (which input)   spec  (what property)

Shipped specs:
  - MarginSpec(true_class)        classifier argmax is preserved
  - ThresholdSpec(upper, lower)   every (or selected) output stays in a box
  - ReconErrorSpec(tau)           autoencoder reconstruction score <= tau

Custom properties: implement `holds(output) -> bool` on any object; the
`output` is whatever the paired model's `reach_output(...)` returns (a
logits/hidden `Zono` for RNNModel, a `(z_x_hat_seq, z_x_seq)` tuple for
LSTMAutoencoder). Soundness is the spec author's responsibility: `holds`
must return True only if the property holds for *every* point in the set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from cert_rnn.verify import bisect_epsilon, spec_c_score_ub
from cert_rnn.zono import Zono


@runtime_checkable
class Spec(Protocol):
    """Anything with a sound `holds(output) -> bool` check."""

    def holds(self, output) -> bool: ...


@dataclass(frozen=True)
class MarginSpec:
    """Classifier robustness: logit[true_class] provably dominates every
    other logit over the perturbation set (argmax cannot change).

    Operates on a logits `Zono` (RNNModel.reach_output with a head).
    """

    true_class: int

    def holds(self, z_logits: Zono) -> bool:
        C = z_logits.dim
        tc = self.true_class
        if not (0 <= tc < C):
            raise ValueError(f"true_class {tc} out of range [0, {C})")
        others = [c for c in range(C) if c != tc]
        diffs = np.zeros((len(others), C), dtype=np.float64)
        for i, c in enumerate(others):
            diffs[i, tc] = 1.0
            diffs[i, c] = -1.0
        lb, _ = z_logits.affine_map(diffs).get_ranges()
        return bool(np.all(lb > 0))


@dataclass(frozen=True)
class ThresholdSpec:
    """Box bound on the output: every selected output element stays
    `<= upper` and `>= lower` over the perturbation set.

    `upper`/`lower` may be scalars or per-element arrays; either may be
    None to leave that side unbounded. `indices` restricts the check to a
    subset of output dimensions (default: all). Operates on an output
    `Zono` (RNNModel.reach_output).
    """

    upper: float | np.ndarray | None = None
    lower: float | np.ndarray | None = None
    indices: Sequence[int] | None = None

    def holds(self, z: Zono) -> bool:
        lb, ub = z.get_ranges()
        if self.indices is not None:
            idx = list(self.indices)
            lb, ub = lb[idx], ub[idx]
        ok = True
        if self.upper is not None:
            ok = ok and bool(np.all(ub <= self.upper))
        if self.lower is not None:
            ok = ok and bool(np.all(lb >= self.lower))
        return ok


@dataclass(frozen=True)
class ReconErrorSpec:
    """Autoencoder false-alarm (Spec C): the sound upper bound on the
    reconstruction score `||AE(x') - x'||_2^2 / N` stays `<= tau` over the
    perturbation set. Operates on a `(z_x_hat_seq, z_x_seq)` tuple
    (LSTMAutoencoder.reach_output).
    """

    tau: float

    def holds(self, ae_output) -> bool:
        z_x_hat_seq, z_x_seq = ae_output
        return spec_c_score_ub(z_x_hat_seq, z_x_seq) <= self.tau


@dataclass
class CertResult:
    """Outcome of `certify`. `radius` is the certified epsilon (the
    min-over-frames for single_frame); `per_frame` is the per-frame array
    (None for multi_frame).

    Timing is always recorded, so one `certify` call carries everything:
    `seconds` (total wall clock), `n_reach_calls` (abstract forward
    passes executed), `sec_per_reach`, and for single_frame the
    per-frame wall clock `per_frame_seconds`.

    If `certify` was given a `frames=` subset, `frames` holds the checked
    indices and `radius` is the min over THOSE frames only -- unchecked
    frames are NaN in `per_frame` and carry no guarantee.
    """

    radius: float
    per_frame: np.ndarray | None
    threat_model: str
    spec: str
    eps_init: float
    n_iters: int
    seconds: float = float("nan")
    n_reach_calls: int = 0
    per_frame_seconds: np.ndarray | None = field(default=None, repr=False)
    frames: tuple | None = None

    @property
    def certified(self) -> bool:
        return self.radius > 0.0

    @property
    def sec_per_reach(self) -> float:
        if self.n_reach_calls <= 0:
            return float("nan")
        return self.seconds / self.n_reach_calls

    def __str__(self) -> str:
        head = (
            f"CertResult(radius={self.radius:.6g}, certified={self.certified}, "
            f"threat_model={self.threat_model!r}, spec={self.spec})"
        )
        if np.isfinite(self.seconds):
            head += (
                f"\n  time: {self.seconds:.2f}s total "
                f"({self.n_reach_calls} reach calls @ {self.sec_per_reach:.3f}s)"
            )
        if self.frames is not None:
            head += (
                f"\n  frames checked: {list(self.frames)} "
                f"(radius is the min over these frames ONLY)"
            )
        if self.per_frame is not None:
            n = int(np.sum(np.isfinite(self.per_frame)))
            head += f"\n  per_frame (min over {n}): " + np.array2string(
                self.per_frame, precision=4, threshold=12
            )
        return head


def certify(
    model,
    x: np.ndarray,
    spec: Spec,
    *,
    threat_model: str = "single_frame",
    eps_init: float = 0.5,
    n_iters: int = 12,
    frames: Sequence[int] | None = None,
) -> CertResult:
    """Certify `spec` on `model` at input `x` via Algorithm 1 bisection.

    `model` is a cert_rnn wrapper (RNNModel / LSTMAutoencoder); `spec` is
    any object with a sound `holds(output)` check compatible with the
    model's `reach_output`. `single_frame` bisects each frame independently
    and reports the min; `multi_frame` perturbs all frames jointly.

    `frames` (single_frame only) restricts the bisection to a subset of
    frame indices -- the returned radius is the min over THOSE frames
    only, so it is a cost-saving preview, not a guarantee about the
    unchecked frames (which are NaN in per_frame).

    Wall-clock timing and reach-call counts are always recorded on the
    result (`seconds`, `n_reach_calls`, `sec_per_reach`,
    `per_frame_seconds`), so no separate timing call is needed.
    """
    if not hasattr(model, "reach_output"):
        raise TypeError(
            "model must be a cert_rnn model wrapper (RNNModel / LSTMAutoencoder) "
            f"exposing reach_output(); got {type(model).__name__}"
        )
    x = np.asarray(x, dtype=np.float64)
    n_calls = 0

    def at(eps: float, t_pert: int | None) -> bool:
        nonlocal n_calls
        n_calls += 1
        return spec.holds(model.reach_output(x, eps, threat_model, t_pert))

    t_start = time.perf_counter()
    if threat_model == "single_frame":
        T = x.shape[0]
        if frames is None:
            sel = list(range(T))
        else:
            sel = sorted({int(t) for t in frames})
            bad = [t for t in sel if not (0 <= t < T)]
            if bad:
                raise ValueError(f"frames {bad} out of range [0, {T})")
            if not sel:
                raise ValueError("frames must be non-empty")
        per_frame = np.full(T, np.nan)
        per_frame_seconds = np.full(T, np.nan)
        for t in sel:
            t0 = time.perf_counter()
            per_frame[t] = bisect_epsilon(
                lambda e, _t=t: at(e, _t), eps_init, n_iters
            )
            per_frame_seconds[t] = time.perf_counter() - t0
        return CertResult(
            float(np.min(per_frame[sel])), per_frame, threat_model, repr(spec),
            eps_init, n_iters,
            seconds=time.perf_counter() - t_start,
            n_reach_calls=n_calls,
            per_frame_seconds=per_frame_seconds,
            frames=tuple(sel) if frames is not None else None,
        )
    if threat_model == "multi_frame":
        if frames is not None:
            raise ValueError(
                "frames= only applies to threat_model='single_frame' "
                "(multi_frame perturbs all frames jointly in one bisection)"
            )
        eps = bisect_epsilon(lambda e: at(e, None), eps_init, n_iters)
        return CertResult(
            eps, None, threat_model, repr(spec), eps_init, n_iters,
            seconds=time.perf_counter() - t_start,
            n_reach_calls=n_calls,
        )
    raise ValueError(f"unknown threat_model {threat_model!r}")
