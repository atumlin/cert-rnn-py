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

from dataclasses import dataclass
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
    (None for multi_frame)."""

    radius: float
    per_frame: np.ndarray | None
    threat_model: str
    spec: str
    eps_init: float
    n_iters: int

    @property
    def certified(self) -> bool:
        return self.radius > 0.0

    def __str__(self) -> str:
        head = (
            f"CertResult(radius={self.radius:.6g}, certified={self.certified}, "
            f"threat_model={self.threat_model!r}, spec={self.spec})"
        )
        if self.per_frame is not None:
            head += f"\n  per_frame (min over {len(self.per_frame)}): " + np.array2string(
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
) -> CertResult:
    """Certify `spec` on `model` at input `x` via Algorithm 1 bisection.

    `model` is a cert_rnn wrapper (RNNModel / LSTMAutoencoder); `spec` is
    any object with a sound `holds(output)` check compatible with the
    model's `reach_output`. `single_frame` bisects each frame independently
    and reports the min; `multi_frame` perturbs all frames jointly.
    """
    if not hasattr(model, "reach_output"):
        raise TypeError(
            "model must be a cert_rnn model wrapper (RNNModel / LSTMAutoencoder) "
            f"exposing reach_output(); got {type(model).__name__}"
        )
    x = np.asarray(x, dtype=np.float64)

    def at(eps: float, t_pert: int | None) -> bool:
        return spec.holds(model.reach_output(x, eps, threat_model, t_pert))

    if threat_model == "single_frame":
        T = x.shape[0]
        per_frame = np.zeros(T)
        for t in range(T):
            per_frame[t] = bisect_epsilon(
                lambda e, _t=t: at(e, _t), eps_init, n_iters
            )
        return CertResult(
            float(per_frame.min()), per_frame, threat_model, repr(spec), eps_init, n_iters
        )
    if threat_model == "multi_frame":
        eps = bisect_epsilon(lambda e: at(e, None), eps_init, n_iters)
        return CertResult(eps, None, threat_model, repr(spec), eps_init, n_iters)
    raise ValueError(f"unknown threat_model {threat_model!r}")
