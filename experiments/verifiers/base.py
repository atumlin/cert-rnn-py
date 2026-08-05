"""Verifier adapter contract + the shared bisection harness.

Fairness rule: every lower-bound tool answers the same yes/no query
("does Spec C provably hold at eps?") and is driven by the SAME
Algorithm-1 bisection (same eps_init, same n_iters, same frame loop).
Tools differ only in the abstract domain answering the query. Attack
tools answer the dual query ("is there a concrete violation at eps?")
and produce an UPPER bound via the mirrored bisection.

An adapter therefore implements one method:

    holds(case, eps, threat_model, t_pert) -> bool        (bound tools)
    violates(case, eps, threat_model, t_pert) -> bool     (attack tools)

and inherits `certify_radius` from the harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from experiments.common import Timeout


@dataclass
class Case:
    """One verification problem: a trained LSTM-AE + anchor + threshold."""

    benchmark_id: str
    anchor_id: str
    encoder: dict
    decoder: dict
    head: dict
    anchor: np.ndarray          # (T, D) float64
    tau: float
    meta: dict = field(default_factory=dict)   # H, T, D, L_enc, L_dec, params, ...

    @property
    def T(self) -> int:
        return self.anchor.shape[0]

    @property
    def D(self) -> int:
        return self.anchor.shape[1]


@dataclass
class AdapterResult:
    status: str                       # common.STATUSES
    radius: float | None = None
    radius_kind: str = "lower"
    per_frame: np.ndarray | None = None
    seconds: float = 0.0
    n_bound_calls: int = 0
    tool_meta: dict = field(default_factory=dict)
    error: str | None = None


def bisect(check, eps_init: float, n_iters: int) -> float:
    """Du et al. Algorithm 1: largest eps (down to eps_init * 2^-n_iters
    granularity) where `check(eps)` is True. Identical to
    cert_rnn.verify.bisect_epsilon; duplicated here so adapters need not
    import engine internals."""
    eps, step, best = eps_init, eps_init, 0.0
    for _ in range(n_iters + 1):
        if check(eps):
            best = max(best, eps)
            eps += step / 2
        else:
            eps -= step / 2
        step /= 2
    return best


def bisect_crossing(score_at, tau: float, eps_init: float, n_iters: int
                    ) -> tuple[float, int]:
    """Value-guided radius search: same soundness contract as `bisect`
    (returns the largest eps VERIFIED to satisfy score_at(eps) <= tau,
    at granularity eps_init * 2^-n_iters), but uses the bound VALUES to
    place queries instead of blind halving.

    score_at(eps) -> sound score upper bound; monotone nondecreasing in
    eps. Strategy: bracket by doubling/halving, then regula falsi in
    log(score) space (the bound grows ~geometrically in eps), clamped to
    stay inside the bracket. Every returned radius was directly verified
    by a call with score <= tau -- the search heuristic cannot affect
    soundness, only query count. Returns (radius, n_queries)."""
    import math

    granule = eps_init * (0.5 ** n_iters)
    lo, lo_val = 0.0, None          # certified side (score <= tau)
    hi, hi_val = None, None         # falsified side
    log_tau = math.log(tau)
    history: list[tuple[float, float]] = []   # (eps, score) evaluated
    eps = eps_init
    n = 0
    while (hi is None or (hi - lo) > granule) and n < n_iters + 4:
        s = score_at(eps)
        n += 1
        history.append((eps, s))
        if s <= tau:
            lo, lo_val = eps, s
        else:
            hi, hi_val = eps, s

        if hi is not None and lo > 0.0:
            # bracketed: secant on (log eps, log score), clamped inside
            le_lo, ls_lo = math.log(lo), math.log(max(lo_val, 1e-300))
            le_hi, ls_hi = math.log(hi), math.log(hi_val)
            if ls_hi - ls_lo > 1e-12:
                le = le_lo + (log_tau - ls_lo) * (le_hi - le_lo) / (ls_hi - ls_lo)
                eps_new = math.exp(le)
            else:
                eps_new = 0.5 * (lo + hi)
            margin = max(granule, 0.05 * (hi - lo))
            eps = min(max(eps_new, lo + margin), hi - margin) \
                if hi - lo > 2 * margin else 0.5 * (lo + hi)
        elif hi is not None:
            # no certified point yet: extrapolate the crossing from the
            # last two scores in log-log space instead of blind halving
            if len(history) >= 2 and history[-2][1] > 0 and s > 0:
                (e1, s1), (e2, s2) = history[-2], history[-1]
                d = math.log(s2) - math.log(s1)
                if abs(d) > 1e-12 and e1 != e2:
                    le = math.log(e2) + (log_tau - math.log(s2)) \
                        * (math.log(e2) - math.log(e1)) / d
                    eps = min(max(math.exp(le), granule), hi * 0.75)
                else:
                    eps = hi / 2.0
            else:
                eps = hi / 2.0
            if eps < granule:
                break
        else:
            # no falsified point yet: certified at eps_init, grow
            eps = 2.0 * eps
    return lo, n


class BoundVerifier:
    """Base for sound lower-bound tools. Subclasses set `name` and
    implement holds()."""

    name: str = "?"
    radius_kind = "lower"

    def holds(self, case: Case, eps: float, threat_model: str,
              t_pert: int | None) -> bool:
        raise NotImplementedError

    def supports(self, case: Case) -> bool:
        return True

    def certify_radius(
        self,
        case: Case,
        *,
        threat_model: str = "single_frame",
        frames: list[int] | None = None,
        eps_init: float = 0.5,
        n_iters: int = 12,
        timeout_s: float | None = None,
    ) -> AdapterResult:
        if not self.supports(case):
            return AdapterResult(status="unsupported")
        tm = Timeout(timeout_s)
        n_calls = 0

        def checked(eps: float, t_pert: int | None) -> bool:
            nonlocal n_calls
            if tm.expired:
                raise _TimeoutSignal
            n_calls += 1
            return self.holds(case, eps, threat_model, t_pert)

        try:
            if threat_model == "single_frame":
                sel = frames if frames is not None else list(range(case.T))
                per_frame = np.full(case.T, np.nan)
                for t in sel:
                    per_frame[t] = bisect(
                        lambda e, _t=t: checked(e, _t), eps_init, n_iters)
                radius = float(np.nanmin(per_frame[sel]))
            elif threat_model == "multi_frame":
                per_frame = None
                radius = bisect(lambda e: checked(e, None), eps_init, n_iters)
            else:
                raise ValueError(f"unknown threat_model {threat_model!r}")
        except _TimeoutSignal:
            return AdapterResult(status="timeout", seconds=tm.elapsed,
                                 n_bound_calls=n_calls)
        except Exception as exc:  # tool crash is a result, not a run-killer
            return AdapterResult(status="error", seconds=tm.elapsed,
                                 n_bound_calls=n_calls, error=repr(exc))
        status = "certified" if radius > 0 else "uncertified"
        return AdapterResult(status=status, radius=radius, per_frame=per_frame,
                             seconds=tm.elapsed, n_bound_calls=n_calls)


class AttackVerifier:
    """Base for falsification tools (PGD): produce the smallest eps at
    which a concrete spec violation is FOUND -- an upper bound on the
    true robustness radius (and hence on any sound certified radius)."""

    name: str = "?"
    radius_kind = "upper"

    def violates(self, case: Case, eps: float, threat_model: str,
                 t_pert: int | None) -> bool:
        raise NotImplementedError

    def supports(self, case: Case) -> bool:
        return True

    def certify_radius(
        self,
        case: Case,
        *,
        threat_model: str = "single_frame",
        frames: list[int] | None = None,
        eps_init: float = 0.5,
        n_iters: int = 12,
        timeout_s: float | None = None,
    ) -> AdapterResult:
        """Mirrored bisection: largest eps where NO violation is found.
        The reported radius is that eps + one granule -- i.e. the first
        eps where the attack succeeded (upper bound). If the attack never
        succeeds up to eps_init, the upper bound is +inf (reported NaN)."""
        tm = Timeout(timeout_s)
        n_calls = 0

        def safe(eps: float, t_pert: int | None) -> bool:
            nonlocal n_calls
            if tm.expired:
                raise _TimeoutSignal
            n_calls += 1
            return not self.violates(case, eps, threat_model, t_pert)

        try:
            if threat_model == "single_frame":
                sel = frames if frames is not None else list(range(case.T))
                per_frame = np.full(case.T, np.nan)
                for t in sel:
                    per_frame[t] = bisect(
                        lambda e, _t=t: safe(e, _t), eps_init, n_iters)
                radius = float(np.nanmin(per_frame[sel]))
            else:
                per_frame = None
                radius = bisect(lambda e: safe(e, None), eps_init, n_iters)
        except _TimeoutSignal:
            return AdapterResult(status="timeout", radius_kind="upper",
                                 seconds=tm.elapsed, n_bound_calls=n_calls)
        except Exception as exc:
            return AdapterResult(status="error", radius_kind="upper",
                                 seconds=tm.elapsed, n_bound_calls=n_calls,
                                 error=repr(exc))
        # `radius` is the largest attack-free eps; +granule = first
        # successful attack eps when one was found inside the search range.
        # If the attack never succeeded (radius climbed to the bisection
        # ceiling ~2*eps_init), the upper bound is uninformative.
        granule = eps_init * (0.5 ** n_iters)
        ceiling = eps_init * (2.0 - 0.5 ** n_iters) - 1e-12
        if radius >= ceiling:
            return AdapterResult(
                status="no_violation", radius=None, radius_kind="upper",
                per_frame=per_frame, seconds=tm.elapsed,
                n_bound_calls=n_calls,
                tool_meta={"largest_attack_free_eps": radius},
            )
        return AdapterResult(
            status="falsified", radius=radius + granule, radius_kind="upper",
            per_frame=per_frame, seconds=tm.elapsed, n_bound_calls=n_calls,
            tool_meta={"largest_attack_free_eps": radius},
        )


class _TimeoutSignal(Exception):
    pass
