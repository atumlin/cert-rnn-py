"""Adapter: interval bound propagation (IBP) baseline.

A from-scratch sound interval-arithmetic forward pass through the same
LSTM-AE model dicts the zonotope engine consumes. This is the cheapest
sound domain: no relational information, so bounds blow up with depth --
it anchors the fast-but-loose end of the verifiability/scalability
trade-off, exactly the role IBP plays in feedforward comparisons.

Soundness argument per op:
  affine   W x + b with W = W+ + W-  ->  standard interval matmul
  sigmoid/tanh  monotone  ->  apply to endpoints
  hadamard a*b  ->  min/max over the four endpoint products
No other ops appear in the LSTM cell."""

from __future__ import annotations

import numpy as np

from experiments.verifiers.base import BoundVerifier, Case


def _affine(lo, hi, W, b=None):
    Wp, Wn = np.maximum(W, 0.0), np.minimum(W, 0.0)
    out_lo = Wp @ lo + Wn @ hi
    out_hi = Wp @ hi + Wn @ lo
    if b is not None:
        out_lo, out_hi = out_lo + b, out_hi + b
    return out_lo, out_hi


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _mul(alo, ahi, blo, bhi):
    p = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return p.min(axis=0), p.max(axis=0)


def _lstm_step_stack_interval(x_lo, x_hi, h, c, layers):
    inp_lo, inp_hi = x_lo, x_hi
    for i, lyr in enumerate(layers):
        H = lyr["b"].shape[0] // 4
        pre_lo1, pre_hi1 = _affine(inp_lo, inp_hi, lyr["W_in"], lyr["b"])
        pre_lo2, pre_hi2 = _affine(h[i][0], h[i][1], lyr["W_rec"])
        pre_lo, pre_hi = pre_lo1 + pre_lo2, pre_hi1 + pre_hi2

        sl = lambda a, k: a[k * H:(k + 1) * H]
        i_lo, i_hi = _sigmoid(sl(pre_lo, 0)), _sigmoid(sl(pre_hi, 0))
        f_lo, f_hi = _sigmoid(sl(pre_lo, 1)), _sigmoid(sl(pre_hi, 1))
        g_lo, g_hi = np.tanh(sl(pre_lo, 2)), np.tanh(sl(pre_hi, 2))
        o_lo, o_hi = _sigmoid(sl(pre_lo, 3)), _sigmoid(sl(pre_hi, 3))

        t1_lo, t1_hi = _mul(f_lo, f_hi, c[i][0], c[i][1])
        t2_lo, t2_hi = _mul(i_lo, i_hi, g_lo, g_hi)
        c_lo, c_hi = t1_lo + t2_lo, t1_hi + t2_hi
        h_lo, h_hi = _mul(o_lo, o_hi, np.tanh(c_lo), np.tanh(c_hi))

        h[i] = (h_lo, h_hi)
        c[i] = (c_lo, c_hi)
        inp_lo, inp_hi = h_lo, h_hi
    return h, c


def interval_ae_diff_bounds(encoder, decoder, head, x, eps, threat_model,
                            t_pert):
    """Interval bounds on diff = AE(x') - x' per (t, d) over the ball."""
    T, D = x.shape
    H = encoder["H"]

    def in_box(t):
        if threat_model == "multi_frame" or (
                threat_model == "single_frame" and t == t_pert):
            return x[t] - eps, x[t] + eps
        return x[t].copy(), x[t].copy()

    h = [(np.zeros(H), np.zeros(H)) for _ in encoder["layers"]]
    c = [(np.zeros(H), np.zeros(H)) for _ in encoder["layers"]]
    for t in range(T):
        lo, hi = in_box(t)
        h, c = _lstm_step_stack_interval(lo, hi, h, c, encoder["layers"])
    lat_lo, lat_hi = h[-1]

    hd = [(np.zeros(H), np.zeros(H)) for _ in decoder["layers"]]
    cd = [(np.zeros(H), np.zeros(H)) for _ in decoder["layers"]]
    diff_lo = np.empty((T, D))
    diff_hi = np.empty((T, D))
    for t in range(T):
        hd, cd = _lstm_step_stack_interval(lat_lo, lat_hi, hd, cd,
                                           decoder["layers"])
        xh_lo, xh_hi = _affine(hd[-1][0], hd[-1][1], head["W"], head["b"])
        in_lo, in_hi = in_box(t)
        diff_lo[t] = xh_lo - in_hi
        diff_hi[t] = xh_hi - in_lo
    return diff_lo, diff_hi


def interval_score_bounds(encoder, decoder, head, x, eps, threat_model,
                          t_pert):
    """(score_lb, score_ub): sound bounds on ||AE(x')-x'||^2/N."""
    dlo, dhi = interval_ae_diff_bounds(encoder, decoder, head, x, eps,
                                       threat_model, t_pert)
    comp_max = np.maximum(np.abs(dlo), np.abs(dhi))
    comp_min = np.maximum(0.0, np.maximum(dlo, -dhi))
    N = dlo.size
    return float(np.sum(comp_min ** 2)) / N, float(np.sum(comp_max ** 2)) / N


class IntervalIBP(BoundVerifier):
    name = "interval-ibp"

    def holds(self, case: Case, eps: float, threat_model: str,
              t_pert: int | None) -> bool:
        _, ub = interval_score_bounds(case.encoder, case.decoder, case.head,
                                      case.anchor, eps, threat_model, t_pert)
        return ub <= case.tau


class IntervalIBPMasking(BoundVerifier):
    name = "interval-ibp-masking"

    def holds(self, case: Case, eps: float, threat_model: str,
              t_pert: int | None) -> bool:
        lb, _ = interval_score_bounds(case.encoder, case.decoder, case.head,
                                      case.anchor, eps, threat_model, t_pert)
        return lb >= case.tau
