"""Stratified box generators for the soundness suite.

Strata cover the nine sign configurations of ([lx,ux], [ly,uy]) from Du et
al.'s Table 8 (four pure-sign quadrants, four one-axis straddles, origin box)
plus the numerical regimes the enumeration's tolerance gates care about:
saturation, tight widths, very wide boxes, and widths straddling the 1e-12
point-degeneracy gate in transformers.py.

Every generator takes an explicit seeded rng; the calling test logs the seed.
Returned arrays are (lx, ux, ly, uy), each shape (n,).
"""

from __future__ import annotations

import numpy as np

# The nine Table 8 sign strata. p = strictly positive interval, n = strictly
# negative, s = straddles 0. (x_kind, y_kind) per stratum.
SIGN_STRATA = {
    "x_pos_y_pos": ("p", "p"),
    "x_pos_y_neg": ("p", "n"),
    "x_neg_y_pos": ("n", "p"),
    "x_neg_y_neg": ("n", "n"),
    "x_straddle_y_pos": ("s", "p"),
    "x_straddle_y_neg": ("s", "n"),
    "x_pos_y_straddle": ("p", "s"),
    "x_neg_y_straddle": ("n", "s"),
    "origin": ("s", "s"),
}

REGIME_STRATA = ("saturated", "wide_saturated", "tight", "wide", "near_degenerate")

ALL_STRATA = tuple(SIGN_STRATA) + REGIME_STRATA


def _interval(rng: np.random.Generator, n: int, kind: str):
    """One interval batch of the given sign kind."""
    if kind == "p":
        lo = rng.uniform(0.0, 4.0, n)
        w = rng.uniform(1e-3, 4.0, n)
        return lo, lo + w
    if kind == "n":
        hi = -rng.uniform(0.0, 4.0, n)
        w = rng.uniform(1e-3, 4.0, n)
        return hi - w, hi
    if kind == "s":
        return -rng.uniform(1e-3, 4.0, n), rng.uniform(1e-3, 4.0, n)
    raise ValueError(kind)


def _regime_interval(rng: np.random.Generator, n: int, regime: str):
    if regime == "saturated":
        c = rng.choice([-1.0, 1.0], n) * rng.uniform(10.0, 30.0, n)
        w = rng.uniform(1e-3, 3.0, n)
    elif regime == "wide_saturated":
        # x-range starting deep in saturation AND wide: where the
        # interior-critical quartic has a near-double root at p->1 and
        # the dropped-candidate magnitude is largest (finding F-1, ~1e-5)
        lo = rng.choice([-1.0, 1.0], n) * rng.uniform(8.0, 20.0, n)
        w = rng.uniform(2.0, 30.0, n)
        return np.minimum(lo, lo + np.sign(lo) * w), np.maximum(lo, lo + np.sign(lo) * w)
    elif regime == "tight":
        c = rng.uniform(-3.0, 3.0, n)
        w = rng.uniform(1e-6, 1e-3, n)
    elif regime == "wide":
        c = rng.uniform(-5.0, 5.0, n)
        w = rng.uniform(40.0, 100.0, n)
    elif regime == "near_degenerate":
        # widths log-spaced straddling the 1e-12 point gate
        c = rng.uniform(-3.0, 3.0, n)
        w = 10.0 ** rng.uniform(-13.5, -9.0, n)
    else:
        raise ValueError(regime)
    return c - 0.5 * w, c + 0.5 * w


def sigtanh_tilt(lx, ux, ly, uy):
    """The shipped corner-fit tilt for sigma(x)tanh(y) (transformers.py)."""
    sl = 1 / (1 + np.exp(-lx)); su = 1 / (1 + np.exp(-ux))
    tly = np.tanh(ly); tuy = np.tanh(uy)
    A = (su - sl) * (tly + tuy) / (2 * (ux - lx))
    B = (sl + su) * (tuy - tly) / (2 * (uy - ly))
    return A, B


def quartic_conditioning(A, B):
    """For the interior-critical quartic p^4-(2+B)p^3+(1+2B)p^2-Bp-A^2 of
    the sigma*tanh residual: return (min pairwise root separation among
    real roots in (0,1), max conditioning 1/(p(1-p)) among them). Boxes
    with tiny separation are where the |imag|<1e-10 realness gate and
    root-recovery error bite (finding F-1). NaN if no real root in (0,1)."""
    A = np.atleast_1d(A); B = np.atleast_1d(B)
    sep = np.full(A.shape, np.nan)
    cond = np.full(A.shape, np.nan)
    for i in range(A.shape[0]):
        r = np.roots([1.0, -(2.0 + B[i]), 1.0 + 2.0 * B[i], -B[i], -A[i] ** 2])
        pr = np.sort(r.real[(np.abs(r.imag) < 1e-6)])
        pr = pr[(pr > 0) & (pr < 1)]
        if pr.size:
            cond[i] = float(np.max(1.0 / (pr * (1 - pr))))
            allr = np.sort(r.real)
            sep[i] = float(np.min(np.diff(allr))) if allr.size > 1 else np.inf
    return sep, cond


def near_degenerate_quartic_boxes(rng: np.random.Generator, n: int,
                                  n_candidates: int = 40000) -> tuple:
    """Adversarial stratum: sigma*tanh boxes whose interior-critical
    quartic has (nearly) coincident roots or a root near p=0/1, i.e.
    poorly conditioned root recovery. Draws many candidate boxes across
    the saturation/wide regimes, scores by min root separation and by
    conditioning 1/(p(1-p)), keeps the n worst by each criterion.
    Returns (lx, ux, ly, uy)."""
    lx = np.empty(0); ux = np.empty(0); ly = np.empty(0); uy = np.empty(0)
    for regime in ("saturated", "wide", "tight"):
        for other in ("s", "p", "n"):
            m = n_candidates // 9
            rlx, rux = _regime_interval(rng, m, regime)
            gly, guy = _interval(rng, m, other)
            lx = np.r_[lx, rlx]; ux = np.r_[ux, rux]
            ly = np.r_[ly, gly]; uy = np.r_[uy, guy]
    A, B = sigtanh_tilt(lx, ux, ly, uy)
    sep, cond = quartic_conditioning(A, B)
    ok = np.isfinite(sep) & np.isfinite(cond)
    idx_sep = np.argsort(np.where(ok, sep, np.inf))[: n // 2]
    idx_cond = np.argsort(np.where(ok, -cond, np.inf))[: n - n // 2]
    idx = np.unique(np.r_[idx_sep, idx_cond])
    return lx[idx], ux[idx], ly[idx], uy[idx]


def stratified_boxes(rng: np.random.Generator, n_per: int) -> dict:
    """Dict stratum -> (lx, ux, ly, uy), n_per boxes each.

    Sign strata place x and y per SIGN_STRATA. Regime strata apply the
    regime to one or both axes (chosen per box) with the other axis drawn
    as a generic straddling interval, so regime boxes still spread over
    sign configurations.
    """
    out = {}
    for name, (kx, ky) in SIGN_STRATA.items():
        lx, ux = _interval(rng, n_per, kx)
        ly, uy = _interval(rng, n_per, ky)
        out[name] = (lx, ux, ly, uy)
    for regime in REGIME_STRATA:
        rlx, rux = _regime_interval(rng, n_per, regime)
        rly, ruy = _regime_interval(rng, n_per, regime)
        glx, gux = _interval(rng, n_per, "s")
        gly, guy = _interval(rng, n_per, "s")
        if regime == "wide_saturated":
            # y moderate (|y| <= 5) so tanh is not saturated: the
            # dangerous configuration is saturated x with live y
            gly = rng.uniform(-5.0, 4.0, n_per); guy = gly + rng.uniform(0.3, 4.0, n_per)
            which = np.zeros(n_per, dtype=int)   # x only
        else:
            which = rng.integers(0, 3, n_per)  # 0: x only, 1: y only, 2: both
        lx = np.where(which == 1, glx, rlx)
        ux = np.where(which == 1, gux, rux)
        ly = np.where(which == 0, gly, rly)
        uy = np.where(which == 0, guy, ruy)
        out[regime] = (lx, ux, ly, uy)
    return out
