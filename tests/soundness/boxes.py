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

REGIME_STRATA = ("saturated", "tight", "wide", "near_degenerate")

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
        which = rng.integers(0, 3, n_per)  # 0: x only, 1: y only, 2: both
        lx = np.where(which == 1, glx, rlx)
        ux = np.where(which == 1, gux, rux)
        ly = np.where(which == 0, gly, rly)
        uy = np.where(which == 0, guy, ruy)
        out[regime] = (lx, ux, ly, uy)
    return out
