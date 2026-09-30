"""Level 1 sampling methodology (soundness.md T-1.1 .. T-1.3).

Sample in eps-space only. Every sample set includes interior draws, the
eps sign corners {-1,+1}^p (subsampled when 2^p is large), per-generator
edge points (all coordinates at +-1 except one swept), and — when a 2-D
projection is supplied — the enumerated polygon vertices and edge points.
Adversarial refinement is coordinate ascent on the violation.
"""

from __future__ import annotations

import numpy as np


def sample_alphas(rng: np.random.Generator, p: int, n_interior: int,
                  n_corners: int = 256, n_edge: int = 8) -> np.ndarray:
    """(n, p) alpha samples: interior + sign corners + edge points."""
    parts = []
    if p == 0:
        return np.zeros((1, 0))
    parts.append(rng.uniform(-1.0, 1.0, (n_interior, p)))
    if p <= 12:
        # all 2^p corners
        idx = np.arange(2 ** p)
        corners = ((idx[:, None] >> np.arange(p)[None, :]) & 1) * 2.0 - 1.0
    else:
        corners = rng.choice([-1.0, 1.0], size=(n_corners, p))
    parts.append(corners)
    # edge points: one coordinate free, others at random signs
    base = rng.choice([-1.0, 1.0], size=(n_edge * p, p))
    which = np.repeat(np.arange(p), n_edge)
    base[np.arange(n_edge * p), which] = rng.uniform(-1.0, 1.0, n_edge * p)
    parts.append(base)
    return np.vstack(parts)


def coordinate_ascent(alpha0: np.ndarray, violation_fn, n_rounds: int = 6,
                      steps=(0.5, 0.25, 0.1, 0.03, 0.01)) -> tuple:
    """Maximize violation_fn(alpha) over the cube by coordinate ascent
    from alpha0. Returns (best_alpha, best_value)."""
    a = alpha0.copy()
    best = violation_fn(a)
    p = a.shape[0]
    for _ in range(n_rounds):
        improved = False
        for i in range(p):
            for st in steps:
                for sgn in (+1.0, -1.0):
                    trial = a.copy()
                    trial[i] = np.clip(trial[i] + sgn * st, -1.0, 1.0)
                    v = violation_fn(trial)
                    if v > best:
                        best, a, improved = v, trial, True
        if not improved:
            break
    return a, best
