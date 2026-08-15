"""ZRLT Tier 1 soundness (soundness.md T-2.1 / T-2.2 + cover property).

The Tier 1 construction (cert_rnn.tier1) is sound iff (a) the admitted
grid cells cover the joint zonotope Z and (b) each cell's (C1, C2) is a
correct enclosure of the residual over that cell (the R-tests' subject).
This file tests (a) directly with eps-space samples including sign corners
and edge points, and the resulting ordering and enclosure at the
transformer level. Baseline bit-identity under mode "box" is asserted so
the switch cannot silently perturb the baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.tier1 import admitted_cells, c1c2_over_zono
from cert_rnn.transformers import (
    _sigid_plane_batch,
    _sigmoid,
    _sigtanh_plane_batch,
    bilinear_mode,
    bilinear_sigmoid_identity,
    bilinear_sigmoid_tanh,
    get_bilinear_mode,
)
from cert_rnn.zono import PredAllocator, Zono, align_pred_space

from tests.soundness.sampling import coordinate_ascent, sample_alphas

SEED = 20260814


def _pair(rng, K, p, corr, scale_x=1.0, scale_y=1.0, ids=None):
    """Operand pair sharing p preds; corr in [0,1] blends y's generators
    toward x's (1 = perfectly correlated sliver)."""
    ids = ids or tuple(range(100, 100 + p))
    Vx = rng.normal(0, 0.3, (K, p)) * scale_x
    Vy_ind = rng.normal(0, 0.3, (K, p)) * scale_y
    Vy = corr * Vx * (scale_y / scale_x) + (1 - corr) * Vy_ind
    zx = Zono(rng.normal(0, 1.0, K) * scale_x, Vx, ids)
    zy = Zono(rng.normal(0, 1.0, K) * scale_y, Vy, ids)
    return zx, zy


@pytest.mark.soundness
@pytest.mark.parametrize("corr", [0.0, 0.7, 0.95, 1.0])
@pytest.mark.parametrize("n", [4, 12, 24])
def test_admitted_cells_cover_zonotope(corr, n):
    """Every reachable (x, y) — sampled in eps-space with all sign
    corners, edge points and interior draws — lies in some admitted
    (closed) cell. This is the covering property that makes Tier 1 sound."""
    rng = np.random.default_rng(SEED)
    print(f"\n[seed={SEED}] cover test corr={corr} n={n}")
    for p in (1, 2, 5, 9):
        for _ in range(15):
            zx, zy = _pair(rng, 3, p, corr, scale_x=rng.uniform(0.2, 6),
                           scale_y=rng.uniform(0.2, 6))
            _, (Vx, Vy) = align_pred_space(zx, zy)
            lbx, ubx = zx.get_ranges(); lby, uby = zy.get_ranges()
            al = sample_alphas(rng, p, 300)
            for k in range(3):
                cells = admitted_cells(zx.c[k], zy.c[k], Vx[k], Vy[k],
                                       lbx[k], ubx[k], lby[k], uby[k], n)
                if cells is None:
                    continue
                clx, cux, cly, cuy = cells
                x = zx.c[k] + al @ Vx[k]
                y = zy.c[k] + al @ Vy[k]
                tol = 1e-12 * max(1.0, ubx[k] - lbx[k], uby[k] - lby[k],
                                  abs(zx.c[k]), abs(zy.c[k]))
                inside = ((x[:, None] >= clx[None, :] - tol) &
                          (x[:, None] <= cux[None, :] + tol) &
                          (y[:, None] >= cly[None, :] - tol) &
                          (y[:, None] <= cuy[None, :] + tol)).any(axis=1)
                assert inside.all(), (
                    f"{(~inside).sum()} reachable points outside every "
                    f"admitted cell (p={p}, corr={corr}, n={n}, coord {k})")


@pytest.mark.soundness
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
def test_t22_tightness_ordering(kind):
    """T-2.2: Tier 1 (C1, C2) is never looser than the box (C1, C2), for
    the SAME (A, B). Raw values (no clamping in tier1.py), so a violation
    beyond fp is a finding."""
    rng = np.random.default_rng(SEED)
    plane = _sigtanh_plane_batch if kind == "sigtanh" else _sigid_plane_batch
    worst = 0.0
    n_touched = 0
    for corr in (0.0, 0.5, 0.9, 1.0):
        for scale in (0.5, 2.0, 6.0):
            for _ in range(10):
                zx, zy = _pair(rng, 12, 6, corr, scale_x=scale, scale_y=1.5)
                _, (Vx, Vy) = align_pred_space(zx, zy)
                lbx, ubx = zx.get_ranges(); lby, uby = zy.get_ranges()
                A, B, C1b, C2b = plane(lbx, ubx, lby, uby)
                C1z, C2z = c1c2_over_zono(kind, A, B, zx.c, zy.c, Vx, Vy,
                                          lbx, ubx, lby, uby, C1b, C2b)
                sc = np.maximum(1.0, np.maximum(np.abs(C1b), np.abs(C2b)))
                worst = max(worst, float(np.max((C1b - C1z) / sc)),
                            float(np.max((C2z - C2b) / sc)))
                n_touched += int(np.sum((C1z != C1b) | (C2z != C2b)))
    print(f"\n[seed={SEED}] T-2.2 {kind}: worst ordering violation "
          f"{worst:.3e}; {n_touched} coordinates tightened")
    assert n_touched > 0, "Tier 1 never engaged — test is vacuous"
    assert worst <= 1e-12, f"Tier 1 looser than box by {worst:.3e} rel"


@pytest.mark.soundness
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
def test_t21_enclosure_zono_mode(kind):
    """T-2.1: under mode 'zono', every eps-space sample (interior, sign
    corners, edge points, plus coordinate-ascent refinement) satisfies
    |f - affine| <= fresh width. Both K<8 and K>=8 dispatch."""
    rng = np.random.default_rng(SEED)
    fn = bilinear_sigmoid_tanh if kind == "sigtanh" else bilinear_sigmoid_identity
    worst = 0.0
    with bilinear_mode("zono"):
        for K in (3, 12):
            for corr in (0.0, 0.8, 1.0):
                for scale in (0.5, 2.0):
                    for _ in range(8):
                        p = 6
                        zx, zy = _pair(rng, K, p, corr, scale_x=scale, scale_y=1.5)
                        zo = fn(zx, zy, PredAllocator(1000))
                        al = sample_alphas(rng, p, 300)
                        x = zx.c[None, :] + al @ zx.V.T
                        y = zy.c[None, :] + al @ zy.V.T
                        f = _sigmoid(x) * np.tanh(y) if kind == "sigtanh" else x * _sigmoid(y)
                        aff = zo.c[None, :] + al @ zo.V[:, :p].T
                        width = np.sum(np.abs(zo.V[:, p:]), axis=1)
                        v = (np.abs(f - aff) - width[None, :]) / np.maximum(1.0, np.abs(f))
                        worst = max(worst, float(v.max()))
                        i, k = np.unravel_index(np.argmax(v), v.shape)

                        def viol(a, k=k):
                            xx = zx.c[k] + zx.V[k] @ a; yy = zy.c[k] + zy.V[k] @ a
                            ff = _sigmoid(xx) * np.tanh(yy) if kind == "sigtanh" else xx * _sigmoid(yy)
                            return (abs(ff - (zo.c[k] + zo.V[k, :p] @ a)) - width[k]) / max(1.0, abs(ff))

                        _, wa = coordinate_ascent(al[i], viol)
                        worst = max(worst, wa)
    print(f"\n[seed={SEED}] T-2.1 {kind} zono-mode: worst rel violation {worst:.3e}")
    assert worst <= 1e-12, f"Tier 1 enclosure violation {worst:.3e}"


@pytest.mark.soundness
def test_box_mode_is_bit_identical_baseline():
    """Default mode is 'box' and produces the identical zonotope as before
    the switch existed (guards the baseline against accidental drift)."""
    assert get_bilinear_mode() == "box"
    rng = np.random.default_rng(SEED)
    for K in (3, 12):
        zx, zy = _pair(rng, K, 5, 0.6)
        a1 = bilinear_sigmoid_tanh(zx, zy, PredAllocator(1000))
        with bilinear_mode("zono"):
            a2 = bilinear_sigmoid_tanh(zx, zy, PredAllocator(1000))
        a3 = bilinear_sigmoid_tanh(zx, zy, PredAllocator(1000))
        assert np.array_equal(a1.c, a3.c) and np.array_equal(a1.V, a3.V)
        assert a1.pred_ids == a3.pred_ids
        # and zono mode is genuinely different (tighter) on correlated operands
        assert not np.array_equal(a1.V, a2.V)
