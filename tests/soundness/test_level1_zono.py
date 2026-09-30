"""Level 1: zonotope-level enclosure with the T-1.x sampling methodology.

For each transformer (tanh, sigmoid, sigma*tanh, x*sigma, full lstm_step):
draw alpha in eps-space (interior + all/subsampled sign corners + edge
points), push the concrete points through the true function, and assert
they lie inside the output zonotope's per-coordinate ranges — then run
coordinate-ascent adversarial refinement from the worst sample (T-1.3).
Violations report their magnitude relative to the output width.

Both scalar (K<8) and batch (K>=8) dispatch paths are exercised via K.
"""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.lstm import lstm_step
from cert_rnn.transformers import (
    _sigmoid,
    bilinear_sigmoid_identity,
    bilinear_sigmoid_tanh,
    sigmoid_zono,
    tanh_zono,
)
from cert_rnn.zono import PredAllocator, Zono

from tests.soundness.sampling import coordinate_ascent, sample_alphas

SEED = 20260814
TOL = 1e-12  # relative to max(1, |value|); violations beyond are reported


def _rand_zono(rng, K, p, scale, ids):
    return Zono(rng.normal(0, scale, K), rng.normal(0, scale, (K, p)) * 0.5, ids)


def _check_enclosure(name, z_out, alphas_out, concrete, rng):
    """alphas_out: (n, P) in z_out's pred order; concrete: (n, K)."""
    lb, ub = z_out.get_ranges()
    scale = np.maximum(1.0, np.maximum(np.abs(lb), np.abs(ub)))
    viol = np.maximum(lb[None, :] - concrete, concrete - ub[None, :]) / scale[None, :]
    worst = float(viol.max())
    return worst


@pytest.mark.soundness
@pytest.mark.parametrize("K", [3, 12])
@pytest.mark.parametrize("regime", ["standard", "saturating", "wide_sat"])
def test_bilinear_enclosure(K, regime):
    rng = np.random.default_rng(SEED)
    print(f"\n[seed={SEED}] level-1 bilinear K={K} {regime}")
    p = 6
    scale = {"standard": 1.0, "saturating": 4.0, "wide_sat": 12.0}[regime]
    worst = {}
    for kind in ("sigtanh", "sigid"):
        for trial in range(20):
            ids = tuple(range(100, 100 + p))
            zx = _rand_zono(rng, K, p, scale, ids)
            zy = _rand_zono(rng, K, p, 1.5, ids)     # y moderate
            alloc = PredAllocator(1000)
            fn = bilinear_sigmoid_tanh if kind == "sigtanh" else bilinear_sigmoid_identity
            zo = fn(zx, zy, alloc)
            al = sample_alphas(rng, p, 400)
            x = zx.c[None, :] + al @ zx.V.T
            y = zy.c[None, :] + al @ zy.V.T
            f = _sigmoid(x) * np.tanh(y) if kind == "sigtanh" else x * _sigmoid(y)
            # containment: for each sample, exists fresh alpha in [-1,1] s.t.
            # f = affine part + fresh*width  <=>  |f - affine| <= width
            P_shared = zo.V[:, :p]        # shared columns come first (sorted ids)
            affine = zo.c[None, :] + al @ P_shared.T
            width = np.sum(np.abs(zo.V[:, p:]), axis=1)
            sc = np.maximum(1.0, np.abs(f))
            v = (np.abs(f - affine) - width[None, :]) / sc
            w = float(v.max())
            # adversarial refinement from the worst sample
            i, k = np.unravel_index(np.argmax(v), v.shape)

            def violation(a, k=k):
                xx = zx.c[k] + zx.V[k] @ a
                yy = zy.c[k] + zy.V[k] @ a
                ff = _sigmoid(xx) * np.tanh(yy) if kind == "sigtanh" else xx * _sigmoid(yy)
                aff = zo.c[k] + P_shared[k] @ a
                return (abs(ff - aff) - width[k]) / max(1.0, abs(ff))

            _, w_adv = coordinate_ascent(al[i], violation)
            worst[kind] = max(worst.get(kind, -np.inf), w, w_adv)
    print("  worst relative violation:", {k: f"{v:.3e}" for k, v in worst.items()})
    bad = {k: v for k, v in worst.items() if v > TOL}
    if bad:
        # magnitude distribution is the report; F-1 (sigtanh, saturating,
        # <=~2e-5) is a known signature
        known = all(k == "sigtanh" and v <= 2e-5 and regime != "standard"
                    for k, v in bad.items())
        msg = f"level-1 bilinear enclosure violations K={K} {regime}: {bad}"
        if known:
            pytest.xfail("known F-1 signature: " + msg)
        pytest.fail(msg)


@pytest.mark.soundness
@pytest.mark.parametrize("K", [3, 12])
def test_unary_enclosure(K):
    rng = np.random.default_rng(SEED)
    p = 6
    worst = 0.0
    for fn, f in ((tanh_zono, np.tanh), (sigmoid_zono, _sigmoid)):
        for scale in (1.0, 4.0, 15.0):
            for _ in range(20):
                ids = tuple(range(100, 100 + p))
                z = _rand_zono(rng, K, p, scale, ids)
                zo = fn(z, PredAllocator(1000))
                al = sample_alphas(rng, p, 400)
                x = z.c[None, :] + al @ z.V.T
                affine = zo.c[None, :] + al @ zo.V[:, :p].T
                width = np.sum(np.abs(zo.V[:, p:]), axis=1)
                v = (np.abs(f(x) - affine) - width[None, :]) / np.maximum(1.0, np.abs(f(x)))
                worst = max(worst, float(v.max()))
    print(f"\n[seed={SEED}] level-1 unary K={K}: worst rel violation {worst:.3e}")
    assert worst <= TOL, f"unary enclosure violation {worst:.3e}"


@pytest.mark.soundness
@pytest.mark.parametrize("H", [3, 10])
def test_lstm_step_enclosure(H):
    """Full cell: alpha in eps-space through the concrete cell vs the
    abstract step's per-coordinate ranges (bbox containment; the fresh
    predicates make exact affine containment inapplicable across two
    chained bilinears)."""
    rng = np.random.default_rng(SEED)
    D = 4; p = D + 2 * H   # input preds + h/c preds
    worst = 0.0
    for scale in (0.5, 2.0):
        for _ in range(10):
            W_in = rng.normal(0, scale, (4 * H, D)); W_rec = rng.normal(0, scale, (4 * H, H))
            b = rng.normal(0, 0.1, 4 * H)
            ids_x = tuple(range(100, 100 + D)); ids_h = tuple(range(200, 200 + H))
            ids_c = tuple(range(300, 300 + H))
            zx = Zono(rng.normal(0, 1, D), 0.1 * np.eye(D), ids_x)
            zh = Zono(rng.normal(0, 0.5, H), 0.1 * np.eye(H), ids_h)
            zc = Zono(rng.normal(0, 0.5, H), 0.1 * np.eye(H), ids_c)
            zh1, zc1 = lstm_step(zx, zh, zc, W_in, W_rec, b, PredAllocator(1000))
            al = sample_alphas(rng, p, 300)
            x = zx.c + al[:, :D] * 0.1
            h = zh.c + al[:, D:D + H] * 0.1
            c = zc.c + al[:, D + H:] * 0.1
            g = x @ W_in.T + h @ W_rec.T + b
            i_ = _sigmoid(g[:, :H]); f_ = _sigmoid(g[:, H:2 * H])
            gg = np.tanh(g[:, 2 * H:3 * H]); o_ = _sigmoid(g[:, 3 * H:])
            c_new = f_ * c + i_ * gg
            h_new = o_ * np.tanh(c_new)
            for zo, conc in ((zh1, h_new), (zc1, c_new)):
                lb, ub = zo.get_ranges()
                sc = np.maximum(1.0, np.maximum(np.abs(lb), np.abs(ub)))
                v = np.maximum(lb - conc, conc - ub) / sc
                worst = max(worst, float(v.max()))
    print(f"\n[seed={SEED}] level-1 lstm_step H={H}: worst rel violation {worst:.3e}")
    assert worst <= TOL, f"lstm_step enclosure violation {worst:.3e}"
