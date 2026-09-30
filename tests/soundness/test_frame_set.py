"""threat_model="frame_set": a chosen set of frames perturbed jointly, the
others pinned (the concentrated attack). Input construction only -- the
same transformers run -- so the checks are: exact agreement with the two
existing threat models at the extremes (one frame == single_frame, all
frames == multi_frame), eps-space enclosure + score soundness on sampled
inputs (interior, sign corners, edge points, coordinate-ascent
refinement), and bit-identical parallel searches."""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.analysis import concrete_lstm_ae_forward, reconstruction_score
from cert_rnn.transformers import bilinear_mode
from cert_rnn.verify import (certify_radius_spec_c, lstm_ae_reach,
                             spec_c_score_ub, spec_c_score_ub_joint)
from cert_rnn.zono import reset_pred_allocator
from tests.soundness.sampling import coordinate_ascent, sample_alphas
from tests.soundness.test_sequence_decoder import _seq_ae

SEED = 20260930
TOL = 1e-12


def _latent_ae(seed):
    _, ae = _seq_ae(seed)
    dec = dict(ae.decoder)
    dec.pop("input")
    return ae.encoder, dec, ae.head


def _models(seed):
    _, ae = _seq_ae(seed)
    return {"sequence": (ae.encoder, ae.decoder, ae.head), "latent": _latent_ae(seed)}


def _same(a, b):
    for za, zb in zip(a[0] + a[1], b[0] + b[1]):
        assert np.array_equal(za.c, zb.c) and np.array_equal(za.V, zb.V)


@pytest.mark.parametrize("layout", ["sequence", "latent"])
@pytest.mark.parametrize("mode", ["box", "zono"])
def test_extremes_match_existing_threat_models(layout, mode):
    enc, dec, head = _models(1)[layout]
    x = np.random.default_rng(SEED).uniform(0.2, 0.8, (5, 3))
    with bilinear_mode(mode):
        for t in (0, 2, 4):
            reset_pred_allocator(0)
            a = lstm_ae_reach(enc, dec, head, x, 0.05, "single_frame", t)
            reset_pred_allocator(0)
            b = lstm_ae_reach(enc, dec, head, x, 0.05, "frame_set", (t,))
            _same(a, b)
        reset_pred_allocator(0)
        a = lstm_ae_reach(enc, dec, head, x, 0.05, "multi_frame", None)
        reset_pred_allocator(0)
        b = lstm_ae_reach(enc, dec, head, x, 0.05, "frame_set", tuple(range(5)))
        _same(a, b)


def test_frame_set_rejects_bad_frames():
    enc, dec, head = _models(2)["sequence"]
    x = np.zeros((5, 3))
    for bad in ((), (5,), (-1, 2)):
        with pytest.raises(ValueError):
            lstm_ae_reach(enc, dec, head, x, 0.05, "frame_set", bad)
    with pytest.raises(ValueError):
        certify_radius_spec_c(enc, dec, head, x, 1.0, threat_model="frame_set")


@pytest.mark.soundness
@pytest.mark.parametrize("layout", ["sequence", "latent"])
@pytest.mark.parametrize("mode", ["box", "zono"])
@pytest.mark.parametrize("frames", [(1, 3), (0, 1, 2), (2, 4)])
def test_enclosure_and_score(layout, mode, frames):
    """Every eps-space sample over the chosen frames: reconstruction inside
    the reach's affine form + fresh width, concrete score <= both bounds;
    unperturbed frames stay pinned."""
    rng = np.random.default_rng(SEED)
    T, D = 5, 3
    worst_enc = worst_score = 0.0
    for trial in range(4):
        enc, dec, head = _models(10 + trial)[layout]
        x0 = rng.uniform(0.2, 0.8, (T, D))
        eps = float(rng.choice([0.01, 0.05, 0.15]))
        reset_pred_allocator(0)
        with bilinear_mode(mode):
            zxh, zx = lstm_ae_reach(enc, dec, head, x0, eps, "frame_set", frames)
        for t in range(T):
            assert (zx[t].V.shape[1] == 0) == (t not in frames)
        col, k = {}, 0
        for t in frames:
            for pid in zx[t].pred_ids:
                col[pid] = k
                k += 1
        ubs = (spec_c_score_ub(zxh, zx), spec_c_score_ub_joint(zxh, zx))

        def to_x(a):
            x = x0.copy()
            x[list(frames)] += eps * a.reshape(len(frames), D)
            return x

        def enc_violation(a):
            xh = concrete_lstm_ae_forward(enc, dec, head, to_x(a))
            v = -np.inf
            for t, z in enumerate(zxh):
                idx = [j for j, pid in enumerate(z.pred_ids) if pid in col]
                aff = z.c + (z.V[:, idx] @ a[[col[z.pred_ids[j]] for j in idx]]
                             if idx else 0.0)
                fresh = np.sum(np.abs(np.delete(z.V, idx, axis=1)), axis=1)
                v = max(v, float(np.max((np.abs(xh[t] - aff) - fresh)
                                        / np.maximum(1.0, np.abs(xh[t])))))
            return v

        def score_violation(a):
            s = reconstruction_score(enc, dec, head, to_x(a))
            return (s - min(ubs)) / max(1.0, s)

        al = sample_alphas(rng, k, 150)
        ev = np.array([enc_violation(a) for a in al])
        sv = np.array([score_violation(a) for a in al])
        _, e_ref = coordinate_ascent(al[int(np.argmax(ev))], enc_violation, n_rounds=3)
        _, s_ref = coordinate_ascent(al[int(np.argmax(sv))], score_violation, n_rounds=3)
        worst_enc = max(worst_enc, float(ev.max()), e_ref)
        worst_score = max(worst_score, float(sv.max()), s_ref)
    print(f"\n[seed={SEED}] frame_set {layout} {mode} {frames}: worst enclosure "
          f"{worst_enc:.3e}, worst score {worst_score:.3e}")
    assert worst_enc <= TOL, f"enclosure violation {worst_enc:.3e}"
    assert worst_score <= TOL, f"score bound violation {worst_score:.3e}"


@pytest.mark.soundness
@pytest.mark.parametrize("mode", ["box", "zono"])
@pytest.mark.parametrize("score_bound", ["componentwise", "joint"])
def test_parallel_identity_frame_set(mode, score_bound):
    enc, dec, head = _models(20)["sequence"]
    x = np.random.default_rng(SEED).uniform(0, 1, (6, 3))
    tau = 3.0 * reconstruction_score(enc, dec, head, x)
    kw = dict(threat_model="frame_set", frames=(1, 2, 4), score_bound=score_bound)
    with bilinear_mode(mode):
        reset_pred_allocator(0)
        r0, pf0 = certify_radius_spec_c(enc, dec, head, x, tau, 0.5, 12, **kw)
        assert pf0 is None
        for search, n_workers in (("bisect", 4), ("kary", 1), ("kary", 4)):
            reset_pred_allocator(31)
            r, _ = certify_radius_spec_c(enc, dec, head, x, tau, 0.5, 12, search=search,
                                         probes=15, n_workers=n_workers, **kw)
            assert r == r0, (search, n_workers, r, r0)
