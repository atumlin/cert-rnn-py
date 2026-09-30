"""Per-step-code LSTM-AE (decoder["input"] == "sequence"): decoder step t
reads the encoder's top-layer h at step t instead of the final h.

No new over-approximation: the reach composes the same LSTM-step and
affine transformers in a different order. These tests check that the
wiring is right (torch parity, eps=0 exactness) and that the composition
stays sound: every eps-space sample (interior, sign corners, edge points,
plus coordinate-ascent refinement) of the input set maps to a
reconstruction inside the reach's affine form + fresh width, and its
concrete score never exceeds either score upper bound.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from cert_rnn import LSTMAutoencoder
from cert_rnn.analysis import concrete_lstm_ae_forward, reconstruction_score
from cert_rnn.transformers import bilinear_mode
from cert_rnn.verify import (certify_radius_spec_c, lstm_ae_reach,
                             spec_c_score_ub, spec_c_score_ub_joint)
from cert_rnn.zono import reset_pred_allocator
from tests.soundness.sampling import coordinate_ascent, sample_alphas

SEED = 20260929
TOL = 1e-12   # relative to max(1, |value|), as in the Level-1 suite


class _SeqAE(nn.Module):
    """Reference torch model: encoder LSTM, decoder LSTM reading the
    encoder's hidden state at every step, per-step linear head."""

    def __init__(self, D, H):
        super().__init__()
        self.enc = nn.LSTM(D, H, batch_first=True)
        self.dec = nn.LSTM(H, H, batch_first=True)
        self.head = nn.Linear(H, D)

    def forward(self, x):
        codes, _ = self.enc(x)
        y, _ = self.dec(codes)
        return self.head(y)


def _seq_ae(seed, D=3, H=4):
    torch.manual_seed(seed)
    m = _SeqAE(D, H).double()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(2.0)          # stronger nonlinearity than the default init
    ae = LSTMAutoencoder.from_torch(m.enc, m.dec, m.head, decoder_input="sequence")
    return m, ae


def test_from_torch_records_decoder_input():
    _, ae = _seq_ae(0)
    assert ae.decoder["input"] == "sequence"
    m = _SeqAE(3, 4).double()
    assert "input" not in LSTMAutoencoder.from_torch(m.enc, m.dec, m.head).decoder
    with pytest.raises(ValueError):
        LSTMAutoencoder.from_torch(m.enc, m.dec, m.head, decoder_input="nope")


@pytest.mark.parametrize("T", [1, 5, 12])
def test_torch_parity(T):
    m, ae = _seq_ae(1)
    X = np.random.default_rng(SEED).uniform(0, 1, (7, T, 3))
    with torch.no_grad():
        ref = m(torch.as_tensor(X)).numpy()
    got = concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, X)
    assert np.max(np.abs(got - ref)) < 1e-12
    s = reconstruction_score(ae.encoder, ae.decoder, ae.head, X)
    assert np.allclose(s, ((ref - X) ** 2).mean(axis=(1, 2)), rtol=0, atol=1e-13)


def test_layouts_differ():
    """Guard against the flag being ignored: same weights, different wiring."""
    _, ae = _seq_ae(2)
    lat = dict(ae.decoder)
    lat.pop("input")
    x = np.random.default_rng(SEED).uniform(0, 1, (6, 3))
    a = concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, x)
    b = concrete_lstm_ae_forward(ae.encoder, lat, ae.head, x)
    assert np.max(np.abs(a - b)) > 1e-3


@pytest.mark.parametrize("threat,t_pert", [("multi_frame", None),
                                           ("single_frame", 0),
                                           ("single_frame", 4)])
def test_eps0_reach_is_concrete(threat, t_pert):
    _, ae = _seq_ae(3)
    x = np.random.default_rng(SEED).uniform(0, 1, (5, 3))
    zxh, _ = lstm_ae_reach(ae.encoder, ae.decoder, ae.head, x, 0.0, threat, t_pert)
    ref = concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, x)
    for t, z in enumerate(zxh):
        assert np.max(np.abs(z.c - ref[t])) < 1e-12
        assert np.all(np.abs(z.V) < 1e-12)


def _input_columns(z_x_seq, T, D, threat, t_pert):
    """pred_id -> alpha index for the input generators."""
    col = {}
    frames = range(T) if threat == "multi_frame" else [t_pert]
    k = 0
    for t in frames:
        for pid in z_x_seq[t].pred_ids:
            col[pid] = k
            k += 1
    return col, k


@pytest.mark.soundness
@pytest.mark.parametrize("mode", ["box", "zono"])
@pytest.mark.parametrize("threat,t_pert", [("multi_frame", None),
                                           ("single_frame", 0),
                                           ("single_frame", 3)])
def test_enclosure_and_score(mode, threat, t_pert):
    """Every eps-space sample: reconstruction inside the reach's affine form
    + fresh width, concrete score <= both score upper bounds."""
    rng = np.random.default_rng(SEED)
    T, D = 5, 3
    worst_enc = worst_score = 0.0
    for trial in range(6):
        _, ae = _seq_ae(10 + trial)
        x0 = rng.uniform(0.2, 0.8, (T, D))
        eps = float(rng.choice([0.01, 0.05, 0.15]))
        reset_pred_allocator(0)
        with bilinear_mode(mode):
            zxh, zx = lstm_ae_reach(ae.encoder, ae.decoder, ae.head, x0, eps,
                                    threat, t_pert)
        col, p = _input_columns(zx, T, D, threat, t_pert)
        ubs = (spec_c_score_ub(zxh, zx), spec_c_score_ub_joint(zxh, zx))
        frames = list(range(T)) if threat == "multi_frame" else [t_pert]

        def to_x(a):
            x = x0.copy()
            x[frames] += eps * a.reshape(len(frames), D)
            return x

        def enc_violation(a):
            xh = concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, to_x(a))
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
            s = reconstruction_score(ae.encoder, ae.decoder, ae.head, to_x(a))
            return (s - min(ubs)) / max(1.0, s)

        al = sample_alphas(rng, p, 200)
        ev = np.array([enc_violation(a) for a in al])
        sv = np.array([score_violation(a) for a in al])
        _, e_ref = coordinate_ascent(al[int(np.argmax(ev))], enc_violation, n_rounds=3)
        _, s_ref = coordinate_ascent(al[int(np.argmax(sv))], score_violation, n_rounds=3)
        worst_enc = max(worst_enc, float(ev.max()), e_ref)
        worst_score = max(worst_score, float(sv.max()), s_ref)
    print(f"\n[seed={SEED}] sequence-decoder {mode} {threat} t={t_pert}: worst "
          f"enclosure {worst_enc:.3e}, worst score {worst_score:.3e}")
    assert worst_enc <= TOL, f"enclosure violation {worst_enc:.3e}"
    assert worst_score <= TOL, f"score bound violation {worst_score:.3e}"


@pytest.mark.soundness
@pytest.mark.parametrize("threat", ["single_frame", "multi_frame"])
@pytest.mark.parametrize("mode", ["box", "zono"])
def test_parallel_identity_sequence(threat, mode):
    """Parallel/k-ary searches stay bit-identical for the per-step layout."""
    _, ae = _seq_ae(20)
    x = np.random.default_rng(SEED).uniform(0, 1, (6, 3))
    tau = 3.0 * reconstruction_score(ae.encoder, ae.decoder, ae.head, x)
    args = (ae.encoder, ae.decoder, ae.head, x, tau, 0.5, 12, threat)
    with bilinear_mode(mode):
        reset_pred_allocator(0)
        r0, pf0 = certify_radius_spec_c(*args)
        for search, n_workers in (("bisect", 4), ("kary", 1), ("kary", 4)):
            reset_pred_allocator(999)
            r, pf = certify_radius_spec_c(*args, search=search, probes=15,
                                          n_workers=n_workers)
            assert r == r0, (search, n_workers, r, r0)
            if pf0 is not None:
                assert np.array_equal(pf, pf0)
