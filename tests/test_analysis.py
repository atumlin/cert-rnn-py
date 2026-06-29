"""Tests for cert_rnn.analysis (diagnostics on the LSTM-AE).

Self-contained: builds a tiny LSTM-AE inline via from_torch (no external
example data), so it runs in CI. Anchors the concrete numpy forward to the
abstract engine (the verifier's eps=0 center IS the concrete reconstruction),
checks tightness soundness (UB >= empirical max), and reach_stats shapes.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from cert_rnn import LSTMAutoencoder, analysis
from cert_rnn.verify import lstm_ae_reach

T, D, H = 8, 6, 4


def _tiny_ae(seed: int = 0) -> LSTMAutoencoder:
    """A 2-layer encoder / 2-layer decoder uniform-H LSTM-AE with the head."""
    torch.manual_seed(seed)
    enc = [nn.LSTMCell(D, H).double(), nn.LSTMCell(H, H).double()]
    dec = [nn.LSTMCell(H, H).double(), nn.LSTMCell(H, H).double()]
    head = nn.Linear(H, D).double()
    return LSTMAutoencoder.from_torch(enc, dec, head)


@pytest.fixture(scope="module")
def ae():
    return _tiny_ae()


@pytest.fixture(scope="module")
def anchor():
    return np.random.default_rng(0).standard_normal((T, D))


def test_concrete_forward_matches_reach_center(ae, anchor):
    """At eps=0 the verifier's reconstruction zonotope is a point; its center
    must equal the concrete numpy forward to machine precision."""
    x_hat = analysis.concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, anchor)
    z_xh, _ = lstm_ae_reach(ae.encoder, ae.decoder, ae.head, anchor, 0.0, "multi_frame", None)
    center = np.array([z.c for z in z_xh])          # (T, D)
    assert x_hat.shape == anchor.shape
    assert np.allclose(x_hat, center, atol=1e-9, rtol=0)


def test_score_equals_reach_ub_at_zero(ae, anchor):
    """Concrete score equals the verifier's score_ub at eps=0 (point set)."""
    s = analysis.reconstruction_score(ae.encoder, ae.decoder, ae.head, anchor)
    stats0 = analysis.reach_stats(ae.encoder, ae.decoder, ae.head, anchor, 0.0)
    assert abs(stats0["score_ub"] - s) <= 1e-9


def test_concrete_forward_batches(ae, anchor):
    """Batched forward equals per-sample forward."""
    X = np.stack([anchor, anchor * 0.9, anchor + 0.01])
    xb = analysis.concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, X)
    assert xb.shape == X.shape
    for i in range(len(X)):
        xi = analysis.concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, X[i])
        assert np.allclose(xb[i], xi, atol=1e-12)


def test_reach_stats_shapes(ae, anchor):
    st = analysis.reach_stats(ae.encoder, ae.decoder, ae.head, anchor, 0.02)
    assert st["recon_width"].shape == (T, D)
    assert st["err_width"].shape == (T, D)
    assert st["n_pred"] > 0
    assert st["score_ub"] >= 0.0
    # widths are zero at eps=0 (point set)
    st0 = analysis.reach_stats(ae.encoder, ae.decoder, ae.head, anchor, 0.0)
    assert np.allclose(st0["err_width"], 0.0, atol=1e-9)


@pytest.mark.parametrize("eps", [0.01, 0.02, 0.05])
def test_tightness_is_sound(ae, anchor, eps):
    """Certified upper bound must dominate the empirical worst case."""
    r = analysis.tightness(ae.encoder, ae.decoder, ae.head, anchor, eps, n_samples=1500, seed=0)
    assert r["sound"]
    assert r["score_ub"] >= r["empirical_max"] - 1e-9
    assert r["ratio"] >= 1.0 - 1e-9


def test_score_vs_eps_monotone(ae, anchor):
    rows = analysis.score_vs_eps(ae.encoder, ae.decoder, ae.head, anchor, [0.0, 0.01, 0.02, 0.05])
    vals = [s for _, s in rows]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))  # sound bound grows with eps


def test_time_certify_reports_factors(ae, anchor):
    """time_certify returns the cost factors alongside the wall-clock."""
    tc = analysis.time_certify(ae.encoder, ae.decoder, ae.head, anchor, tau=1e9,
                               threat_model="multi_frame", n_iters=4)
    assert tc["T"] == T and tc["H"] == H
    assert tc["n_enc_layers"] == 2 and tc["n_dec_layers"] == 2
    assert tc["n_reach_calls"] == 4               # n_iters x 1 for multi_frame
    assert tc["seconds"] > 0.0 and tc["sec_per_reach"] > 0.0


def test_wrapper_methods(ae, anchor):
    """The LSTMAutoencoder convenience methods delegate correctly."""
    assert np.allclose(ae.reconstruct(anchor),
                       analysis.concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, anchor))
    st = ae.reach_stats(anchor, 0.02)
    assert "n_pred" in st and "err_width" in st
    t = ae.tightness(anchor, 0.02, n_samples=500)
    assert t["sound"]
    assert ae.time_reach(anchor, 0.02, repeat=1) > 0.0
