"""Tests for cert_rnn.analysis (diagnostics on the LSTM-AE).

Anchors the concrete numpy forward to the abstract engine (the verifier's
eps=0 center IS the concrete reconstruction), checks tightness soundness
(UB >= empirical max), and the shapes/keys of reach_stats.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples" / "lstm_ae_ieee9"
sys.path.insert(0, str(EXAMPLES))

# noqa: E402
from ae_loader import load_lstm_ae  # type: ignore[import]

from cert_rnn import analysis
from cert_rnn.verify import lstm_ae_reach


@pytest.fixture(scope="module")
def model():
    return load_lstm_ae("S")


def test_concrete_forward_matches_reach_center(model):
    """At eps=0 the verifier's reconstruction zonotope is a point; its center
    must equal the concrete numpy forward to machine precision."""
    enc, dec, head, anchor = model["encoder"], model["decoder"], model["head"], model["anchor"]
    x_hat = analysis.concrete_lstm_ae_forward(enc, dec, head, anchor)
    z_xh, _ = lstm_ae_reach(enc, dec, head, anchor, 0.0, "multi_frame", None)
    center = np.array([z.c for z in z_xh])          # (T, D)
    assert x_hat.shape == anchor.shape
    assert np.allclose(x_hat, center, atol=1e-9, rtol=0)


def test_score_matches_anchor_score(model):
    """Concrete score equals the checkpoint's recorded anchor_score and the
    verifier's score_ub at eps=0."""
    enc, dec, head, anchor = model["encoder"], model["decoder"], model["head"], model["anchor"]
    s = analysis.reconstruction_score(enc, dec, head, anchor)
    assert abs(s - model["anchor_score"]) <= 1e-6
    stats = analysis.reach_stats(enc, dec, head, anchor, 0.0)
    assert abs(stats["score_ub"] - s) <= 1e-9


def test_concrete_forward_batches(model):
    """Batched forward equals per-sample forward."""
    enc, dec, head, anchor = model["encoder"], model["decoder"], model["head"], model["anchor"]
    X = np.stack([anchor, anchor * 0.9, anchor + 0.01])
    xb = analysis.concrete_lstm_ae_forward(enc, dec, head, X)
    assert xb.shape == X.shape
    for i in range(len(X)):
        xi = analysis.concrete_lstm_ae_forward(enc, dec, head, X[i])
        assert np.allclose(xb[i], xi, atol=1e-12)


def test_reach_stats_shapes(model):
    enc, dec, head, anchor = model["encoder"], model["decoder"], model["head"], model["anchor"]
    T, D = anchor.shape
    st = analysis.reach_stats(enc, dec, head, anchor, 0.02)
    assert st["recon_width"].shape == (T, D)
    assert st["err_width"].shape == (T, D)
    assert st["n_pred"] > 0
    assert st["score_ub"] >= 0.0
    # widths are non-negative and zero at eps=0
    st0 = analysis.reach_stats(enc, dec, head, anchor, 0.0)
    assert np.allclose(st0["err_width"], 0.0, atol=1e-9)


@pytest.mark.parametrize("eps", [0.01, 0.02, 0.05])
def test_tightness_is_sound(model, eps):
    """Certified upper bound must dominate the empirical worst case."""
    enc, dec, head, anchor = model["encoder"], model["decoder"], model["head"], model["anchor"]
    r = analysis.tightness(enc, dec, head, anchor, eps, n_samples=1500, seed=0)
    assert r["sound"]
    assert r["score_ub"] >= r["empirical_max"] - 1e-9
    assert r["ratio"] >= 1.0 - 1e-9


def test_score_vs_eps_monotone(model):
    enc, dec, head, anchor = model["encoder"], model["decoder"], model["head"], model["anchor"]
    rows = analysis.score_vs_eps(enc, dec, head, anchor, [0.0, 0.01, 0.02, 0.05])
    vals = [s for _, s in rows]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))  # sound bound grows with eps


def test_wrapper_methods(model):
    """The LSTMAutoencoder convenience methods delegate correctly."""
    ae = model["model"]
    anchor = model["anchor"]
    assert np.allclose(ae.reconstruct(anchor),
                       analysis.concrete_lstm_ae_forward(ae.encoder, ae.decoder, ae.head, anchor))
    st = ae.reach_stats(anchor, 0.02)
    assert "n_pred" in st and "err_width" in st
    t = ae.tightness(anchor, 0.02, n_samples=500)
    assert t["sound"]
    assert ae.time_reach(anchor, 0.02, repeat=1) > 0.0
