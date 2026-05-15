"""Verification: bisection correctness, spec_a/c semantics, and end-to-end
soundness checked by sampling concrete points from the certified ball.

Soundness contract: if certify_radius_spec_X returns eps_cert > 0, then
EVERY concrete sample inside the eps_cert-ball must satisfy the spec.
This is the user-facing guarantee.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from cert_rnn import Zono
from cert_rnn.from_torch import lstm_to_model_dict
from cert_rnn.verify import (
    bisect_epsilon,
    certify_radius_spec_a,
    certify_radius_spec_c,
    lstm_ae_reach,
    lstm_reach,
    spec_a_margin,
    spec_c_holds,
    spec_c_score_ub,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _concrete_lstm_forward(layers, x_seq):
    """Plain numpy LSTM stack forward. Returns top-layer h_T."""
    H = layers[0]["W_rec"].shape[1]
    L = len(layers)
    h = [np.zeros(H) for _ in range(L)]
    c = [np.zeros(H) for _ in range(L)]
    for t in range(x_seq.shape[0]):
        inp = x_seq[t]
        for i in range(L):
            W_in = layers[i]["W_in"]
            W_rec = layers[i]["W_rec"]
            b = layers[i]["b"]
            pre = W_in @ inp + W_rec @ h[i] + b
            ii = _sigmoid(pre[:H])
            ff = _sigmoid(pre[H : 2 * H])
            gg = np.tanh(pre[2 * H : 3 * H])
            oo = _sigmoid(pre[3 * H : 4 * H])
            c[i] = ff * c[i] + ii * gg
            h[i] = oo * np.tanh(c[i])
            inp = h[i]
    return h[-1]


def _concrete_ae_score(encoder, decoder, head, x_seq):
    """||AE(x) - x||_2^2 / N from concrete forwards through the AE."""
    T, D = x_seq.shape
    # Encoder
    H = encoder["H"]
    L_enc = encoder["L"]
    h_enc = [np.zeros(H) for _ in range(L_enc)]
    c_enc = [np.zeros(H) for _ in range(L_enc)]
    for t in range(T):
        inp = x_seq[t]
        for i in range(L_enc):
            W_in = encoder["layers"][i]["W_in"]
            W_rec = encoder["layers"][i]["W_rec"]
            b = encoder["layers"][i]["b"]
            pre = W_in @ inp + W_rec @ h_enc[i] + b
            ii = _sigmoid(pre[:H])
            ff = _sigmoid(pre[H : 2 * H])
            gg = np.tanh(pre[2 * H : 3 * H])
            oo = _sigmoid(pre[3 * H : 4 * H])
            c_enc[i] = ff * c_enc[i] + ii * gg
            h_enc[i] = oo * np.tanh(c_enc[i])
            inp = h_enc[i]
    latent = h_enc[-1]

    L_dec = decoder["L"]
    h_dec = [np.zeros(H) for _ in range(L_dec)]
    c_dec = [np.zeros(H) for _ in range(L_dec)]
    score = 0.0
    for t in range(T):
        inp = latent
        for i in range(L_dec):
            W_in = decoder["layers"][i]["W_in"]
            W_rec = decoder["layers"][i]["W_rec"]
            b = decoder["layers"][i]["b"]
            pre = W_in @ inp + W_rec @ h_dec[i] + b
            ii = _sigmoid(pre[:H])
            ff = _sigmoid(pre[H : 2 * H])
            gg = np.tanh(pre[2 * H : 3 * H])
            oo = _sigmoid(pre[3 * H : 4 * H])
            c_dec[i] = ff * c_dec[i] + ii * gg
            h_dec[i] = oo * np.tanh(c_dec[i])
            inp = h_dec[i]
        x_hat = head["W"] @ h_dec[-1] + head["b"]
        score += float(np.sum((x_hat - x_seq[t]) ** 2))
    return score / (T * D)


# ---------- bisection correctness ----------


def test_bisect_finds_threshold():
    """For certify_fn = (eps < 0.3), Algorithm 1 should converge near 0.3."""
    threshold = 0.3
    eps_cert = bisect_epsilon(lambda eps: eps < threshold, eps_init=0.5, n_iters=14)
    # Algorithm 1 converges to within ~0.5^(n_iters+1) of the threshold.
    # Cert eps must never exceed threshold (soundness) and be close.
    assert eps_cert < threshold
    assert threshold - eps_cert < 0.5 ** 12


def test_bisect_returns_zero_when_never_certified():
    eps_cert = bisect_epsilon(lambda eps: False, eps_init=0.5, n_iters=10)
    assert eps_cert == 0.0


def test_bisect_eps_init_certifies_returns_at_least_eps_init():
    eps_cert = bisect_epsilon(lambda eps: True, eps_init=0.5, n_iters=10)
    assert eps_cert >= 0.5


# ---------- lstm_reach contains nominal forward ----------


def _build_tiny_lstm(D=2, H=3, C=3, L=1, seed=0):
    torch.manual_seed(seed)
    rec = nn.LSTM(D, H, num_layers=L, batch_first=True).double()
    fc = nn.Linear(H, C).double()
    return lstm_to_model_dict(rec, fc)


def test_lstm_reach_contains_nominal():
    """Reach set at any eps must contain the nominal (eps=0) forward."""
    model = _build_tiny_lstm()
    rng = np.random.default_rng(123)
    x_seq = rng.standard_normal((4, model["D"]))
    h_nom = _concrete_lstm_forward(model["layers"], x_seq)
    for eps in [0.0, 0.01, 0.1]:
        z_h_top_seq = lstm_reach(model, x_seq, eps, "single_frame", t_pert=1)
        lb, ub = z_h_top_seq[-1].get_ranges()
        assert np.all(h_nom >= lb - 1e-12) and np.all(h_nom <= ub + 1e-12)


# ---------- spec_a end-to-end soundness ----------


def _argmax_logits(model, x_seq):
    h_T = _concrete_lstm_forward(model["layers"], x_seq)
    logits = model["head"]["W"] @ h_T + model["head"]["b"]
    return int(np.argmax(logits)), logits


def test_spec_a_small_eps_certifies_clear_margin():
    """A small eps must certify when the nominal logit margin is wide."""
    model = _build_tiny_lstm(seed=1)
    rng = np.random.default_rng(5)
    x_seq = rng.standard_normal((4, model["D"]))
    pred, _ = _argmax_logits(model, x_seq)
    assert spec_a_margin(model, x_seq, eps=0.0, true_class=pred,
                         threat_model="single_frame", t_pert=0) is True
    # Tiny eps should still certify on this clean input.
    assert spec_a_margin(model, x_seq, eps=1e-6, true_class=pred,
                         threat_model="single_frame", t_pert=0) is True


def test_spec_a_certified_radius_is_sound():
    """Whatever radius bisection certifies, no concrete sample inside it
    must change the argmax."""
    model = _build_tiny_lstm(seed=2)
    rng = np.random.default_rng(7)
    x_seq = rng.standard_normal((3, model["D"]))
    pred, _ = _argmax_logits(model, x_seq)

    eps_cert, per_frame = certify_radius_spec_a(
        model, x_seq, pred, eps_init=0.5, n_iters=10, threat_model="single_frame"
    )
    # If we got a positive radius, sample within each frame's radius and check.
    if eps_cert <= 0:
        pytest.skip("model never certified; nothing to validate")

    sample_rng = np.random.default_rng(101)
    T, D = x_seq.shape
    n_viol = 0
    n_samples = 200
    for t_pert in range(T):
        eps_t = per_frame[t_pert]
        if eps_t <= 0:
            continue
        for _ in range(n_samples):
            x = x_seq.copy()
            x[t_pert] = x_seq[t_pert] + eps_t * (2 * sample_rng.random(D) - 1)
            ap, _ = _argmax_logits(model, x)
            if ap != pred:
                n_viol += 1
    assert n_viol == 0, f"spec_a soundness violated: {n_viol} misclassifications"


def test_spec_a_multi_frame_minkowski_soundness():
    """Multi-frame perturbation must also be sound (the Minkowski regression)."""
    model = _build_tiny_lstm(D=2, H=2, C=3, L=1, seed=4)
    rng = np.random.default_rng(13)
    x_seq = rng.standard_normal((3, model["D"]))
    pred, _ = _argmax_logits(model, x_seq)
    eps_cert, _ = certify_radius_spec_a(
        model, x_seq, pred, eps_init=0.5, n_iters=10, threat_model="multi_frame"
    )
    if eps_cert <= 0:
        pytest.skip("model never certified; nothing to validate")

    sample_rng = np.random.default_rng(17)
    T, D = x_seq.shape
    n_viol = 0
    for _ in range(200):
        x = x_seq + eps_cert * (2 * sample_rng.random((T, D)) - 1)
        ap, _ = _argmax_logits(model, x)
        if ap != pred:
            n_viol += 1
    assert n_viol == 0, f"multi-frame spec_a unsound: {n_viol}/200 violations"


# ---------- spec_c end-to-end soundness ----------


def _build_tiny_ae(D=3, H=4, T=4, seed=0):
    torch.manual_seed(seed)
    enc_rec = nn.LSTM(D, H, num_layers=1, batch_first=True).double()
    dec_rec = nn.LSTM(H, H, num_layers=1, batch_first=True).double()
    head_lin = nn.Linear(H, D).double()
    encoder = lstm_to_model_dict(enc_rec)
    decoder = lstm_to_model_dict(dec_rec)
    head = {
        "W": head_lin.weight.detach().numpy().astype(np.float64),
        "b": head_lin.bias.detach().numpy().astype(np.float64),
    }
    return encoder, decoder, head


def test_spec_c_score_ub_over_bounds_concrete():
    """score_ub must >= max concrete score over samples in the eps-ball."""
    enc, dec, head = _build_tiny_ae(D=3, H=4, T=4, seed=3)
    rng = np.random.default_rng(21)
    x_anchor = rng.standard_normal((4, 3))
    eps = 0.05
    t_pert = 1
    z_xh, z_x = lstm_ae_reach(enc, dec, head, x_anchor, eps,
                              threat_model="single_frame", t_pert=t_pert)
    ub = spec_c_score_ub(z_xh, z_x)

    sample_rng = np.random.default_rng(31)
    max_concrete = 0.0
    n_viol = 0
    for _ in range(200):
        x = x_anchor.copy()
        x[t_pert] = x_anchor[t_pert] + eps * (2 * sample_rng.random(3) - 1)
        s = _concrete_ae_score(enc, dec, head, x)
        max_concrete = max(max_concrete, s)
        if s > ub + 1e-9:
            n_viol += 1
    assert n_viol == 0, (
        f"spec_c unsound: {n_viol} concrete scores exceed ub. "
        f"max_concrete={max_concrete:.4f}, ub={ub:.4f}"
    )


def test_certify_radius_spec_c_returns_positive_for_tight_anchor():
    """Anchor that already has score ~ 0, tau set generously: bisection
    should return a positive radius and respect tau on samples."""
    enc, dec, head = _build_tiny_ae(D=3, H=4, T=4, seed=5)
    rng = np.random.default_rng(41)
    x_anchor = rng.standard_normal((4, 3))
    anchor_score = _concrete_ae_score(enc, dec, head, x_anchor)
    tau = anchor_score * 4 + 1.0  # generous margin

    eps_cert, per_frame = certify_radius_spec_c(
        enc, dec, head, x_anchor, tau, eps_init=0.5, n_iters=10,
        threat_model="single_frame",
    )
    assert eps_cert > 0
    # Validate by sampling.
    sample_rng = np.random.default_rng(43)
    T, D = x_anchor.shape
    n_viol = 0
    for t_pert in range(T):
        eps_t = per_frame[t_pert]
        if eps_t <= 0:
            continue
        for _ in range(100):
            x = x_anchor.copy()
            x[t_pert] = x_anchor[t_pert] + eps_t * (2 * sample_rng.random(D) - 1)
            s = _concrete_ae_score(enc, dec, head, x)
            if s > tau + 1e-9:
                n_viol += 1
    assert n_viol == 0, f"spec_c radius unsound: {n_viol} concrete scores exceed tau"


def test_certify_radius_spec_c_multi_frame_minkowski():
    """Multi-frame Spec C must be sound under Minkowski lstm_step."""
    enc, dec, head = _build_tiny_ae(D=2, H=3, T=3, seed=8)
    rng = np.random.default_rng(53)
    x_anchor = rng.standard_normal((3, 2))
    anchor_score = _concrete_ae_score(enc, dec, head, x_anchor)
    tau = anchor_score * 4 + 1.0

    eps_cert, _ = certify_radius_spec_c(
        enc, dec, head, x_anchor, tau, eps_init=0.5, n_iters=10,
        threat_model="multi_frame",
    )
    if eps_cert <= 0:
        pytest.skip("model never certified; nothing to validate")

    sample_rng = np.random.default_rng(59)
    T, D = x_anchor.shape
    n_viol = 0
    for _ in range(100):
        x = x_anchor + eps_cert * (2 * sample_rng.random((T, D)) - 1)
        s = _concrete_ae_score(enc, dec, head, x)
        if s > tau + 1e-9:
            n_viol += 1
    assert n_viol == 0, f"multi-frame spec_c unsound: {n_viol}/100 violations"
