"""End-to-end soundness of lstm_step / lstm_step_stack / rnn_step.

Three soundness conditions:
    1. Single-frame perturbation (same ball reused across all T steps).
       This is the Algorithm-1 production pattern.
    2. Fresh-per-timestep perturbation (independent ball at every step,
       disjoint pred_ids). The MATLAB reference's zero-padding step is
       UNSOUND under this threat model; the Minkowski pred_ids scheme
       must make it sound. This test is the regression that closes the
       largest known limitation.
    3. Stacked multi-layer LSTM under single-frame.

Soundness check is LP-feasibility (necessary + sufficient).
"""

import numpy as np

from cert_rnn import Zono
from cert_rnn.audit import lp_feasible
from cert_rnn.lstm import lstm_state_init, lstm_step, lstm_step_stack
from cert_rnn.rnn import rnn_step


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_concrete_forward(x_seq, W_in, W_rec, b, h0, c0):
    H = h0.shape[0]
    h, c = h0.copy(), c0.copy()
    for t in range(x_seq.shape[1]):
        x = x_seq[:, t]
        pre = W_in @ x + W_rec @ h + b
        i = _sigmoid(pre[:H])
        f = _sigmoid(pre[H : 2 * H])
        g = np.tanh(pre[2 * H : 3 * H])
        o = _sigmoid(pre[3 * H : 4 * H])
        c = f * c + i * g
        h = o * np.tanh(c)
    return h, c


def _rnn_concrete_forward(x_seq, W_in, W_rec, b, h0):
    h = h0.copy()
    for t in range(x_seq.shape[1]):
        x = x_seq[:, t]
        h = np.tanh(W_in @ x + W_rec @ h + b)
    return h


def test_lstm_step_singleframe_repeated_input():
    """Algorithm 1 pattern: same ball reused at every step. T=3."""
    rng = np.random.default_rng(20260504)
    D, H, T, eps = 2, 2, 3, 0.2
    W_in = 0.5 * rng.standard_normal((4 * H, D))
    W_rec = 0.3 * rng.standard_normal((4 * H, H))
    b = 0.1 * rng.standard_normal(4 * H)
    mu = np.array([0.5, -0.3])

    z_x = Zono.from_box(mu, eps)
    z_h = Zono.point(np.zeros(H))
    z_c = Zono.point(np.zeros(H))
    for _ in range(T):
        z_h, z_c = lstm_step(z_x, z_h, z_c, W_in, W_rec, b)

    sampling_rng = np.random.default_rng(7)
    alphas = 2 * sampling_rng.random((D, 200)) - 1
    n_viol = 0
    for s in range(200):
        x = mu + eps * alphas[:, s]
        x_seq = np.tile(x[:, None], (1, T))
        h_T, _ = _lstm_concrete_forward(
            x_seq, W_in, W_rec, b, np.zeros(H), np.zeros(H)
        )
        if not lp_feasible(z_h, h_T):
            n_viol += 1
    assert n_viol == 0, f"single-frame: {n_viol}/200 outside cert bound"


def test_lstm_step_fresh_per_timestep_minkowski():
    """Threat model the MATLAB reference is UNSOUND on: each frame has
    its own independent eps-ball. With Minkowski pred_ids this must be
    sound."""
    rng = np.random.default_rng(20260505)
    D, H, T, eps = 2, 2, 3, 0.1
    W_in = 0.5 * rng.standard_normal((4 * H, D))
    W_rec = 0.3 * rng.standard_normal((4 * H, H))
    b = 0.1 * rng.standard_normal(4 * H)
    centers = rng.standard_normal((D, T)) * 0.3

    z_x_seq = [Zono.from_box(centers[:, t], eps) for t in range(T)]
    z_h = Zono.point(np.zeros(H))
    z_c = Zono.point(np.zeros(H))
    for t in range(T):
        z_h, z_c = lstm_step(z_x_seq[t], z_h, z_c, W_in, W_rec, b)

    sampling_rng = np.random.default_rng(11)
    n_viol = 0
    for _ in range(200):
        x_seq = centers + eps * (2 * sampling_rng.random((D, T)) - 1)
        h_T, _ = _lstm_concrete_forward(
            x_seq, W_in, W_rec, b, np.zeros(H), np.zeros(H)
        )
        if not lp_feasible(z_h, h_T):
            n_viol += 1
    assert n_viol == 0, f"Minkowski multi-frame: {n_viol}/200 violations"


def test_rnn_step_singleframe_repeated_input():
    rng = np.random.default_rng(20260506)
    D, H, T, eps = 2, 3, 5, 0.2
    W_in = 0.5 * rng.standard_normal((H, D))
    W_rec = 0.3 * rng.standard_normal((H, H))
    b = 0.1 * rng.standard_normal(H)
    mu = np.array([0.5, -0.3])
    z_x = Zono.from_box(mu, eps)
    z_h = Zono.point(np.zeros(H))
    for _ in range(T):
        z_h = rnn_step(z_x, z_h, W_in, W_rec, b)

    sampling_rng = np.random.default_rng(13)
    n_viol = 0
    for _ in range(200):
        alphas = 2 * sampling_rng.random(D) - 1
        x = mu + eps * alphas
        x_seq = np.tile(x[:, None], (1, T))
        h_T = _rnn_concrete_forward(x_seq, W_in, W_rec, b, np.zeros(H))
        if not lp_feasible(z_h, h_T):
            n_viol += 1
    assert n_viol == 0


def test_lstm_stack_multi_layer_singleframe():
    rng = np.random.default_rng(20260507)
    D, H, T, L, eps = 2, 3, 3, 2, 0.1
    layers = []
    for i in range(L):
        in_size = D if i == 0 else H
        layers.append(
            {
                "W_in": 0.5 * rng.standard_normal((4 * H, in_size)),
                "W_rec": 0.3 * rng.standard_normal((4 * H, H)),
                "b": 0.1 * rng.standard_normal(4 * H),
            }
        )
    mu = np.array([0.4, -0.2])
    z_x = Zono.from_box(mu, eps)
    z_h_layers, z_c_layers = lstm_state_init(H, L)
    for _ in range(T):
        z_h_layers, z_c_layers = lstm_step_stack(
            z_x, z_h_layers, z_c_layers, layers
        )

    sampling_rng = np.random.default_rng(17)
    n_viol = 0
    for _ in range(100):
        alphas = 2 * sampling_rng.random(D) - 1
        x = mu + eps * alphas
        h_layers = [np.zeros(H) for _ in range(L)]
        c_layers = [np.zeros(H) for _ in range(L)]
        for _t in range(T):
            inp = x
            for i in range(L):
                W_in = layers[i]["W_in"]
                W_rec = layers[i]["W_rec"]
                b = layers[i]["b"]
                pre = W_in @ inp + W_rec @ h_layers[i] + b
                ii = _sigmoid(pre[:H])
                ff = _sigmoid(pre[H : 2 * H])
                gg = np.tanh(pre[2 * H : 3 * H])
                oo = _sigmoid(pre[3 * H : 4 * H])
                c_layers[i] = ff * c_layers[i] + ii * gg
                h_layers[i] = oo * np.tanh(c_layers[i])
                inp = h_layers[i]
        if not lp_feasible(z_h_layers[-1], h_layers[-1]):
            n_viol += 1
    assert n_viol == 0
