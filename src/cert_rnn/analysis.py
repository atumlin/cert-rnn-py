"""Diagnostics for LSTM-autoencoder verification: cost, overapproximation,
and bound tightness.

These are *analysis* helpers layered on the sound engine -- they answer
"how expensive is this?", "how loose is the certified bound?", and "how far
can the reconstruction swing over the perturbation set?".  None of them
weaken soundness; ``reach_stats``/``score_vs_eps`` read the verifier's own
zonotopes, and ``tightness`` compares the sound upper bound against a concrete
(sampled) lower bound on the true worst case.

The concrete forward (``concrete_lstm_ae_forward``) is a plain batched numpy
LSTM-AE evaluation that mirrors ``cert_rnn.verify.lstm_ae_reach`` exactly
(same [i,f,g,o] gate order, zero initial state, latent = encoder final
top-layer hidden, decoder reads the latent each step, per-step linear head).
It matches the abstract engine's center at eps=0 to machine precision -- see
tests/test_analysis.py.

All functions take the plain model dicts (``encoder``/``decoder``/``head``)
the engine consumes, and are also exposed as lightweight methods on
``cert_rnn.LSTMAutoencoder`` (e.g. ``ae.reach_stats(anchor, eps)``).  They
return data (no printing) so callers can format as they like.
"""

from __future__ import annotations

import time

import numpy as np

from cert_rnn.transformers import _sigmoid
from cert_rnn.verify import (
    ThreatModel,
    certify_radius_spec_c,
    lstm_ae_reach,
    spec_c_score_ub,
)
from cert_rnn.zono import zono_sub


# --------------------------------------------------------------------------- #
# concrete (numpy) forward -- mirrors lstm_ae_reach semantics exactly
# --------------------------------------------------------------------------- #
def _stack_step(x, h_list, c_list, layers):
    """One timestep through an LSTM stack (in place); returns top hidden.

    x: (N, in0).  layers: list of {"W_in","W_rec","b"} with gate order
    [i, f, g, o].  Updates h_list/c_list per layer; layer i feeds layer i+1.
    """
    inp = x
    for li, ly in enumerate(layers):
        W_in, W_rec, b = ly["W_in"], ly["W_rec"], ly["b"]
        H = W_rec.shape[1]
        g = inp @ W_in.T + h_list[li] @ W_rec.T + b          # (N, 4H)
        i = _sigmoid(g[:, :H])
        f = _sigmoid(g[:, H:2 * H])
        gg = np.tanh(g[:, 2 * H:3 * H])
        o = _sigmoid(g[:, 3 * H:])
        c_new = f * c_list[li] + i * gg
        h_new = o * np.tanh(c_new)
        h_list[li], c_list[li] = h_new, c_new
        inp = h_new
    return inp


def concrete_lstm_ae_forward(encoder, decoder, head, X):
    """Plain batched reconstruction AE(X). X is (T, D) or (N, T, D).

    Returns x_hat with the same leading shape.  Matches lstm_ae_reach's center.
    """
    X = np.asarray(X, dtype=np.float64)
    single = X.ndim == 2
    if single:
        X = X[None]
    N, T, D = X.shape

    enc_layers, dec_layers = encoder["layers"], decoder["layers"]
    h_e = [np.zeros((N, ly["W_rec"].shape[1])) for ly in enc_layers]
    c_e = [np.zeros_like(h) for h in h_e]
    for t in range(T):
        _stack_step(X[:, t, :], h_e, c_e, enc_layers)
    latent = h_e[-1]                                         # (N, H)

    h_d = [np.zeros((N, ly["W_rec"].shape[1])) for ly in dec_layers]
    c_d = [np.zeros_like(h) for h in h_d]
    Wh, bh = head["W"], head["b"]
    x_hat = np.empty((N, T, D))
    for t in range(T):
        top = _stack_step(latent, h_d, c_d, dec_layers)
        x_hat[:, t, :] = top @ Wh.T + bh
    return x_hat[0] if single else x_hat


def reconstruction_score(encoder, decoder, head, X):
    """Concrete anomaly score mean((AE(x)-x)**2). Scalar for (T,D), else (N,)."""
    X = np.asarray(X, dtype=np.float64)
    single = X.ndim == 2
    Xb = X[None] if single else X
    x_hat = concrete_lstm_ae_forward(encoder, decoder, head, Xb)
    s = ((x_hat - Xb) ** 2).mean(axis=(1, 2))
    return float(s[0]) if single else s


# --------------------------------------------------------------------------- #
# overapproximation (reads the verifier's own zonotopes)
# --------------------------------------------------------------------------- #
def _t_pert(threat_model, t_pert):
    if threat_model == "single_frame" and t_pert is None:
        return 0
    return t_pert


def reach_stats(encoder, decoder, head, x_anchor, eps,
                threat_model: ThreatModel = "multi_frame", t_pert=None) -> dict:
    """Zonotope overapproximation stats for the reconstruction over the eps-ball.

    Returns:
        score_ub        sound upper bound on the reconstruction score
        n_pred          # zonotope generators in the reconstruction (size/precision)
        recon_width     (T, D) interval width ub-lb of x_hat
        err_width       (T, D) interval width of (x_hat - x)
        mean_err_width, max_err_width   scalar summaries
    The widths are the overapproximation: how far each reconstructed value can
    provably swing over the input perturbation set.
    """
    tp = _t_pert(threat_model, t_pert)
    z_xh, z_x = lstm_ae_reach(encoder, decoder, head, x_anchor, float(eps),
                              threat_model, tp)
    recon_w, err_w, n_pred = [], [], 0
    for zh, zx in zip(z_xh, z_x):
        lb, ub = zh.get_ranges()
        recon_w.append(ub - lb)
        d = zono_sub(zh, zx)
        dlb, dub = d.get_ranges()
        err_w.append(dub - dlb)
        n_pred = max(n_pred, zh.n_pred)
    recon_w = np.asarray(recon_w)
    err_w = np.asarray(err_w)
    return {
        "eps": float(eps),
        "score_ub": float(spec_c_score_ub(z_xh, z_x)),
        "n_pred": int(n_pred),
        "recon_width": recon_w,
        "err_width": err_w,
        "mean_err_width": float(err_w.mean()),
        "max_err_width": float(err_w.max()),
    }


def score_vs_eps(encoder, decoder, head, x_anchor, eps_list,
                 threat_model: ThreatModel = "multi_frame", t_pert=None):
    """Sound worst-case score at each eps (the robustness curve).

    Returns list of (eps, score_ub).
    """
    tp = _t_pert(threat_model, t_pert)
    rows = []
    for eps in eps_list:
        z_xh, z_x = lstm_ae_reach(encoder, decoder, head, x_anchor, float(eps),
                                  threat_model, tp)
        rows.append((float(eps), float(spec_c_score_ub(z_xh, z_x))))
    return rows


# --------------------------------------------------------------------------- #
# bound tightness (sound UB vs sampled worst case)
# --------------------------------------------------------------------------- #
def tightness(encoder, decoder, head, x_anchor, eps, n_samples: int = 2000,
              threat_model: ThreatModel = "multi_frame", t_pert=None,
              seed: int = 0) -> dict:
    """Compare the certified upper bound to an empirical (sampled) worst case.

    Draws ``n_samples`` perturbations from the L_inf eps-ball, scores them with
    the concrete forward, and compares the max to the sound upper bound.  The
    UB must be >= the empirical max (soundness); gap/ratio quantify looseness.
    The empirical max is itself a lower bound on the true worst case, so the
    reported gap is a (conservative) over-estimate of the real looseness.

    multi_frame perturbs every frame; single_frame perturbs only frame t_pert.
    """
    x_anchor = np.asarray(x_anchor, dtype=np.float64)
    rng = np.random.default_rng(seed)
    P = rng.uniform(-eps, eps, size=(n_samples,) + x_anchor.shape)
    if threat_model == "single_frame":
        tp = 0 if t_pert is None else t_pert
        mask = np.zeros_like(x_anchor)
        mask[tp] = 1.0
        P = P * mask
    X = x_anchor[None] + P
    emp_max = float(np.max(reconstruction_score(encoder, decoder, head, X)))

    tp = _t_pert(threat_model, t_pert)
    z_xh, z_x = lstm_ae_reach(encoder, decoder, head, x_anchor, float(eps),
                              threat_model, tp)
    ub = float(spec_c_score_ub(z_xh, z_x))
    return {
        "eps": float(eps),
        "score_ub": ub,
        "empirical_max": emp_max,
        "gap": ub - emp_max,
        "ratio": (ub / emp_max) if emp_max > 0 else float("inf"),
        "sound": bool(ub >= emp_max - 1e-9),
        "n_samples": int(n_samples),
    }


# --------------------------------------------------------------------------- #
# timings
# --------------------------------------------------------------------------- #
def time_reach(encoder, decoder, head, x_anchor, eps,
               threat_model: ThreatModel = "multi_frame", t_pert=None,
               repeat: int = 5) -> float:
    """Best-of-``repeat`` wall-clock (s) for one abstract forward pass."""
    tp = _t_pert(threat_model, t_pert)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        lstm_ae_reach(encoder, decoder, head, x_anchor, float(eps), threat_model, tp)
        best = min(best, time.perf_counter() - t0)
    return best


def time_certify(encoder, decoder, head, x_anchor, tau,
                 threat_model: ThreatModel = "multi_frame", n_iters: int = 12,
                 eps_init: float = 0.5) -> dict:
    """Wall-clock (s) for a full certified-radius computation, with the cost
    factors that drive it.

    The cost is (number of abstract forward passes) x (cost per pass):
        n_reach_calls = n_iters x (T if single_frame else 1)
        cost per pass scales with T, hidden size H, and stack depth.

    Returns a dict with the timing AND the factors:
        radius, seconds, per_frame, threat_model, n_iters,
        T, D, H, n_enc_layers, n_dec_layers,
        n_reach_calls, sec_per_reach.
    """
    T = int(np.asarray(x_anchor).shape[0])
    t0 = time.perf_counter()
    radius, per_frame = certify_radius_spec_c(
        encoder, decoder, head, x_anchor, tau, eps_init, n_iters, threat_model
    )
    dt = time.perf_counter() - t0
    n_reach = n_iters * (T if threat_model == "single_frame" else 1)
    return {
        "radius": float(radius),
        "seconds": dt,
        "per_frame": per_frame,
        "threat_model": threat_model,
        "n_iters": int(n_iters),
        "T": T,
        "D": int(encoder["D"]),
        "H": int(encoder["H"]),
        "n_enc_layers": int(encoder["L"]),
        "n_dec_layers": int(decoder["L"]),
        "n_reach_calls": int(n_reach),
        "sec_per_reach": dt / n_reach if n_reach else float("nan"),
    }
