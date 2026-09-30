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
from dataclasses import dataclass

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

    from cert_rnn.verify import decoder_input

    enc_layers, dec_layers = encoder["layers"], decoder["layers"]
    h_e = [np.zeros((N, ly["W_rec"].shape[1])) for ly in enc_layers]
    c_e = [np.zeros_like(h) for h in h_e]
    codes = []
    for t in range(T):
        codes.append(_stack_step(X[:, t, :], h_e, c_e, enc_layers).copy())
    if decoder_input(decoder) == "latent":
        codes = [codes[-1]] * T                              # (N, H) each

    h_d = [np.zeros((N, ly["W_rec"].shape[1])) for ly in dec_layers]
    c_d = [np.zeros_like(h) for h in h_d]
    Wh, bh = head["W"], head["b"]
    x_hat = np.empty((N, T, D))
    for t in range(T):
        top = _stack_step(codes[t], h_d, c_d, dec_layers)
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
# preflight (input-format validation before certify)
# --------------------------------------------------------------------------- #
@dataclass
class PreflightReport:
    """Result of `preflight`. `ok` is True iff no check FAILed (WARNs and
    infos don't block). `checks` is a list of (status, name, detail)
    tuples with status in {"pass", "fail", "warn", "info"}. `title` is
    the caller's label for the run; `subject` is a one-line summary of
    what was checked (model dims, anchor shape, tau). Print the report
    for a human-readable summary."""

    ok: bool
    checks: list
    title: str = ""
    subject: str = ""

    def __str__(self) -> str:
        name = f" [{self.title}]" if self.title else ""
        lines = [
            f"Preflight{name}: "
            f"{'OK -- safe to certify' if self.ok else 'FAILED -- fix before certifying'}"
        ]
        if self.subject:
            lines.append(f"  checking: {self.subject}")
        for status, name, detail in self.checks:
            lines.append(f"  [{status.upper():4s}] {name}: {detail}")
        return "\n".join(lines)


def preflight(encoder, decoder, head, x_anchor, tau=None,
              torch_model=None, title: str = "") -> PreflightReport:
    """Validate the anchor (and optionally tau / torch parity) BEFORE a
    long certify run. Milliseconds; catches the mistakes that otherwise
    surface as cryptic engine errors or silently-wrong results.

    Checks, in order (structural failures stop early):
      1. anchor is numeric and convertible to float64
      2. anchor is a single (T, D) window -- catches batched (N, T, D)
         input, 1-D input needing reshape(-1, 1), and transposed (D, T)
      3. anchor's feature axis matches the model's D
      4. all values finite (no NaN/Inf)
      5. [info] value range -- eyeball that it matches training scaling
      6. [info] concrete anchor score; if `tau` given, checks the spec
         is not already violated at eps=0 (certify would return 0)
      7. if `torch_model` given (a callable mapping a (1, T, D) tensor
         to a (1, T, D) tensor), end-to-end parity: the tool's forward
         must reproduce the torch model's reconstruction score. This is
         THE topology check -- run it once per model.

    The model-side wiring (layer sizes, decoder-reads-latent, head dims)
    is already validated at construction by from_torch; preflight covers
    the anchor side that construction cannot see.
    """
    checks: list = []
    D, H = int(encoder["D"]), int(encoder["H"])
    try:
        anchor_shape = tuple(np.asarray(x_anchor).shape)
    except Exception:
        anchor_shape = "?"
    subject = (
        f"model D={D} H={H} L_enc={int(encoder['L'])} L_dec={int(decoder['L'])}"
        f" | anchor {anchor_shape}"
        + (f" | tau={float(tau):.6g}" if isinstance(tau, (int, float, np.floating)) else
           f" | tau={tau!r}" if tau is not None else "")
        + (" | torch parity: yes" if torch_model is not None else "")
    )

    def add(status: str, name: str, detail: str) -> None:
        checks.append((status, name, detail))

    def done() -> PreflightReport:
        return PreflightReport(
            not any(s == "fail" for s, _, _ in checks), checks, title, subject
        )

    # 1. numeric / convertible
    try:
        x = np.asarray(x_anchor, dtype=np.float64)
    except (TypeError, ValueError) as e:
        add("fail", "dtype", f"anchor is not convertible to float64: {e}")
        return done()
    src_dtype = getattr(np.asarray(x_anchor), "dtype", "unknown")
    add("pass", "dtype", f"convertible to float64 (source dtype: {src_dtype})")

    # 2. rank
    if x.ndim == 3:
        add("fail", "shape",
            f"got {x.shape} -- looks like a BATCH of windows; certify takes "
            f"ONE (T, D) window. Index it first, e.g. X[i].")
        return done()
    if x.ndim == 1:
        hint = " reshape(-1, 1) to make it (T, 1)." if D == 1 else ""
        add("fail", "shape",
            f"got 1-D shape {x.shape}; expected (T, D=({D})).{hint}")
        return done()
    if x.ndim != 2:
        add("fail", "shape", f"got {x.ndim}-D shape {x.shape}; expected (T, D)")
        return done()
    T, d = x.shape

    # 3. feature axis
    if d != D:
        hint = (" Axis 0 matches D -- the anchor looks TRANSPOSED; pass x.T."
                if T == D else "")
        add("fail", "feature dim",
            f"anchor last axis is {d} but the model expects D={D}.{hint}")
        return done()
    add("pass", "shape", f"(T={T}, D={D}) matches the model (H={H})")
    add("info", "seq length",
        f"T={T}; the engine accepts any T -- confirm it equals your "
        f"training window length (cost per reach grows ~T^2)")

    # 4. finiteness
    n_bad = int(np.size(x) - np.sum(np.isfinite(x)))
    if n_bad:
        add("fail", "finite", f"{n_bad} NaN/Inf value(s) in the anchor")
        return done()
    add("pass", "finite", "no NaN/Inf")

    # 5. scaling eyeball
    add("info", "value range",
        f"[{x.min():.4g}, {x.max():.4g}], mean {x.mean():.4g} -- must be in "
        f"the same scaling/normalization used at training time")

    # 6. concrete score vs tau
    score = reconstruction_score(encoder, decoder, head, x)
    add("info", "anchor score", f"{score:.6g} (concrete, eps=0)")
    if tau is not None:
        if not np.isfinite(tau) or tau <= 0:
            add("fail", "tau", f"tau={tau} must be a finite positive number")
        elif score > tau:
            add("warn", "tau",
                f"anchor score {score:.6g} > tau={tau:.6g} -- the spec is "
                f"already violated at eps=0; certify will return radius 0")
        else:
            add("pass", "tau",
                f"tau={tau:.6g} leaves {tau / score:.2f}x headroom over the "
                f"anchor score")

    # 7. torch parity (the topology check)
    if torch_model is not None:
        import torch

        try:
            params = list(torch_model.parameters())
        except AttributeError:
            params = []
        dtypes = sorted({str(p.dtype) for p in params})
        if len(dtypes) > 1:
            add("fail", "torch parity",
                f"torch model has MIXED parameter dtypes {dtypes} -- its own "
                f"forward will fail. A common cause is an in-place .double() "
                f"on a submodule during conversion (nn.Module.double() "
                f"mutates); restore with model.float() and drop the "
                f".double() -- from_torch casts to float64 internally.")
            return done()
        if params:
            t_dtype, t_device = params[0].dtype, params[0].device
        else:
            t_dtype, t_device = torch.float32, "cpu"
        try:
            with torch.no_grad():
                xt = torch.as_tensor(x, dtype=t_dtype, device=t_device).unsqueeze(0)
                recon = torch_model(xt).squeeze(0).cpu().numpy().astype(np.float64)
        except Exception as e:  # report, don't crash the preflight
            add("fail", "torch parity",
                f"torch model forward raised {type(e).__name__}: {e}")
            return done()
        if recon.shape != x.shape:
            add("fail", "torch parity",
                f"torch model returned shape {recon.shape}, expected {x.shape}")
            return done()
        model_score = float(((recon - x) ** 2).mean())
        diff = abs(model_score - score)
        tol = 1e-4 * max(1.0, abs(model_score))
        if diff <= tol:
            add("pass", "torch parity",
                f"tool score {score:.6g} == torch score {model_score:.6g} "
                f"(|diff|={diff:.2e})")
        else:
            add("fail", "torch parity",
                f"tool score {score:.6g} != torch score {model_score:.6g} "
                f"(|diff|={diff:.2e}) -- the extracted model does NOT "
                f"compute the same function; check the topology mapping "
                f"(decoder input = {decoder.get('input', 'latent')}, "
                f"per-step head)")
    return done()


# --------------------------------------------------------------------------- #
# smoke test (seconds-fast pipeline check + cost forecast)
# --------------------------------------------------------------------------- #
def smoke_test(encoder, decoder, head, x_anchor, tau,
               n_frames: int = 8, n_iters: int = 3,
               threat_model: ThreatModel = "multi_frame",
               eps_init: float = 0.5, full_n_iters: int = 12) -> dict:
    """Fast end-to-end check on a truncated anchor, plus a cost forecast
    for the full-length run. Runs in seconds; run this BEFORE committing
    to a multi-minute/-hour certify.

    On ``x_anchor[:n_frames]`` it runs:
      1. an eps=0 internal parity check -- the abstract engine's score at
         eps=0 must equal the concrete numpy forward's score (catches
         model-dict / shape / topology problems). NOTE: this checks the
         engine against itself; parity against your ORIGINAL torch model
         should be checked once separately (verify_ae.py step 4).
      2. one timed abstract reach -- the unit of cost everything scales in,
      3. a short Algorithm-1 bisection (``n_iters``) end-to-end.
    It then times a second reach at ``2*n_frames`` (when the anchor is long
    enough), fits the empirical cost-growth exponent in T, and forecasts:
      - ``est_sec_per_reach_full``: one reach at the full length,
      - ``est_certify_multi_frame_s``:  (full_n_iters+1) reaches,
      - ``est_certify_single_frame_s``: T_full * (full_n_iters+1) reaches.

    Returns a dict with all of the above plus ``ok`` (parity passed and
    the smoke bisection completed).
    """
    x_anchor = np.asarray(x_anchor, dtype=np.float64)
    T_full = int(x_anchor.shape[0])
    Ts = int(min(n_frames, T_full))
    x_s = x_anchor[:Ts]
    tp = _t_pert(threat_model, None)

    # 1. eps=0 parity: input zonos are points, so the abstract score upper
    # bound collapses to the concrete score (up to fp noise).
    z_xh, z_x = lstm_ae_reach(encoder, decoder, head, x_s, 0.0,
                              "single_frame", 0)
    ub0 = spec_c_score_ub(z_xh, z_x)
    concrete = reconstruction_score(encoder, decoder, head, x_s)
    parity_diff = abs(ub0 - concrete)
    parity_ok = bool(parity_diff < 1e-4)

    # 2. one timed reach at the truncated length
    t0 = time.perf_counter()
    z_xh, z_x = lstm_ae_reach(encoder, decoder, head, x_s, float(eps_init),
                              threat_model, tp)
    dt1 = time.perf_counter() - t0
    score_ub_smoke = spec_c_score_ub(z_xh, z_x)

    # 3. short bisection end-to-end (exercises the full certify path)
    t0 = time.perf_counter()
    smoke_radius, _ = certify_radius_spec_c(
        encoder, decoder, head, x_s, tau, eps_init, n_iters, threat_model
    )
    dt_cert = time.perf_counter() - t0

    # 4. fit the cost-growth exponent from a second, longer reach; fall
    # back to the theoretical ~quadratic growth when the anchor is short.
    T2 = int(min(2 * Ts, T_full))
    if T2 > Ts and dt1 > 0:
        t0 = time.perf_counter()
        lstm_ae_reach(encoder, decoder, head, x_anchor[:T2], float(eps_init),
                      threat_model, _t_pert(threat_model, None))
        dt2 = time.perf_counter() - t0
        growth = float(np.log(dt2 / dt1) / np.log(T2 / Ts)) if dt2 > dt1 else 2.0
        est_reach_full = dt2 * (T_full / T2) ** growth
    else:
        growth = 2.0
        est_reach_full = dt1 * (T_full / Ts) ** growth if Ts else float("nan")

    evals = full_n_iters + 1  # bisect_epsilon: n_iters steps + final re-check
    return {
        "ok": parity_ok and np.isfinite(smoke_radius),
        "T_full": T_full,
        "T_smoke": Ts,
        "threat_model": threat_model,
        "parity_abs_diff": float(parity_diff),
        "parity_ok": parity_ok,
        "smoke_radius": float(smoke_radius),
        "smoke_score_ub_at_eps_init": float(score_ub_smoke),
        "smoke_certify_seconds": float(dt_cert),
        "sec_per_reach_smoke": float(dt1),
        "growth_exponent": growth,
        "est_sec_per_reach_full": float(est_reach_full),
        "est_certify_multi_frame_s": float(evals * est_reach_full),
        "est_certify_single_frame_s": float(T_full * evals * est_reach_full),
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
