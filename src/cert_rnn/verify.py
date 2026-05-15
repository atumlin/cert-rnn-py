"""Cert-RNN verification: Algorithm 1 bisection, reach-set, Spec A, Spec C.

Two specs:
  - Spec A (classifier margin): certify that logit[true_class] dominates
    every other logit over the entire eps-ball -- i.e., argmax does not
    change. Sound via componentwise diff bbox.
  - Spec C (autoencoder false-alarm): certify that
        score(x') := ||AE(x') - x'||_2^2 / N <= tau
    for every x' in the eps-ball, where AE is encoder+decoder+per-step
    linear head. Sound upper bound:
        score_ub = (1/N) * sum_{t,d} (|c_diff[t,d]| + sum_p |V_diff[t,d,p]|)^2
    componentwise worst-case squared, summed. Over-bounds
    max-of-sum-of-squares; tighter joint bounds need a quadratic-over-box
    solver (out of scope).

Threat models:
  - 'single_frame' (Algorithm 1): one frame perturbed by eps; pin the
    others. The sample's certified radius is min over frames.
  - 'multi_frame': every frame perturbed independently with disjoint
    pred_ids. Sound under this port's Minkowski-padded lstm_step; the
    MATLAB reference is unsound here.

Algorithm 1 (Du et al., CCS 2021): start at eps_init; at iteration l
in [2, n_iters+1], add 0.5^l if certify holds at eps, else subtract
0.5^l. Track the largest eps that ever certified.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from cert_rnn.lstm import lstm_state_init, lstm_step_stack
from cert_rnn.rnn import rnn_step
from cert_rnn.zono import Zono, zono_sub

ThreatModel = Literal["single_frame", "multi_frame"]


# ---------- Algorithm 1 ----------


def bisect_epsilon(
    certify_fn: Callable[[float], bool],
    eps_init: float = 0.5,
    n_iters: int = 12,
) -> float:
    """Du et al. Algorithm 1 bisection on epsilon.

    Returns the largest eps for which certify_fn(eps) returned True.
    Returns 0.0 if no eps in the search trajectory certified.
    """
    eps = eps_init
    best = 0.0
    for ell in range(2, n_iters + 2):
        eps = max(eps, 0.0)
        ok = certify_fn(eps)
        if ok:
            best = max(best, eps)
            eps = eps + 0.5 ** ell
        else:
            eps = eps - 0.5 ** ell
    if eps > 0 and certify_fn(eps):
        best = max(best, eps)
    return best


# ---------- input zono construction ----------


def _build_input_zonos(
    x_seq: np.ndarray, eps: float, threat_model: ThreatModel, t_pert: int | None
) -> list[Zono]:
    T, D = x_seq.shape
    if threat_model == "single_frame":
        if t_pert is None:
            raise ValueError("single_frame requires t_pert")
        if not (0 <= t_pert < T):
            raise ValueError(f"t_pert {t_pert} out of range [0, {T})")
        out = []
        for t in range(T):
            if t == t_pert and eps > 0:
                out.append(Zono.from_box(x_seq[t], eps))
            else:
                out.append(Zono.point(x_seq[t]))
        return out
    if threat_model == "multi_frame":
        return [
            Zono.from_box(x_seq[t], eps) if eps > 0 else Zono.point(x_seq[t])
            for t in range(T)
        ]
    raise ValueError(f"unknown threat_model {threat_model!r}")


# ---------- reach-set ----------


def lstm_reach(
    model_dict: dict,
    x_seq: np.ndarray,
    eps: float,
    threat_model: ThreatModel = "single_frame",
    t_pert: int | None = None,
) -> list[Zono]:
    """Forward an LSTM-stack model over an eps-perturbed input sequence.

    Returns the per-timestep top-layer hidden zonotope list. If
    model_dict has a 'head', the classifier head is NOT applied here --
    callers compose it via Zono.affine_map.
    """
    H, L = model_dict["H"], model_dict["L"]
    z_x_seq = _build_input_zonos(x_seq, eps, threat_model, t_pert)
    z_h, z_c = lstm_state_init(H, L)
    z_h_top_seq: list[Zono] = []
    for t in range(x_seq.shape[0]):
        z_h, z_c = lstm_step_stack(z_x_seq[t], z_h, z_c, model_dict["layers"])
        z_h_top_seq.append(z_h[-1])
    return z_h_top_seq


def lstm_ae_reach(
    encoder: dict,
    decoder: dict,
    head: dict,
    x_anchor: np.ndarray,
    eps: float,
    threat_model: ThreatModel = "single_frame",
    t_pert: int | None = None,
) -> tuple[list[Zono], list[Zono]]:
    """Forward an LSTM autoencoder: encoder over x_anchor, decoder reads
    the latent (encoder's final top-layer h) at every step, per-step
    head produces a reconstruction zono per timestep.

    Returns (z_x_hat_seq, z_x_seq) for downstream spec_c_score_ub.
    """
    H = encoder["H"]
    L_enc, L_dec = encoder["L"], decoder["L"]
    T, D = x_anchor.shape
    z_x_seq = _build_input_zonos(x_anchor, eps, threat_model, t_pert)

    z_h_enc, z_c_enc = lstm_state_init(H, L_enc)
    for t in range(T):
        z_h_enc, z_c_enc = lstm_step_stack(
            z_x_seq[t], z_h_enc, z_c_enc, encoder["layers"]
        )
    z_latent = z_h_enc[-1]

    z_h_dec, z_c_dec = lstm_state_init(H, L_dec)
    z_x_hat_seq: list[Zono] = []
    for _t in range(T):
        z_h_dec, z_c_dec = lstm_step_stack(
            z_latent, z_h_dec, z_c_dec, decoder["layers"]
        )
        z_x_hat_seq.append(z_h_dec[-1].affine_map(head["W"], head["b"]))
    return z_x_hat_seq, z_x_seq


# ---------- specs ----------


def spec_a_margin(
    model_dict: dict,
    x_seq: np.ndarray,
    eps: float,
    true_class: int,
    threat_model: ThreatModel = "single_frame",
    t_pert: int | None = None,
) -> bool:
    """Certify: logit[true_class] > logit[c] for every c != true_class
    over the entire eps-ball.

    Sound check: build the (C-1, C) difference matrix D where
        D[i, true_class] = +1, D[i, c_i] = -1
    apply it to the logits zono, and assert the resulting bbox has lb > 0
    on every row.
    """
    if "head" not in model_dict:
        raise ValueError("spec_a_margin requires model_dict['head']")
    z_h_top_seq = lstm_reach(model_dict, x_seq, eps, threat_model, t_pert)
    z_h_T = z_h_top_seq[-1]
    z_logits = z_h_T.affine_map(model_dict["head"]["W"], model_dict["head"]["b"])
    C = z_logits.dim
    if not (0 <= true_class < C):
        raise ValueError(f"true_class {true_class} out of range [0, {C})")
    others = [c for c in range(C) if c != true_class]
    diffs = np.zeros((len(others), C), dtype=np.float64)
    for i, c in enumerate(others):
        diffs[i, true_class] = 1.0
        diffs[i, c] = -1.0
    z_diff = z_logits.affine_map(diffs)
    lb, _ = z_diff.get_ranges()
    return bool(np.all(lb > 0))


def spec_c_score_ub(z_x_hat_seq: list[Zono], z_x_seq: list[Zono]) -> float:
    """Sound upper bound on score(x') = ||AE(x') - x'||_2^2 / N over the
    perturbation set defined by z_x_seq.

        score_ub = (1/N) * sum_{t, d} (|c_diff[t,d]| + sum_p |V_diff[t,d,p]|)^2

    Componentwise worst-case |diff| squared, summed. Over-bounds
    max-of-sum-of-squares; tighter joint bounds would need a
    quadratic-over-box solver.
    """
    if len(z_x_hat_seq) != len(z_x_seq):
        raise ValueError("z_x_hat_seq and z_x_seq must have the same length")
    T = len(z_x_hat_seq)
    if T == 0:
        return 0.0
    D = z_x_hat_seq[0].dim
    N = T * D
    score_ub = 0.0
    for t in range(T):
        z_diff = zono_sub(z_x_hat_seq[t], z_x_seq[t])
        radius = np.sum(np.abs(z_diff.V), axis=1)
        comp_max = np.abs(z_diff.c) + radius
        score_ub += float(np.sum(comp_max ** 2))
    return score_ub / N


def spec_c_holds(
    encoder: dict,
    decoder: dict,
    head: dict,
    x_anchor: np.ndarray,
    eps: float,
    tau: float,
    threat_model: ThreatModel = "single_frame",
    t_pert: int | None = None,
) -> bool:
    """Spec C wrapper: True iff sound score_ub <= tau."""
    z_xh, z_x = lstm_ae_reach(
        encoder, decoder, head, x_anchor, eps, threat_model, t_pert
    )
    return spec_c_score_ub(z_xh, z_x) <= tau


# ---------- certified radius (bisection over eps) ----------


def certify_radius_spec_a(
    model_dict: dict,
    x_seq: np.ndarray,
    true_class: int,
    eps_init: float = 0.5,
    n_iters: int = 12,
    threat_model: ThreatModel = "single_frame",
) -> tuple[float, np.ndarray | None]:
    """Bisect epsilon for Spec A.

    single_frame: bisect per frame; return (min over frames, per-frame array).
    multi_frame:  bisect once; return (eps, None).
    """
    if threat_model == "single_frame":
        T = x_seq.shape[0]
        per_frame = np.zeros(T)
        for t in range(T):
            per_frame[t] = bisect_epsilon(
                lambda eps, _t=t: spec_a_margin(
                    model_dict, x_seq, eps, true_class, "single_frame", _t
                ),
                eps_init,
                n_iters,
            )
        return float(per_frame.min()), per_frame
    eps_cert = bisect_epsilon(
        lambda eps: spec_a_margin(
            model_dict, x_seq, eps, true_class, "multi_frame", None
        ),
        eps_init,
        n_iters,
    )
    return eps_cert, None


def certify_radius_spec_c(
    encoder: dict,
    decoder: dict,
    head: dict,
    x_anchor: np.ndarray,
    tau: float,
    eps_init: float = 0.5,
    n_iters: int = 12,
    threat_model: ThreatModel = "single_frame",
) -> tuple[float, np.ndarray | None]:
    """Bisect epsilon for Spec C. Same shape as certify_radius_spec_a."""
    if threat_model == "single_frame":
        T = x_anchor.shape[0]
        per_frame = np.zeros(T)
        for t in range(T):
            per_frame[t] = bisect_epsilon(
                lambda eps, _t=t: spec_c_holds(
                    encoder, decoder, head, x_anchor, eps, tau, "single_frame", _t
                ),
                eps_init,
                n_iters,
            )
        return float(per_frame.min()), per_frame
    eps_cert = bisect_epsilon(
        lambda eps: spec_c_holds(
            encoder, decoder, head, x_anchor, eps, tau, "multi_frame", None
        ),
        eps_init,
        n_iters,
    )
    return eps_cert, None
