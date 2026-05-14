"""Adversarial red-team fuzz for the Cert-RNN engine.

Ports test_red_team.m and extends with multi-frame coverage. For each
scenario:
  1. Build random LSTM weights (varying D, H, L, T, weight scale, center regime).
  2. Build cert input zonotope under the chosen threat model:
       - 'single_frame': perturb ONE frame by eps; pin all others.
                         This is the Algorithm-1 production pattern.
       - 'multi_frame':  every frame has its own independent eps-ball
                         (disjoint pred_ids). The MATLAB reference is
                         UNSOUND on this; the Minkowski fix in this
                         port must keep it sound.
  3. Sample concrete inputs from the perturbation set.
  4. Bounding-box check (necessary; matches MATLAB pattern). Reports
     the worst per-coordinate violation across all samples.

Scale: 30 scenarios per threat model, 100 samples each → 6,000 concrete
forwards. Caps are smaller than MATLAB's (D<=4, H<=4, L<=2, T<=5) to
stay within a reasonable pytest wall-time budget; the per-transformer
LP audit (test_transformers.py) and end-to-end LP check
(test_lstm_step.py) carry the tighter soundness contract.
"""

import numpy as np
import pytest

from cert_rnn import Zono
from cert_rnn.lstm import lstm_state_init, lstm_step_stack

CFG = {
    "n_scenarios_per_model": 30,
    "n_samples": 100,
    "D_choices": [1, 2, 4],
    "H_choices": [1, 2, 4],
    "L_choices": [1, 2],
    "T_choices": [1, 2, 5],
    "eps_choices": [1e-5, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1],
    "weight_scales": [0.1, 0.5, 1.0, 2.0],
    "center_regimes": ["tight", "standard", "saturating"],
    "tol": 1e-9,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _sample_scenario(rng):
    sc = {
        "D": rng.choice(CFG["D_choices"]).item(),
        "H": rng.choice(CFG["H_choices"]).item(),
        "L": rng.choice(CFG["L_choices"]).item(),
        "T": rng.choice(CFG["T_choices"]).item(),
        "eps": float(rng.choice(CFG["eps_choices"])),
        "weight_scale": float(rng.choice(CFG["weight_scales"])),
        "center_regime": str(rng.choice(CFG["center_regimes"])),
    }
    sc["num_classes"] = max(2, min(10, sc["H"]))
    return sc


def _build_layers(sc, rng):
    layers = []
    for i in range(sc["L"]):
        in_size = sc["D"] if i == 0 else sc["H"]
        layers.append(
            {
                "W_in": sc["weight_scale"] * rng.standard_normal((4 * sc["H"], in_size)),
                "W_rec": sc["weight_scale"] * rng.standard_normal((4 * sc["H"], sc["H"])),
                "b": 0.1 * rng.standard_normal(4 * sc["H"]),
            }
        )
    return layers


def _sample_center(sc, rng):
    regime = sc["center_regime"]
    T, D = sc["T"], sc["D"]
    if regime == "tight":
        return 0.3 * rng.standard_normal((T, D))
    if regime == "standard":
        return rng.random((T, D))
    return 3.0 * rng.standard_normal((T, D))  # saturating


def _build_classifier_head(sc, rng):
    return (
        sc["weight_scale"] * rng.standard_normal((sc["num_classes"], sc["H"])),
        0.1 * rng.standard_normal(sc["num_classes"]),
    )


def _concrete_forward(layers, x_seq, sc, W_fc, b_fc):
    H, L = sc["H"], sc["L"]
    h_layers = [np.zeros(H) for _ in range(L)]
    c_layers = [np.zeros(H) for _ in range(L)]
    T = x_seq.shape[0]
    for t in range(T):
        inp = x_seq[t]
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
    return W_fc @ h_layers[-1] + b_fc


def _cert_forward_singleframe(layers, x_center, sc, W_fc, b_fc, t_pert):
    z_h, z_c = lstm_state_init(sc["H"], sc["L"])
    T, D = x_center.shape
    for t in range(T):
        c_t = x_center[t]
        if t == t_pert and sc["eps"] > 0:
            z_x = Zono.from_box(c_t, sc["eps"])
        else:
            z_x = Zono.point(c_t)
        z_h, z_c = lstm_step_stack(z_x, z_h, z_c, layers)
    return z_h[-1].affine_map(W_fc, b_fc)


def _cert_forward_multiframe(layers, x_center, sc, W_fc, b_fc):
    """Every frame has its own eps-ball with DISJOINT pred_ids.
    The threat model the MATLAB reference is unsound on."""
    z_h, z_c = lstm_state_init(sc["H"], sc["L"])
    T, D = x_center.shape
    for t in range(T):
        z_x = Zono.from_box(x_center[t], sc["eps"])
        z_h, z_c = lstm_step_stack(z_x, z_h, z_c, layers)
    return z_h[-1].affine_map(W_fc, b_fc)


def _bbox_check(z_logits, sc, layers, x_center, W_fc, b_fc, threat_model, sampling_rng):
    """Sample concrete inputs, run forward, count bbox violations."""
    lb, ub = z_logits.get_ranges()
    T, D = x_center.shape
    n_violate = 0
    max_under = -np.inf
    max_over = -np.inf
    if threat_model == "single_frame":
        t_pert = sc["_t_pert"]
        for _ in range(CFG["n_samples"]):
            x = x_center.copy()
            x[t_pert] = x_center[t_pert] + sc["eps"] * (
                2 * sampling_rng.random(D) - 1
            )
            logits = _concrete_forward(layers, x, sc, W_fc, b_fc)
            below = float(np.max(lb - logits))
            above = float(np.max(logits - ub))
            max_under = max(max_under, below)
            max_over = max(max_over, above)
            if below > CFG["tol"] or above > CFG["tol"]:
                n_violate += 1
    else:
        for _ in range(CFG["n_samples"]):
            x = x_center + sc["eps"] * (2 * sampling_rng.random((T, D)) - 1)
            logits = _concrete_forward(layers, x, sc, W_fc, b_fc)
            below = float(np.max(lb - logits))
            above = float(np.max(logits - ub))
            max_under = max(max_under, below)
            max_over = max(max_over, above)
            if below > CFG["tol"] or above > CFG["tol"]:
                n_violate += 1
    return n_violate, max(max_under, max_over)


def _run_red_team(threat_model, scenario_seed):
    """Drive N scenarios under one threat model. Returns total violations."""
    rng = np.random.default_rng(scenario_seed)
    total_viol = 0
    worst = 0.0
    for s in range(CFG["n_scenarios_per_model"]):
        sc = _sample_scenario(rng)
        layers = _build_layers(sc, rng)
        W_fc, b_fc = _build_classifier_head(sc, rng)
        x_center = _sample_center(sc, rng)

        if threat_model == "single_frame":
            sc["_t_pert"] = int(rng.integers(0, sc["T"]))
            z_logits = _cert_forward_singleframe(
                layers, x_center, sc, W_fc, b_fc, sc["_t_pert"]
            )
        else:
            z_logits = _cert_forward_multiframe(
                layers, x_center, sc, W_fc, b_fc
            )

        sampling_rng = np.random.default_rng(scenario_seed * 7919 + s)
        n_viol, worst_here = _bbox_check(
            z_logits, sc, layers, x_center, W_fc, b_fc, threat_model, sampling_rng
        )
        total_viol += n_viol
        worst = max(worst, worst_here)
    return total_viol, worst


def test_red_team_single_frame():
    n_viol, worst = _run_red_team("single_frame", scenario_seed=20260507)
    assert n_viol == 0, (
        f"single-frame: {n_viol} bbox violations across "
        f"{CFG['n_scenarios_per_model']}x{CFG['n_samples']} samples; "
        f"worst violation magnitude {worst:.3e}"
    )


def test_red_team_multi_frame_minkowski():
    """Threat model the MATLAB reference is unsound on. The Minkowski
    pred_ids alignment in this port must drive violations to zero."""
    n_viol, worst = _run_red_team("multi_frame", scenario_seed=20260508)
    assert n_viol == 0, (
        f"multi-frame: {n_viol} bbox violations across "
        f"{CFG['n_scenarios_per_model']}x{CFG['n_samples']} samples; "
        f"worst violation magnitude {worst:.3e}"
    )
