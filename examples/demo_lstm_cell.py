"""End-to-end Cert-RNN walkthrough on a tiny PyTorch LSTM.

Run:
    python examples/demo_lstm_cell.py

What this script does, top to bottom:
  1. Build a random PyTorch nn.LSTMCell + nn.Linear classifier (float64).
  2. Extract them into a Cert-RNN model dict via cert_rnn.from_torch.
  3. Pick a concrete input x*, find its predicted class.
  4. Run the abstract forward at eps=0 to sanity-check the model dict.
  5. Bisect epsilon (Du et al. Algorithm 1) to find the largest single-frame
     perturbation under which the predicted class is provably preserved.
  6. Sample 500 concrete x' inside that ball, run the concrete forward,
     confirm no sampled point flips the prediction.

If you read this file end to end, you have seen the entire public API.
"""

from __future__ import annotations

import os
# Pin BLAS to single thread. cert_rnn does many small numpy ops; default
# BLAS threading hurts more than it helps and burns CPU on busy machines.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

from cert_rnn import certify_radius_spec_a, lstm_reach, lstm_to_model_dict


# ----- 1. Build a tiny LSTM ---------------------------------------------------

torch.manual_seed(20260514)
D, H, T, C = 4, 6, 8, 3            # input dim, hidden, timesteps, num classes
cell = nn.LSTMCell(D, H).double().eval()
fc = nn.Linear(H, C).double().eval()


# ----- 2. Extract a Cert-RNN model dict ---------------------------------------

model = lstm_to_model_dict(cell, fc)
# model is a plain dict consumed by cert_rnn.lstm.lstm_step_stack:
#   { "type": "lstm", "gate_order": "ifgo",
#     "D": 4, "H": 6, "L": 1,
#     "layers": [{"W_in": (4H, D), "W_rec": (4H, H), "b": (4H,)}],
#     "head":   {"W": (C, H),       "b": (C,)} }
print(f"Model: D={model['D']} H={model['H']} L={model['L']} C={fc.out_features}")


# ----- 3. Pick a concrete input and find its predicted class ------------------

rng = np.random.default_rng(42)
x_seq = rng.standard_normal((T, D))   # (T, D)

def concrete_logits(x: np.ndarray) -> np.ndarray:
    """Plain PyTorch forward, for comparison with the cert engine."""
    x_t = torch.tensor(x, dtype=torch.float64).unsqueeze(1)   # (T, 1, D)
    h = torch.zeros(1, H, dtype=torch.float64)
    c = torch.zeros(1, H, dtype=torch.float64)
    for t in range(T):
        h, c = cell(x_t[t], (h, c))
    return fc(h)[0].detach().numpy()

logits = concrete_logits(x_seq)
true_class = int(np.argmax(logits))
print(f"\nNominal logits: {np.array2string(logits, precision=3)}")
print(f"Predicted class: {true_class}")


# ----- 4. eps=0 parity check ---------------------------------------------------
# Point-zonotope inputs propagate exactly through every transformer
# (degenerate-box branches), so cert forward at eps=0 must match PyTorch.

z_h_top_seq = lstm_reach(model, x_seq, eps=0.0,
                         threat_model="single_frame", t_pert=0)
z_logits = z_h_top_seq[-1].affine_map(model["head"]["W"], model["head"]["b"])
cert_logits = z_logits.c
print(f"\ncert eps=0 logits: {np.array2string(cert_logits, precision=3)}")
print(f"max |cert - pytorch|: {float(np.max(np.abs(cert_logits - logits))):.2e}")


# ----- 5. Algorithm 1 bisection: max eps that preserves the prediction --------
# certify_radius_spec_a runs Du et al. Algorithm 1 per frame for the
# classifier-margin spec (logit[true_class] provably dominates every other
# logit over the eps-ball) and returns (min-over-frames, per-frame array).
# The margin-matrix construction and bisection live in cert_rnn.verify --
# no need to hand-roll them here.

print("\nBisecting epsilon per frame (Algorithm 1, single-frame perturbation):")
cert_radius, eps_per_frame = certify_radius_spec_a(
    model, x_seq, true_class, eps_init=0.5, n_iters=12, threat_model="single_frame"
)
for t in range(T):
    print(f"  frame {t}: certified eps = {eps_per_frame[t]:.4f}")

print(f"\nCertified single-frame radius (min over T): {cert_radius:.4f}")


# ----- 6. Soundness sanity via sampling --------------------------------------
# For every t_pert, sample concrete x' inside the certified ball; the
# predicted class must never change.

if cert_radius > 0:
    sample_rng = np.random.default_rng(2026)
    n_samples = 500
    n_violations = 0
    for t_pert in range(T):
        eps_t = eps_per_frame[t_pert]
        if eps_t <= 0:
            continue
        for _ in range(n_samples):
            x = x_seq.copy()
            x[t_pert] = x_seq[t_pert] + eps_t * (2 * sample_rng.random(D) - 1)
            if int(np.argmax(concrete_logits(x))) != true_class:
                n_violations += 1
    total = T * n_samples
    print(
        f"\nSampled {total} concrete x' inside the certified per-frame balls; "
        f"{n_violations} flipped the prediction."
    )
    assert n_violations == 0, "soundness violation"
    print("Cert bound holds.")
else:
    print("\nModel never certified at eps_init=0.5; nothing to sample.")
