"""Template: certify your own trained LSTM classifier.

Copy this file and replace the three marked sections with your model,
input, and property. This is the minimal user path -- no MATLAB
cross-validation, no sampling harness (see docs/quickstart.md for more).

Run:
    python examples/verify_my_model.py
"""

from __future__ import annotations

# Pin BLAS to a single thread BEFORE importing numpy/cert_rnn -- cert_rnn
# does many small ops and multi-threaded BLAS only adds contention.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch.nn as nn

from cert_rnn import MarginSpec, RNNModel

# ---- 1. YOUR MODEL ----------------------------------------------------------
# An nn.LSTM / nn.LSTMCell / nn.RNN (+ optional nn.Linear head). Load your
# checkpoint here; the random module below is only a placeholder.
import torch

torch.manual_seed(0)
recurrent = nn.LSTMCell(8, 16).double().eval()   # <-- replace with your layer
head = nn.Linear(16, 4).double().eval()          # <-- replace with your head

model = RNNModel.from_torch(recurrent, head)

# ---- 2. YOUR INPUT ----------------------------------------------------------
x_seq = np.random.default_rng(0).standard_normal((20, 8))   # <-- (T, D)
# In real use this is your ground-truth label. For this self-contained
# template we use the model's own nominal prediction so the demo certifies
# a real (non-zero) radius.
true_class = int(np.argmax(model.reach_output(x_seq, 0.0, "single_frame", 0).c))

# ---- 3. CERTIFY -------------------------------------------------------------
# Largest single-frame L_inf radius under which the prediction provably holds.
result = model.certify(x_seq, MarginSpec(true_class))

print(result)
print(f"\ncertified radius: {result.radius:.6f}")
print(f"certified: {result.certified}")
