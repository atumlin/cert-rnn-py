"""Vanilla (Elman) tanh-RNN abstract step.

    h_t = tanh(W_in @ x_t + W_rec @ h_{t-1} + b)

The Cert-RNN paper (Du et al., CCS 2021, Table 2) evaluates only
tanh-activated vanilla RNNs; ReLU variants would substitute a sound
ReLU zono transformer but are outside the paper's scope.
"""

from __future__ import annotations

import numpy as np

from cert_rnn.transformers import tanh_zono
from cert_rnn.zono import PredAllocator, Zono, get_default_allocator, zono_add


def rnn_step(
    z_x: Zono,
    z_h_prev: Zono,
    W_in: np.ndarray,
    W_rec: np.ndarray,
    b: np.ndarray,
    allocator: PredAllocator | None = None,
) -> Zono:
    alloc = allocator if allocator is not None else get_default_allocator()
    if W_in.shape[0] != W_rec.shape[0] or W_in.shape[0] != b.shape[0]:
        raise ValueError("rnn_step: W_in / W_rec / b row counts must match")
    H = W_in.shape[0]
    if W_rec.shape[1] != H:
        raise ValueError("rnn_step: W_rec must be H x H")
    if z_h_prev.dim != H:
        raise ValueError(f"rnn_step: z_h_prev dim must equal H={H}")
    if z_x.dim != W_in.shape[1]:
        raise ValueError("rnn_step: z_x dim must equal W_in col count")

    z_in_proj = z_x.affine_map(W_in, b)
    z_rec_proj = z_h_prev.affine_map(W_rec, None)
    z_pre = zono_add(z_in_proj, z_rec_proj)
    return tanh_zono(z_pre, alloc)
