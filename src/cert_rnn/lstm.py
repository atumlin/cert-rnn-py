"""LSTM time-step abstract transformer (Minkowski-padded).

Predicate alignment via pred_ids keeps each alpha-variable identifiable
across timesteps; combinations (zono_add, bilinear transformers) embed
their inputs into a unified predicate space where shared pred_ids share
a column and unshared get disjoint columns. This is the fix relative to
the MATLAB reference, which used positional zero-padding and silently
aliased fresh-per-timestep predicates with state predicates -- the
all-frame threat model became unsound under that pattern.

Gate order [i, f, g, o]: matches PyTorch nn.LSTM and MATLAB Deep
Learning Toolbox.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cert_rnn.transformers import (
    bilinear_sigmoid_identity,
    bilinear_sigmoid_tanh,
)
from cert_rnn.zono import PredAllocator, Zono, get_default_allocator, zono_add


def lstm_state_init(H: int, L: int) -> tuple[list[Zono], list[Zono]]:
    """Build L cell-arrays of zero-point Zonos (h, c) for L stacked layers."""
    z_h_layers = [Zono.point(np.zeros(H)) for _ in range(L)]
    z_c_layers = [Zono.point(np.zeros(H)) for _ in range(L)]
    return z_h_layers, z_c_layers


def lstm_step(
    z_x: Zono,
    z_h_prev: Zono,
    z_c_prev: Zono,
    W_in: np.ndarray,
    W_rec: np.ndarray,
    b: np.ndarray,
    allocator: PredAllocator | None = None,
) -> tuple[Zono, Zono]:
    """One LSTM time step using Cert-RNN transformers.

    Update:
        [i_pre; f_pre; g_pre; o_pre] = W_in @ x + W_rec @ h_prev + b
        c_t = sigma(f_pre) * c_prev + sigma(i_pre) * tanh(g_pre)
        h_t = sigma(o_pre) * tanh(c_t)

    Predicate alignment is Minkowski-correct via pred_ids; no
    shared-prefix assumption. Sound under both Algorithm-1 single-frame
    and fresh-per-timestep ("all frames perturbed independently") inputs.
    """
    alloc = allocator if allocator is not None else get_default_allocator()
    if W_in.shape[0] != W_rec.shape[0] or W_in.shape[0] != b.shape[0]:
        raise ValueError("lstm_step: W_in / W_rec / b row counts must match")
    if W_in.shape[0] % 4 != 0:
        raise ValueError("lstm_step: W_in must have 4*H rows")
    H = W_in.shape[0] // 4
    if z_h_prev.dim != H or z_c_prev.dim != H:
        raise ValueError(f"lstm_step: h_prev / c_prev dim must equal H={H}")
    if z_x.dim != W_in.shape[1]:
        raise ValueError("lstm_step: z_x dim must equal W_in col count")

    z_in_proj = z_x.affine_map(W_in, b)
    z_rec_proj = z_h_prev.affine_map(W_rec, None)
    z_pre = zono_add(z_in_proj, z_rec_proj)

    z_i_pre = z_pre.slice_rows(0, H)
    z_f_pre = z_pre.slice_rows(H, 2 * H)
    z_g_pre = z_pre.slice_rows(2 * H, 3 * H)
    z_o_pre = z_pre.slice_rows(3 * H, 4 * H)

    # c_t = f_t * c_prev + i_t * g_t
    z_c_term1 = bilinear_sigmoid_identity(z_c_prev, z_f_pre, alloc)
    z_c_term2 = bilinear_sigmoid_tanh(z_i_pre, z_g_pre, alloc)
    z_c = zono_add(z_c_term1, z_c_term2)

    # h_t = sigma(o_pre) * tanh(c_t)
    z_h = bilinear_sigmoid_tanh(z_o_pre, z_c, alloc)
    return z_h, z_c


def lstm_step_stack(
    z_x: Zono,
    z_h_layers: list[Zono],
    z_c_layers: list[Zono],
    layers: Sequence[dict],
    allocator: PredAllocator | None = None,
) -> tuple[list[Zono], list[Zono]]:
    """One time step of a stacked (multi-layer) LSTM."""
    L = len(layers)
    if len(z_h_layers) != L or len(z_c_layers) != L:
        raise ValueError(f"lstm_step_stack: state lists must match L={L}")
    new_h: list[Zono] = [None] * L  # type: ignore[list-item]
    new_c: list[Zono] = [None] * L  # type: ignore[list-item]
    inp = z_x
    for i in range(L):
        lyr = layers[i]
        new_h[i], new_c[i] = lstm_step(
            inp,
            z_h_layers[i],
            z_c_layers[i],
            lyr["W_in"],
            lyr["W_rec"],
            lyr["b"],
            allocator,
        )
        inp = new_h[i]
    return new_h, new_c
