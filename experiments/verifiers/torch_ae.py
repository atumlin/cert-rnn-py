"""Explicit-op torch reconstruction of an LSTM-AE from cert_rnn dicts.

Rebuilds the exact computation `lstm_ae_reach` abstracts -- encoder over
the sequence, latent broadcast to every decoder step, per-step linear
head -- using only matmul / sigmoid / tanh / elementwise-mul, so that
(a) PGD gets exact gradients of the concrete score, and (b) auto_LiRPA
can trace the graph (its bound rules cover these primitives; an nn.LSTM
black-box would not trace).

Weights come from the SAME dicts the zonotope engine consumes -- there
is no second checkpoint-loading path to drift out of sync."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TorchLSTMAE(nn.Module):
    """Batched concrete forward. x: (B, T, D) -> x_hat: (B, T, D)."""

    def __init__(self, encoder: dict, decoder: dict, head: dict,
                 dtype=torch.float64):
        super().__init__()
        self.T_default = None
        self.enc = self._pack(encoder, dtype, "enc")
        self.dec = self._pack(decoder, dtype, "dec")
        self.register_buffer("head_W", torch.as_tensor(head["W"], dtype=dtype))
        self.register_buffer("head_b", torch.as_tensor(head["b"], dtype=dtype))
        self.H = encoder["H"]

    def _pack(self, md: dict, dtype, tag: str):
        layers = []
        for i, lyr in enumerate(md["layers"]):
            names = {}
            for key in ("W_in", "W_rec", "b"):
                buf = torch.as_tensor(np.asarray(lyr[key]), dtype=dtype)
                bname = f"{tag}{i}_{key}"
                self.register_buffer(bname, buf)
                names[key] = bname
            layers.append(names)
        return layers

    def _step_stack(self, x_t, h, c, layers):
        inp = x_t
        for i, names in enumerate(layers):
            W_in = getattr(self, names["W_in"])
            W_rec = getattr(self, names["W_rec"])
            b = getattr(self, names["b"])
            pre = inp @ W_in.T + h[i] @ W_rec.T + b
            H = b.shape[0] // 4
            ig = torch.sigmoid(pre[..., 0 * H:1 * H])
            fg = torch.sigmoid(pre[..., 1 * H:2 * H])
            gg = torch.tanh(pre[..., 2 * H:3 * H])
            og = torch.sigmoid(pre[..., 3 * H:4 * H])
            c[i] = fg * c[i] + ig * gg
            h[i] = og * torch.tanh(c[i])
            inp = h[i]
        return h, c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        zeros = lambda: [x.new_zeros(B, self.H) for _ in self.enc]
        h, c = zeros(), zeros()
        for t in range(T):
            h, c = self._step_stack(x[:, t, :], h, c, self.enc)
        latent = h[-1]
        hd = [x.new_zeros(B, self.H) for _ in self.dec]
        cd = [x.new_zeros(B, self.H) for _ in self.dec]
        outs = []
        for _t in range(T):
            hd, cd = self._step_stack(latent, hd, cd, self.dec)
            outs.append(hd[-1] @ self.head_W.T + self.head_b)
        return torch.stack(outs, dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """(B,) reconstruction score mean((AE(x)-x)^2) -- the anomaly score."""
        diff = self.forward(x) - x
        return diff.pow(2).flatten(1).mean(dim=1)


class SingleFrameResidualAE(nn.Module):
    """auto_LiRPA-facing wrapper for the single_frame threat model: the
    module input is ONLY the perturbed frame x_t (B, D); all other frames
    are baked in as constants. Output is the flattened residual
    AE(x') - x' of shape (B, T*D), so an L_inf box on the input yields
    interval bounds on every residual component."""

    def __init__(self, ae: TorchLSTMAE, anchor: np.ndarray, t_pert: int,
                 dtype=torch.float64):
        super().__init__()
        self.ae = ae
        self.t_pert = t_pert
        self.register_buffer("anchor",
                             torch.as_tensor(anchor, dtype=dtype))  # (T, D)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        B = x_t.shape[0]
        T, D = self.anchor.shape
        frames = []
        for t in range(T):
            if t == self.t_pert:
                frames.append(x_t)
            else:
                frames.append(self.anchor[t].unsqueeze(0).expand(B, D))
        x = torch.stack(frames, dim=1)          # (B, T, D)
        return (self.ae(x) - x).flatten(1)      # (B, T*D)


class MultiFrameResidualAE(nn.Module):
    """auto_LiRPA-facing wrapper for the multi_frame threat model: input
    is the whole sequence, flattened residual out."""

    def __init__(self, ae: TorchLSTMAE):
        super().__init__()
        self.ae = ae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.ae(x) - x).flatten(1)
