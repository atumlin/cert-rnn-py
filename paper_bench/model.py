"""v3 LSTM autoencoder: training model, cert-rnn conversion, and the
explicit-op score graph shared by the linear-relaxation tools and PGD.

Model (one code per timestep): an L-layer LSTM encoder compresses each
second's D features to H = round(0.8 D) numbers; an L-layer LSTM decoder
reads the encoder's top-layer hidden state at the same step; a linear
head maps back to D features. Score = mean squared reconstruction error
over the window; alarm iff score > tau.

`ScoreGraph` rebuilds the trained network from the SAME cert-rnn dicts the
zonotope engine verifies, using only matmul / sigmoid / tanh / mul, with
the score folded in, so every tool bounds byte-identical weights and the
spec becomes linear in the graph output (score <= tau).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def code_size(n_features: int) -> int:
    return round(0.8 * n_features)        # 80% kept (Zone-A: 64 -> 51)


class V3AE(nn.Module):
    """Training model. x: (B, T, D) -> x_hat: (B, T, D)."""

    def __init__(self, n_features: int, layers: int = 1):
        super().__init__()
        H = code_size(n_features)
        self.encoder = nn.LSTM(n_features, H, num_layers=layers, batch_first=True)
        self.decoder = nn.LSTM(H, H, num_layers=layers, batch_first=True)
        self.head = nn.Linear(H, n_features)

    def forward(self, x):
        codes, _ = self.encoder(x)            # one code per second (top layer, every step)
        y, _ = self.decoder(codes)            # decoder step t reads code t
        return self.head(y)


def to_certrnn(model: V3AE):
    import copy
    from cert_rnn import LSTMAutoencoder
    m = copy.deepcopy(model).double().cpu().eval()          # leave the caller's model untouched
    return LSTMAutoencoder.from_torch(m.encoder, m.decoder, m.head, decoder_input="sequence")


class ScoreGraph(nn.Module):
    """Explicit-op score network built from cert-rnn dicts.

    forward(x_flat): (B, T*D) -> (B, 1) score. Inputs outside the attacked
    seconds are given zero-width bounds by the caller (full-input encoding).
    """

    def __init__(self, ae, T: int, dtype=torch.float32):
        super().__init__()
        self.T, self.D, self.H = T, ae.encoder["D"], ae.encoder["H"]
        self._enc = self._pack(ae.encoder, "enc", dtype)
        self._dec = self._pack(ae.decoder, "dec", dtype)
        # zero initial states are made as x_t @ 0 so the batch size stays dynamic in an ONNX trace
        self.register_buffer("zeros_DH", torch.zeros(self.D, self.H, dtype=dtype))
        self.register_buffer("head_W", torch.as_tensor(np.asarray(ae.head["W"]), dtype=dtype))
        self.register_buffer("head_b", torch.as_tensor(np.asarray(ae.head["b"]), dtype=dtype))

    def _pack(self, md, tag, dtype):
        names = []
        for i, ly in enumerate(md["layers"]):
            row = {}
            for k in ("W_in", "W_rec", "b"):
                n = f"{tag}{i}_{k}"
                self.register_buffer(n, torch.as_tensor(np.asarray(ly[k]), dtype=dtype))
                row[k] = n
            names.append(row)
        return names

    def _step(self, x_t, h, c, layers):
        inp = x_t
        for i, n in enumerate(layers):
            pre = inp @ getattr(self, n["W_in"]).T + h[i] @ getattr(self, n["W_rec"]).T + getattr(self, n["b"])
            H = self.H
            ig, fg = torch.sigmoid(pre[:, :H]), torch.sigmoid(pre[:, H:2 * H])
            gg, og = torch.tanh(pre[:, 2 * H:3 * H]), torch.sigmoid(pre[:, 3 * H:])
            c[i] = fg * c[i] + ig * gg
            h[i] = og * torch.tanh(c[i])
            inp = h[i]
        return inp

    def reconstruct(self, x):                 # x: (B, T, D)
        zero = x[:, 0, :] @ self.zeros_DH                       # (B, H) of zeros, batch-safe
        he, ce = [zero] * len(self._enc), [zero] * len(self._enc)
        hd, cd = [zero] * len(self._dec), [zero] * len(self._dec)
        outs = []
        for t in range(self.T):
            code = self._step(x[:, t, :], he, ce, self._enc)
            top = self._step(code, hd, cd, self._dec)
            outs.append(top @ self.head_W.T + self.head_b)
        return torch.stack(outs, dim=1)

    def forward(self, x_flat):
        x = x_flat.reshape(-1, self.T, self.D)
        d = self.reconstruct(x) - x
        return (d * d).flatten(1).mean(dim=1, keepdim=True)


def window_scores(model: V3AE, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([((model(b) - b) ** 2).mean(dim=(1, 2)).numpy()
                               for b in torch.split(torch.as_tensor(X), batch)])
