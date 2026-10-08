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


class AttackScore(nn.Module):
    """The same score as a function of the ATTACKED readings only (B, k*D).

    * Every other reading, and the zero initial state, is a constant buffer, so
      bounding tools treat it as exact. (Zero-width perturbed inputs instead
      leave float-rounding widths that alpha-CROWN's sigmoid/tanh relaxations
      blow up to ~1e20.)
    * Only MatMul / Add / Mul / Sigmoid / Tanh on buffers: no Slice, ReduceSum
      or scalar Div, whose scalar constants alpha-beta-CROWN's GenBaB (BBPS)
      branching heuristic does not accept.
    * Constant (1, .) and batched (B, .) terms are combined by broadcasting with
      the batched term first (onnx2pytorch adds in place).
    """

    def __init__(self, g: ScoreGraph, anchor: np.ndarray, frames):
        super().__init__()
        self.T, self.D, self.H, self.frames = g.T, g.D, g.H, sorted(frames)
        dtype, H, D = g.head_W.dtype, g.H, g.D
        reg = lambda n, v: self.register_buffer(n, torch.as_tensor(np.asarray(v), dtype=dtype))
        anchor = np.asarray(anchor).reshape(g.T, D)
        k = len(self.frames)
        for j in range(k):                          # z @ sel_j = attacked reading j
            sel = np.zeros((k * D, D)); sel[j * D:(j + 1) * D] = np.eye(D)
            reg(f"sel{j}", sel)
        for t in range(g.T):
            reg(f"anchor{t}", anchor[t:t + 1])
        reg("zero_H", np.zeros((1, H)))
        reg("ones_D", np.ones((D, 1)))
        reg("inv_TD", np.full((1, 1), 1.0 / (g.T * D)))
        reg("head_WT", g.head_W.detach().cpu().numpy().T)
        reg("head_b", g.head_b.detach().cpu().numpy().reshape(-1))
        self._layers = {}
        for tag, layers in (("enc", g._enc), ("dec", g._dec)):
            names = []
            for i, n in enumerate(layers):
                W_in = getattr(g, n["W_in"]).detach().cpu().numpy()
                W_rec = getattr(g, n["W_rec"]).detach().cpu().numpy()
                bias = getattr(g, n["b"]).detach().cpu().numpy().reshape(-1)
                row = {}
                for q, gate in enumerate("ifgo"):     # PyTorch gate order i, f, g, o
                    rows = slice(q * H, (q + 1) * H)
                    for kind, val in (("Wx", W_in[rows].T), ("Wh", W_rec[rows].T), ("b", bias[rows])):
                        nm = f"{tag}{i}_{gate}_{kind}"
                        reg(nm, val)
                        row[gate, kind] = nm
                names.append(row)
            self._layers[tag] = names

    @staticmethod
    def _add(a, b, a_batched):
        return a + b if a_batched else b + a

    def _step(self, x_t, h, c, tag, x_batched, state_batched):
        inp, batched = x_t, x_batched
        for i, row in enumerate(self._layers[tag]):
            B = lambda gate, kind: getattr(self, row[gate, kind])
            pre = {q: self._add(inp @ B(q, "Wx"), h[i] @ B(q, "Wh"), batched) + B(q, "b") for q in "ifgo"}
            ig, fg = torch.sigmoid(pre["i"]), torch.sigmoid(pre["f"])
            gg, og = torch.tanh(pre["g"]), torch.sigmoid(pre["o"])
            out_b = batched or state_batched
            c[i] = self._add(ig * gg, fg * c[i], out_b)
            h[i] = og * torch.tanh(c[i])
            inp, batched = h[i], out_b
        return inp, batched

    def forward(self, z):
        L_enc, L_dec = len(self._layers["enc"]), len(self._layers["dec"])
        he, ce = [self.zero_H] * L_enc, [self.zero_H] * L_enc
        hd, cd = [self.zero_H] * L_dec, [self.zero_H] * L_dec
        total, state_batched = None, False      # state becomes batched at the first attacked second
        for t in range(self.T):
            if t in self.frames:
                x_t, xb = z @ getattr(self, f"sel{self.frames.index(t)}"), True
            else:
                x_t, xb = getattr(self, f"anchor{t}"), False
            code, enc_b = self._step(x_t, he, ce, "enc", xb, state_batched)
            top, _ = self._step(code, hd, cd, "dec", enc_b, state_batched)
            d = (top @ self.head_WT + self.head_b) - x_t
            err = (d * d) @ self.ones_D
            state_batched = enc_b
            total = err if total is None else self._add(err, total, enc_b)
        return total * self.inv_TD


def window_scores(model: V3AE, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([((model(b) - b) ** 2).mean(dim=(1, 2)).numpy()
                               for b in torch.split(torch.as_tensor(X), batch)])
