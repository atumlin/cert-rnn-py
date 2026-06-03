"""Probe whether MATLAB's certified eps on size D is empirically unsound.

Hypothesis: MATLAB's cert eps (0.0247) > Python's cert eps (0.0228) for
size D because MATLAB's positional padding aliases distinct alphas in
the stacked encoder/decoder pipeline, producing tighter-but-potentially-
unsound bounds. If we can find a concrete x' in B(x_anchor, eps_matlab)
at some frame t_pert that violates score(x') <= tau, that's a soundness
violation in MATLAB.

Method:
  1. Rebuild the D autoencoder in PyTorch from the .pt weights
     (2-layer encoder, 2-layer decoder, single-element classifier head).
  2. For each test frame t_pert, run PGD ascent on the concrete score
     inside B(x_anchor, eps_matlab[t]) where eps_matlab[t] is MATLAB's
     certified eps for that frame.
  3. Also run inside B(x_anchor, eps_python[t]) as a control -- here
     Python claims soundness; PGD should not find a violator.
  4. Report any frame where PGD found a violator inside the MATLAB ball.

Frames probed: 0, 5, 10, 15, 20, 25 (every 5th).

Run:
    python scripts/probe_D_cert_unsoundness.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples" / "lstm_ae_ieee9"))
from ae_loader import load_lstm_ae    # noqa: E402

torch.set_default_dtype(torch.float64)


def build_torch_ae_multilayer(bundle: dict) -> nn.Module:
    enc = bundle["encoder"]
    dec = bundle["decoder"]
    head = bundle["head"]
    H = bundle["H"]
    T = bundle["T"]
    D_dim = bundle["D"]
    L_enc = enc["L"]
    L_dec = dec["L"]

    class LstmAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.H = H; self.T = T; self.D = D_dim
            self.L_enc = L_enc; self.L_dec = L_dec
            self.enc_cells = nn.ModuleList()
            for i in range(L_enc):
                in_size = D_dim if i == 0 else H
                cell = nn.LSTMCell(in_size, H)
                lyr = enc["layers"][i]
                with torch.no_grad():
                    cell.weight_ih.copy_(torch.tensor(lyr["W_in"]))
                    cell.weight_hh.copy_(torch.tensor(lyr["W_rec"]))
                    cell.bias_ih.copy_(torch.tensor(lyr["b"] / 2))
                    cell.bias_hh.copy_(torch.tensor(lyr["b"] / 2))
                self.enc_cells.append(cell)
            self.dec_cells = nn.ModuleList()
            for i in range(L_dec):
                in_size = H  # decoder input is always H (the latent or previous-layer h)
                cell = nn.LSTMCell(in_size, H)
                lyr = dec["layers"][i]
                with torch.no_grad():
                    cell.weight_ih.copy_(torch.tensor(lyr["W_in"]))
                    cell.weight_hh.copy_(torch.tensor(lyr["W_rec"]))
                    cell.bias_ih.copy_(torch.tensor(lyr["b"] / 2))
                    cell.bias_hh.copy_(torch.tensor(lyr["b"] / 2))
                self.dec_cells.append(cell)
            self.head = nn.Linear(H, D_dim)
            with torch.no_grad():
                self.head.weight.copy_(torch.tensor(head["W"]))
                self.head.bias.copy_(torch.tensor(head["b"]))

        def encode(self, x):
            hs = [torch.zeros(1, self.H) for _ in range(self.L_enc)]
            cs = [torch.zeros(1, self.H) for _ in range(self.L_enc)]
            for t in range(self.T):
                inp = x[t:t+1]
                for i in range(self.L_enc):
                    hs[i], cs[i] = self.enc_cells[i](inp, (hs[i], cs[i]))
                    inp = hs[i]
            return hs[-1]   # final top hidden

        def decode(self, latent):
            hs = [torch.zeros(1, self.H) for _ in range(self.L_dec)]
            cs = [torch.zeros(1, self.H) for _ in range(self.L_dec)]
            x_hat = []
            for t in range(self.T):
                inp = latent
                for i in range(self.L_dec):
                    hs[i], cs[i] = self.dec_cells[i](inp, (hs[i], cs[i]))
                    inp = hs[i]
                x_hat.append(self.head(hs[-1]))
            return torch.cat(x_hat, dim=0)

        def forward(self, x):
            latent = self.encode(x)
            return self.decode(latent)

        def score(self, x):
            xh = self.forward(x)
            return ((xh - x) ** 2).sum() / (self.T * self.D)

    return LstmAE()


def pgd_attack(net, x_anchor, t_pert, eps, n_steps=120, n_restarts=20):
    D_dim = x_anchor.shape[1]
    step = eps / 5
    best_score = -np.inf
    best_x = None
    for r in range(n_restarts):
        dx = (2 * torch.rand(D_dim) - 1) * eps
        dx.requires_grad_(True)
        x_base = torch.tensor(x_anchor)
        for _ in range(n_steps):
            x = x_base.clone()
            x[t_pert] = x[t_pert] + dx
            s = net.score(x)
            grad = torch.autograd.grad(s, dx)[0]
            with torch.no_grad():
                dx = dx + step * grad.sign()
                dx = dx.clamp(-eps, eps)
            dx.requires_grad_(True)
        with torch.no_grad():
            x = x_base.clone()
            x[t_pert] = x[t_pert] + dx
            s = float(net.score(x).item())
            if s > best_score:
                best_score = s
                best_x = x.detach().numpy().copy()
    return best_score, best_x


def main():
    bundle = load_lstm_ae("D")
    net = build_torch_ae_multilayer(bundle).eval()
    tau = bundle["tau"]
    anchor = bundle["anchor"]

    # Load Python and MATLAB per-frame eps.
    py = dict(np.load(ROOT / "examples" / "lstm_ae_ieee9" / "results" / "certrnn_lstm_ae_D.npz"))
    mat = scipy.io.loadmat(str(ROOT / "examples" / "lstm_ae_ieee9" / "matlab_results" / "certrnn_lstm_ae_D.mat"))
    eps_py = py["eps_per_frame"]
    eps_mat = mat["eps_per_frame"].flatten().astype(np.float64)

    print(f"=== Probing size D (H={bundle['H']} L_enc={bundle['encoder']['L']} L_dec={bundle['decoder']['L']}) ===")
    print(f"  tau={tau} anchor_score={bundle['anchor_score']:.4f}")
    with torch.no_grad():
        s_anchor = float(net.score(torch.tensor(anchor)).item())
    print(f"  PyTorch anchor_score: {s_anchor:.6f} (drift {s_anchor - bundle['anchor_score']:.2e})")
    print()

    frames = [0, 5, 10, 15, 20, 25, 29]
    matlab_violations = []
    print(f"  {'frame':>5} {'eps_py':>10} {'eps_mat':>10} {'PGD@eps_py':>12} {'PGD@eps_mat':>12} {'tau':>6} {'verdict':>20}")
    for t in frames:
        s_py, _ = pgd_attack(net, anchor, t, eps=float(eps_py[t]), n_steps=120, n_restarts=20)
        s_mat, _ = pgd_attack(net, anchor, t, eps=float(eps_mat[t]), n_steps=120, n_restarts=20)
        py_ok = s_py <= tau + 1e-9
        mat_ok = s_mat <= tau + 1e-9
        verdict = "MATLAB UNSOUND" if not mat_ok else ("matlab loose" if py_ok else "both sound")
        if not mat_ok:
            matlab_violations.append((t, eps_mat[t], s_mat))
        print(
            f"  {t:>5} {float(eps_py[t]):>10.5f} {float(eps_mat[t]):>10.5f} "
            f"{s_py:>12.5f} {s_mat:>12.5f} {tau:>6.3f} {verdict:>20}"
        )

    print()
    if matlab_violations:
        print(f"  MATLAB UNSOUNDNESS WITNESSED on {len(matlab_violations)} frame(s):")
        for t, e, s in matlab_violations:
            print(f"    frame {t}: at eps={e:.5f} found x' with score={s:.5f} > tau={tau}")
    else:
        print(f"  No PGD-discoverable MATLAB unsoundness on frames {frames}.")
        print(f"  MATLAB cert may still be unsound at worst-case; PGD is local.")


if __name__ == "__main__":
    main()
