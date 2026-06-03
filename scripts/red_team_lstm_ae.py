"""Red-team validation of LSTM-AE Spec C (autoencoder false-alarm) certificates.

Property under test:
    forall x' in B(x_anchor, eps),  score(x') = ||AE(x') - x'||_2^2 / N <= tau.

Strategy at one frame t_pert:
  1. Bisect the certified eps.
  2. Random sampling at eps: concrete score must be <= tau, otherwise UNSOUND.
  3. Random sampling at 1.5*eps: probes tightness.
  4. Gradient ascent (PGD) on the concrete score inside the eps-ball;
     if it pushes any sampled point above tau, the bound is UNSOUND.

Runs on size S (H=4) which is fast (~30s per frame in PGD).

Run:
    python scripts/red_team_lstm_ae.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples" / "lstm_ae_ieee9"))
from ae_loader import load_lstm_ae    # noqa: E402

torch.set_default_dtype(torch.float64)


def build_torch_ae(bundle: dict) -> nn.Module:
    enc = bundle["encoder"]; dec = bundle["decoder"]; head = bundle["head"]
    H = bundle["H"]; T = bundle["T"]; D = bundle["D"]

    class LstmAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.H = H; self.T = T; self.D = D
            self.enc_cell = nn.LSTMCell(D, H)
            self.dec_cell = nn.LSTMCell(H, H)
            for cell, layer in [(self.enc_cell, enc["layers"][0]),
                                (self.dec_cell, dec["layers"][0])]:
                with torch.no_grad():
                    cell.weight_ih.copy_(torch.tensor(layer["W_in"]))
                    cell.weight_hh.copy_(torch.tensor(layer["W_rec"]))
                    cell.bias_ih.copy_(torch.tensor(layer["b"] / 2))
                    cell.bias_hh.copy_(torch.tensor(layer["b"] / 2))
            self.head = nn.Linear(H, D)
            with torch.no_grad():
                self.head.weight.copy_(torch.tensor(head["W"]))
                self.head.bias.copy_(torch.tensor(head["b"]))

        def forward(self, x):
            # x: (T, D)  -> (T, D)
            h = torch.zeros(1, self.H); c = torch.zeros(1, self.H)
            for t in range(self.T):
                h, c = self.enc_cell(x[t:t+1], (h, c))
            latent = h
            x_hat = []
            h = torch.zeros(1, self.H); c = torch.zeros(1, self.H)
            for t in range(self.T):
                h, c = self.dec_cell(latent, (h, c))
                x_hat.append(self.head(h))
            return torch.cat(x_hat, dim=0)   # (T, D)

        def score(self, x):
            xh = self.forward(x)
            return ((xh - x) ** 2).sum() / (self.T * self.D)

    return LstmAE()


def pgd_score_attack(net, x_anchor, t_pert, eps, n_steps=80, n_restarts=10):
    """Gradient ASCENT on the concrete score inside the single-frame eps-ball.
    Returns the maximum concrete score found and whether it exceeded tau (caller
    checks tau)."""
    D = x_anchor.shape[1]
    step = eps / 5
    best_score = -np.inf
    best_x = None
    for _ in range(n_restarts):
        dx = (2 * torch.rand(D) - 1) * eps
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
    bundle = load_lstm_ae("S")
    net = build_torch_ae(bundle).eval()
    tau = bundle["tau"]
    anchor = bundle["anchor"]
    print(
        f"=== Red team: LSTM-AE IEEE-9 size S ===\n"
        f"  H={bundle['H']} T={bundle['T']} D={bundle['D']} "
        f"tau={tau} anchor_score={bundle['anchor_score']:.4f}"
    )

    # Verify anchor score matches metadata
    with torch.no_grad():
        s_anchor = float(net.score(torch.tensor(anchor)).item())
    print(f"  Python-Torch anchor_score: {s_anchor:.6f}  (meta {bundle['anchor_score']:.6f})")

    # Bisect cert eps at a few frames
    from cert_rnn.verify import bisect_epsilon, spec_c_holds
    frames_to_test = [0, 10, 20, 29]
    summary = []
    rng = np.random.default_rng(2026_05_14)
    for t_pert in frames_to_test:
        eps_cert = bisect_epsilon(
            lambda eps, _t=t_pert: spec_c_holds(
                bundle["encoder"], bundle["decoder"], bundle["head"],
                anchor, eps, tau, threat_model="single_frame", t_pert=_t,
            ),
            eps_init=0.5, n_iters=12,
        )

        D = bundle["D"]
        n_sample = 500
        max_at = -np.inf
        max_15 = -np.inf
        viol_at = viol_15 = 0
        for _ in range(n_sample):
            x = anchor.copy()
            x[t_pert] = anchor[t_pert] + eps_cert * (2 * rng.random(D) - 1)
            with torch.no_grad():
                s = float(net.score(torch.tensor(x)).item())
            max_at = max(max_at, s)
            if s > tau + 1e-9: viol_at += 1
            x = anchor.copy()
            x[t_pert] = anchor[t_pert] + 1.5 * eps_cert * (2 * rng.random(D) - 1)
            with torch.no_grad():
                s = float(net.score(torch.tensor(x)).item())
            max_15 = max(max_15, s)
            if s > tau + 1e-9: viol_15 += 1

        # PGD
        pgd_score_at, _ = pgd_score_attack(net, anchor, t_pert, eps_cert, n_steps=80, n_restarts=10)
        pgd_score_over, _ = pgd_score_attack(net, anchor, t_pert, 1.5 * eps_cert, n_steps=80, n_restarts=10)

        summary.append({
            "t_pert": t_pert, "eps_cert": eps_cert,
            "max_random_at": max_at, "max_random_15": max_15,
            "viol_at": viol_at, "viol_15": viol_15,
            "pgd_at": pgd_score_at, "pgd_15": pgd_score_over,
            "tau": tau,
        })
        print(
            f"\n  frame {t_pert}: cert_eps={eps_cert:.6f}\n"
            f"    random @ eps:    max_score={max_at:.4f}  (tau={tau})  violations: {viol_at}/{n_sample}\n"
            f"    random @ 1.5eps: max_score={max_15:.4f}  violations: {viol_15}/{n_sample}\n"
            f"    PGD @ eps:    score={pgd_score_at:.4f}  {'EXCEEDS tau (UNSOUND)' if pgd_score_at > tau + 1e-9 else 'within tau'}\n"
            f"    PGD @ 1.5eps: score={pgd_score_over:.4f}  {'exceeds tau' if pgd_score_over > tau + 1e-9 else 'still within tau'}"
        )

    print("\n=== Summary ===")
    unsound = sum(1 for r in summary if r["viol_at"] > 0 or r["pgd_at"] > tau + 1e-9)
    print(f"  Soundness violations: {unsound}/{len(summary)} frames tested")
    if unsound == 0:
        print("  -> No empirical evidence of unsoundness on these frames.")


if __name__ == "__main__":
    main()
