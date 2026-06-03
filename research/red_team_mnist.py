"""Red-team validation of MNIST Spec A (classifier margin) certificates.

Strategy for one sample at one t_pert:
  1. Bisect the certified eps using the cert engine.
  2. Sample N random concrete perturbations at *exactly* eps; count
     argmax flips.
  3. Sample N random concrete perturbations at eps*(1+overshoot); count
     argmax flips. If the cert is meaningful at all, going past eps
     should sometimes flip; if it doesn't, the bound is loose but not
     a soundness witness either way.
  4. Run a PGD attack inside the eps-ball using PyTorch autograd; if
     it finds a flipping point, the cert is UNSOUND. If not, that is
     strong empirical evidence of soundness (much stronger than random
     sampling).

Run:
    python scripts/red_team_mnist.py
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
sys.path.insert(0, str(ROOT / "examples" / "mnist_sequence"))
from mnist_loader import load_mnist_lstm   # noqa: E402
from demo_multilayer import cert_radius_singleframe   # noqa: E402

torch.set_default_dtype(torch.float64)


def build_torch_model(bundle: dict) -> tuple[nn.Module, int, int, int, int]:
    """Reconstruct the trained LSTM as a torch.nn.Module from the numpy
    weights in the .mat. Single-layer in MNIST LSTM-1-32; this function
    handles any L."""
    model = bundle["model"]
    H, L, D, C = model["H"], model["L"], model["D"], bundle["num_classes"]

    class StackedLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.cells = nn.ModuleList()
            for i in range(L):
                in_size = D if i == 0 else H
                cell = nn.LSTMCell(in_size, H)
                lyr = model["layers"][i]
                with torch.no_grad():
                    cell.weight_ih.copy_(torch.tensor(lyr["W_in"]))
                    cell.weight_hh.copy_(torch.tensor(lyr["W_rec"]))
                    # Split the folded bias half/half (sum is what matters).
                    cell.bias_ih.copy_(torch.tensor(lyr["b"] / 2))
                    cell.bias_hh.copy_(torch.tensor(lyr["b"] / 2))
                self.cells.append(cell)
            self.fc = nn.Linear(H, C)
            with torch.no_grad():
                self.fc.weight.copy_(torch.tensor(model["head"]["W"]))
                self.fc.bias.copy_(torch.tensor(model["head"]["b"]))

        def forward(self, x):
            # x: (T, D) or (T, B, D)
            if x.ndim == 2:
                x = x.unsqueeze(1)  # (T, 1, D)
            T, B, _ = x.shape
            hs = [torch.zeros(B, H) for _ in range(L)]
            cs = [torch.zeros(B, H) for _ in range(L)]
            for t in range(T):
                inp = x[t]
                for i in range(L):
                    hs[i], cs[i] = self.cells[i](inp, (hs[i], cs[i]))
                    inp = hs[i]
            return self.fc(hs[-1])  # (B, C)

    return StackedLSTM(), D, H, L, C


def pgd_attack(net, x0, y_true, t_pert, eps, n_steps=50, step_size=None, n_restarts=5):
    """Project gradient ascent on -log p(y_true | x') restricted to
    perturbing frame t_pert by [-eps, eps] in L_inf. Returns (best_dx, flipped).
    Picks the worst (max-loss) attack across restarts.
    """
    D = x0.shape[1]
    if step_size is None:
        step_size = eps / 5
    best_dx = None
    best_flipped = False
    best_loss = -np.inf
    for r in range(n_restarts):
        dx = (2 * torch.rand(D) - 1) * eps
        dx.requires_grad_(True)
        x_base = torch.tensor(x0)
        for _ in range(n_steps):
            x = x_base.clone()
            x[t_pert] = x[t_pert] + dx
            logits = net(x).squeeze(0)
            loss = -nn.functional.log_softmax(logits, dim=0)[y_true]
            grad = torch.autograd.grad(loss, dx)[0]
            with torch.no_grad():
                dx = dx + step_size * grad.sign()
                dx = dx.clamp(-eps, eps)
            dx.requires_grad_(True)
        with torch.no_grad():
            x = x_base.clone()
            x[t_pert] = x[t_pert] + dx
            logits = net(x).squeeze(0)
            pred = int(torch.argmax(logits).item())
            flipped = (pred != y_true)
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_dx = dx.detach().clone()
                best_flipped = flipped
    return best_dx, best_flipped


def main():
    bundle = load_mnist_lstm("LSTM-1-32")
    net, D, H, L, C = build_torch_model(bundle)
    net.eval()
    n_test_samples = 3
    print(f"=== Red team: MNIST LSTM-1-32 ({n_test_samples} samples) ===")

    rng = np.random.default_rng(2026_05_14)
    summary = []
    for sample_idx in range(n_test_samples):
        x0 = bundle["X_test"][sample_idx]
        y_true = int(bundle["Y_test"][sample_idx])

        # Verify the model predicts the correct class on the nominal input.
        with torch.no_grad():
            nominal_pred = int(torch.argmax(net(torch.tensor(x0))[0]).item())
        if nominal_pred != y_true:
            print(f"\n  sample {sample_idx}: nominal pred {nominal_pred} != label {y_true}; skipping")
            continue

        # Bisect the certified radius per frame.
        eps_per_frame = np.zeros(x0.shape[0])
        for t in range(x0.shape[0]):
            from cert_rnn.verify import bisect_epsilon
            from cert_rnn.zono import Zono
            from cert_rnn.lstm import lstm_state_init, lstm_step_stack

            # Inline single-frame cert that's faster than reach (state precompute)
            # but uses cert engine.
            def _certifies(eps, t_p=t):
                z_h, z_c = lstm_state_init(H, L)
                T = x0.shape[0]
                for tt in range(T):
                    if tt == t_p and eps > 0:
                        z_x = Zono.from_box(x0[tt], eps)
                    else:
                        z_x = Zono.point(x0[tt])
                    z_h, z_c = lstm_step_stack(z_x, z_h, z_c, bundle["model"]["layers"])
                z_logits = z_h[-1].affine_map(bundle["model"]["head"]["W"], bundle["model"]["head"]["b"])
                others = [c for c in range(C) if c != y_true]
                diffs = np.zeros((len(others), C))
                for i, c in enumerate(others):
                    diffs[i, y_true] = 1.0
                    diffs[i, c] = -1.0
                lb, _ = z_logits.affine_map(diffs).get_ranges()
                return bool(np.all(lb > 0))

            eps_per_frame[t] = bisect_epsilon(_certifies, eps_init=0.5, n_iters=12)
        cert_radius = float(eps_per_frame.min())
        worst_frame = int(eps_per_frame.argmin())

        # Random sampling at the certified boundary
        n_sample = 1000
        T = x0.shape[0]
        flips_at = 0
        flips_just_over = 0
        for _ in range(n_sample):
            x = x0.copy()
            x[worst_frame] = x0[worst_frame] + cert_radius * (2 * rng.random(D) - 1)
            with torch.no_grad():
                pred = int(torch.argmax(net(torch.tensor(x))[0]).item())
            if pred != y_true:
                flips_at += 1
            # Sample at 1.5x the cert radius to see if the bound is tight.
            x_over = x0.copy()
            x_over[worst_frame] = x0[worst_frame] + 1.5 * cert_radius * (2 * rng.random(D) - 1)
            with torch.no_grad():
                pred = int(torch.argmax(net(torch.tensor(x_over))[0]).item())
            if pred != y_true:
                flips_just_over += 1

        # PGD inside the cert ball
        dx, pgd_flipped = pgd_attack(net, x0, y_true, worst_frame, cert_radius,
                                     n_steps=80, n_restarts=10)

        # PGD slightly past the cert ball (sanity: bound shouldn't be too loose)
        _, pgd_over = pgd_attack(net, x0, y_true, worst_frame, 1.5 * cert_radius,
                                 n_steps=80, n_restarts=10)

        result = {
            "sample": sample_idx,
            "y_true": y_true,
            "cert_radius": cert_radius,
            "worst_frame": worst_frame,
            "random_at_eps_flips": flips_at,
            "random_15x_eps_flips": flips_just_over,
            "pgd_at_eps_flipped": pgd_flipped,
            "pgd_15x_eps_flipped": pgd_over,
        }
        summary.append(result)
        print(
            f"\n  sample {sample_idx} (label={y_true}): cert_radius={cert_radius:.5f} at t={worst_frame}\n"
            f"    random @ eps:    {flips_at}/{n_sample} flips    -- soundness (must be 0)\n"
            f"    random @ 1.5eps: {flips_just_over}/{n_sample} flips   -- tightness probe\n"
            f"    PGD @ eps:    {'FLIPPED (UNSOUND)' if pgd_flipped else 'no flip (cert holds)'}\n"
            f"    PGD @ 1.5eps: {'FLIPPED' if pgd_over else 'no flip (bound is loose)'}"
        )

    print("\n=== Summary ===")
    unsound = sum(1 for r in summary if r["random_at_eps_flips"] > 0 or r["pgd_at_eps_flipped"])
    print(f"  Soundness violations: {unsound}/{len(summary)}")
    print(f"  Each sample tested with 1000 random + 10x80-step PGD inside the cert ball.")
    if unsound == 0:
        print("  -> No empirical evidence of unsoundness on these samples.")


if __name__ == "__main__":
    main()
