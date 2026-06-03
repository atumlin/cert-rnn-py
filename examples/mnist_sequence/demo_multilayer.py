"""Replicate demo_multilayer_mnist.m on the Python Cert-RNN engine.

For each MNIST LSTM config:
  1. Parity check at eps=0 vs saved PyTorch logits (max |cert.c - python|).
  2. Algorithm 1 cert radius (single-frame, min over T frames) on
     n_samples test images; reports mean/std/min/max.
  3. Side-by-side comparison against the MATLAB summary.

State precomputation: for single-frame at frame t_pert, the unperturbed
forward up to t_pert-1 is reused across all bisection trials and across
all t_pert >= cached frame. Mirrors the MATLAB optimization in
cert_radius_singleframe.

Usage:
    python demo_multilayer.py                              # default n=5, LSTM-1-32 only
    python demo_multilayer.py --configs LSTM-1-32 LSTM-2-32
    python demo_multilayer.py --configs all --n-samples 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).parent))
from mnist_loader import CONFIG_FILES, load_mnist_lstm  # noqa: E402

from cert_rnn.lstm import lstm_state_init, lstm_step_stack  # noqa: E402
from cert_rnn.verify import bisect_epsilon  # noqa: E402
from cert_rnn.zono import Zono  # noqa: E402

THIS = Path(__file__).parent
RESULTS_DIR = THIS / "results"
MATLAB_RESULTS_DIR = THIS / "matlab_results"


def _precompute_states(model: dict, x_seq: np.ndarray) -> tuple[list, list]:
    """Unperturbed forward; snapshot (h_layers, c_layers) before every step."""
    H, L = model["H"], model["L"]
    z_h, z_c = lstm_state_init(H, L)
    h_states = [z_h]
    c_states = [z_c]
    for t in range(x_seq.shape[0]):
        z_x = Zono.point(x_seq[t])
        z_h, z_c = lstm_step_stack(z_x, z_h, z_c, model["layers"])
        h_states.append(z_h)
        c_states.append(z_c)
    return h_states, c_states


# NOTE: cert_rnn.certify_radius_spec_a is the high-level entry point for this
# exact spec (classifier margin, Algorithm 1, min over frames) -- see
# examples/demo_lstm_cell.py for the one-call usage. This harness keeps its
# own forward below only to reuse the precomputed unperturbed prefix states
# (h_init/c_init), which mirrors the MATLAB cert_radius_singleframe timing
# optimization being reproduced here. Use the high-level API for new code.
def _certifies(
    model: dict,
    x_seq: np.ndarray,
    eps: float,
    correct_class: int,
    t_pert: int,
    h_init: list,
    c_init: list,
) -> bool:
    z_h, z_c = h_init, c_init
    T = x_seq.shape[0]
    for t in range(t_pert, T):
        if t == t_pert and eps > 0:
            z_x = Zono.from_box(x_seq[t], eps)
        else:
            z_x = Zono.point(x_seq[t])
        z_h, z_c = lstm_step_stack(z_x, z_h, z_c, model["layers"])
    z_logits = z_h[-1].affine_map(model["head"]["W"], model["head"]["b"])
    C = z_logits.dim
    others = [c for c in range(C) if c != correct_class]
    diffs = np.zeros((len(others), C))
    for i, c in enumerate(others):
        diffs[i, correct_class] = 1.0
        diffs[i, c] = -1.0
    lb, _ = z_logits.affine_map(diffs).get_ranges()
    return bool(np.all(lb > 0))


def cert_radius_singleframe(model: dict, x_seq: np.ndarray, y_label: int) -> float:
    """Algorithm 1: per-frame bisection, min over frames."""
    T = x_seq.shape[0]
    h_states, c_states = _precompute_states(model, x_seq)
    eps_per_frame = np.zeros(T)
    for t_pert in range(T):
        eps_per_frame[t_pert] = bisect_epsilon(
            lambda eps, _t=t_pert: _certifies(
                model, x_seq, eps, y_label, _t, h_states[_t], c_states[_t]
            ),
            eps_init=0.5,
            n_iters=12,
        )
    return float(eps_per_frame.min())


def perstep_widths(model: dict, x_seq: np.ndarray, eps: float, t_pert: int) -> np.ndarray:
    """Per-timestep max(ub - lb) of the top-layer hidden state when one
    frame is perturbed by eps. Mirrors MATLAB perstep_widths(...) for
    direct numerical comparison."""
    H, L = model["H"], model["L"]
    z_h, z_c = lstm_state_init(H, L)
    T = x_seq.shape[0]
    widths = np.zeros(T)
    for t in range(T):
        if t == t_pert and eps > 0:
            z_x = Zono.from_box(x_seq[t], eps)
        else:
            z_x = Zono.point(x_seq[t])
        z_h, z_c = lstm_step_stack(z_x, z_h, z_c, model["layers"])
        lb, ub = z_h[-1].get_ranges()
        widths[t] = float(np.max(ub - lb))
    return widths


def parity_at_eps_zero(bundle: dict, n_parity: int) -> float:
    """Max abs diff between cert forward (eps=0) and saved PyTorch logits."""
    model = bundle["model"]
    n = min(n_parity, len(bundle["X_test"]))
    max_err = 0.0
    for i in range(n):
        x = bundle["X_test"][i]
        z_h, z_c = lstm_state_init(model["H"], model["L"])
        for t in range(x.shape[0]):
            z_h, z_c = lstm_step_stack(Zono.point(x[t]), z_h, z_c, model["layers"])
        logits = (
            model["head"]["W"] @ z_h[-1].c + model["head"]["b"]
        )
        ref = bundle["python_predictions"][i]
        max_err = max(max_err, float(np.max(np.abs(logits - ref))))
    return max_err


def run_config(config: str, n_samples: int, n_parity: int = 5, verbose: bool = True) -> dict:
    bundle = load_mnist_lstm(config)
    model = bundle["model"]

    if verbose:
        print(
            f"\n=== {config} ===  L={model['L']} H={model['H']} D={model['D']} T={bundle['T']}"
        )

    parity = parity_at_eps_zero(bundle, n_parity)
    if verbose:
        print(f"  parity (Cert vs PyTorch, eps=0) over {n_parity} samples: max {parity:.2e}")

    n = min(n_samples, len(bundle["X_test"]))
    radii = np.zeros(n)
    t0 = time.perf_counter()
    for i in range(n):
        radii[i] = cert_radius_singleframe(model, bundle["X_test"][i], int(bundle["Y_test"][i]))
        if verbose:
            print(f"  [{i+1:2d}/{n}] radius = {radii[i]:.4f}  running mean = {radii[:i+1].mean():.4f}")
    total_t = time.perf_counter() - t0

    out = {
        "config": config,
        "L": model["L"], "H": model["H"],
        "n_samples": n,
        "mean_radius": float(radii.mean()),
        "std_radius": float(radii.std(ddof=1)) if n > 1 else 0.0,
        "min_radius": float(radii.min()),
        "max_radius": float(radii.max()),
        "time_s_total": total_t,
        "time_s_per_sample": total_t / n,
        "parity_err": parity,
        "radii": radii,
    }
    if verbose:
        print(
            f"  certified radius: mean={out['mean_radius']:.4f}  std={out['std_radius']:.4f}  "
            f"min={out['min_radius']:.4f}  max={out['max_radius']:.4f}\n"
            f"  time / sample:    {out['time_s_per_sample']:.1f}s"
        )
    return out


def matlab_row(config: str) -> dict | None:
    path = MATLAB_RESULTS_DIR / "demo_multilayer_results.mat"
    if not path.exists():
        return None
    s = scipy.io.loadmat(str(path))["summary"]
    names = [str(n[0]) for n in s["name"][0, 0].flatten()]
    if config not in names:
        return None
    i = names.index(config)
    return {
        "mean_radius": float(s["mean_radius"][0, 0][i, 0]),
        "std_radius": float(s["std_radius"][0, 0][i, 0]),
        "min_radius": float(s["min_radius"][0, 0][i, 0]),
        "max_radius": float(s["max_radius"][0, 0][i, 0]),
        "time_s_per_sample": float(s["time_s"][0, 0][i, 0]),
        "parity_err": float(s["parity_err"][0, 0][i, 0]),
    }


def print_diff(out: dict, mlab: dict | None) -> None:
    if mlab is None:
        return
    print(f"\n  vs MATLAB (n_samples likely differs unless --n-samples matches):")
    print(f"    mean: py={out['mean_radius']:.4f}  matlab={mlab['mean_radius']:.4f}  diff={out['mean_radius']-mlab['mean_radius']:+.4f}")
    print(f"    min:  py={out['min_radius']:.4f}  matlab={mlab['min_radius']:.4f}")
    print(f"    max:  py={out['max_radius']:.4f}  matlab={mlab['max_radius']:.4f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--configs", nargs="+", default=["LSTM-1-32"],
                   help="configs to run; 'all' expands to every available")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--n-parity", type=int, default=5)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    configs = (list(CONFIG_FILES) if args.configs == ["all"] else args.configs)
    RESULTS_DIR.mkdir(exist_ok=True)
    all_out = {}
    for cfg in configs:
        out = run_config(cfg, args.n_samples, args.n_parity)
        all_out[cfg] = out
        print_diff(out, matlab_row(cfg))
        if not args.no_save:
            save = RESULTS_DIR / f"{cfg.lower().replace('-', '_')}_results.npz"
            np.savez(save, **{k: v for k, v in out.items() if k != "config"})
            print(f"  saved {save.relative_to(THIS.parent.parent)}")

    print("\n=== summary ===")
    print(f"  {'config':<10}  L  H   mean_r    std       min       max       sec/samp   parity")
    for cfg, out in all_out.items():
        print(
            f"  {cfg:<10}  {out['L']}  {out['H']:2d}  "
            f"{out['mean_radius']:.4f}    {out['std_radius']:.4f}    "
            f"{out['min_radius']:.4f}    {out['max_radius']:.4f}    "
            f"{out['time_s_per_sample']:6.1f}     {out['parity_err']:.2e}"
        )
    print("  paper LSTM-1-32 ref: mean=0.0187 std=0.0087")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
