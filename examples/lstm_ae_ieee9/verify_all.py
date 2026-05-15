"""Replicate verify_certrnn_lstm_ae.m for all four IEEE-9 LSTM-AE sizes.

For each size, runs Algorithm 1 bisection per frame, saves results to
results/certrnn_lstm_ae_<size>.npz, and prints a side-by-side comparison
against the MATLAB numbers (matlab_results/certrnn_lstm_ae_<size>.mat).

Usage:
    python verify_all.py                         # S, M, D (default; L is slow)
    python verify_all.py --sizes S               # one size
    python verify_all.py --sizes S M L D         # everything (L takes ~1h)
    python verify_all.py --frames 0 5 10         # subset of frames
    python verify_all.py --n-iters 12            # bisection depth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).parent))
from ae_loader import load_lstm_ae  # noqa: E402

from cert_rnn.verify import bisect_epsilon, spec_c_holds  # noqa: E402

THIS = Path(__file__).parent
RESULTS_DIR = THIS / "results"
MATLAB_RESULTS_DIR = THIS / "matlab_results"


def verify_one_size(
    size: str,
    frames: list[int] | None = None,
    eps_init: float = 0.5,
    n_iters: int = 12,
    verbose: bool = True,
) -> dict:
    m = load_lstm_ae(size)
    if frames is None:
        frames = list(range(m["T"]))
    eps_per_frame = np.zeros(m["T"])
    times_per_frame = np.zeros(m["T"])

    if verbose:
        print(
            f"\n=== size={size} H={m['H']} T={m['T']} D={m['D']} "
            f"tau={m['tau']} anchor_score={m['anchor_score']:.4f} ===\n"
            f"  bisecting {len(frames)} frame(s), {n_iters} iterations, eps_init={eps_init}"
        )

    for t_pert in frames:
        t0 = time.perf_counter()
        eps = bisect_epsilon(
            lambda eps, _t=t_pert: spec_c_holds(
                m["encoder"], m["decoder"], m["head"], m["anchor"], eps, m["tau"],
                threat_model="single_frame", t_pert=_t,
            ),
            eps_init=eps_init,
            n_iters=n_iters,
        )
        dt = time.perf_counter() - t0
        eps_per_frame[t_pert] = eps
        times_per_frame[t_pert] = dt
        if verbose:
            print(f"  frame {t_pert:2d}: certified eps = {eps:.6f}  ({dt:.2f}s)")

    cert_radius = float(eps_per_frame[frames].min())
    out = {
        "size": size,
        "cert_radius": cert_radius,
        "eps_per_frame": eps_per_frame,
        "times_per_frame": times_per_frame,
        "frames": np.array(frames, dtype=int),
        "tau": m["tau"],
        "anchor_score": m["anchor_score"],
        "anchor_index": m["anchor_index"],
        "T": m["T"], "D": m["D"], "H": m["H"],
        "total_time_s": float(times_per_frame[frames].sum()),
    }

    if verbose:
        print(
            f"\n  certified eps (min over frames): {cert_radius:.6f}\n"
            f"  total wall time: {out['total_time_s']:.1f}s"
        )

    return out


def diff_against_matlab(out: dict, mat_path: Path, atol: float) -> None:
    if not mat_path.exists():
        print(f"  [no MATLAB reference at {mat_path}; skipping diff]")
        return
    m = scipy.io.loadmat(str(mat_path))
    mlab_eps = m["eps_per_frame"].flatten().astype(np.float64)
    mlab_cert = float(m["cert_radius"].item())
    diff = out["eps_per_frame"] - mlab_eps
    max_abs = float(np.max(np.abs(diff)))
    print(
        f"\n  vs MATLAB:  cert_radius py={out['cert_radius']:.6f} "
        f"matlab={mlab_cert:.6f}\n"
        f"              max |eps_per_frame - matlab| = {max_abs:.3e}  "
        f"(atol={atol:.0e})  "
        f"=> {'PASS' if max_abs <= atol else 'FAIL'}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sizes", nargs="+", default=["S", "M", "D"],
                   choices=["S", "M", "L", "D"], help="sizes to verify (default: S M D)")
    p.add_argument("--frames", nargs="+", type=int, default=None,
                   help="frame indices to bisect; default = all 30")
    p.add_argument("--n-iters", type=int, default=12)
    p.add_argument("--eps-init", type=float, default=0.5)
    p.add_argument("--atol", type=float, default=1e-6,
                   help="tolerance for the MATLAB diff")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)
    RESULTS_DIR.mkdir(exist_ok=True)

    for size in args.sizes:
        out = verify_one_size(
            size,
            frames=args.frames,
            eps_init=args.eps_init,
            n_iters=args.n_iters,
        )
        diff_against_matlab(out, MATLAB_RESULTS_DIR / f"certrnn_lstm_ae_{size}.mat", args.atol)
        if not args.no_save:
            save_path = RESULTS_DIR / f"certrnn_lstm_ae_{size}.npz"
            np.savez(save_path, **{k: v for k, v in out.items() if k != "size"})
            print(f"  saved {save_path.relative_to(THIS.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
