"""Replicate reach_set_spec_c.m: at a fixed eps, scan each frame and report
the per-(t, d) reach-set bounds on x_hat and diff, plus the sound score_ub.

Usage:
    python reach_set_eps.py --size S --eps 0.01
    python reach_set_eps.py --size M --eps 0.025 --frames 0 5 10
    python reach_set_eps.py --size S --eps 0.01 --topk 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ae_loader import load_lstm_ae  # noqa: E402

from cert_rnn.verify import lstm_ae_reach, spec_c_score_ub  # noqa: E402
from cert_rnn.zono import zono_sub  # noqa: E402

THIS = Path(__file__).parent
RESULTS_DIR = THIS / "results"


def forward_and_bound(m: dict, t_pert: int, eps: float) -> dict:
    T, D = m["T"], m["D"]
    z_xh_seq, z_x_seq = lstm_ae_reach(
        m["encoder"], m["decoder"], m["head"], m["anchor"], eps,
        threat_model="single_frame", t_pert=t_pert,
    )
    xhat_lb = np.zeros((T, D))
    xhat_ub = np.zeros((T, D))
    diff_lb = np.zeros((T, D))
    diff_ub = np.zeros((T, D))
    for t in range(T):
        lb, ub = z_xh_seq[t].get_ranges()
        xhat_lb[t], xhat_ub[t] = lb, ub
        z_diff = zono_sub(z_xh_seq[t], z_x_seq[t])
        radius = np.sum(np.abs(z_diff.V), axis=1)
        diff_lb[t] = z_diff.c - radius
        diff_ub[t] = z_diff.c + radius
    score_ub = spec_c_score_ub(z_xh_seq, z_x_seq)
    return {
        "xhat_lb": xhat_lb, "xhat_ub": xhat_ub,
        "diff_lb": diff_lb, "diff_ub": diff_ub,
        "score_ub": score_ub,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--size", required=True, choices=["S", "M", "L", "D"])
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--frames", nargs="+", type=int, default=None)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    m = load_lstm_ae(args.size)
    if args.frames is None:
        frames = list(range(m["T"]))
    else:
        frames = args.frames
    print(
        f"\n=== reach-set + Spec C verdict ===\n"
        f"  size={args.size} eps={args.eps} tau={m['tau']} "
        f"anchor_score={m['anchor_score']:.4f} (margin={m['tau']-m['anchor_score']:.4f})\n"
        f"  D={m['D']} T={m['T']} H={m['H']}  scanning {len(frames)} frame(s)\n"
    )

    score_ub_per_frame = np.zeros(len(frames))
    diff_max_per_frame = np.zeros(len(frames))
    worst_idx = -1
    worst = None
    t0 = time.perf_counter()
    for fi, t_pert in enumerate(frames):
        r = forward_and_bound(m, t_pert, args.eps)
        score_ub_per_frame[fi] = r["score_ub"]
        diff_max_per_frame[fi] = float(
            np.max(np.abs(np.concatenate([r["diff_lb"].ravel(), r["diff_ub"].ravel()])))
        )
        verdict = "VERIFIED" if r["score_ub"] <= m["tau"] else "NOT VERIFIED"
        print(f"  frame {t_pert:2d}: score_ub = {r['score_ub']:.4f}  (tau = {m['tau']})  {verdict}")
        if worst is None or r["score_ub"] > worst["score_ub"]:
            worst = r
            worst_idx = t_pert
    dt = time.perf_counter() - t0

    n_verified = int(np.sum(score_ub_per_frame <= m["tau"]))
    print(
        f"\n=== verdict summary ===\n"
        f"  frames verified: {n_verified} / {len(frames)}\n"
        f"  Spec C at eps={args.eps}: {'VERIFIED' if n_verified == len(frames) else 'NOT VERIFIED'}\n"
        f"  worst score_ub: {float(score_ub_per_frame.max()):.4f} at t*={worst_idx} (tau={m['tau']})\n"
        f"  worst |diff|:   {float(diff_max_per_frame.max()):.4f}\n"
        f"  wall time:      {dt:.2f}s"
    )

    abs_diff_box = np.maximum(np.abs(worst["diff_lb"]), np.abs(worst["diff_ub"]))
    flat_idx = np.argsort(-abs_diff_box.ravel())
    print(f"\n  top {args.topk} worst-cell |diff| from frame t*={worst_idx}:")
    print("   rank   t   d   |diff|_max   xhat range          diff range")
    for rk in range(min(args.topk, flat_idx.size)):
        t_idx, d_idx = np.unravel_index(flat_idx[rk], abs_diff_box.shape)
        print(
            f"   {rk+1:4d}  {t_idx:3d} {d_idx:3d}  {abs_diff_box[t_idx, d_idx]:10.4f}   "
            f"[{worst['xhat_lb'][t_idx, d_idx]:+.3f}, {worst['xhat_ub'][t_idx, d_idx]:+.3f}]   "
            f"[{worst['diff_lb'][t_idx, d_idx]:+.3f}, {worst['diff_ub'][t_idx, d_idx]:+.3f}]"
        )

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / f"reach_{args.size}_eps{args.eps:g}.npz"
        np.savez(
            out_path,
            size=args.size, eps=args.eps, tau=m["tau"], anchor_score=m["anchor_score"],
            frames=np.array(frames, dtype=int),
            score_ub_per_frame=score_ub_per_frame,
            diff_max_per_frame=diff_max_per_frame,
            worst_frame=worst_idx,
            worst_xhat_lb=worst["xhat_lb"], worst_xhat_ub=worst["xhat_ub"],
            worst_diff_lb=worst["diff_lb"], worst_diff_ub=worst["diff_ub"],
            worst_score_ub=worst["score_ub"],
        )
        print(f"\n  saved {out_path.relative_to(THIS.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
