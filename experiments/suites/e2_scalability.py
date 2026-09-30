"""Suite E2 -- scalability: wall-clock vs model size, one axis at a time.

Question: how does each tool's certify cost grow with hidden width H,
sequence length T, input dim D, and depth L -- and where does each tool
time out? Produces the data for scaling curves and a VNN-COMP-style
cactus plot (instances solved vs time).

Design: synthetic family around center (H16, T30, D9, L1); each sweep
varies ONE axis. Per point we certify a fixed small frame subset (default
3 frames) so the cost of a cell is bounded and comparable across points;
per-reach cost is what scales, and the frame subset multiplies it
uniformly for every tool.

    python -m experiments.suites.e2_scalability                  # all sweeps
    python -m experiments.suites.e2_scalability --sweep H --quick
"""

from __future__ import annotations

import argparse

from experiments.common import pin_blas_single_thread

SWEEPS = {
    "H": [4, 8, 16, 32, 64, 128],
    "T": [10, 20, 30, 60, 120],
    "D": [9, 18, 36, 72],
    "L": [1, 2, 3],
}
CENTER = {"H": 16, "T": 30, "D": 9, "L": 1}


def main(argv=None):
    pin_blas_single_thread()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep", nargs="+", default=list(SWEEPS),
                   choices=list(SWEEPS))
    p.add_argument("--tools", nargs="+",
                   default=["certrnn-zono", "interval-ibp", "lirpa-ibp",
                            "lirpa-backward"])
    p.add_argument("--frames", nargs="+", type=int, default=[0, 14, 29],
                   help="frame subset per point (clipped to T-1)")
    p.add_argument("--n-iters", type=int, default=8,
                   help="fewer iterations than E1: E2 measures cost growth, "
                        "and cost per iteration is eps-independent")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="first 3 points per sweep, 1 frame, 1 anchor")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    from experiments.benchmarks.synthetic import SynthSpec
    from experiments.benchmarks import synthetic
    from experiments.suites._runner import run_matrix

    for axis in args.sweep:
        values = SWEEPS[axis][:3] if args.quick else SWEEPS[axis]
        for v in values:
            shape = dict(CENTER)
            shape[axis] = v
            spec = SynthSpec(H=shape["H"], T=shape["T"], D=shape["D"],
                             L=shape["L"], seed=args.seed)
            cases = synthetic.get_cases(spec, "normal",
                                        n_anchors=1 if args.quick else 2)
            frames = [f for f in args.frames if f < shape["T"]]
            if args.quick:
                frames = frames[:1]
            run_matrix(
                "e2", args.tools, cases,
                frames=frames, n_iters=args.n_iters,
                timeout_s=args.timeout,
                tag=f"{args.tag + '_' if args.tag else ''}sweep{axis}",
            )


if __name__ == "__main__":
    main()
