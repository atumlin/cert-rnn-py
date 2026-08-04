"""Suite E4 -- tightness + ablations on the tool under study.

  (a) Certified-vs-empirical gap: certrnn-zono radius vs PGD upper bound
      per frame (how much radius does soundness cost?).
  (b) Bisection depth: radius/time vs n_iters in {6, 9, 12, 15}.
  (c) Threat model: single_frame vs multi_frame radii on the same anchor.

All on IEEE-9 (S and D by default: the byte-exact case and the
multi-layer case where the MATLAB reference is aliasing-tightened).

    python -m experiments.suites.e4_tightness --quick
"""

from __future__ import annotations

import argparse

from experiments.common import pin_blas_single_thread


def main(argv=None):
    pin_blas_single_thread()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sizes", nargs="+", default=["S", "D"],
                   choices=["S", "M", "L", "D"])
    p.add_argument("--frames", nargs="+", type=int, default=None)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    from experiments.benchmarks import ieee9
    from experiments.suites._runner import run_matrix

    cases = ieee9.get_cases(args.sizes)
    frames = args.frames if args.frames is not None else (
        [0, 15, 29] if args.quick else None)

    # (a) gap
    run_matrix("e4", ["certrnn-zono", "pgd"], cases, frames=frames,
               timeout_s=args.timeout,
               tag=f"{args.tag + '_' if args.tag else ''}gap")
    # (b) bisection depth
    for n in ([9, 12] if args.quick else [6, 9, 12, 15]):
        run_matrix("e4", ["certrnn-zono"], cases, frames=frames, n_iters=n,
                   timeout_s=args.timeout,
                   tag=f"{args.tag + '_' if args.tag else ''}iters{n}")
    # (c) threat model
    run_matrix("e4", ["certrnn-zono"], cases, threat_model="multi_frame",
               timeout_s=args.timeout,
               tag=f"{args.tag + '_' if args.tag else ''}multiframe")


if __name__ == "__main__":
    main()
