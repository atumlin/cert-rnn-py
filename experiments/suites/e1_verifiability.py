"""Suite E1 -- verifiability: certified-radius quality across tools.

Question: at equal bisection budget, how large a radius does each sound
domain certify, and how close is each to the PGD empirical ceiling?

Grid: IEEE-9 S/M/L/D (the paper's headline table) x
      {certrnn-zono, interval-ibp, lirpa-ibp, lirpa-backward} + pgd.
Single-frame threat model, eps_init=0.5, 12 iterations -- identical to
docs/lstm_ae_results.md so the MATLAB reference column can be reused.

    python -m experiments.suites.e1_verifiability                # full
    python -m experiments.suites.e1_verifiability --sizes S --frames 0 5
"""

from __future__ import annotations

import argparse

from experiments.common import pin_blas_single_thread


def main(argv=None):
    pin_blas_single_thread()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sizes", nargs="+", default=["S", "M", "L", "D"],
                   choices=["S", "M", "L", "D"])
    p.add_argument("--tools", nargs="+",
                   default=["certrnn-zono", "interval-ibp", "lirpa-ibp",
                            "lirpa-backward", "pgd"])
    p.add_argument("--frames", nargs="+", type=int, default=None)
    p.add_argument("--n-iters", type=int, default=12)
    p.add_argument("--eps-init", type=float, default=0.5)
    p.add_argument("--timeout", type=float, default=3600.0,
                   help="per (tool, case) budget, seconds")
    p.add_argument("--tag", default="")
    p.add_argument("--smd", action="store_true",
                   help="also run the SMD entity benchmark (needs data)")
    args = p.parse_args(argv)

    from experiments.benchmarks import ieee9, public_tsad
    from experiments.suites._runner import run_matrix

    cases = ieee9.get_cases(args.sizes)
    if args.smd:
        cases += [c for c in public_tsad.get_cases()
                  if c.meta["label"] == "normal"]
    run_matrix("e1", args.tools, cases, frames=args.frames,
               eps_init=args.eps_init, n_iters=args.n_iters,
               timeout_s=args.timeout, tag=args.tag)


if __name__ == "__main__":
    main()
