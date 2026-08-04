"""Suite E3 -- certified anomaly detection: both sides of the threshold.

The detector-level guarantees an operator actually wants:

  false-alarm robustness   normal window, certify score_ub <= tau over
                           the ball (no adversarial false positive);
  masking robustness       anomalous window, certify score_lb >= tau
                           over the ball (no adversarial false negative
                           -- an attacker cannot HIDE the anomaly).

From per-window certified radii the tabulator derives certified-robust
FPR/TPR at any target eps: the fraction of windows whose certified
radius covers eps. PGD (both directions) upper-bounds each radius.

    python -m experiments.suites.e3_detection                # synthetic
    python -m experiments.suites.e3_detection --smd          # + SMD entity
"""

from __future__ import annotations

import argparse

from experiments.common import pin_blas_single_thread


def main(argv=None):
    pin_blas_single_thread()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tools", nargs="+",
                   default=["certrnn-zono", "interval-ibp"],
                   help="base tool names; -masking variants auto-derived")
    p.add_argument("--n-anchors", type=int, default=8)
    p.add_argument("--n-iters", type=int, default=10)
    p.add_argument("--threat-model", default="multi_frame",
                   choices=["single_frame", "multi_frame"],
                   help="multi_frame: attacker perturbs the whole window -- "
                        "the operationally meaningful detector threat model")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smd", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    from experiments.benchmarks import public_tsad, synthetic
    from experiments.benchmarks.synthetic import SynthSpec
    from experiments.suites._runner import run_matrix

    spec = SynthSpec(seed=args.seed)
    normal = synthetic.get_cases(spec, "normal", n_anchors=args.n_anchors)
    anom = synthetic.get_cases(spec, "anom", n_anchors=args.n_anchors)
    if args.smd:
        smd = public_tsad.get_cases(n_normal=args.n_anchors,
                                    n_anom=args.n_anchors)
        normal += [c for c in smd if c.meta["label"] == "normal"]
        anom += [c for c in smd if c.meta["label"] == "anom"]

    # keep only anchors on the correct side of tau (certifiable subjects)
    run_matrix("e3", args.tools + ["pgd"], normal,
               threat_model=args.threat_model, n_iters=args.n_iters,
               timeout_s=args.timeout, spec_label="specC<=tau",
               tag=f"{args.tag + '_' if args.tag else ''}normal")
    run_matrix("e3", [t + "-masking" for t in args.tools] + ["pgd-masking"],
               anom, threat_model=args.threat_model, n_iters=args.n_iters,
               timeout_s=args.timeout, spec_label="specC>=tau",
               tag=f"{args.tag + '_' if args.tag else ''}anom")


if __name__ == "__main__":
    main()
