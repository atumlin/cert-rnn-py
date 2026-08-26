#!/usr/bin/env python3
"""Detached Cert-RNN+ verification driver for the Zone-A LSTM-AE (v2).

Runs the heavy certification stages outside the notebook so an SSH drop
costs nothing: launch it inside tmux, detach, come back later, load
`verify_results_v2.json` in the notebook and draw the figures.

    tmux new -s certify
    python verify_zoneA_v2.py --job verify_job_v2.pkl \
        --out verify_results_v2.json --workers 12 2>&1 | tee -a verify_v2.log
    # Ctrl-b d to detach;  tmux attach -t certify  to return

The job file is a numpy-only pickle written by the notebook
({encoder, decoder, head, anchor, tau}) -- this script needs neither
torch nor the notebook's model classes.

Stages (each checkpointed to --out as soon as it finishes; rerunning
skips completed stages, so the script is resume-safe):

  box_multi          baseline Cert-RNN radius, multi_frame (previous strategy)
  zono_multi         Cert-RNN+ ZRLT Tier 1 radius, multi_frame
  zono_joint_multi   Tier 1 + joint quadratic score bound, multi_frame
  curves             sound score-vs-eps curves under both gate modes
  zono_single_frame  Tier 1 per-frame radii (parallel k-ary search)

The refined stages run with the SPEED-UPS ON BY DEFAULT: the search
probes in cheap box mode for the early rounds and switches to Tier-1
zono only for the last --zono-last-rounds rounds (default 3), where the
radius is actually decided. Both gate modes are sound, so every mixed
schedule certifies only true radii; the result lands between the
all-box and the all-zono radius (and is never below all-box). Pass
--zono-last-rounds -1 for pure Tier-1 everywhere (slow but tightest);
the single-frame stage additionally uses the parallel k-ary probe walk
(--workers). The multi_frame stages print one line per probe -- the
tmux heartbeat.
"""

# Pin BLAS to one thread BEFORE importing numpy: cert-rnn does many small
# ops; parallelism comes from the probe workers, not BLAS.
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import pickle
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# cert_rnn need not be installed in this interpreter: this script lives in
# <repo>/examples/zoneA_lstm_ae/, so <repo>/src is two levels up.
try:
    import cert_rnn  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import cert_rnn  # noqa: F401


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"meta": {}, "stages": {}}


def save_results(path: str, res: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, path)  # atomic: a crash never truncates the results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", default="verify_job_v2.pkl")
    ap.add_argument("--out", default="verify_results_v2.json")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--eps-init", type=float, default=0.5)
    ap.add_argument("--n-iters", type=int, default=12)
    ap.add_argument("--zono-last-rounds", type=int, default=3,
                    help="Tier-1 cutover: only the last N search rounds run "
                         "in zono mode (sound; the speed-up). -1 = pure "
                         "Tier-1 in every round (slow but tightest)")
    ap.add_argument("--stages", default="all",
                    help="comma list from: box_multi,zono_multi,"
                         "zono_joint_multi,curves,zono_single_frame")
    ap.add_argument("--single-search", choices=["bisect", "kary"],
                    default="bisect",
                    help="single-frame probe walk. bisect (default): "
                         "13 probes/frame, frames run in parallel across "
                         "--workers — best when workers <~ T. kary: 46 "
                         "probes/frame in 4 rounds — only pays when the "
                         "pool dwarfs the probe demand (workers >> T)")
    args = ap.parse_args()

    from cert_rnn import LSTMAutoencoder, ReconErrorSpec
    from cert_rnn.transformers import bilinear_mode
    from cert_rnn.verify import certify_radius_spec_c, spec_c_score_ub_joint

    with open(args.job, "rb") as f:
        job = pickle.load(f)
    ae = LSTMAutoencoder(job["encoder"], job["decoder"], job["head"])
    anchor = np.asarray(job["anchor"], dtype=np.float64)
    tau = float(job["tau"])
    T, D = anchor.shape

    res = load_results(args.out)
    res["meta"].update({
        "host": socket.gethostname(), "workers": args.workers,
        "eps_init": args.eps_init, "n_iters": args.n_iters,
        "zono_last_rounds": args.zono_last_rounds,
        "tau": tau, "T": T, "D": D, "H": ae.H,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    S = res["stages"]
    wanted = (["box_multi", "zono_multi", "zono_joint_multi", "curves",
               "zono_single_frame"] if args.stages == "all"
              else [s.strip() for s in args.stages.split(",")])

    def done(name: str) -> bool:
        if name in S:
            log(f"stage {name}: already done (radius="
                f"{S[name].get('radius')}), skipping")
            return True
        return False

    def finish(name: str, payload: dict, t0: float) -> None:
        payload["seconds"] = round(time.perf_counter() - t0, 2)
        payload["done_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        S[name] = payload
        save_results(args.out, res)
        log(f"stage {name}: DONE {payload}")

    @dataclass(frozen=True)
    class JointReconErrorSpec:
        """Spec C via the joint quadratic score bound (never looser)."""
        tau: float

        def holds(self, ae_output) -> bool:
            z_xh, z_x = ae_output
            return spec_c_score_ub_joint(z_xh, z_x) <= self.tau

    def certify_multi(spec) -> "object":
        # spec-level serial certify: multi_frame bisection is inherently
        # sequential, and the progress callback is the tmux heartbeat
        return ae.certify(anchor, spec, threat_model="multi_frame",
                          eps_init=args.eps_init, n_iters=args.n_iters,
                          progress=log)

    zono_last = args.zono_last_rounds
    if zono_last is not None and zono_last < 0:
        zono_last = None                      # pure Tier-1 everywhere

    def certify_multi_cutover(spec) -> float:
        """Algorithm-1 bisection (multi_frame) with the sound box->zono
        cutover: rounds before the last `zono_last` probe in cheap box
        mode, the final rounds in Tier-1 zono. Sound for any schedule
        (both modes are certified bounds); result >= the all-box radius
        and <= the all-zono radius."""
        eps, step, best = args.eps_init, args.eps_init, 0.0
        total = args.n_iters + 1
        for i in range(total):
            mode = ("zono" if zono_last is None or i >= total - zono_last
                    else "box")
            t0 = time.perf_counter()
            with bilinear_mode(mode):
                ok = spec.holds(ae.reach(anchor, eps, "multi_frame", None))
            log(f"probe {i + 1}/{total} [{mode}] eps={eps:.6g} -> "
                f"{'holds' if ok else 'fails'} "
                f"({time.perf_counter() - t0:.1f}s)")
            if ok:
                best = max(best, eps)
                eps += step / 2
            else:
                eps -= step / 2
            step /= 2
        return best

    for name in wanted:
        if done(name):
            continue
        log(f"stage {name}: START")
        t0 = time.perf_counter()

        if name == "box_multi":
            r = certify_multi(ReconErrorSpec(tau))
            finish(name, {"radius": r.radius}, t0)

        elif name == "zono_multi":
            radius = certify_multi_cutover(ReconErrorSpec(tau))
            finish(name, {"radius": radius,
                          "zono_last_rounds": zono_last}, t0)

        elif name == "zono_joint_multi":
            radius = certify_multi_cutover(JointReconErrorSpec(tau))
            finish(name, {"radius": radius,
                          "zono_last_rounds": zono_last}, t0)

        elif name == "curves":
            need = [s for s in ("box_multi", "zono_multi") if s not in S]
            if need:
                log(f"stage curves: needs {need} first, skipping")
                continue
            r_box, r_zono = S["box_multi"]["radius"], S["zono_multi"]["radius"]
            eps_grid = sorted(set(np.round(np.concatenate([
                np.linspace(0.25, 2.0, 8) * max(r_box, 1e-6),
                [r_box, r_zono]]), 10)))
            log(f"curves over {len(eps_grid)} eps points, both modes")
            curve_box = ae.score_vs_eps(anchor, eps_grid,
                                        threat_model="multi_frame")
            with bilinear_mode("zono"):
                curve_zono = ae.score_vs_eps(anchor, eps_grid,
                                             threat_model="multi_frame")
            finish(name, {
                "eps": [float(e) for e, _ in curve_box],
                "score_ub_box": [float(s) for _, s in curve_box],
                "score_ub_zono": [float(s) for _, s in curve_zono],
            }, t0)

        elif name == "zono_single_frame":
            # parallel per-frame walk + k-ary probes: this is where
            # --workers actually pays (radii bit-identical to serial)
            with bilinear_mode("zono"):
                radius, per_frame = certify_radius_spec_c(
                    ae.encoder, ae.decoder, ae.head, anchor, tau,
                    eps_init=args.eps_init, n_iters=args.n_iters,
                    threat_model="single_frame", n_workers=args.workers,
                    search=args.single_search, probes=15,
                    zono_last_rounds=zono_last)
            finish(name, {"radius": float(radius),
                          "per_frame": [float(x) for x in per_frame]}, t0)

        else:
            log(f"unknown stage {name!r}, skipping")

    log(f"all requested stages finished -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
