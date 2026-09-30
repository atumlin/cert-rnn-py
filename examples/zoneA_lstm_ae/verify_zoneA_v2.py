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

Default stages (the single-frame head-to-head, Cert-RNN+ first; both
run the SAME parallel per-frame walk and the SAME componentwise score
bound, so the comparison isolates the gate refinement):

  zono_single_frame  Cert-RNN+ ZRLT Tier 1 per-frame radii (cutover on)
  box_single_frame   Cert-RNN baseline per-frame radii

Extra stages via --stages (or --stages all):

  box_multi          baseline Cert-RNN radius, multi_frame
  zono_multi         Cert-RNN+ ZRLT Tier 1 radius, multi_frame
  zono_joint_multi   Tier 1 + joint quadratic score bound, multi_frame
  curves             sound score-vs-eps curves under both gate modes

Tier-1 stages run Tier-1 in EVERY probe by default (--zono-last-rounds
-1). --zono-last-rounds N (e.g. 3) is the optional speed-up: cheap box
probes in the early rounds, Tier-1 only in the last N; both gate modes
are sound, so the result lands between the all-box and all-zono radius.

Parallelism (radii bit-identical to the serial Algorithm 1):
  single_frame  every frame's whole bisection is one pool task
                (free-running, earliest = slowest frames first)
  multi_frame   serial bisection by default; --multi-search kary evaluates
                --probes eps values per round in parallel (13 serial probes
                -> 4 rounds at --probes 15). Measured: Tier-1 passes are
                memory-bandwidth bound, so kary was 1.7x faster at 1 layer
                but 0.8x (slower) at 2 layers on a 28-core machine
  inside probes --tier1-threads threads over neurons in the Tier-1 gate
                step (0 = auto: spare cores / tasks in flight, max 4)

Attacks (--attack, both modes):
  whole         every reading of every second within +-eps (worst case,
                covers gradual attacks)
  seconds       each single second on its own (spike attack)
  concentrated  the --seconds set jointly, e.g. --seconds 4,5,6
  all           whole + seconds (+ concentrated when --seconds is given)
In radius mode --attack picks the stage families (concentrated stages are
named box_/zono_/zono_joint_seconds_4-5-6); without it, --stages applies.

Fixed-eps CHECK mode (--eps 0.05, or a list --eps 0.01,0.05): no radius
search. ONE pass per window x attack x gate mode at exactly that eps, all
in parallel, each scored with both score bounds. A concrete attack at the
same eps labels every result:
  VERIFIED        sound score bound <= tau: no false alarm within +-eps
  COUNTEREXAMPLE  a real input within +-eps scores > tau; the input is
                  saved to <out>_cex/*.npy (--minimize-cex also saves the
                  fewest-readings version; --no-save-cex turns it off)
  UNKNOWN         bound > tau, no counterexample found
  UNSOUND         a real input beats a certified bound or radius -- a
                  soundness finding, reported loudly
Saved as stage "check_eps_<eps>" in --out.

Batch jobs: a job file with "anchors" (K, T, D) (+ optional "anchor_info")
runs every window; results go to res["anchors"][k]["stages"], with a
headline summary per eps (res["summary"]) or per radius stage
(res["summary_radius"]). Windows already above tau at eps=0 are false
alarms as they stand and are reported, not checked.
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


_CK: dict = {}

ATTACK_FAMILIES = ("whole", "seconds", "concentrated")


def parse_attacks(attack_arg, seconds_arg, T):
    """-> list of attack items: ("whole", None), ("seconds", t) for every t,
    ("concentrated", (s1, s2, ...))."""
    fams = (["whole", "seconds"] + (["concentrated"] if seconds_arg else [])
            if attack_arg in (None, "all") else
            [a.strip() for a in attack_arg.split(",")])
    bad = [a for a in fams if a not in ATTACK_FAMILIES]
    if bad:
        raise SystemExit(f"unknown --attack {bad}; choose from {ATTACK_FAMILIES} or all")
    secs = None
    if seconds_arg:
        secs = tuple(sorted({int(s) for s in str(seconds_arg).split(",")}))
        if secs[0] < 0 or secs[-1] >= T:
            raise SystemExit(f"--seconds {secs} out of range [0, {T})")
    if "concentrated" in fams and not secs:
        raise SystemExit("--attack concentrated needs --seconds, e.g. --seconds 4,5,6")
    items = []
    if "whole" in fams:
        items.append(("whole", None))
    if "seconds" in fams:
        items += [("seconds", t) for t in range(T)]
    if "concentrated" in fams:
        items.append(("concentrated", secs))
    return items


def _spec(item):
    """attack item -> (threat_model, t_pert)."""
    fam, a = item
    if fam == "whole":
        return "multi_frame", None
    if fam == "seconds":
        return "single_frame", a
    return "frame_set", a


def _mask(item, shape):
    fam, a = item
    m = np.zeros(shape)
    if fam == "whole":
        m[:] = 1.0
    elif fam == "seconds":
        m[a] = 1.0
    else:
        m[list(a)] = 1.0
    return m


def _label(item):
    fam, a = item
    return ("whole" if fam == "whole" else f"sec{a}" if fam == "seconds"
            else "secs" + "-".join(map(str, a)))


def _check_init(payload: dict, threads: int) -> None:
    from cert_rnn import tier1
    _CK.clear()
    _CK.update(payload)
    tier1.THREADS = threads


def _check_job(job):
    """One job of the fixed-eps check.
    ("pass", k, item, mode): one reach at eps for anchor k -> both score
        upper bounds.
    ("attack", k, item): concrete attack at eps -> (score, input)."""
    from cert_rnn import LSTMAutoencoder
    from cert_rnn.transformers import bilinear_mode
    from cert_rnn.verify import (lstm_ae_reach, spec_c_score_ub,
                                 spec_c_score_ub_joint)
    x = _CK["anchors"][job[1]]
    t0 = time.perf_counter()
    if job[0] == "attack":
        ae = LSTMAutoencoder(_CK["enc"], _CK["dec"], _CK["head"])
        score, xa = concrete_attack(ae, x, _CK["eps"], _mask(job[2], x.shape),
                                    return_input=True)
        return {"score": score, "x": xa, "seconds": round(time.perf_counter() - t0, 2)}
    _, _, item, mode = job
    tm, tp = _spec(item)
    with bilinear_mode(mode):
        zxh, zx = lstm_ae_reach(_CK["enc"], _CK["dec"], _CK["head"], x,
                                _CK["eps"], tm, tp)
    return {"ub_componentwise": float(spec_c_score_ub(zxh, zx)),
            "ub_joint": float(spec_c_score_ub_joint(zxh, zx)),
            "seconds": round(time.perf_counter() - t0, 2)}


def concrete_attack(ae, anchor, eps, mask, steps=20, h=1e-5, return_input=False):
    """Largest score found inside the eps-box (masked entries) by signed
    finite-difference gradient ascent -- a LOWER bound on the true worst
    score (any value > tau is a real counterexample). return_input=True
    also returns the input that reached it."""
    idx = np.flatnonzero(mask.ravel())
    x = anchor.copy()
    best, best_x = float(ae.score(x[None])[0]), x.copy()
    for _ in range(steps):
        E = np.zeros((len(idx), anchor.size))
        E[np.arange(len(idx)), idx] = h
        E = E.reshape((-1,) + anchor.shape)
        g = (ae.score(x[None] + E) - ae.score(x[None] - E)) / (2 * h)
        step = np.zeros(anchor.size)
        step[idx] = 0.25 * eps * np.sign(g)
        x = np.clip(x + step.reshape(anchor.shape), anchor - eps * mask,
                    anchor + eps * mask)
        s = float(ae.score(x[None])[0])
        if s > best:
            best, best_x = s, x.copy()
    xc = anchor + eps * mask * np.sign(x - anchor + 1e-300)   # snap to a corner
    s = float(ae.score(xc[None])[0])
    if s > best:
        best, best_x = s, xc
    return (best, best_x) if return_input else best


def minimize_counterexample(ae, anchor, x, tau):
    """Greedily reset perturbed readings to their anchor values, least
    important first, while the score stays > tau. Returns the reduced input
    (every kept change is still within the original eps box)."""
    x = x.copy()
    while True:
        idx = np.flatnonzero(np.abs(x - anchor).ravel() > 0)
        if idx.size <= 1:
            return x
        cand = np.repeat(x[None], idx.size, axis=0).reshape(idx.size, -1)
        cand[np.arange(idx.size), idx] = anchor.ravel()[idx]
        s = ae.score(cand.reshape((-1,) + anchor.shape))
        j = int(np.argmax(s))
        if s[j] <= tau:
            return x
        x = cand[j].reshape(anchor.shape)


def describe_counterexample(anchor, x, top=10):
    d = x - anchor
    order = np.argsort(-np.abs(d).ravel())[:top]
    T, D = anchor.shape
    return {"linf": float(np.max(np.abs(d))),
            "n_changed": int(np.sum(np.abs(d) > 0)),
            "seconds_changed": sorted({int(i // D) for i in np.flatnonzero(np.abs(d).ravel() > 0)}),
            "top_changes": [{"second": int(i // D), "feature": int(i % D),
                             "delta": float(d.ravel()[i])}
                            for i in order if d.ravel()[i] != 0]}


def certified_radius_for(S, item):
    """Largest certified radius in S (radius-search stages) for this attack."""
    fam, a = item
    rs = []
    if fam == "whole":
        rs = [S[n]["radius"] for n in ("box_multi", "zono_multi", "zono_joint_multi")
              if n in S and S[n].get("radius") is not None]
    elif fam == "seconds":
        rs = [S[n]["per_frame"][a] for n in ("box_single_frame", "zono_single_frame")
              if n in S and S[n].get("per_frame")]
    else:
        tag = "-".join(map(str, a))
        rs = [S[n]["radius"] for n in (f"box_seconds_{tag}", f"zono_seconds_{tag}",
                                       f"zono_joint_seconds_{tag}") if n in S]
    return max(rs) if rs else None


def verdict(ub, atk, tau):
    if ub <= tau:
        # a sound bound can never sit below a real input's score
        return "UNSOUND" if atk > tau else "VERIFIED"
    return "COUNTEREXAMPLE" if atk > tau else "UNKNOWN"


def window_verdict(verdicts):
    """Worst verdict over a window's attacks (e.g. its 10 seconds)."""
    for v in ("UNSOUND", "COUNTEREXAMPLE", "UNKNOWN", "VERIFIED"):
        if v in verdicts:
            return v
    return "VERIFIED"


def run_check(args, ae, anchors, infos, tau, res, eps_list, items, batch):
    """Fixed-eps check for every anchor x attack item x gate mode, one pass
    each, all jobs of one eps in one process pool."""
    from concurrent.futures import ProcessPoolExecutor
    K, T, D = anchors.shape
    cex_dir = Path(args.out).with_suffix("")
    cex_dir = cex_dir.parent / (cex_dir.name + "_cex")
    s0 = ae.score(anchors)
    live = [k for k in range(K) if s0[k] <= tau]
    for k in range(K):
        if s0[k] > tau:
            log(f"window {infos[k].get('name', k)}: already above tau at eps=0 "
                f"(score {s0[k]:.6g} > {tau:.6g}) -- a false alarm as is, not checked")
    for eps in eps_list:
        name = f"check_eps_{eps:g}"
        stages_of = [(res["stages"] if not batch else
                      res["anchors"].setdefault(str(k), {"info": infos[k], "stages": {}})["stages"])
                     for k in range(K)]
        if all(name in stages_of[k] for k in live):
            log(f"{name}: already done, skipping")
            continue
        log(f"{name}: START ({len(live)} windows x {len(items)} attacks x 2 gate modes, "
            f"one pass each, plus a concrete attack per window x attack)")
        t_all = time.perf_counter()
        todo = [k for k in live if name not in stages_of[k]]
        # slowest first: Tier-1 whole-window, Tier-1 early seconds, attacks, box
        jobs = ([("pass", k, it, "zono") for it in items for k in todo]
                + [("attack", k, it) for it in items for k in todo]
                + [("pass", k, it, "box") for it in items for k in todo])
        in_flight = min(len(jobs), args.workers)
        threads = (args.tier1_threads if args.tier1_threads > 0 else
                   max(1, min(4, (os.cpu_count() or 1) // max(1, in_flight))))
        payload = {"enc": ae.encoder, "dec": ae.decoder, "head": ae.head,
                   "anchors": anchors, "eps": eps}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_check_init,
                                 initargs=(payload, threads)) as ex:
            out = dict(zip(jobs, ex.map(_check_job, jobs, chunksize=1)))
        passes_s = time.perf_counter() - t_all

        for k in todo:
            S = stages_of[k]
            entries = {}
            for it in items:
                atk = out[("attack", k, it)]
                e = {"attack": _label(it), "attack_score": atk["score"]}
                for mode in ("box", "zono"):
                    r = out[("pass", k, it, mode)]
                    e[mode] = dict(r, verdict_componentwise=verdict(r["ub_componentwise"], atk["score"], tau),
                                   verdict_joint=verdict(r["ub_joint"], atk["score"], tau))
                if atk["score"] > tau and not args.no_save_cex:
                    cex_dir.mkdir(parents=True, exist_ok=True)
                    stem = (f"a{k:03d}_" if batch else "") + f"eps{eps:g}_{_label(it)}"
                    np.save(cex_dir / f"{stem}.npy", atk["x"])
                    c = {"file": str(cex_dir / f"{stem}.npy"), "score": atk["score"],
                         **describe_counterexample(anchors[k], atk["x"])}
                    assert c["linf"] <= eps * (1 + 1e-9), "counterexample left the eps box"
                    if args.minimize_cex:
                        xm = minimize_counterexample(ae, anchors[k], atk["x"], tau)
                        np.save(cex_dir / f"{stem}_min.npy", xm)
                        c["minimized"] = {"file": str(cex_dir / f"{stem}_min.npy"),
                                          "score": float(ae.score(xm[None])[0]),
                                          **describe_counterexample(anchors[k], xm)}
                    e["counterexample"] = c
                r_cert = certified_radius_for(S, it)
                if r_cert is not None and eps <= r_cert and atk["score"] > tau:
                    e["zono"]["verdict_joint"] = e["box"]["verdict_joint"] = "UNSOUND"
                    log(f"*** SOUNDNESS FINDING: window {k} attack {_label(it)}: real "
                        f"counterexample at eps={eps:g} <= certified radius {r_cert:g} ***")
                entries[it] = e
                if "UNSOUND" in (e["box"]["verdict_joint"], e["zono"]["verdict_joint"]):
                    log(f"*** SOUNDNESS FINDING: window {k} attack {_label(it)} at "
                        f"eps={eps:g}: bound <= tau but a real input scores > tau ***")
            payload = {"eps": eps, "tau": tau,
                       "passes_seconds": round(passes_s, 2),
                       "done_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            if ("whole", None) in entries:
                payload["whole_window"] = entries[("whole", None)]
            secs = [dict(t=it[1], **entries[it]) for it in items if it[0] == "seconds"]
            if secs:
                payload["one_second"] = secs
            conc = [dict(seconds=list(it[1]), **entries[it]) for it in items
                    if it[0] == "concentrated"]
            if conc:
                payload["concentrated"] = conc[0]
            S[name] = payload
        payload_all_s = round(time.perf_counter() - t_all, 2)
        if batch:
            res.setdefault("summary", {})[name] = summarize_check(
                res, name, [k for k in range(K)], s0, tau, items)
        save_results(args.out, res)
        log(f"{name}: DONE in {payload_all_s:.1f}s (passes {passes_s:.1f}s), tau={tau:.6g}")
        if batch:
            print_summary(res["summary"][name], name)
        else:
            print_check(res["stages"][name])


def summarize_check(res, name, ks, s0, tau, items):
    """Headline counts over windows (linked gates, joint score bound)."""
    fams = [f for f in ATTACK_FAMILIES if any(it[0] == f for it in items)]
    out = {"windows": len(ks), "already_alarm": int(sum(s0[k] > tau for k in ks))}
    for fam in fams:
        counts = {"VERIFIED": 0, "COUNTEREXAMPLE": 0, "UNKNOWN": 0, "UNSOUND": 0}
        for k in ks:
            st = res["anchors"][str(k)]["stages"].get(name)
            if st is None:
                continue
            if fam == "whole":
                vs = [st["whole_window"]["zono"]["verdict_joint"]]
            elif fam == "seconds":
                vs = [e["zono"]["verdict_joint"] for e in st["one_second"]]
            else:
                vs = [st["concentrated"]["zono"]["verdict_joint"]]
            counts[window_verdict(vs)] += 1
        out[fam] = counts
    return out


def print_summary(sm, name):
    log(f"  HEADLINE {name}: {sm['windows']} windows "
        f"({sm['already_alarm']} already above tau at eps=0, not checked)")
    for fam in ATTACK_FAMILIES:
        if fam in sm:
            c = sm[fam]
            log(f"    {fam:13s} VERIFIED {c['VERIFIED']:4d} | COUNTEREXAMPLE "
                f"{c['COUNTEREXAMPLE']:4d} | UNKNOWN {c['UNKNOWN']:4d}"
                + (f" | UNSOUND {c['UNSOUND']}" if c["UNSOUND"] else ""))


def print_check(st):
    def line(label, e):
        cex = "  -> counterexample saved" if "counterexample" in e else ""
        log(f"  {label:14s} attack {e['attack_score']:.6g} | "
            f"box {e['box']['ub_joint']:.6g} {e['box']['verdict_joint']} | "
            f"linked gates {e['zono']['ub_joint']:.6g} {e['zono']['verdict_joint']}{cex}")
    if "whole_window" in st:
        line("whole-window", st["whole_window"])
    for e in st.get("one_second", []):
        line(f"second {e['t']}", e)
    if "concentrated" in st:
        line("secs " + ",".join(map(str, st["concentrated"]["seconds"])), st["concentrated"])


def stages_for_attacks(items):
    names = []
    fams = {it[0] for it in items}
    if "seconds" in fams:
        names += ["zono_single_frame", "box_single_frame"]
    if "whole" in fams:
        names += ["box_multi", "zono_multi", "zono_joint_multi"]
    for it in items:
        if it[0] == "concentrated":
            tag = "-".join(map(str, it[1]))
            names += [f"box_seconds_{tag}", f"zono_seconds_{tag}", f"zono_joint_seconds_{tag}"]
    return names


def run_radius(args, ae, anchor, tau, S, wanted, save):
    """Certified-radius stages for one anchor (results into S)."""
    from cert_rnn import ReconErrorSpec, tier1
    from cert_rnn.transformers import bilinear_mode
    from cert_rnn.verify import certify_radius_spec_c, spec_c_score_ub_joint
    T = anchor.shape[0]

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
        save()
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

    def set_threads(in_flight: int) -> None:
        tier1.THREADS = (args.tier1_threads if args.tier1_threads > 0 else
                         max(1, min(4, (os.cpu_count() or 1) // max(1, in_flight))))

    def certify_set(mode: str, score_bound: str, threat: str, frames=None) -> float:
        """multi_frame / frame_set search through certify_radius_spec_c:
        kary (parallel probes) or serial bisect per --multi-search."""
        kary = args.multi_search == "kary"
        set_threads(min(args.probes, args.workers) if kary else 1)
        with bilinear_mode(mode):
            radius, _ = certify_radius_spec_c(
                ae.encoder, ae.decoder, ae.head, anchor, tau,
                eps_init=args.eps_init, n_iters=args.n_iters,
                threat_model=threat, frames=frames,
                search="kary" if kary else "bisect", probes=args.probes,
                n_workers=args.workers if kary else 1, score_bound=score_bound,
                zono_last_rounds=(zono_last if mode == "zono" else None))
        return radius

    for name in wanted:
        if done(name):
            continue
        log(f"stage {name}: START")
        t0 = time.perf_counter()

        if "_seconds_" in name:            # concentrated: chosen seconds jointly
            prefix, tag = name.split("_seconds_")
            frames = tuple(int(s) for s in tag.split("-"))
            mode = "box" if prefix == "box" else "zono"
            sb = "joint" if prefix == "zono_joint" else "componentwise"
            radius = certify_set(mode, sb, "frame_set", frames)
            finish(name, {"radius": radius, "attack_seconds": list(frames),
                          "search": args.multi_search, "tier1_threads": tier1.THREADS,
                          "zono_last_rounds": zono_last if mode == "zono" else None}, t0)
            continue

        if args.multi_search == "kary" and name in ("box_multi", "zono_multi",
                                                     "zono_joint_multi"):
            mode = "box" if name == "box_multi" else "zono"
            sb = "joint" if name == "zono_joint_multi" else "componentwise"
            radius = certify_set(mode, sb, "multi_frame")
            finish(name, {"radius": radius, "search": "kary",
                          "probes": args.probes, "tier1_threads": tier1.THREADS,
                          "zono_last_rounds": zono_last if mode == "zono" else None},
                   t0)
            continue
        if name in ("box_multi", "zono_multi", "zono_joint_multi"):
            set_threads(1)

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

        elif name in ("zono_single_frame", "box_single_frame"):
            # parallel per-frame walk: this is where --workers pays
            # (radii bit-identical to the serial Algorithm 1)
            mode = "zono" if name.startswith("zono") else "box"
            set_threads(min(T, args.workers))
            with bilinear_mode(mode):
                radius, per_frame = certify_radius_spec_c(
                    ae.encoder, ae.decoder, ae.head, anchor, tau,
                    eps_init=args.eps_init, n_iters=args.n_iters,
                    threat_model="single_frame", n_workers=args.workers,
                    search=args.single_search, probes=15,
                    zono_last_rounds=(zono_last if mode == "zono" else None))
            payload = {"radius": float(radius),
                       "per_frame": [float(x) for x in per_frame],
                       "tier1_threads": tier1.THREADS}
            if mode == "zono":
                payload["zono_last_rounds"] = zono_last
            finish(name, payload, t0)

        else:
            log(f"unknown stage {name!r}, skipping")


def summarize_radius(res, wanted):
    out = {}
    for name in wanted:
        rs = [a["stages"][name]["radius"] for a in res["anchors"].values()
              if name in a["stages"] and a["stages"][name].get("radius") is not None]
        if rs:
            out[name] = {"windows": len(rs), "min": float(np.min(rs)),
                         "median": float(np.median(rs)), "max": float(np.max(rs))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", default="verify_job_v2.pkl")
    ap.add_argument("--out", default="verify_results_v2.json")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--eps-init", type=float, default=0.5)
    ap.add_argument("--n-iters", type=int, default=12)
    ap.add_argument("--zono-last-rounds", type=int, default=-1,
                    help="-1 (default) = Tier-1 in every search round. N = "
                         "cutover speed-up: only the last N rounds in zono "
                         "mode, earlier rounds in box mode (sound)")
    ap.add_argument("--multi-search", choices=["kary", "bisect"], default="bisect",
                    help="multi_frame search. bisect (default): serial walk "
                         "with a per-probe heartbeat. kary: --probes eps values "
                         "per round in parallel -- Tier-1 passes are memory-"
                         "bandwidth bound (8 concurrent passes each run ~2.7x "
                         "slower), so this only pays on some machines")
    ap.add_argument("--probes", type=int, default=15,
                    help="k-ary probes per round (2^a - 1)")
    ap.add_argument("--tier1-threads", type=int, default=0,
                    help="threads over neurons inside each Tier-1 gate step; "
                         "0 = auto (spare cores / tasks in flight, max 4)")
    ap.add_argument("--stages", default="default",
                    help="'default' = zono_single_frame,box_single_frame "
                         "(the single-frame head-to-head); 'all' adds the "
                         "multi_frame stages and curves; or a comma list "
                         "from: zono_single_frame,box_single_frame,"
                         "box_multi,zono_multi,zono_joint_multi,curves. "
                         "Ignored when --attack is given")
    ap.add_argument("--attack", default=None,
                    help="which attacks to run: whole (every second), seconds "
                         "(each single second), concentrated (the --seconds "
                         "set jointly), a comma list, or all. Check mode "
                         "default: all; radius mode: selects the stage families")
    ap.add_argument("--seconds", default=None,
                    help="seconds perturbed jointly by the concentrated attack, "
                         "e.g. 4,5,6 (0-based)")
    ap.add_argument("--eps", default=None,
                    help="fixed-eps CHECK mode instead of the radius search, "
                         "e.g. 0.05 or 0.01,0.05 (normalized units)")
    ap.add_argument("--no-save-cex", action="store_true",
                    help="check mode: do not save counterexample inputs")
    ap.add_argument("--minimize-cex", action="store_true",
                    help="check mode: also save a minimized counterexample "
                         "(fewest changed readings that still cross tau)")
    ap.add_argument("--single-search", choices=["bisect", "kary"],
                    default="bisect",
                    help="single-frame probe walk. bisect (default): "
                         "13 probes/frame, frames run in parallel across "
                         "--workers — best when workers <~ T. kary: 46 "
                         "probes/frame in 4 rounds — only pays when the "
                         "pool dwarfs the probe demand (workers >> T)")
    args = ap.parse_args()

    from cert_rnn import LSTMAutoencoder

    with open(args.job, "rb") as f:
        job = pickle.load(f)
    ae = LSTMAutoencoder(job["encoder"], job["decoder"], job["head"])
    batch = "anchors" in job
    anchors = np.asarray(job["anchors"] if batch else [job["anchor"]], dtype=np.float64)
    infos = list(job.get("anchor_info") or [{"name": f"window {k}"} for k in range(len(anchors))])
    tau = float(job["tau"])
    K, T, D = anchors.shape

    res = load_results(args.out)
    res["meta"].update({
        "host": socket.gethostname(), "workers": args.workers,
        "eps_init": args.eps_init, "n_iters": args.n_iters,
        "zono_last_rounds": args.zono_last_rounds,
        "multi_search": args.multi_search, "probes": args.probes,
        "tier1_threads": args.tier1_threads,
        "tau": tau, "T": T, "D": D, "H": ae.H,
        "decoder_input": ae.decoder.get("input", "latent"),
        "windows": K, "batch": batch,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if batch:
        res.setdefault("anchors", {})
        for k in range(K):
            res["anchors"].setdefault(str(k), {"info": infos[k], "stages": {}})

    if args.eps is not None:
        items = parse_attacks(args.attack, args.seconds, T)
        run_check(args, ae, anchors, infos, tau, res,
                  [float(e) for e in str(args.eps).split(",")], items, batch)
        return 0

    if args.attack is not None:
        wanted = stages_for_attacks(parse_attacks(args.attack, args.seconds, T))
    else:
        DEFAULT = ["zono_single_frame", "box_single_frame"]
        ALL = DEFAULT + ["box_multi", "zono_multi", "zono_joint_multi", "curves"]
        wanted = (DEFAULT if args.stages == "default"
                  else ALL if args.stages == "all"
                  else [s.strip() for s in args.stages.split(",")])

    s0 = ae.score(anchors)
    for k in range(K):
        S = res["anchors"][str(k)]["stages"] if batch else res["stages"]
        if batch:
            log(f"=== window {k + 1}/{K}: {infos[k].get('name', k)} ===")
        if s0[k] > tau:
            log(f"window {k}: already above tau at eps=0 (score {s0[k]:.6g}); "
                f"no radius to certify, skipping")
            continue
        run_radius(args, ae, anchors[k], tau, S, wanted,
                   lambda: save_results(args.out, res))
    if batch:
        res["summary_radius"] = summarize_radius(res, wanted)
        save_results(args.out, res)
        for name, v in res["summary_radius"].items():
            log(f"  {name:24s} over {v['windows']} windows: min {v['min']:.6g} "
                f"median {v['median']:.6g} max {v['max']:.6g}")
    log(f"all requested stages finished -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
