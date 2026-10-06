"""Step 4: certified radius (secondary metric). Same bisection for every tool
(Du et al. Algorithm 1: eps_init 0.5, halve the step each round, keep the
largest eps that verified). PGD runs the mirror search and reports the
smallest eps at which it finds a real counterexample (an upper bound).

Resumable at the probe level: the search state is saved after every probe
(work/radius_state/...); a finished search is one line in
results/<hardware>/radius.jsonl.

    python -m paper_bench.radius --hardware recommended
    python -m paper_bench.radius --bench grid9 --tools certrnn certrnn_plus crown
"""

from __future__ import annotations

import argparse
import json
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from paper_bench.check import tool_order
from paper_bench.common import (ResultLog, single_thread_env, atomic_write_json, attack_frames, load_config, model_dir,
                                model_ids, results_dir, work_dir)

_W: dict = {}


def search(tool, ctx, bench, seed, w, attack, state_path, rcfg, hardware):
    """Run (or resume) one bisection; returns the final record."""
    frames = attack_frames(attack, ctx.T)
    is_attack = hasattr(tool, "attack")
    st = json.load(open(state_path)) if state_path.exists() else \
        {"eps": rcfg["eps_init"], "step": rcfg["eps_init"], "best": None, "i": 0, "probes": []}
    n_probes = rcfg["n_iters"] + 1
    while st["i"] < n_probes:
        v = tool.check(ctx, w, frames, st["eps"])
        v.pop("_x", None); v.pop("trace", None)
        st["probes"].append({"eps": st["eps"], **v})
        if is_attack:                                   # PGD: smallest eps with a counterexample
            hit = v["status"] == "falsified"
            if hit:
                st["best"] = st["eps"] if st["best"] is None else min(st["best"], st["eps"])
            st["eps"] += -st["step"] / 2 if hit else st["step"] / 2
        else:                                           # verifier: largest eps that verified
            ok = v["status"] == "verified"
            if ok:
                st["best"] = max(st["best"] or 0.0, st["eps"])
            st["eps"] += st["step"] / 2 if ok else -st["step"] / 2
        st["step"] /= 2
        st["i"] += 1
        atomic_write_json(state_path, st)              # resume point after every probe
    probes = st["probes"]
    return dict(
        key=f"{tool.name}|{bench}|{seed}|{w}|{attack}", tool=tool.name, bench=bench, seed=seed, window=w,
        attack=attack, frames=frames, tau=ctx.tau, hardware=hardware, device=tool.device,
        radius=st["best"] if st["best"] is not None else (None if is_attack else 0.0),
        kind="attack_upper_bound" if is_attack else "certified_lower_bound",
        wall=round(sum(p.get("wall", 0) for p in probes), 3), cpu=round(sum(p.get("cpu", 0) for p in probes), 3),
        n_probes=len(probes), n_timeout=sum(p["status"] == "timeout" for p in probes),
        n_error=sum(p["status"] == "error" for p in probes), time=time.strftime("%Y-%m-%d %H:%M:%S"))


def _init_worker(mdir, name, tcfg, hardware):
    from cert_rnn import tier1
    from paper_bench.tools import ModelContext, make_tool
    tier1.THREADS = 1
    _W.update(ctx=ModelContext(mdir), tool=make_tool(name, tcfg, hardware))


def _worker(args):
    return search(_W["tool"], _W["ctx"], *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", choices=["recommended", "fair"], default="recommended")
    ap.add_argument("--bench", nargs="*")
    ap.add_argument("--tools", nargs="*")
    ap.add_argument("--windows", type=int, help="only the first N verified windows (pilot)")
    ap.add_argument("--seeds", nargs="*", type=int, help="only these seeds")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    from paper_bench.tools import ModelContext, make_tool

    log = ResultLog(results_dir(cfg) / args.hardware / "radius.jsonl")
    workers = cfg["hardware"][args.hardware]["cpu_workers"]
    rcfg, attacks = cfg["protocol"]["radius"], cfg["protocol"]["attacks"]
    for bench, seed in model_ids(cfg, args.bench):
        if args.seeds is not None and seed not in args.seeds:
            continue
        mdir = model_dir(cfg, bench, seed)
        if not (mdir / "meta.json").exists():
            print(f"[{bench}/seed{seed}] not trained yet; skipping")
            continue
        ctx = ModelContext(mdir)
        for name in tool_order(cfg, bench, args.tools):
            tcfg = cfg["tools"][name]
            if tcfg.get("slow") and seed not in rcfg.get("slow_tool_seeds", [0]):
                continue                                   # slow tools: radius on the first seed only
            done = log.done()
            sdir = work_dir(cfg) / "radius_state" / args.hardware / name
            sdir.mkdir(parents=True, exist_ok=True)
            jobs = [(bench, seed, w, a, sdir / f"{bench}_seed{seed}_w{w}_{a}.json", rcfg, args.hardware)
                    for w in range(min(len(ctx.windows), args.windows or len(ctx.windows))) for a in attacks
                    if f"{name}|{bench}|{seed}|{w}|{a}" not in done]
            if not jobs:
                print(f"[{bench}/seed{seed}] {name}: done")
                continue
            print(f"[{bench}/seed{seed}] {name}: {len(jobs)} searches to run", flush=True)
            t0 = time.perf_counter()
            if tcfg["kind"] == "certrnn" and workers > 1:
                single_thread_env()
                # spawn, not fork: the parent already started torch thread pools / CUDA (PGD),
                # and forked children can deadlock on locks held at fork time
                with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn"),
                                         initializer=_init_worker,
                                         initargs=(mdir, name, tcfg, args.hardware)) as ex:
                    for fut in as_completed([ex.submit(_worker, j) for j in jobs]):
                        log.append(fut.result())
            else:
                tool = make_tool(name, tcfg, args.hardware)
                for j in jobs:
                    log.append(search(tool, ctx, *j))
            print(f"[{bench}/seed{seed}] {name}: finished in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
