"""Step 3: fixed-eps checks (main table). For every model, verified window,
attack, eps and tool, one query -> one JSONL line. Resumable: finished
queries are skipped on restart.

    python -m paper_bench.check --hardware recommended
    python -m paper_bench.check --hardware fair --bench grid9 --tools certrnn_plus crown
    python -m paper_bench.check --bench zoneA            # at Sandia (models already trained there)

Order per (tool, window, attack): eps from largest to smallest.
  * PGD runs first for every query; a real counterexample at eps means no
    sound tool can verify it -> FALSIFIED, filled in for every verifier
    without running it (marked implied="pgd").
  * verified at eps => verified at every smaller eps (the smaller box lies
    inside the larger one) -> filled in, marked implied="larger_eps" -- only when
    protocol.infer_smaller_eps is true (off by default, so every query is timed).
Results: results/<hardware>/check.jsonl. PGD counterexample inputs are data
and go to work/cex/ (never to results/).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from paper_bench.common import (ResultLog, RunLog, single_thread_env, attack_frames, load_config, model_dir, model_ids,
                                results_dir, work_dir)

_W: dict = {}      # per-worker state (process pool)


def key(tool, bench, seed, w, attack, eps):
    return f"{tool}|{bench}|{seed}|{w}|{attack}|{eps:g}"


def tool_order(cfg, bench, wanted):
    """PGD first, fast verifiers, then slow ones (only where the benchmark allows)."""
    names = []
    for n, t in cfg["tools"].items():
        if wanted and n not in wanted and not t.get("attack"):
            continue
        if t.get("slow") and not cfg["benchmarks"][bench].get("slow_tools"):
            continue
        names.append(n)
    return sorted(names, key=lambda n: (not cfg["tools"][n].get("attack"), bool(cfg["tools"][n].get("slow"))))


def run_group(tool, ctx, bench, seed, w, attack, eps_list, falsified, done, hardware, cex_dir, infer=True):
    """All eps for one (tool, window, attack), largest first. Returns new records."""
    frames = attack_frames(attack, ctx.T)
    out, verified_at = [], None
    for eps in sorted(eps_list, reverse=True):
        k = key(tool.name, bench, seed, w, attack, eps)
        if k in done:
            if done[k] == "verified":
                verified_at = eps
            continue
        rec = dict(key=k, tool=tool.name, bench=bench, seed=seed, window=w, attack=attack, frames=frames,
                   eps=eps, tau=ctx.tau, hardware=hardware, device=tool.device, implied=None)
        if not hasattr(tool, "attack") and (w, attack, eps) in falsified:
            rec.update(status="falsified", implied="pgd", ub=None, wall=0.0, cpu=0.0)
        elif infer and verified_at is not None:
            rec.update(status="verified", implied="larger_eps", ub=None, wall=0.0, cpu=0.0)
        else:
            v = tool.check(ctx, w, frames, eps)
            x = v.pop("_x", None)
            if x is not None and v["status"] == "falsified" and cex_dir is not None:
                cex_dir.mkdir(parents=True, exist_ok=True)
                np.save(cex_dir / f"w{w}_{attack}_eps{eps:g}.npy", x)
            v.pop("trace", None)
            rec.update(v)
            if v["status"] == "verified":
                verified_at = eps
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out.append(rec)
    return out


def _init_worker(mdir, tool_name, tcfg, hardware):
    from cert_rnn import tier1
    from paper_bench.tools import ModelContext, make_tool
    tier1.THREADS = 1
    _W.update(ctx=ModelContext(mdir), tool=make_tool(tool_name, tcfg, hardware))


def _worker(args):
    return run_group(_W["tool"], _W["ctx"], *args)


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

    log = ResultLog(results_dir(cfg) / args.hardware / "check.jsonl")
    txt = RunLog(cfg, args.hardware, "check", sys.argv)
    workers = cfg["hardware"][args.hardware]["cpu_workers"]
    eps_list, attacks = cfg["protocol"]["eps"], cfg["protocol"]["attacks"]
    for bench, seed in model_ids(cfg, args.bench):
        if args.seeds is not None and seed not in args.seeds:
            continue
        mdir = model_dir(cfg, bench, seed)
        if not (mdir / "meta.json").exists():
            print(f"[{bench}/seed{seed}] not trained yet -- run train first; skipping")
            txt.note(f"{bench}/seed{seed}: not trained yet; skipped")
            continue
        ctx = ModelContext(mdir)
        cex_dir = work_dir(cfg) / "cex" / args.hardware / f"{bench}_seed{seed}"
        for name in tool_order(cfg, bench, args.tools):
            tcfg = cfg["tools"][name]
            recs = log.records()
            done = {r["key"]: r["status"] for r in recs}
            falsified = {(r["window"], r["attack"], r["eps"]) for r in recs
                         if r["tool"] == "pgd" and r["bench"] == bench and r["seed"] == seed
                         and r["status"] == "falsified"}
            n_win = min(len(ctx.windows), args.windows or len(ctx.windows))
            groups = [(w, a) for w in range(n_win) for a in attacks
                      if any(key(name, bench, seed, w, a, e) not in done for e in eps_list)]
            if not groups:
                print(f"[{bench}/seed{seed}] {name}: done")
                txt.note(f"{bench}/seed{seed} {name}: already complete")
                continue
            print(f"[{bench}/seed{seed}] {name}: {len(groups)} (window, attack) groups to run", flush=True)
            txt.note(f"{bench}/seed{seed} {name}: {len(groups)} (window, attack) groups to run")
            t0, n = time.perf_counter(), 0
            infer = cfg["protocol"].get("infer_smaller_eps", True)
            jobs = [(bench, seed, w, a, eps_list, falsified, done, args.hardware, cex_dir, infer) for w, a in groups]
            # CPU tools (zonotopes, CPU auto_LiRPA) run as a pool of single-threaded workers
            pooled = (tcfg["kind"] in ("certrnn", "lirpa") and tcfg.get("device") == "cpu") or \
                     (tcfg["kind"] == "abcrown" and tcfg.get("max_workers", 1) > 1)   # abcrown: subprocesses
            if pooled and workers > 1:
                single_thread_env()
                # spawn, not fork: the parent already started torch thread pools / CUDA (PGD),
                # and forked children can deadlock on locks held at fork time
                with ProcessPoolExecutor(min(workers, tcfg.get("max_workers", workers), len(jobs)), mp_context=mp.get_context("spawn"),
                                         initializer=_init_worker,
                                         initargs=(mdir, name, tcfg, args.hardware)) as ex:
                    for fut in as_completed([ex.submit(_worker, j) for j in jobs]):
                        for r in fut.result():
                            log.append(r); txt.check(r); n += 1
            else:
                tool = make_tool(name, tcfg, args.hardware)
                for j in jobs:
                    for r in run_group(tool, ctx, *j):
                        log.append(r); txt.check(r); n += 1
            print(f"[{bench}/seed{seed}] {name}: {n} queries recorded in {time.perf_counter() - t0:.0f}s", flush=True)
            txt.summary(f"{bench}/seed{seed} {name} (all records so far)",
                        [r for r in log.records() if r["tool"] == name and r["bench"] == bench and r["seed"] == seed])
    final_tables(txt, log.records(), cfg, args)
    txt.close()


def final_tables(txt, recs, cfg, args):
    """Verified count per model x tool (with the PGD limit), and a soundness check."""
    tools = [n for n in cfg["tools"] if n != "pgd"]
    models = sorted({(r["bench"], r["seed"]) for r in recs
                     if (args.bench is None or r["bench"] in args.bench) and (args.seeds is None or r["seed"] in args.seeds)},
                    key=lambda m: (list(cfg["benchmarks"]).index(m[0]) if m[0] in cfg["benchmarks"] else 99, m[1]))
    rows = {}
    for b, s in models:
        g = [r for r in recs if r["bench"] == b and r["seed"] == s]
        n_pgd = sum(r["tool"] == "pgd" for r in g)
        row = {"max verifiable": f"{n_pgd - sum(r['tool'] == 'pgd' and r['status'] == 'falsified' for r in g)}/{n_pgd}"}
        for t in tools:
            gt = [r for r in g if r["tool"] == t]
            if gt:
                row[t] = f"{sum(r['status'] == 'verified' for r in gt)}/{len(gt)}"
        rows[f"{b}/seed{s}"] = row
    txt.table("verified queries (all records so far)", rows, ["max verifiable"] + tools)
    fals = {r["key"].split("|", 1)[1] for r in recs if r["tool"] == "pgd" and r["status"] == "falsified"}
    ver = [r for r in recs if r["tool"] != "pgd" and r["status"] == "verified"]
    bad = [r["key"] for r in ver if r["key"].split("|", 1)[1] in fals]
    txt.note(f"soundness: {len(bad)} of {len(ver)} verified verdicts contradicted by a PGD counterexample"
             + (f"  <-- CHECK: {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    main()
