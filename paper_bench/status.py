"""Progress report: data, training and verification completion.

    python -m paper_bench.status [--hardware recommended] [--quick]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from paper_bench.check import tool_order
from paper_bench.common import ResultLog, data_dir, load_config, model_dir, model_ids, results_dir
from paper_bench.data import LABELS_F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default="recommended")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--data-dir")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    n_att, n_eps = len(cfg["protocol"]["attacks"]), len(cfg["protocol"]["eps"])
    check = ResultLog(results_dir(cfg) / args.hardware / "check.jsonl").records()
    radius = ResultLog(results_dir(cfg) / args.hardware / "radius.jsonl").records()
    c_done = Counter((r["tool"], r["bench"], r["seed"]) for r in check)
    r_done = Counter((r["tool"], r["bench"], r["seed"]) for r in radius)
    c_stat = Counter((r["tool"], r["status"]) for r in check)

    print(f"== {args.hardware} ==")
    for bench, seed in model_ids(cfg):
        spec = cfg["benchmarks"][bench]
        if spec["source"] == "grid":
            data_ok = (data_dir(cfg, bench) / LABELS_F).exists()
        else:
            data_ok = bool(args.data_dir)
        md = model_dir(cfg, bench, seed)
        if (md / "meta.json").exists():
            meta = json.load(open(md / "meta.json"))
            train = f"trained (tau {meta['tau']:.3g}, AUC {meta['detection']['roc_auc']:.3f})"
            n_w = meta["anchors"]["count"]
        elif (md / "ckpt.pt").exists():
            train, n_w = "training (checkpoint saved)", 0
        else:
            train, n_w = "not trained", 0
        print(f"\n{bench}/seed{seed}: data {'ok' if data_ok else 'missing'} | {train}")
        for t in tool_order(cfg, bench, None):
            exp_c = n_w * n_att * n_eps
            slow_skip = cfg["tools"][t].get("slow") and seed not in cfg["protocol"]["radius"].get("slow_tool_seeds", [0])
            exp_r = 0 if slow_skip else n_w * n_att
            c, r = c_done[(t, bench, seed)], r_done[(t, bench, seed)]
            print(f"  {t:13s} check {c:4d}/{exp_c:<4d} {'done' if exp_c and c >= exp_c else '':5s}"
                  f" radius {r:3d}/{exp_r:<3d} {'done' if exp_r and r >= exp_r else ''}")
    if c_stat:
        print("\nverdicts so far (check):")
        for t in sorted({k[0] for k in c_stat}):
            print(f"  {t:13s} " + ", ".join(f"{s} {n}" for (tt, s), n in sorted(c_stat.items()) if tt == t))


if __name__ == "__main__":
    main()
