"""Turn results/*.jsonl into the paper's tables (markdown to stdout).

    python -m experiments.tabulate results/e1.jsonl
    python -m experiments.tabulate results/e3_normal.jsonl --certified-at 0.01
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common import read_records


def radius_table(records: list[dict]) -> str:
    """benchmark x tool -> radius (lower bound; PGD column is the upper
    bound ceiling), plus seconds."""
    tools = sorted({r["tool"] for r in records})
    benches = sorted({(r["benchmark"], r["anchor_id"]) for r in records})
    lines = ["| benchmark / anchor | " + " | ".join(tools) + " |",
             "|---" * (len(tools) + 1) + "|"]
    cell = defaultdict(dict)
    for r in records:
        cell[(r["benchmark"], r["anchor_id"])][r["tool"]] = r
    for key in benches:
        row = [f"{key[0]} {key[1]}"]
        for t in tools:
            r = cell[key].get(t)
            if r is None:
                row.append("--")
            elif r["status"] in ("certified", "falsified"):
                mark = "^" if r["radius_kind"] == "upper" else ""
                row.append(f"{r['radius']:.5f}{mark} ({r['seconds']:.0f}s)")
            else:
                row.append(r["status"])
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("`^` = attack upper bound, not a certificate.")
    return "\n".join(lines)


def certified_at(records: list[dict], eps: float) -> str:
    """Certified-robust detection rates at a target eps: the fraction of
    windows per (tool, spec) whose certified radius covers eps."""
    groups = defaultdict(list)
    for r in records:
        if r["radius_kind"] != "lower":
            continue
        groups[(r["tool"], r["spec"])].append(
            (r["status"] == "certified") and r["radius"] is not None
            and r["radius"] >= eps)
    lines = [f"Certified rate at eps={eps}:",
             "| tool | spec | certified / total |", "|---|---|---|"]
    for (tool, spec), oks in sorted(groups.items()):
        lines.append(f"| {tool} | {spec} | {sum(oks)}/{len(oks)} "
                     f"({100 * sum(oks) / len(oks):.0f}%) |")
    return "\n".join(lines)


def scaling_table(records: list[dict]) -> str:
    """For E2: tool x model-shape -> sec/bound-call (the scaling signal)."""
    lines = ["| benchmark | tool | status | s/bound-call | total s |",
             "|---|---|---|---|---|"]
    for r in records:
        spc = (r["seconds"] / r["n_bound_calls"]
               if r.get("n_bound_calls") else float("nan"))
        lines.append(f"| {r['benchmark']} | {r['tool']} | {r['status']} "
                     f"| {spc:.3f} | {r['seconds']:.1f} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--certified-at", type=float, default=None)
    p.add_argument("--scaling", action="store_true")
    args = p.parse_args(argv)
    records = []
    for path in args.paths:
        records += read_records(path)
    if args.certified_at is not None:
        print(certified_at(records, args.certified_at))
    elif args.scaling:
        print(scaling_table(records))
    else:
        print(radius_table(records))


if __name__ == "__main__":
    main()
