"""Paper figures from results/*.jsonl (matplotlib, PDF out).

    python -m experiments.plots cactus results/e2_sweepH.jsonl
    python -m experiments.plots scaling results/e2_sweepH.jsonl --axis H
    python -m experiments.plots radius results/e1.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import read_records  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"

STYLE = {
    "certrnn-zono": dict(color="#1a6faf", marker="o", label="Cert-RNN zonotope (ours)"),
    "interval-ibp": dict(color="#888888", marker="s", label="Interval (IBP)"),
    "lirpa-ibp": dict(color="#b8860b", marker="^", label="auto_LiRPA IBP"),
    "lirpa-backward": dict(color="#a02c2c", marker="v", label="auto_LiRPA CROWN"),
    "pgd": dict(color="#2e7d32", marker="x", label="PGD upper bound"),
}


def _style(tool):
    return STYLE.get(tool, dict(marker="."))


def cactus(records, out):
    """Instances 'solved' (certified radius >= its own median target)
    within t seconds -- VNN-COMP-style cost profile per tool."""
    per_tool = defaultdict(list)
    for r in records:
        if r["status"] in ("certified", "falsified") and r["seconds"] is not None:
            per_tool[r["tool"]].append(r["seconds"])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for tool, ts in sorted(per_tool.items()):
        ts = sorted(ts)
        ax.step(ts, range(1, len(ts) + 1), where="post", **_style(tool))
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("instances completed")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def scaling(records, axis, out):
    """sec/bound-call vs the swept model axis (log-log)."""
    pts = defaultdict(list)
    for r in records:
        v = r["model_meta"].get(axis if axis != "L" else "L_enc")
        if v is None or not r.get("n_bound_calls"):
            continue
        pts[r["tool"]].append((v, r["seconds"] / r["n_bound_calls"],
                               r["status"]))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for tool, xs in sorted(pts.items()):
        xs.sort()
        ax.plot([a for a, _, s in xs if s != "timeout"],
                [b for _, b, s in xs if s != "timeout"], **_style(tool))
        to = [(a, b) for a, b, s in xs if s == "timeout"]
        if to:
            ax.scatter([a for a, _ in to], [b for _, b in to], marker="X",
                       s=60, color=_style(tool).get("color", "k"))
    ax.set_xlabel(axis)
    ax.set_ylabel("seconds / bound call")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def radius(records, out):
    """Grouped bars: certified radius per benchmark per tool, PGD ceiling
    as a horizontal tick."""
    benches = sorted({r["benchmark"] for r in records})
    tools = [t for t in STYLE if any(r["tool"] == t for r in records)]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    w = 0.8 / max(1, len(tools))
    for j, tool in enumerate(tools):
        vals = []
        for b in benches:
            rs = [r["radius"] for r in records
                  if r["tool"] == tool and r["benchmark"] == b
                  and r["radius"] is not None]
            vals.append(sum(rs) / len(rs) if rs else 0.0)
        if tool == "pgd":
            for i, v in enumerate(vals):
                ax.hlines(v, i - 0.4, i + 0.4, **{k: v2 for k, v2 in
                          _style(tool).items() if k in ("color", "label")
                          and (i == 0 or k != "label")})
        else:
            ax.bar([i + (j - len(tools) / 2) * w for i in range(len(benches))],
                   vals, width=w, color=_style(tool).get("color"),
                   label=_style(tool).get("label", tool))
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches, fontsize=7)
    ax.set_ylabel("certified radius (mean over anchors)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["cactus", "scaling", "radius"])
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--axis", default="H")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    FIG_DIR.mkdir(exist_ok=True)
    records = []
    for path in args.paths:
        records += read_records(path)
    out = args.out or FIG_DIR / f"{args.kind}_{args.paths[0].stem}.pdf"
    if args.kind == "cactus":
        cactus(records, out)
    elif args.kind == "scaling":
        scaling(records, args.axis, out)
    else:
        radius(records, out)


if __name__ == "__main__":
    main()
