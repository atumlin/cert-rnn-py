"""Step 6: tables and figures from whatever results exist (safe to run on
partial results). Writes to results/<hardware>/: CSV + LaTeX tables and PDF
figures. Only numbers -- no data, models or counterexample inputs.

    python -m paper_bench.analyze [--hardware recommended] [--quick] [--seeds 0]

With --seeds, outputs go to results/<hardware>/seeds_<k>/.

Outputs
  detection.csv/.tex      per model: tau, ROC-AUC, F1, recall, false-alarm rate
  check_main.csv/.tex     per benchmark x tool x attack family: % verified at each eps,
                          median seconds per query, timeouts/errors
  certified_rate.pdf      % verified vs eps (one panel per attack family)
  trajectory.pdf/.csv     % of queries verified within t s per query, per model + pooled
  verified_by_cutoff.csv  verified per model x tool at each report cutoff (300 s, 1800 s)
  *_<c>s.*                check_main / paired_check / certified_rate at each cutoff
  radius.csv/.tex         certified radius per tool x attack: min / median, time; PGD upper bound
  paired_check.csv        Cert-RNN+ vs each verifier on identical fixed-eps queries
  paired.csv              (radius) Cert-RNN+ vs each tool on identical (window, attack): wins/ties/losses
  scaling.csv/.pdf        median seconds per query vs model size (features, window, depth)
  soundness.csv           verified verdicts contradicted by a real counterexample (should be 0)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from paper_bench.common import ResultLog, load_config, model_dir, model_ids, results_dir

FAMILY = {"persistent": "persistent", "burst": "burst",
          "transient_first": "transient", "transient_middle": "transient", "transient_last": "transient"}
LABEL = {"certrnn": "Cert-RNN", "certrnn_plus": "Cert-RNN+", "ibp": "IBP", "crown": "CROWN",
         "alpha_crown": r"$\alpha$-CROWN", "abcrown": r"$\alpha,\beta$-CROWN", "pgd": "PGD (attack)"}


def _tex(df: pd.DataFrame, path, caption: str):
    try:
        df.to_latex(path, index=False, float_format="%.4g", caption=caption, escape=False)
    except Exception:                                             # noqa: BLE001 (jinja2 missing etc.)
        pass


def detection(cfg, out):
    rows = []
    for b, s in model_ids(cfg):
        p = model_dir(cfg, b, s) / "meta.json"
        if p.exists():
            m = json.load(open(p))
            rows.append({"benchmark": b, "seed": s, "features": m["D"], "code": m["H"], "window": m["T"],
                         "layers": m["layers"], "tau": m["tau"], **m["detection"],
                         "train_s": m["train_seconds"]})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out / "detection.csv", index=False)
        _tex(df, out / "detection.tex", "Detection performance of the trained detectors.")
    return df


def check_tables(cfg, out, recs, sfx=""):
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["family"] = df["attack"].map(FAMILY)
    ran = df[df["implied"].isna()]
    rows = []
    for (b, t, f), g in df.groupby(["bench", "tool", "family"]):
        row = {"benchmark": b, "tool": LABEL.get(t, t), "attack": f}
        for e, ge in g.groupby("eps"):
            ok = "falsified" if t == "pgd" else "verified"
            row[f"% {ok} @{e:g}"] = round(100 * (ge["status"] == ok).mean(), 1)
        gr = ran[(ran.bench == b) & (ran.tool == t) & (ran.family == f)]
        row["median s/query"] = round(float(gr["wall"].median()), 3) if len(gr) else np.nan
        row["timeouts"] = int((g["status"] == "timeout").sum())
        row["errors"] = int((g["status"] == "error").sum())
        row["queries"] = len(g)
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"check_main{sfx}.csv", index=False)
    _tex(tab, out / f"check_main{sfx}.tex", "Fixed-$\\epsilon$ verification: share of queries verified and time per query.")
    return df


def paired_check(out, df, sfx=""):
    """Cert-RNN+ vs each verifier on identical queries (benchmark, seed, window, attack, eps)."""
    if df.empty or "certrnn_plus" not in set(df.tool):
        return
    key = ["bench", "seed", "window", "attack", "eps"]
    v = df[df.tool != "pgd"].assign(ok=lambda d: d.status == "verified").pivot_table(
        index=key, columns="tool", values="ok", aggfunc="first")
    rows = []
    for t in [t for t in LABEL if t in v.columns and t not in ("certrnn_plus", "pgd")]:
        p = v[["certrnn_plus", t]].dropna().astype(bool)
        for b in sorted(set(p.index.get_level_values("bench"))):
            q = p.xs(b, level="bench")
            rows.append({"benchmark": b, "vs": LABEL[t], "queries": len(q),
                         "both verified": int((q.certrnn_plus & q[t]).sum()),
                         "only Cert-RNN+": int((q.certrnn_plus & ~q[t]).sum()),
                         f"only other": int((~q.certrnn_plus & q[t]).sum()),
                         "neither": int((~q.certrnn_plus & ~q[t]).sum())})
    pd.DataFrame(rows).to_csv(out / f"paired_check{sfx}.csv", index=False)


def figures(df, out, bench="grid9", sfx=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = df[df.bench == bench] if len(df) else df          # one model family per figure (no pooling)
    if df.empty:
        return
    tools = [t for t in LABEL if t in set(df.tool) and t != "pgd"]
    fams = [f for f in ("persistent", "burst", "transient") if f in set(df.family)]
    fig, axs = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.2), squeeze=False, sharey=True)
    for ax, f in zip(axs[0], fams):
        g = df[df.family == f]
        for t in tools:
            s = g[g.tool == t].groupby("eps")["status"].apply(lambda x: 100 * (x == "verified").mean())
            ax.plot(s.index, s.values, marker="o", label=LABEL[t])
        p = g[g.tool == "pgd"].groupby("eps")["status"].apply(lambda x: 100 * (1 - (x == "falsified").mean()))
        if len(p):
            ax.plot(p.index, p.values, "k--", lw=1, label="upper limit (1 - PGD falsified)")
        ax.set(xscale="log", xlabel=r"$\epsilon$", title=f"{bench}: {f} FDI")
    axs[0][0].set_ylabel("% queries verified")
    axs[0][-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out / f"certified_rate{sfx}.pdf"); plt.close(fig)



def at_cutoff(recs, cutoff):
    """Verdicts under a per-query time limit `cutoff` <= the run's limit: a query verified
    after `cutoff` seconds counts as a timeout. (One run at the largest limit gives all.)"""
    out = []
    for r in recs:
        if r.get("status") == "verified" and not r.get("implied") and r.get("wall", 0) > cutoff:
            r = {**r, "status": "timeout", "wall": float(cutoff)}
        out.append(r)
    return out


def verified_by_cutoff(cfg, out, recs, cutoffs):
    """Verified queries per model x tool at each time limit, with the PGD limit."""
    df = pd.DataFrame(recs)
    if df.empty:
        return
    rows = []
    for (b, s), g in df.groupby(["bench", "seed"]):
        pg = g[g.tool == "pgd"]
        row = {"benchmark": b, "seed": s, "queries": len(pg),
               "max verifiable": int(len(pg) - (pg.status == "falsified").sum())}
        for t in [t for t in LABEL if t != "pgd" and t in set(g.tool)]:
            gt = g[g.tool == t]
            for c in cutoffs:
                v = gt[(gt.status == "verified") & (gt.implied.isna() & (gt.wall <= c) | gt.implied.notna())]
                row[f"{LABEL[t]} @{c}s"] = int(len(v))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "verified_by_cutoff.csv", index=False)


def trajectory(cfg, out, recs, cutoffs):
    """Share of queries verified within t seconds per query (x: t, log scale; y: %),
    one panel per model plus a pooled panel; PGD limit dashed; report cutoffs marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(recs)
    if df.empty:
        return
    if (df.implied == "larger_eps").any():
        print("trajectory: some verdicts were inferred (no time) -- rerun with infer_smaller_eps: false")
    tmax = max(cutoffs)
    ts = np.logspace(-2, np.log10(tmax), 300)
    benches = [b for b in cfg["benchmarks"] if b in set(df.bench)] + ["all models"]
    n = len(benches)
    ncol = min(4, n)
    fig, axs = plt.subplots((n + ncol - 1) // ncol, ncol, figsize=(4.0 * ncol, 3.1 * ((n + ncol - 1) // ncol)),
                            squeeze=False, sharey=True)
    rows = []
    for ax, b in zip(axs.flat, benches):
        g = df if b == "all models" else df[df.bench == b]
        for t in [t for t in LABEL if t != "pgd" and t in set(g.tool)]:
            gt = g[g.tool == t]
            if b == "all models":                       # pooled: only the models this tool ran on
                gt = gt[gt.bench.isin(set(gt.bench))]
            N = len(gt)
            w = gt[(gt.status == "verified")]["wall"].where(gt.implied.isna(), 0.0).to_numpy()
            y = [100.0 * (w <= x).sum() / N for x in ts]
            ax.plot(ts, y, label=LABEL[t] + ("" if b != "all models" else f" ({gt.bench.nunique()} models)"),
                    lw=2.2 if t == "certrnn_plus" else 1.4)
            for c in cutoffs:
                rows.append({"benchmark": b, "tool": LABEL[t], "cutoff_s": c, "queries": N,
                             "% verified": round(100.0 * (w <= c).sum() / N, 1)})
        pg = g[g.tool == "pgd"]
        if len(pg):
            ax.axhline(100.0 * (1 - (pg.status == "falsified").mean()), color="k", ls="--", lw=1)
        for c in cutoffs[:-1]:
            ax.axvline(c, color="0.6", ls=":", lw=1)
        ax.set(xscale="log", xlim=(1e-2, tmax), ylim=(0, 100), title=b, xlabel="time per query (s)")
    for ax in axs.flat[len(benches):]:
        ax.axis("off")
    for r in range(axs.shape[0]):
        axs[r][0].set_ylabel("% queries verified")
    axs.flat[len(benches) - 1].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Verified within t seconds per query (dashed: PGD limit; dotted: report cutoff)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "trajectory.pdf"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "trajectory.csv", index=False)


def radius_tables(out, recs):
    df = pd.DataFrame(recs)
    if df.empty:
        return
    df["family"] = df["attack"].map(FAMILY)
    tab = (df.groupby(["bench", "tool", "family"])
             .agg(windows=("radius", "size"), min_radius=("radius", "min"), median_radius=("radius", "median"),
                  median_search_s=("wall", "median"), timeouts=("n_timeout", "sum"), errors=("n_error", "sum"))
             .reset_index())
    tab["tool"] = tab["tool"].map(lambda t: LABEL.get(t, t))
    tab.to_csv(out / "radius.csv", index=False)
    _tex(tab, out / "radius.tex", "Certified radius (PGD: smallest falsifying $\\epsilon$, an upper bound).")
    ver = df[df.kind == "certified_lower_bound"]
    piv = ver.pivot_table(index=["bench", "seed", "window", "attack"], columns="tool", values="radius")
    rows = []
    if "certrnn_plus" in piv:
        for t in piv.columns:
            if t == "certrnn_plus":
                continue
            d = (piv["certrnn_plus"] - piv[t]).dropna()
            rows.append({"vs": LABEL.get(t, t), "pairs": len(d), "Cert-RNN+ larger": int((d > 1e-12).sum()),
                         "tie": int((d.abs() <= 1e-12).sum()), "Cert-RNN+ smaller": int((d < -1e-12).sum())})
    pgd = df[df.tool == "pgd"].set_index(["bench", "seed", "window", "attack"])["radius"]
    if len(pgd) and "certrnn_plus" in piv:
        r = (piv["certrnn_plus"] / pgd).dropna()
        rows.append({"vs": "tightness: Cert-RNN+ / PGD upper bound (median)", "pairs": len(r),
                     "Cert-RNN+ larger": round(float(r.median()), 3)})
    pd.DataFrame(rows).to_csv(out / "paired.csv", index=False)


def scaling(cfg, out, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df.empty:
        return
    ran = df[df["implied"].isna() & (df.status != "error")]       # timeouts count at their (budget) time
    rows = []
    for b, g in ran.groupby("bench"):
        spec = cfg["benchmarks"].get(b, {})
        p = model_dir(cfg, b, cfg["benchmarks"][b]["seeds"][0]) / "meta.json"
        D = json.load(open(p))["D"] if p.exists() else None
        for t, gt in g.groupby("tool"):
            rows.append({"benchmark": b, "tool": LABEL.get(t, t), "features": D, "window": spec.get("window"),
                         "layers": spec.get("layers"), "median s/query": float(gt["wall"].median())})
    sc = pd.DataFrame(rows)
    sc.to_csv(out / "scaling.csv", index=False)
    axes = [("features", ["grid9", "grid14", "grid30"]), ("window", ["grid9", "grid9_T20", "grid9_T30"]),
            ("layers", ["grid9", "grid9_L2"])]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (col, benches) in zip(axs, axes):
        s = sc[sc.benchmark.isin(benches)]
        for t, g in s.groupby("tool"):
            g = g.sort_values(col)
            ax.plot(g[col], g["median s/query"], marker="o", label=t)
        ax.set(yscale="log", xlabel=col, ylabel="median s / query")
    axs[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "scaling.pdf"); plt.close(fig)


def soundness(out, df):
    if df.empty:
        return
    fals = set(map(tuple, df[df.status == "falsified"][["bench", "seed", "window", "attack", "eps"]].values))
    v = df[(df.status == "verified") & (df.tool != "pgd")]
    bad = v[[tuple(x) in fals for x in v[["bench", "seed", "window", "attack", "eps"]].values]]
    pd.DataFrame({"verified verdicts": [len(v)], "contradicted by a real counterexample": [len(bad)]}) \
        .to_csv(out / "soundness.csv", index=False)
    if len(bad):
        bad.to_csv(out / "soundness_violations.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default="recommended")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--seeds", nargs="*", type=int,
                    help="only these seeds (compare every tool on the same queries)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    src = results_dir(cfg) / args.hardware
    out = src if args.seeds is None else src / ("seeds_" + "_".join(map(str, args.seeds)))
    out.mkdir(parents=True, exist_ok=True)
    keep = (lambda r: True) if args.seeds is None else (lambda r: r.get("seed") in args.seeds)
    detection(cfg, out)
    recs = [r for r in ResultLog(src / "check.jsonl").records() if keep(r)]
    cutoffs = sorted(cfg["protocol"].get("report_cutoffs", [max(t["budget"] for t in cfg["tools"].values())]))
    for c in cutoffs:                                   # every table/figure at each time limit
        dc = check_tables(cfg, out, at_cutoff(recs, c), sfx=f"_{c}s")
        paired_check(out, dc, sfx=f"_{c}s")
        figures(dc, out, sfx=f"_{c}s")
    verified_by_cutoff(cfg, out, recs, cutoffs)
    trajectory(cfg, out, recs, cutoffs)
    df = check_tables(cfg, out, recs)                   # uncut (largest limit), for scaling / soundness
    radius_tables(out, [r for r in ResultLog(src / "radius.jsonl").records() if keep(r)])
    scaling(cfg, out, df)
    soundness(out, df)
    print(f"wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
