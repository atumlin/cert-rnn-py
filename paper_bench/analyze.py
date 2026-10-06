"""Step 6: tables and figures from whatever results exist (safe to run on
partial results). Writes to results/<hardware>/: CSV + LaTeX tables and PDF
figures. Only numbers -- no data, models or counterexample inputs.

    python -m paper_bench.analyze [--hardware recommended] [--quick]

Outputs
  detection.csv/.tex      per model: tau, ROC-AUC, F1, recall, false-alarm rate
  check_main.csv/.tex     per benchmark x tool x attack family: % verified at each eps,
                          median seconds per query, timeouts/errors
  certified_rate.pdf      % verified vs eps (one panel per attack family)
  cactus.pdf              queries verified vs time budget (log time)
  radius.csv/.tex         certified radius per tool x attack: min / median, time; PGD upper bound
  paired.csv              Cert-RNN+ vs each tool on identical (window, attack): wins/ties/losses
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


def check_tables(cfg, out, recs):
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
    tab.to_csv(out / "check_main.csv", index=False)
    _tex(tab, out / "check_main.tex", "Fixed-$\\epsilon$ verification: share of queries verified and time per query.")
    return df


def figures(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
        ax.set(xscale="log", xlabel=r"$\epsilon$", title=f"{f} FDI")
    axs[0][0].set_ylabel("% queries verified")
    axs[0][-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "certified_rate.pdf"); plt.close(fig)

    ran = df[df["implied"].isna() & (df.tool != "pgd")]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for t in tools:
        w = np.sort(ran[(ran.tool == t) & (ran.status == "verified")]["wall"].to_numpy())
        if len(w):
            ax.step(np.maximum(w, 1e-3), np.arange(1, len(w) + 1), where="post", label=LABEL[t])
    ax.set(xscale="log", xlabel="time per query (s)", ylabel="queries verified", title="Cactus plot")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "cactus.pdf"); plt.close(fig)


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
    ran = df[df["implied"].isna() & df.status.isin(["verified", "unknown"])]
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
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    out = results_dir(cfg) / args.hardware
    out.mkdir(parents=True, exist_ok=True)
    detection(cfg, out)
    df = check_tables(cfg, out, ResultLog(out / "check.jsonl").records())
    figures(df, out)
    radius_tables(out, ResultLog(out / "radius.jsonl").records())
    scaling(cfg, out, df)
    soundness(out, df)
    print(f"wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
