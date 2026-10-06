"""Step 2: train every (benchmark, seed) model with the v3 recipe, record
detection metrics and pick the verified windows. Resumable: a checkpoint is
written every epoch; a model with meta.json is finished and skipped.

    python -m paper_bench.train                          # all grid benchmarks
    python -m paper_bench.train --bench zoneA --data-dir /path/to/zoneA

Outputs per model (work/models/<bench>/seed<k>/):
    ckpt.pt       per-epoch checkpoint (weights, optimizer, history, best)
    model.pt      final weights (best validation epoch)
    anchors.npz   the windows to verify (scaled inputs) -- data, stays local
    meta.json     tau, detection metrics, shapes, timings  (written last = done)
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from paper_bench.common import atomic_write_json, data_dir, load_config, model_dir, model_ids
from paper_bench.data import prepare, select_anchors
from paper_bench.model import V3AE, code_size, window_scores


def _save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def tau_from(rule: str, train_s, val_s) -> float:
    split, pct = rule.split("_p")
    return float(np.percentile({"train": train_s, "val": val_s}[split], float(pct)))


def train_one(cfg, bench: str, seed: int, data_override=None) -> None:
    out = model_dir(cfg, bench, seed)
    if (out / "meta.json").exists():
        print(f"[{bench}/seed{seed}] done, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    spec, tc = cfg["benchmarks"][bench], cfg["train"]
    T = spec["window"]
    prep = prepare(data_dir(cfg, bench, data_override), T)
    D = len(prep.features)
    torch.set_num_threads(tc["threads"])
    torch.manual_seed(seed)
    model = V3AE(D, spec["layers"])
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    crit = torch.nn.MSELoss()
    hist, best, best_val, epoch0, t_train = dict(train=[], val=[]), None, np.inf, 0, 0.0
    ck = out / "ckpt.pt"
    if ck.exists():                                            # resume mid-training
        s = torch.load(ck, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        hist, best, best_val, epoch0, t_train = s["hist"], s["best"], s["best_val"], s["epoch"], s["t_train"]
        print(f"[{bench}/seed{seed}] resuming after epoch {epoch0}")
    Xtr, Xva = torch.as_tensor(prep.train), torch.as_tensor(prep.val)
    bs = tc["batch_size"]
    for epoch in range(epoch0 + 1, tc["epochs"] + 1):
        t0 = time.perf_counter()
        g = torch.Generator().manual_seed(seed * 100003 + epoch)   # batch order fixed per epoch
        perm = torch.randperm(len(Xtr), generator=g)
        model.train()
        losses = []
        for i in range(0, len(perm) - bs + 1, bs):
            b = Xtr[perm[i:i + bs]]
            opt.zero_grad()
            loss = crit(model(b), b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            losses.append(loss.item())
        val = float(window_scores(model, prep.val).mean())
        hist["train"].append(float(np.mean(losses))); hist["val"].append(val)
        if val < best_val:
            best_val, best = val, copy.deepcopy(model.state_dict())
        t_train += time.perf_counter() - t0
        _save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "best": best,
               "best_val": best_val, "epoch": epoch, "t_train": t_train}, ck)
        if epoch % 10 == 0 or epoch == tc["epochs"]:
            print(f"[{bench}/seed{seed}] epoch {epoch}: train {hist['train'][-1]:.3g} val {val:.3g}", flush=True)

    model.load_state_dict(best)
    model.eval()
    s_tr, s_va = window_scores(model, prep.train), window_scores(model, prep.val)
    s_n, s_a = window_scores(model, prep.test_normal), window_scores(model, prep.test_anomaly)
    tau = tau_from(tc["tau_rule"], s_tr, s_va)
    y = np.r_[np.zeros(len(s_n)), np.ones(len(s_a))]
    sc = np.r_[s_n, s_a]
    pred = sc > tau
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    idx = select_anchors(prep, s_n, tau, tc["n_windows"], T, tc["window_seed"] + seed)
    _save(best, out / "model.pt")
    np.savez(out / "anchors.npz", windows=prep.test_normal[idx].astype(np.float64), index=idx,
             scores=s_n[idx], source=prep.test_normal_src[idx], start=prep.test_normal_start[idx])
    meta = dict(bench=bench, seed=seed, T=T, D=D, H=code_size(D), layers=spec["layers"],
                tau=tau, tau_rule=tc["tau_rule"], epochs=len(hist["val"]), best_val=best_val,
                train_seconds=round(t_train, 1), history=hist,
                n_windows=dict(train=len(prep.train), val=len(prep.val), test_normal=len(s_n), test_anomaly=len(s_a)),
                detection=dict(roc_auc=float(roc_auc_score(y, sc)), recall=rec, precision=prec,
                               f1=2 * prec * rec / max(prec + rec, 1e-12), false_alarm_rate=fp / max(fp + tn, 1)),
                anchors=dict(count=len(idx), scores=[float(v) for v in s_n[idx]], margin_min=float(tau - s_n[idx].max())))
    atomic_write_json(out / "meta.json", meta)                 # written last: marks the model done
    d = meta["detection"]
    print(f"[{bench}/seed{seed}] done: D={D} H={meta['H']} tau={tau:.3g} AUC {d['roc_auc']:.3f} "
          f"F1 {d['f1']:.3f} FA {d['false_alarm_rate']:.3f} | {len(idx)} windows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*")
    ap.add_argument("--data-dir")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    benches = args.bench or [b for b, s in cfg["benchmarks"].items() if s["source"] == "grid"]
    for b, s in model_ids(cfg, benches):
        train_one(cfg, b, s, args.data_dir)


if __name__ == "__main__":
    main()
