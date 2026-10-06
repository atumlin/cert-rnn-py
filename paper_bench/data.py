"""Benchmark data: generation (grid stand-ins) and loading (Zone-A format).

Every benchmark is stored in the Zone-A file layout, so one loader serves
the public stand-ins and the real Sandia data alike:

    SubFeatures_baseline3hr_zoneA.csv          normal operation (1 Hz)
    SubFeatures_cyberphysical_0_1_zoneA.csv    scenario with events (1 Hz)
    CyPhyScenarioLabels_2Class_0_1_zoneA.csv   timestamp, labels (0/1)

Grid stand-ins: AC power flow (pandapower) of an IEEE test case, one
solve per second, slowly drifting autocorrelated loads + PMU-style noise.
Features per second = [Vmag, Vang_deg, P_mw, Q_mvar] x buses
(case9: 36, case14: 56, case30: 120). The scenario contains physical
events (line outage, load spike, load drop) and cyber events on the
measurements (bias, frozen values, slow ramp).

Preparation (`prepare`) is the v3 recipe: files kept in time order,
chronological train/val/test split of the baseline, MinMax scaler fit on
the training block only, sliding windows that never cross a NaN row, a
time gap or a split boundary; a window is abnormal if any second is.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_F = "SubFeatures_baseline3hr_zoneA.csv"
TEST_F = "SubFeatures_cyberphysical_0_1_zoneA.csv"
LABELS_F = "CyPhyScenarioLabels_2Class_0_1_zoneA.csv"

PHYSICAL = ("line_outage", "load_spike", "load_drop")
CYBER = ("fdi_bias", "freeze", "ramp")
MEAS_NOISE = np.array([5e-4, 0.02, 0.2, 0.2])   # Vmag pu, Vang deg, P MW, Q MVAr


# --------------------------------------------------------------------------- #
# generation: grid stand-ins
# --------------------------------------------------------------------------- #
def _net(case: str):
    import pandapower.networks as pn
    return {"case9": pn.case9, "case14": pn.case14, "case30": pn.case30}[case]()


def _features(net) -> np.ndarray:
    rb = net.res_bus
    return np.stack([rb.vm_pu.to_numpy(), rb.va_degree.to_numpy(),
                     rb.p_mw.to_numpy(), rb.q_mvar.to_numpy()], axis=1).ravel()


def _feature_names(n_bus: int) -> list[str]:
    return [f"bus{b}_{q}" for b in range(n_bus) for q in ("Vmag", "Vang_deg", "P_mw", "Q_mvar")]


def _simulate(case: str, n_seconds: int, rng, t_offset: int, events=(), state=None):
    """state carries the normal load process between calls (per-load drift
    phases and the AR(1) deviations), so the scenario CONTINUES the baseline's
    normal operation instead of starting a different one."""
    import logging
    import pandapower as pp
    warnings.filterwarnings("ignore")
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    net = _net(case)
    base = net.load[["p_mw", "q_mvar"]].to_numpy(copy=True)
    lines = net.line["in_service"].to_numpy(copy=True)
    n_bus = len(net.bus)
    noise = np.tile(MEAS_NOISE, n_bus)
    if state is None:
        state = {"phase": rng.uniform(0, 2 * np.pi, len(base)), "x_ar": np.zeros_like(base)}
    x_ar, phase = state["x_ar"].copy(), state["phase"]
    by_second = {s: ev for ev in events for s in range(ev["start"], ev["end"])}
    X = np.empty((n_seconds, 4 * n_bus))
    frozen = None
    for s in range(n_seconds):
        x_ar = 0.98 * x_ar + rng.normal(0, 0.004, x_ar.shape)        # AR(1), std ~2%
        drift = 0.06 * np.sin(2 * np.pi * (t_offset + s) / 5400 + phase)
        pq = base * (1.0 + drift[:, None] + x_ar)
        net.line.loc[:, "in_service"] = lines
        ev = by_second.get(s)
        if ev and ev["type"] == "line_outage":
            net.line.at[ev["line"], "in_service"] = False
        if ev and ev["type"] in ("load_spike", "load_drop"):
            pq[ev["load"]] *= ev["factor"]
        net.load.loc[:, "p_mw"], net.load.loc[:, "q_mvar"] = pq[:, 0], pq[:, 1]
        try:
            pp.runpp(net)
        except Exception:                                          # islanded by an outage
            net.line.loc[:, "in_service"] = lines
            pp.runpp(net)
        x = _features(net) + rng.normal(0, noise)
        if ev and "bus" in ev:
            cols = slice(4 * ev["bus"], 4 * ev["bus"] + 4)
            if ev["type"] == "fdi_bias":
                x[cols] += np.asarray(ev["offset"])
            elif ev["type"] == "ramp":
                x[cols] += np.asarray(ev["offset"]) * (s - ev["start"]) / (ev["end"] - ev["start"])
            elif ev["type"] == "freeze":
                if frozen is None or frozen[0] is not ev:
                    frozen = (ev, X[s - 1, cols].copy())
                x[cols] = frozen[1]
        X[s] = x
    return X, {"phase": phase, "x_ar": x_ar}


def _events(case: str, rng, n_seconds: int, n_events: int) -> list[dict]:
    net = _net(case)
    kinds = [PHYSICAL[i // 2 % 3] if i % 2 == 0 else CYBER[i // 2 % 3] for i in range(n_events)]
    rng.shuffle(kinds)
    slot = n_seconds // n_events
    out = []
    for i, kind in enumerate(kinds):
        dur = int(rng.integers(30, min(90, slot - 50)))
        start = i * slot + int(rng.integers(20, slot - dur - 20))
        ev = {"type": kind, "start": start, "end": start + dur}
        if kind == "line_outage":
            ev["line"] = int(rng.integers(0, len(net.line)))
        elif kind in ("load_spike", "load_drop"):
            ev["load"] = int(rng.integers(0, len(net.load)))
            ev["factor"] = float(rng.uniform(1.4, 1.8) if kind == "load_spike" else rng.uniform(0.2, 0.5))
        else:
            ev["bus"] = int(rng.integers(1, len(net.bus)))
            if kind != "freeze":
                ev["offset"] = (np.array([0.01, 1.0, 5.0, 5.0]) * rng.choice([-1, 1], 4)).tolist()
        out.append(ev)
    return out


def generate_grid(case: str, out_dir: str | Path, baseline_s: int = 10800,
                  scenario_s: int = 3600, n_events: int = 24, seed: int = 7) -> Path:
    """Write a grid stand-in benchmark in the Zone-A layout; returns out_dir."""
    out = Path(out_dir)
    if (out / LABELS_F).exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = _feature_names(len(_net(case).bus))

    def frame(X, start):
        df = pd.DataFrame(X, columns=names)
        df.insert(0, "timestamp", pd.date_range(start, periods=len(X), freq="s").strftime("%Y-%m-%d %H:%M:%S"))
        return df

    X_base, state = _simulate(case, baseline_s, rng, 0)
    frame(X_base, "2025-06-01").to_csv(out / TRAIN_F)
    events = _events(case, rng, scenario_s, n_events)
    X_scen, _ = _simulate(case, scenario_s, rng, baseline_s, events, state)     # same normal process
    scen = frame(X_scen, "2025-06-02")
    scen.to_csv(out / TEST_F)
    lab = np.zeros(len(scen), dtype=int)
    for ev in events:
        lab[ev["start"]:ev["end"]] = 1
    pd.DataFrame({"timestamp": scen["timestamp"], "labels": lab}).to_csv(out / LABELS_F, index=False)
    json.dump({"case": case, "seed": seed, "events": events}, open(out / "events.json", "w"), indent=1)
    return out


# --------------------------------------------------------------------------- #
# loading + v3 preparation
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    features: list
    data_min: np.ndarray            # MinMax scaler (fit on the training block)
    data_range: np.ndarray
    train: np.ndarray               # (N, T, D) float32 windows, scaled
    val: np.ndarray
    test_normal: np.ndarray
    test_anomaly: np.ndarray
    test_normal_src: np.ndarray     # "base" or "scen" per normal test window
    test_normal_start: np.ndarray   # start row of each normal test window in its file


def _load(data_dir):
    d = Path(data_dir)
    base = pd.read_csv(d / TRAIN_F)
    scen = pd.read_csv(d / TEST_F)
    n = len(scen)
    scen = scen.merge(pd.read_csv(d / LABELS_F), on="timestamp", how="left")
    assert len(scen) == n, "duplicate timestamps in the labels file"
    scen = scen.rename(columns={"labels": "target"}).fillna({"target": 1})   # unlabeled = abnormal
    base["target"] = 0
    for f in (base, scen):
        f.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    feats = [c for c in base.columns if c not in ("timestamp", "target")]
    assert feats == [c for c in scen.columns if c not in ("timestamp", "target")]
    return base, scen, feats


def _segments(ts: pd.Series) -> np.ndarray:
    """Segment id per row; a new segment starts at a time gap > 1.5x the median step."""
    t = pd.to_numeric(ts, errors="coerce")
    if t.isna().any():
        t = pd.to_datetime(ts, errors="coerce")
        if t.isna().any():
            return np.zeros(len(ts), dtype=int)
        t = (t - t.iloc[0]).dt.total_seconds()
    dt = t.diff().to_numpy()[1:]
    return np.concatenate([[0], np.cumsum(dt > 1.5 * np.median(dt))])


def _windows(frame: pd.DataFrame, feats, lo, rng_, window: int, stride: int = 1):
    X = ((frame[feats].to_numpy(np.float64) - lo) / rng_).astype(np.float32)
    y = frame["target"].to_numpy()
    ok = ~np.isnan(X).any(axis=1)
    seg = _segments(frame["timestamp"].reset_index(drop=True))
    starts = np.array([s for s in range(0, len(X) - window + 1, stride)
                       if ok[s:s + window].all() and seg[s] == seg[s + window - 1]], dtype=int)
    Xw = np.stack([X[s:s + window] for s in starts]) if len(starts) else np.zeros((0, window, len(feats)), np.float32)
    yw = np.array([int(y[s:s + window].max()) for s in starts], dtype=int)
    return Xw, yw, starts


def prepare(data_dir, window: int = 10, split=(0.70, 0.15, 0.15)) -> Prepared:
    base, scen, feats = _load(data_dir)
    n = len(base)
    i_tr, i_va = int(split[0] * n), int((split[0] + split[1]) * n)
    tr = base.iloc[:i_tr]
    lo = np.nanmin(tr[feats].to_numpy(np.float64), axis=0)
    span = np.nanmax(tr[feats].to_numpy(np.float64), axis=0) - lo
    span[span == 0] = 1.0                                      # constant features (as sklearn)
    w = lambda f: _windows(f, feats, lo, span, window)
    Xtr, _, _ = w(tr)
    Xva, _, _ = w(base.iloc[i_tr:i_va])
    Xbt, _, sbt = w(base.iloc[i_va:].reset_index(drop=True))
    Xsc, ysc, ssc = w(scen)
    normal = np.concatenate([Xbt, Xsc[ysc == 0]])
    return Prepared(
        features=feats, data_min=lo, data_range=span, train=Xtr, val=Xva,
        test_normal=normal, test_anomaly=Xsc[ysc == 1],
        test_normal_src=np.array(["base"] * len(Xbt) + ["scen"] * int((ysc == 0).sum())),
        test_normal_start=np.concatenate([sbt, ssc[ysc == 0]]))


def select_anchors(prep: Prepared, scores: np.ndarray, tau: float, n: int, window: int,
                   seed: int = 0) -> np.ndarray:
    """Indices of up to n normal test windows that are below tau, chosen at random
    but non-overlapping within each source file (starts >= window apart)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scores))
    taken: dict = {"base": [], "scen": []}
    out = []
    for i in order:
        if scores[i] > tau:
            continue
        src, s = prep.test_normal_src[i], prep.test_normal_start[i]
        if all(abs(s - t) >= window for t in taken[src]):
            taken[src].append(s)
            out.append(int(i))
            if len(out) == n:
                break
    return np.array(sorted(out), dtype=int)
