"""Shared helpers: config, paths, crash-safe files, resumable result logs,
attack definitions, per-query timeouts."""

from __future__ import annotations

import copy
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# config + paths
# --------------------------------------------------------------------------- #
def load_config(path: str | None = None, quick: bool = False) -> dict:
    cfg = yaml.safe_load(open(path or HERE / "config.yaml"))
    if quick:                                   # tiny smoke version (reproduce.sh --quick)
        q = cfg["quick"]
        cfg["benchmarks"] = {b: {**cfg["benchmarks"][b], "seeds": q["seeds"]} for b in q["benchmarks"]}
        cfg["train"].update(n_windows=q["n_windows"], epochs=q["epochs"])
        cfg["protocol"].update(eps=q["eps"], attacks=q["attacks"])
        cfg["grid"].update(baseline_s=q["baseline_s"], scenario_s=q["scenario_s"], n_events=q["n_events"])
        for t in cfg["tools"].values():
            t["budget"] = min(t["budget"], q["budget_cap"])
        cfg["work_dir"] = cfg["work_dir"] + "_quick"
        cfg["results_dir"] = cfg["results_dir"] + "_quick"
    cfg["_quick"] = quick
    return cfg


def work_dir(cfg) -> Path:
    return HERE / cfg["work_dir"]


def results_dir(cfg) -> Path:
    return HERE / cfg["results_dir"]


def data_dir(cfg, bench: str, override: str | None = None) -> Path:
    if cfg["benchmarks"][bench]["source"] == "dir":
        if not override:
            raise SystemExit(f"benchmark {bench} reads an existing data folder: pass --data-dir")
        return Path(override)
    case = cfg["benchmarks"][bench]["case"]
    return work_dir(cfg) / "data" / case          # benchmarks sharing a grid case share data


def model_dir(cfg, bench: str, seed: int) -> Path:
    return work_dir(cfg) / "models" / bench / f"seed{seed}"


def model_ids(cfg, benches=None):
    """(benchmark, seed) pairs, in config order."""
    for b, spec in cfg["benchmarks"].items():
        if benches and b not in benches:
            continue
        for s in spec["seeds"]:
            yield b, s


# --------------------------------------------------------------------------- #
# crash-safe files
# --------------------------------------------------------------------------- #
def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ResultLog:
    """Append-only JSONL: one line per finished unit of work, flushed to disk
    immediately. On restart, `done()` lists what is already finished."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> list:
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:       # a line cut off by a crash
                        pass
        return out

    def done(self) -> set:
        return {r["key"] for r in self.records() if "key" in r}

    def append(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())


# --------------------------------------------------------------------------- #
# attacks
# --------------------------------------------------------------------------- #
def attack_frames(attack: str, T: int) -> list:
    """Seconds perturbed by an attack (0-based)."""
    mid = T // 2
    if attack == "persistent":
        return list(range(T))
    if attack == "burst":
        return [4, 5, 6] if T == 10 else [mid - 1, mid, mid + 1]
    if attack == "transient_first":
        return [0]
    if attack == "transient_middle":
        return [mid]
    if attack == "transient_last":
        return [T - 1]
    raise ValueError(f"unknown attack {attack!r}")


# --------------------------------------------------------------------------- #
# timing / timeouts
# --------------------------------------------------------------------------- #
class Timeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Raise Timeout if the block runs longer than `seconds` (main thread, Unix)."""
    def handler(*_):
        raise Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 0.01))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class Stopwatch:
    """Wall-clock and CPU seconds (this process + finished children)."""

    def __enter__(self):
        self.w0 = time.perf_counter()
        t = os.times()
        self.c0 = t.user + t.system + t.children_user + t.children_system
        return self

    def __exit__(self, *exc):
        t = os.times()
        self.wall = time.perf_counter() - self.w0
        self.cpu = t.user + t.system + t.children_user + t.children_system - self.c0
        return False


THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def single_thread_env() -> None:
    """Make spawned pool workers start single-threaded. Must run in the parent
    before the pool starts: a worker imports numpy/torch while unpickling its
    task, before any initializer, so setting these inside the worker is too late
    (24 workers x 28 BLAS threads each made the machine ~25x oversubscribed)."""
    for v in THREAD_VARS:
        os.environ[v] = "1"


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
