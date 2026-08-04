"""Shared experiment infrastructure: result records, env capture, runtime.

Every suite writes one JSONL file per run under experiments/results/.
A record is one (tool, benchmark, anchor, frame-set, threat-model) cell;
the plotting/tabulation layer aggregates cells, never re-runs tools.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
GENERATED_DIR = EXPERIMENTS_DIR / "benchmarks" / "generated"

# Status vocabulary (fixed; plots key off these):
#   certified   tool returned a sound certified radius (radius > 0)
#   uncertified tool ran but certified radius 0 (spec fails at smallest eps)
#   falsified   attack tool found a concrete violation (radius is an UPPER bound)
#   no_violation attack tool found NO violation anywhere in its search range
#               (upper bound uninformative; radius is None)
#   timeout     wall-clock budget exceeded
#   error       tool crashed / raised
#   unsupported tool cannot express this model or spec
STATUSES = ("certified", "uncertified", "falsified", "no_violation",
            "timeout", "error", "unsupported")


@dataclass
class Record:
    suite: str
    tool: str                 # verifier registry name
    benchmark: str            # benchmark id, e.g. "ieee9-S" or "synth-H32-T30-D9-L1"
    anchor_id: str            # which anchor/window
    threat_model: str         # single_frame | multi_frame
    spec: str                 # "specC<=tau" | "specC>=tau" | ...
    status: str
    radius: float | None = None       # certified lb (or attack ub for falsified)
    radius_kind: str = "lower"        # lower | upper
    frames: list | None = None        # frame subset (single_frame), None = all
    seconds: float | None = None
    n_bound_calls: int | None = None
    tau: float | None = None
    model_meta: dict = field(default_factory=dict)   # H, T, D, L_enc, L_dec, params
    tool_meta: dict = field(default_factory=dict)    # method options, versions
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")


class ResultsWriter:
    """Append-only JSONL writer; one line per Record, flushed immediately
    so long sweeps are resumable/inspectable mid-run."""

    def __init__(self, suite: str, tag: str = ""):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{suite}{'_' + tag if tag else ''}.jsonl"
        self.path = RESULTS_DIR / name
        self._fh = open(self.path, "a", buffering=1)
        self._fh.write(json.dumps({"_env": capture_env(), "_suite": suite,
                                   "_started_unix": time.time()}) + "\n")

    def write(self, rec: Record) -> None:
        self._fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")

    def close(self) -> None:
        self._fh.close()


def read_records(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        if not d.get("_env"):
            out.append(d)
    return out


def capture_env() -> dict:
    """Hardware/software provenance stamped into every results file."""
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "blas_threads": {k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    for mod in ("numpy", "scipy", "torch", "auto_LiRPA"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        env["git_rev"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EXPERIMENTS_DIR, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        env["git_rev"] = None
    return env


def pin_blas_single_thread() -> None:
    """All timing suites must call this BEFORE importing numpy-heavy code.

    Falls back to cert_rnn.runtime if available (same policy, one home)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def count_params(encoder: dict, decoder: dict, head: dict) -> int:
    n = 0
    for md in (encoder, decoder):
        for lyr in md["layers"]:
            n += lyr["W_in"].size + lyr["W_rec"].size + lyr["b"].size
    n += head["W"].size + head["b"].size
    return int(n)


class Timeout:
    """Cooperative wall-clock budget. Adapters check .expired between
    bisection evaluations (per-reach granularity; a single reach call is
    never interrupted, so record `seconds` can overshoot slightly)."""

    def __init__(self, seconds: float | None):
        self.budget = seconds
        self.t0 = time.perf_counter()

    @property
    def expired(self) -> bool:
        return self.budget is not None and (time.perf_counter() - self.t0) > self.budget

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
