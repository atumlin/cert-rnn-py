"""Shared (tools x cases) matrix driver used by every suite."""

from __future__ import annotations

import numpy as np

from experiments.common import Record, ResultsWriter
from experiments.verifiers import get_verifier
from experiments.verifiers.base import Case


def run_matrix(
    suite: str,
    tool_names: list[str],
    cases: list[Case],
    *,
    threat_model: str = "single_frame",
    frames: list[int] | None = None,
    eps_init: float = 0.5,
    n_iters: int = 12,
    timeout_s: float | None = 900.0,
    tag: str = "",
    spec_label: str = "specC<=tau",
    verbose: bool = True,
) -> list[Record]:
    writer = ResultsWriter(suite, tag)
    records = []
    for case in cases:
        for name in tool_names:
            tool = get_verifier(name)
            if verbose:
                print(f"[{suite}] {case.benchmark_id}/{case.anchor_id} "
                      f"x {name} ...", flush=True)
            r = tool.certify_radius(
                case, threat_model=threat_model, frames=frames,
                eps_init=eps_init, n_iters=n_iters, timeout_s=timeout_s)
            rec = Record(
                suite=suite, tool=name, benchmark=case.benchmark_id,
                anchor_id=case.anchor_id, threat_model=threat_model,
                spec=spec_label, status=r.status, radius=r.radius,
                radius_kind=r.radius_kind,
                frames=list(frames) if frames is not None else None,
                seconds=round(r.seconds, 4), n_bound_calls=r.n_bound_calls,
                tau=case.tau, model_meta=case.meta,
                tool_meta={**r.tool_meta,
                           **({"error": r.error} if r.error else {}),
                           "eps_init": eps_init, "n_iters": n_iters},
                extra={"per_frame": (np.round(r.per_frame, 8).tolist()
                                     if r.per_frame is not None else None)},
            )
            writer.write(rec)
            records.append(rec)
            if verbose:
                print(f"    -> {r.status}  radius={r.radius}  "
                      f"({r.seconds:.1f}s, {r.n_bound_calls} bound calls)",
                      flush=True)
    writer.close()
    if verbose:
        print(f"[{suite}] wrote {len(records)} records -> {writer.path}")
    return records
