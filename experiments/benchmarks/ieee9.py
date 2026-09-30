"""Benchmark: the shipped IEEE-9 LSTM-AE checkpoints (S/M/L/D).

Thin wrapper around examples/lstm_ae_ieee9/ae_loader.py producing Case
objects; the MATLAB reference numbers for these live in
examples/lstm_ae_ieee9/matlab_results/ and docs/lstm_ae_results.md."""

from __future__ import annotations

import sys
from pathlib import Path

from experiments.common import count_params
from experiments.verifiers.base import Case

_EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "lstm_ae_ieee9"


def get_cases(sizes=("S", "M", "L", "D")) -> list[Case]:
    sys.path.insert(0, str(_EXAMPLE_DIR))
    try:
        from ae_loader import load_lstm_ae
    finally:
        sys.path.pop(0)
    cases = []
    for size in sizes:
        m = load_lstm_ae(size)
        cases.append(Case(
            benchmark_id=f"ieee9-{size}",
            anchor_id=f"train#{m['anchor_index']}",
            encoder=m["encoder"], decoder=m["decoder"], head=m["head"],
            anchor=m["anchor"], tau=m["tau"],
            meta={
                "H": m["H"], "T": m["T"], "D": m["D"],
                "L_enc": len(m["encoder"]["layers"]),
                "L_dec": len(m["decoder"]["layers"]),
                "params": count_params(m["encoder"], m["decoder"], m["head"]),
                "anchor_score": m["anchor_score"],
                "domain": "power-system (IEEE-9 bus)",
            },
        ))
    return cases
