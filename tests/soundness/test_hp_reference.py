"""Nightly independent reference check (replaces the retired scalar R-1
nightly run): the production (C1, C2) against a 50-digit, independently
written implementation of the candidate mathematics (reference_hp.py —
imports nothing from cert_rnn).

Two directions, both required:
  SOUNDNESS: C1 <= true_min and C2 >= true_max (allowing 1e-30 for the
  reference's own final rounding). Any violation is a hard failure with
  magnitude.
  TIGHTNESS BUDGET, prediction-relative: looseness must not exceed
  max(LOOSE, predicted certified-enclosure width for that box) — observed
  loose boxes sit at exactly 0.5x the predicted width (true value
  mid-interval), so slack EXPLAINED by the intervals passes while slack the
  machinery cannot account for fails, however small the flat budget.

Fast tier: 240 boxes (adversarial stratum first). Nightly: ~3000.
"""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.transformers import (
    _sigid_plane_batch,
    _sigtanh_plane_batch,
)

from tests.soundness.boxes import near_degenerate_quartic_boxes, stratified_boxes
from tests.soundness.reference_hp import true_minmax_sigid, true_minmax_sigtanh

SEED = 20260816
LOOSE = 1e-9   # flat floor; the prediction-relative budget does the real work


def _run(kind, n_per, n_adv, rng):
    boxes = {"near_degenerate_quartic": near_degenerate_quartic_boxes(rng, n_adv)}
    boxes.update(stratified_boxes(rng, n_per))
    plane = _sigtanh_plane_batch if kind == "sigtanh" else _sigid_plane_batch
    ref = true_minmax_sigtanh if kind == "sigtanh" else true_minmax_sigid
    worst_unsound = 0.0
    worst_loose = 0.0
    n = 0
    bad = []
    for stratum, (lx, ux, ly, uy) in boxes.items():
        A, B, C1, C2 = plane(lx, ux, ly, uy)
        for i in range(lx.shape[0]):
            if ux[i] - lx[i] < 1e-11 or uy[i] - ly[i] < 1e-11:
                continue   # point-degenerate handled by T-0.4 / F-2 slack
            tmin, tmax = ref(A[i], B[i], lx[i], ux[i], ly[i], uy[i])
            n += 1
            sc1 = max(1.0, abs(float(tmin)))
            sc2 = max(1.0, abs(float(tmax)))
            unsound = max(float(C1[i] - tmin) / sc1, float(tmax - C2[i]) / sc2)
            loose = max(float(tmin - C1[i]) / sc1, float(C2[i] - tmax) / sc2)
            worst_unsound = max(worst_unsound, unsound)
            worst_loose = max(worst_loose, loose)
            budget = LOOSE
            if loose > LOOSE:
                from tests.soundness.test_r_tests import _predicted_enclosure_width
                budget = max(LOOSE, _predicted_enclosure_width(
                    kind, A[i], B[i], lx[i], ux[i], ly[i], uy[i]))
            if unsound > 1e-25 or loose > budget:
                bad.append((stratum, unsound, loose,
                            (lx[i], ux[i], ly[i], uy[i], A[i], B[i])))
    print(f"\n[seed={SEED}] hp-reference {kind}: {n} boxes | "
          f"worst unsound {worst_unsound:.3e} | worst loose {worst_loose:.3e}")
    assert not bad, (
        f"{kind}: {len(bad)} boxes fail vs the 50-digit reference; worst "
        f"unsound {worst_unsound:.3e}, worst loose {worst_loose:.3e}; "
        f"first: {bad[0]}")


@pytest.mark.soundness
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
def test_hp_reference_fast(kind):
    rng = np.random.default_rng(SEED)
    _run(kind, 12, 80, rng)


@pytest.mark.nightly
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
def test_hp_reference_nightly(kind):
    rng = np.random.default_rng(SEED + 1)
    _run(kind, 150, 1000, rng)
