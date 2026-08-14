"""Differential test: scalar (K<8) vs batched (K>=8) plane transcriptions.

bilinear_sigmoid_tanh / bilinear_sigmoid_identity dispatch on K to one of two
independently hand-written implementations of the same candidate enumeration
(transformers.py: _sigtanh_plane vs _sigtanh_plane_batch, _sigid_plane vs
_sigid_plane_batch). Only the scalar path is exercised by the existing unit
tests, while production benchmarks run K>=8, i.e. the batched path. The two
must agree on (A, B, C1, C2) for every box; a divergence is a transcription
finding in one of them (arbitrated later by the R-3 grid oracle).

Tolerance is relative to magnitude. Divergences beyond tolerance are reported
with their full magnitude distribution per stratum, not just pass/fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.transformers import (
    _sigid_plane,
    _sigid_plane_batch,
    _sigtanh_plane,
    _sigtanh_plane_batch,
)

from tests.soundness.boxes import stratified_boxes

SEED = 20260814
N_PER_STRATUM = 200
REL_TOL = 1e-11  # both paths use companion-matrix eigenvalues; allow fp noise


def _run_scalar(plane_fn, lx, ux, ly, uy):
    n = lx.shape[0]
    A = np.empty(n)
    B = np.empty(n)
    C1 = np.empty(n)
    C2 = np.empty(n)
    for k in range(n):
        A[k], B[k], C1[k], C2[k] = plane_fn(lx[k], ux[k], ly[k], uy[k])
    return {"A": A, "B": B, "C1": C1, "C2": C2}


def _run_batch(batch_fn, lx, ux, ly, uy):
    A, B, C1, C2 = batch_fn(lx, ux, ly, uy)
    return {"A": A, "B": B, "C1": C1, "C2": C2}


@pytest.mark.soundness
@pytest.mark.parametrize(
    "name,scalar_fn,batch_fn",
    [
        ("sigtanh", _sigtanh_plane, _sigtanh_plane_batch),
        ("sigid", _sigid_plane, _sigid_plane_batch),
    ],
)
def test_scalar_batch_agree(name, scalar_fn, batch_fn):
    rng = np.random.default_rng(SEED)
    print(f"\n[seed={SEED}] scalar-vs-batch differential: {name}")
    boxes = stratified_boxes(rng, N_PER_STRATUM)
    failures = []
    for stratum, (lx, ux, ly, uy) in boxes.items():
        s = _run_scalar(scalar_fn, lx, ux, ly, uy)
        b = _run_batch(batch_fn, lx, ux, ly, uy)
        stratum_max = 0.0
        for comp in ("A", "B", "C1", "C2"):
            sv, bv = s[comp], b[comp]
            scale = np.maximum(1.0, np.maximum(np.abs(sv), np.abs(bv)))
            rel = np.abs(sv - bv) / scale
            stratum_max = max(stratum_max, float(np.max(rel)))
            bad = np.flatnonzero(rel > REL_TOL)
            for k in bad:
                failures.append(
                    (stratum, comp, float(rel[k]),
                     (float(lx[k]), float(ux[k]), float(ly[k]), float(uy[k])),
                     float(sv[k]), float(bv[k]))
                )
        print(f"  {stratum:20s} max rel diff {stratum_max:.3e}")
    if failures:
        failures.sort(key=lambda f: -f[2])
        mags = np.array([f[2] for f in failures])
        lines = [
            f"{name}: {len(failures)} scalar/batch divergences beyond "
            f"rel {REL_TOL:g}; magnitude distribution: "
            f"max={mags.max():.3e} p90={np.percentile(mags, 90):.3e} "
            f"median={np.median(mags):.3e}",
        ]
        for stratum, comp, rel, box, sv, bv in failures[:10]:
            lines.append(
                f"  [{stratum}] {comp}: rel={rel:.3e} box={box} "
                f"scalar={sv!r} batch={bv!r}"
            )
        # Finding F-1 (docs/phase0_findings.md §B): known sigtanh C1
        # divergence from near-double quartic roots in the saturation
        # regime, magnitude <= ~1e-7. Divergences matching that exact
        # signature are an expected failure until the eta/root-polish
        # decision lands; anything OUTSIDE the signature is a NEW finding
        # and must fail loudly, not hide inside the known red result.
        def _is_f1(f):
            stratum, comp, rel = f[0], f[1], f[2]
            return (name == "sigtanh" and comp == "C1" and rel <= 1e-7
                    and stratum in ("saturated", "wide"))

        if all(_is_f1(f) for f in failures):
            pytest.xfail("known finding F-1 (sigtanh C1, saturation, "
                         "<=1e-7):\n" + "\n".join(lines))
        pytest.fail("divergence OUTSIDE the F-1 signature:\n"
                    + "\n".join(lines))
