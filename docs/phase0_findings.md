# Phase 0 findings — Cert-RNN+ soundness scaffolding & instrumentation

Status: step 2 of the Phase 0 execution order (differential ∥ M2) complete.
Everything here is measurement/reporting; no bounding code was changed.

## A. Enumeration audit (Priority 1a) — verdict

The C1/C2 candidate enumeration (`src/cert_rnn/transformers.py`) is
**structurally complete** for both bilinears:

- `_c1c2_sigtanh`: 4 corners; vertical-edge stationary points
  (tanh²(y) = 1 − B/σ(x_e), both signs); horizontal-edge stationary points
  (p² − p + A/tanh(y_e) = 0, p = σ(x), both roots); interior critical points
  via closed-form elimination of ∇g = 0 — the quartic
  p⁴ − (2+B)p³ + (1+2B)p² − Bp − A² = 0, all four complex roots obtained by
  `np.roots` (scalar path) / companion-matrix `eigvals` (batch path), so no
  root can be structurally missed. Derivation verified independently.
- `_c1c2_sigid`: interior is saddle-only (Hessian det = −σ′(y)² < 0) and
  horizontal edges are linear in x, so corners + vertical-edge stationary
  points suffice. Verified.

Residual risk is **numerical, not structural**, and fails inward (flattering):
tolerance gates (`|imag| < 1e-10` realness filter, `ratio < 0.25 − 1e-12`,
`p ∈ (1e-12, 1−1e-12)`, ±1e-12 in-box margins) can drop boundary-adjacent
candidates; companion/`np.roots` roots carry no Newton polish; and there is
**no outward rounding η anywhere** (ledger row S14 unimplemented), so nothing
absorbs either effect.

## B. Finding F-1: scalar/batch divergence, inward-biased, saturation regime

`tests/soundness/test_scalar_batch_diff.py` (seed 20260814, 2 600 boxes,
13 strata) — sigid agrees everywhere (max rel diff 3e-17); **sigtanh
diverges** in the `saturated`/`wide` strata:

- 3/2 600 boxes beyond rel 1e-11; magnitudes 1.3e-11, 1.9e-10, **1.78e-8**;
  all on C1.
- Worst box (11.6908, 13.5908) × (−1.7162, 0.2293): the interior quartic has
  a **near-double root at p → 1** (roots 0.99999790 / 1.00000210). `np.roots`
  (balanced companion) and raw `eigvals` recover it with different error, and
  the logit map x = log(p/(1−p)) amplifies root error by 1/(p(1−p)) ≈ 4.7e5.
- Ground truth (1201² grid + L-BFGS-B refinement): true min −0.18687095519.
  **Scalar C1 sits 1.86e-8 INSIDE the true minimum; batch C1 8.2e-10 inside.**
  Both are unsound-direction offsets at the transformer level; production
  (K ≥ 8: MNIST H=32, PMU H=55, IEEE9) runs the batch path, the existing unit
  tests (K < 8) exercise only the scalar path.

Interpretation per the η policy: violation magnitude ~1e-8 in the saturation
stratum — larger than fp noise (η territory ~1e-13), far below logic-bug
scale (~1e-3). Root cause is root-recovery conditioning near double roots,
not a missing candidate. Candidate remedies (NOT applied — awaiting R-1/R-3
magnitude distributions over the full stratified budget): Newton-polish
quartic roots in x-space; magnitude-relative outward η sized by the R-test
distribution (new ledger row required).

Regression: the failing test stays failing by design until the η/polish
decision is made. It reports the full magnitude distribution on failure.

## C. M2 (proposal §10) — ZRLT go/no-go: **GO**

`research/phase0_instrumentation.py` (eps=0.02, seed 20260814) on ieee9-S
(H=4, T=30, D=36) and synth-H16 (H=16, T=30, D=9), both threat models.
headroom = area(joint 2-D zonotope)/(operand box area): 1.0 ⇒ box exact
(nothing to recover), → 0 ⇒ thin sliver (large Tier-1 recoverable slack).

**Operand correlation does NOT decay with t — it grows.**

single_frame (Algorithm 1's threat model, the paper's headline setting):
- ieee9-S: |cos| rises 0.38→0.91 (f·c), 0.64→0.94 (i·g), 0.59→0.96 (o·tanh c)
  across encoder steps; late-encoder headroom **0.16–0.26** (the box has
  4–6× the area of the true joint set).
- synth-H16: |cos| 0.84–0.93 throughout, decoder 0.85–0.96; headroom
  0.18–0.37.
- Z4 block (Tier 2): ALL six pairwise |cos| among (f,i,g,c_prev) at
  0.85–0.95 late-encoder on both subjects — strong inter-gate coupling for
  the joint-cell LP to recover.

multi_frame: mixed — f·c_prev and o·tanh(c) mostly box-like (headroom
0.88–0.98), but i·g holds headroom 0.51–0.71 and the synth decoder sits
~0.52–0.65; Z4 coupling weaker (0.03–0.76, i:g strongest).

Read-out: ZRLT Tier 1 and Tier 2 both have real headroom, strongest exactly
where Cert-RNN's headline numbers live (single-frame). No pivot needed.
Outputs: research/phase0_out/ (m2_gates.csv, m2_z4.csv, per-subject PNGs,
env.json).

## D. Instrumentation integrity

`src/cert_rnn/instrument.py` re-executes `lstm_step`'s body with recording;
`tests/soundness/test_instrument_parity.py` asserts bit-identical (c, V,
pred_ids) against `verify.lstm_ae_reach` under both threat models. The
closed-form area (4·Σ prefix-sum crosses) is verified against a naive O(p²)
double sum and an exact axis-aligned box case — the box case caught an
initial factor-of-2 error in both the formula and the (equally wrong) naive
test, which is why the independent oracle exists.

## E. Measurement 1 — gap over joint zonotope vs gap over box (tilt fixed)

`research/phase0_gap_ratio.py` (eps=0.02, seed 20260814). For every gate
instance recorded during a reach: gap_box = C2−C1 as the code computes it
today; gap_Z = exact residual extrema over the true joint 2-D zonotope
(vertices via `cert_rnn.geometry.zono2d_vertices` + edge-stationary points
by bisection on the directional derivative + interior criticals inside the
polygon + eps-space sample augmentation). ratio = gap_Z/gap_box ≤ 1;
1−ratio = fraction of the gate's error bar Tier 1 would remove at fixed
tilt. Instances with gap_box < 1e-12 (float-noise gates) are excluded from
ratio statistics but included in gap-weighted aggregates.

| combo | n meaningful | tightening mean [p10, p90] | gap-weighted removal |
|---|---|---|---|
| ieee9-S multi_frame | 712 | 14.4% [0.2, 42.1] | 5.5% |
| ieee9-S single_frame | 712 | **51.4%** [29.0, 71.9] | **37.3%** |
| synth-H16 multi_frame | 2848 | 27.9% [0.6, 52.1] | 20.8% |
| synth-H16 single_frame | 863 | 30.7% [0.0, 55.0] | **46.1%** |

Benefit holds at ALL depths (encoder t0→t29 gap-weighted removal stays
~37–62% on ieee9-S single_frame); largest exactly where certified radii are
decided (single-frame, the paper's headline threat model). Validation:
eps-space samples (interior + sign corners) vs [C1_Z, C2_Z] — worst residual
violation 4.4e-4 of gap_box, i.e. reported percentages accurate to ~0.05 pp.

Measurement-tooling bugs found & fixed during validation (both in NEW
measurement code, not the engine): (i) segment-degenerate point-in-polygon
admitted points beyond the segment ends (perfectly-correlated operands);
(ii) edge-slack normalized by max(1, edge_len) was vacuous for the tiny cap
edges of sliver polygons. Both produced impossible gap_Z > gap_box values
that the "ratio must be ≤ 1" invariant caught; regression tests added in
tests/soundness/test_vertex_enum.py.

## F. Measurement 2 — F-1 propagation to the reported radius

`research/phase0_f1_propagation.py`: one-sided inward shift of C1 by 1e-8
injected at EVERY coordinate of ONE gate at ONE timestep (upper estimate of
F-1's reach), swept over gates × timesteps × phases, both threat models,
both subjects. Effect measured on the final certified score bound and
converted to an equivalent radius shift via dscore_ub/deps (finite
difference).

- Worst equivalent radius shift anywhere: **1.87e-9** (ieee9-S,
  single_frame, enc t=2, f·c_prev).
- Max score-level amplification through the network: ×14; most
  configurations damp the injection (amplification < 1). No exponential
  blow-up with early injection.
- Reported-radius resolution is 1.2e-4 (12-round bisection); relevance
  floor ~1e-6. Worst case sits ~500× below the floor and ~65,000× below
  the resolution.

**Verdict: F-1 cannot change any certified radius this repo reports.** It
remains a documented numerical caveat (see §B) pending the η/root-polish
decision. Caveat: measured on these two subjects at eps=0.02; the ~3–4
orders of margin makes the conclusion robust to subject variation.

### E.1 Reporting decision (deliberate)

The single-frame numbers (51% / 37% error-weighted on the power grid) are
led with because single-frame is Algorithm 1's headline threat model. The
all-frames numbers (**14% average / 6% error-weighted on the power grid;
28% / 21% synthetic**) are materially weaker and are reported with equal
prominence in every table and in the brief; the reader must not come away
with the single-frame figure as "the" gain.

### E.2 Independent oracle check (item 1) — measurement is not too low

Brute-force oracle sharing NO code with the measurement path
(`research/phase0_gap_oracle.py`: dense grid over the operand box,
membership by the exact facet-normal H-representation of the 2-D zonotope
plus a 720-direction fan, coordinate-descent polish constrained inside Z).
56 instances across all four combos, stratified over phase/timestep/gate:
**0 instances where the oracle exceeds the measurement by more than 0.1% of
the gap** (worst deficit 2.1e-4 of gap; mean ratio 0.6516 measured vs
0.6512 oracle). The measurement is not under-reporting gap_Z.

Error accounting, corrected: the ~0.05 pp figure previously quoted is
*sampling* error from the augmentation samples — it bounds how far the
measured [C1_Z, C2_Z] can sit *inside* the true range for the sampled
points, i.e. how much the benefit could be OVER-stated. Systematic error in
the other direction (candidate set missing a true extremum ⇒ benefit
UNDER-stated) is not bounded by sampling; it is what the oracle above
checks, and it found none above 2e-4 of gap. (A first oracle version with a
fixed 720-direction fan and no facet normals over-admitted points near
sliver polygons and falsely reported up to 25% deficits; 40-digit
recomputation sided with the measurement, the oracle was corrected, and
this note stays as the record.)

### E.3 Tilt re-optimization (item 2) — verdict: not for Tier 1

`research/phase0_tilt_reopt.py`: 10 subgradient steps on the convex
gap(A,B) from the box tilt, per gate instance, keep-best. Additional
gap-weighted removal beyond fixed-tilt Tier 1:

| combo | fixed tilt | re-optimized | additional |
|---|---|---|---|
| ieee9-S multi_frame | 3.7% | 7.8% | **+4.0 pp** |
| ieee9-S single_frame | 37.7% | 42.4% | **+4.7 pp** |
| synth-H16 multi_frame | 21.9% | 21.9% | +0.0 pp |
| synth-H16 single_frame | 46.6% | 46.6% | +0.0 pp |

Between the two decision thresholds (≈10% ⇒ include, ≈1% ⇒ skip): 4–5 pp
on the small power-grid model, nothing on the wider synthetic one. Judgement:
**skip for Tier 1** — the gain is model-dependent, at most a ~12% relative
improvement on the recovered slack, and it costs ~10× the enumeration work
per gate. Recorded for the paper as an evaluated-and-declined optimization;
revisit only if a later benchmark shows the power-grid pattern is common.

## G. R-tests against the current code (item 3) — three known signatures

Harness: `tests/soundness/test_r_tests.py` (R-1 two-sided subdivision
equality, R-2 odd-symmetry metamorphic, R-3 dense-grid oracle, T-0.4
degenerate boxes) + `test_level1_zono.py` (Level 1 eps-space sampling with
sign corners, edge points, coordinate-ascent adversarial refinement).
Strata: near-degenerate-quartic adversarial stratum FIRST, then nine sign
strata + saturated / **wide_saturated** / tight / wide / near-degenerate
width regimes. Violations are binned by quartic conditioning 1/(p(1−p)) and
each is classified by a 50-digit recomputation; only violations that carry
a known signature xfail, anything else fails loudly.

Results, both code paths, fast tier (14 280 R-1 checks, 680 R-3 boxes):
- **sigid: all R-tests pass** (R-1, R-3, T-0.4 modulo F-2 below).
- **sigtanh: 101 R-1 / 43 R-3 violations, all classified**; magnitudes by
  conditioning bin — cond ≥ 1e5: max **7.4e-6** (wide_saturated),
  3.1e-8 (saturated), 5.8e-9 (near-degenerate stratum); cond < 1e5:
  ≤ 3.4e-9. Scalar and batch paths carry the same violation set (batch's
  worst is 2× scalar's on the same boxes).

**F-1 escalation.** The mechanism is now precisely characterized and it is
a *drop*, not an imprecision: near a double quartic root at p→1 the float64
root carries ~1e-9 error; through 1/(p(1−p)) ≈ 1e6 that error pushes
A/(p(1−p)) across ±1, `arctanh` is rejected, and the in-box interior
critical point is **removed from the candidate list entirely** (verified at
50 digits on box (11.47, 38.53)×(−4.26, −3.51): true critical point at
x=14.77, y=−3.84 with g=−0.9919846 vs reported C1=−0.9919759). Magnitude
grows with x-range width: ≤ 3e-8 for widths ≤ 3, up to **1.7e-5** for
saturated x-ranges 2–30 wide (both paths, ~1/3 of such boxes). The previous
"~1e-8" ceiling in §B was for narrow boxes only.

Consequence for §F: the propagation study is linear in the injected δ, so
scaling its worst equivalent radius shift (1.9e-9 at δ=1e-8) to δ=1.7e-5
gives **~3e-6** — still ~40× below the 1.2e-4 reporting resolution and at
the 1e-6 relevance floor, but with far less margin than reported. The
verdict "cannot change a reported radius" stands on these subjects but is
now conditional on real networks not producing wide saturated pre-activation
boxes at every coordinate of a gate; wide-saturated boxes did not occur in
any recorded gate of the two subjects at eps=0.02. **The fix should move up
from "before the paper" to "before end-to-end Tier 1 numbers are trusted."**

**F-2** (new, tiny): a nonzero interval width below the 1e-12 point gate is
treated as a point (zero residual). Real dropped error ≤ 0.25·max|x|·width
≈ 4e-12 in T-0.4. Needs the outward η, nothing else.

**F-3** (new, tiny): the `ratio > 1e-12` / `p ∈ (1e-12, 1−1e-12)` gates
reject legitimate edge/interior stationary points when σ(x) or 1−σ(x) is
below 1e-12 (|x| > 27.6). Absolute magnitude ≤ ~1e-11 (three instances at
1.3–2.5e-13 rel in the near-degenerate stratum, x ≈ −28.8). Same fix family
as F-1 (evaluate candidates outward over an interval instead of gating).

Bar for proceeding (zero violations across the nightly budget): **not met
on sigtanh; met on sigid.** The R-tests xfail on the three signatures and
will hard-fail on any new one — that is the harness Tier 1 is written
against. Ledger row added (soundness.md S15).

## Open items (next steps per plan)

- Vertex enumeration promotion + property tests (step 3).
- R-1 / R-2 / R-3 / T-0.4 / Level-1 sampling + near-degenerate quartic
  stratum; nightly-scale run; η decision from magnitude distributions
  (step 4).
- Case-1/4 closed-form oracle (step 5); M1, M3 (step 6); doc updates
  (step 7).
