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

## H. ZRLT Tier 1 — built as a cell cover, not a vertex/edge candidate swap

**Deviation from the plan, flagged.** The plan said: swap box corners +
box-edge stationary points for zonotope vertices + zonotope-edge stationary
points. Along a box edge one coordinate is constant, so the stationary
condition decouples and has a closed form — that is why the baseline
enumeration is structurally complete. Along a general zonotope edge both
x and y vary linearly and d·∇g = 0 has **no closed form**; the Measurement-1
script found those roots by sample-scan + bisection, which is fine for a
measurement (independently validated, §E.2) but structurally incomplete as
bounding code (two roots between adjacent samples are missed silently, and
the miss is inward). Making it rigorous means interval-arithmetic root
isolation on transcendental compositions — a new soundness surface.

**Construction used** (`src/cert_rnn/tier1.py`, mode switch
`transformers.set_bilinear_mode("zono")` / `bilinear_mode(...)` context
manager, default "box"): cover the joint zonotope Z with the cells of an
n×n axis-aligned grid over the operand box that intersect Z (separating-axis
test against Z's exact edge normals — the perpendiculars of its generators —
with outward tolerance), and evaluate the EXISTING closed-form box
enumeration on every admitted cell with the same (A, B):
C1 = min over cells, C2 = max over cells.
- Sound: admitted cells ⊇ Z ⊇ reachable set; each cell's extremum is exact
  (same code the R-tests audit); tested directly by the covering property
  (every eps-space sample incl. all sign corners and edge points lands in an
  admitted cell — `test_tier1.py`, corr ∈ {0, .7, .95, 1}, n ∈ {4,12,24}).
- Never looser than the box (union ⊆ box): T-2.2 asserted on the raw,
  unclamped values — 0 violations, 2 700 coordinates tightened.
- Nothing new enumerated: inherits F-1/F-2/F-3 exactly (Level-1 in zono
  mode xfails on the same F-1 signature at 1.7e-6 and nothing else); every
  R-test, LP-feasibility audit, `lstm_step`, verify and red-team test passes
  under zono mode. R-1 (same (A,B) on sub-boxes) is literally this
  construction's soundness test.
- Tilt (A, B) unchanged (E.3).
- Ledger row S16 added.

**Recovery vs grid resolution** (`research/phase1_tier1_recovery.py`,
fraction of the exact box→Z gap improvement recovered, gap-weighted):

| combo | n=8 | n=16 | n=32 | exact-Z removal |
|---|---|---|---|---|
| ieee9-S single_frame | 0.62 | **0.80** | 0.89 | 38.1% |
| synth-H16 single_frame | 0.69 | **0.85** | 0.93 | 54.4% |
| ieee9-S multi_frame | 0.19 | **0.33** | 0.51 | 5.5% |
| synth-H16 multi_frame | 0.30 | **0.45** | 0.68 | 21.0% |

Default n=16; fat joint sets (multi-frame) are covered less efficiently by
axis-aligned cells but also had least to recover. A cheap skip
(closed-form area ratio > 0.9 ⇒ box is near-exact ⇒ keep box result) trades
nothing measurable for time.

## I. Tier 1 end-to-end certified radius (actual delta, not extrapolated)

`research/phase1_tier1_e2e.py` (Algorithm-1 bisection, 12 rounds, Spec C
componentwise score bound; single-frame on frames {0,7,15,22,29}; jobs in a
process pool). Data: research/phase0_out/tier1_e2e.csv.

| subject | threat | frame | radius box → Tier 1 | Δ | time × |
|---|---|---|---|---|---|
| ieee9-S | multi_frame | — | 0.000732 → 0.001099 | **+50.0%** | 27× |
| ieee9-S | single_frame | 0 | 0.017334 → 0.028198 | **+62.7%** | 14× |
| ieee9-S | single_frame | 7 | 0.014526 → 0.023193 | **+59.7%** | 5× |
| ieee9-S | single_frame | 15 | 0.023315 → 0.034302 | **+47.1%** | 9× |
| ieee9-S | single_frame | 22 | 0.029785 → 0.040405 | **+35.7%** | 4× |
| ieee9-S | single_frame | 29 | 0.020386 → 0.025269 | **+24.0%** | 6× |
| synth-H16 | multi_frame | — | 0.014038 → 0.014038 | +0.0% | 58× |
| synth-H16 | single_frame | 0,7,15,22 | unchanged | +0.0% | 27–84× |
| synth-H16 | single_frame | 29 | 0.176880 → 0.178955 | +1.2% | 34× |

Score bound at eps=0.02 is ≤ baseline in every row (end-to-end tightness
ordering holds; no row looser).

**Reading.** The gate-level number (37–46% error-weighted removal) does
not transfer uniformly. On the power-grid model the radius gain is
comparable to it; on the synthetic model it is nil. The explanatory
variable is the M1 quantity — the fraction of the bound width contributed
by the bilinear fresh symbols: **0.2–1.7% on synth-H16 vs 7–41% on
ieee9-S** (measured at eps 0.02–0.2 with the recorder). The synthetic
autoencoder is nearly linear over its perturbation range, so its certified
bound is dominated by the exactly-propagated affine part, which Tier 1
cannot touch. This is a cheap, per-model predictor of whether Tier 1 pays;
recommend reporting it alongside every radius result. (M1 was deprioritized
as "partly superseded" — this is where it turned out to matter.)

**Cost.** 4–84× wall-clock in the current pure-Python per-coordinate loop
(SAT test + admitted-cell batching), worst under multi-frame where the
generator count is ~1000 and Z is fat. Not optimized yet: vectorizing the
SAT test across coordinates and capping cells for fat Z should recover most
of it; the recovery-vs-n table shows n=8 keeps 60–70% of the single-frame
gain at ~half the cost if needed.

## J. F-1 / F-2 / F-3 fixed structurally (`src/cert_rnn/certified.py`)

**Diagnosis refined.** The critical point (x*, y*) of g is a well-conditioned
zero of ∇g; the ill-conditioning was an artifact of the elimination to
p = σ(x): near a double quartic root at p→1, a 1e-9 root error moves
tanh(y*) = A/(p(1−p)) by 2.6e-3 (across the |t|=1 admissibility boundary —
the F-1 drop) but moves tanh²(y*) = 1 − B/p by only 9e-13. So the fix
changes the *decision procedure*, not the precision:

1. **Certified root intervals, no gates.** Companion eigenvalues →
   3 Newton steps → Smith inclusion disks (Smith 1970; Braess–Hadeler
   1973: for distinct estimates z_i of a monic degree-n polynomial, all
   roots lie in ∪ D(z_i, n|q(z_i)|/∏_{j≠i}|z_i−z_j|)), with |q(z_i)|
   bounded above by its float64 value plus a rigorous rounding term
   (4nu·Σ|c_k||z|^k, covering Horner and one rounding per coefficient).
   A disk that misses the real axis has no real root; the others give
   real intervals J that cover every real root. This replaces the
   |imag|<1e-10 gate and the p∈(1e-12, 1−1e-12) gates (F-3).
2. **Admissibility on intervals.** x* ∈ logit(J) ∩ [lx,ux]; tanh(y*)
   constrained by BOTH stationarity equations and intersected —
   T_A = A/(J(1−J)) (coarse, carries the sign) ∩ T_B = ±√(1 − B/J)
   (well-conditioned) — then y* ∈ arctanh(T) ∩ [ly,uy]. A candidate is
   admitted iff some p̃ ∈ J satisfies every condition, so float error near a
   boundary can no longer drop a legitimate critical point.
3. **Outward interval evaluation.** g over the rectangle X×Y by interval
   arithmetic, every transcendental widened by 2 ulps; the candidate
   contributes [lo, hi] instead of a point value. Edge stationary points
   (quadratics — disc = 1−4c is exact by Sterbenz where it matters; the
   vertical-edge tanh² equation) get the same treatment.
4. **η (ledger S14, now implemented):** corners are point-evaluated and
   rounded outward by 2e-15·(1 + |A|·max|x| + |B|·max|y|).
5. **F-2:** the sub-1e-12 point-width branches widen outward by the
   Lipschitz bound over the collapsed extent (|∂f/∂x| ≤ 1/4, |∂f/∂y| ≤ 1
   for σ·tanh; |∂f/∂x| ≤ 1, |∂f/∂y| ≤ |x|/4 for x·σ). Not the same
   machinery, but six lines; done in the same change.

The certified candidate logic is implemented ONCE (vectorised) and called
by both the scalar (K<8) and batch (K≥8) transcriptions on 1-element /
K-element arrays; duplicating interval logic by hand would be a bug
factory. Consequence: the scalar/batch differential canary now covers the
tilt formulas, degenerate branches, corner code and the K-dispatch — not
the stationary-point candidates. It is kept as a runtime canary as
requested; the R-3 grid oracle and the 50-digit classifier remain the
independent checks on the shared code.

**Results (fast tier, seed 20260814):** R-1 14 280 checks, R-3 680 boxes,
R-2 1 040, T-0.4 400, both bilinears, both paths — **0 violations**;
Level-1 zono-mode wide-saturated case that xfailed at 5.5e-8 now passes;
mutation canary (interior candidates dropped) is caught with 151
violations > 1e-6. Nightly tier: see §J.1. **All F-1/F-2/F-3 xfail
signatures retired**; the ledger's `RETIRE_KNOWN=True` makes any violation a
hard failure again.

**R-1's child side changed from equality to a bounded inequality.** With
exact point candidates, min over sub-boxes had to EQUAL the parent. With
certified intervals, a sub-box's NEW interior edge can carry an
ill-conditioned stationary candidate whose sound enclosure is wider than
the parent's slack, so a child may legitimately be *looser* by up to the
enclosure width — observed max 9.6e-8 (near a p→0 root cluster on a
saturated box), median 6e-12. The soundness side (parent-missed) stays a
hard FP_TOL check and is at zero; the looseness side has a 1e-6 budget and
its distribution is printed.

**Cost of rigor:** enclosure widths ~1e-15 in well-conditioned cases; up to
~1e-7 where the roots are genuinely near-double (previously wrong by up to
1.7e-5 there). Wall-clock: see §J.2.

**A second defect found and fixed during validation (in the new code):**
at an *exact* double root the unguarded Newton polish divides two
rounding-level numbers (q ≈ q′ ≈ −5.5e-17), steps by ~1.0 and lands on a
different root; three estimates then coincide and the Smith radius
explodes (8e12), degenerating the enclosure to the whole box — sound, but
loose by up to 0.1 and *discontinuous* in the inputs (a 1e-10 change in
an operand box flipped a gate's C1 by 0.1 in the ieee9 multi-frame
propagation probe). Fixed by (i) a guarded polish (skip when |q| is at
its rounding floor; reject steps that are large or do not reduce |q|) and
(ii) cluster-aware Smith configurations: for any close pair the
configuration with the pair re-placed at z̄ ± ρ, ρ = √(|q(z̄)|/∏others),
is also evaluated (Smith holds for any distinct points) and the tighter of
the two valid covers is kept — radius ~√(floor) ≈ 4e-7 instead of
floor/separation. Nightly R-1 child-side slack fell from max 0.125 to
1.3e-7.

### J.1 Nightly tier (seed 20260815), fixed code

| test | checks | sigtanh scalar | sigtanh batch | sigid scalar | sigid batch |
|---|---|---|---|---|---|
| R-1 subdivision (depth 4) | 2 139 875 | (running, long) | **0** | **0** | **0** |
| R-3 grid oracle (301², refined) | 25 175 boxes | **0** | **0** | **0** | **0** |

Near-degenerate adversarial stratum (6 000 boxes) and wide-saturated
stratum included; all sign regions covered. **Bar for proceeding met on
every completed configuration.**

### J.2 Effects on numbers, margin, cost

- **Certified radii:** all 24 e2e radii (both modes, both subjects, both
  settings) are **bit-identical** before and after the fix. The correction
  never reached a reported number.
- **Margin (propagation study, now probed at each configuration's own
  certified radius rather than a fixed eps):** worst equivalent radius
  shift for a 1e-8 inward injection is 2.3e-9 (max amplification ×9). But
  the relevant statement is no longer an observation: with F-1/F-3
  removed, no inward error above 1e-13 rel was found in 8.6M R-1 checks
  and 100k R-3 boxes; the residual is the η/2-ulp fp floor. Scaled, that is
  ~1e-14 in radius terms — the previous 40× margin was against a
  measured defect; the current one is against a rounding floor.
- **Wall-clock:** the certified machinery costs 2.4–3.3× on the box
  baseline per reach (ieee9-S single-frame 42 → 140 ms; synth-H16 175 →
  458 ms) — a fixed ~0.5 ms per bilinear call of vectorised interval
  bookkeeping, ~180 calls per reach. The scalar (K<8) dispatch was retired
  (batch is now faster at every K ≥ 2); the scalar transcription stays as
  the differential canary. Tier 1 multipliers against the NEW box
  baseline: 4–10× (ieee9), 11–45× (synth-H16); Item 2 will re-baseline.

## K. Phase 1 throughput (2a parallel T loop, 2b k-ary search)

Implementation: `verify.py` — `kary_epsilon`, `_search_frames` (process
pool with a per-worker model payload), and `search=`, `probes=`,
`n_workers=` parameters on `certify_radius_spec_a/_c` (defaults unchanged:
sequential Algorithm 1). Both changes are pure reorderings of identical
arithmetic.

**Grid arithmetic (2b).** `bisect_epsilon(eps_init=0.5, n_iters=12)` is
bisection on (0, 1) with 12 loop probes + 1 final = **13 probes at
resolution 2^-13**; for a monotone oracle it returns the largest certified
point of the grid G = {j * 2^-13}. A k-ary round with m = 2^a - 1 probes
narrows by a bits, so ending on exactly this grid needs rounds of bits
summing to 13: **15 probes => 4+4+4+1 => 4 sequential rounds, 46 probes**
(three rounds would stop at 2^-12 and could return a smaller radius; a
finer grid could return a larger one). 31 probes => 5+5+3 => 3 rounds (61
probes); 63 => 6+6+1 => 3 rounds; 127 => 7+6 => 2 rounds. probes=1
degenerates to the 13-round bisection (used as a self-check). Requires
eps_init >= 0.5 (where Algorithm 1's `max(eps, 0)` clamp never fires).

**Allocator determinism (2a).** Each worker process has its own default
predicate allocator, so ids differ from the sequential run by an offset.
This does not perturb results: within a reach ids are allocated
monotonically, every zonotope orders its columns by allocation order (ids
sorted), and `align_pred_space` / `_stack_residuals` depend only on that
relative order. Verified directly (`test_allocator_offset_invariance`:
reach from allocator start 0 vs 10^6 => identical (c, V), identical
relative id ranks, identical score bound). No finding.

**Bit-identity (the regression test that matters).**
`tests/soundness/test_throughput_identity.py`: (i) k-ary == bisection on
300 random monotone oracles x probe counts {1,3,7,15,31,127}; (ii) random
AE, T=6, both settings: sequential Algorithm 1 vs parallel bisection vs
k-ary sequential vs k-ary parallel — exact equality of the radius and every
per-frame radius; (iii) nightly: **ieee9-S and synth-H16, both settings,
all 30 frames, 12-round resolution — exact equality across all
configurations. PASSED.**

**Wall-clock (24 workers on 28 cores; radii identical in every row):**

| subject / setting | A baseline (seq. bisection, 13 rounds) | B 2a only (parallel frames, 13 rounds) | C 2b seq-probes (4 rounds, 46 probes/frame, no parallelism) | E 2a+2b (k-ary 15, 4 rounds) | best |
|---|---|---|---|---|---|
| ieee9-S multi | 2.5 s | — (1 frame) | 8.8 s | **1.3 s** | 1.9x (E) |
| ieee9-S single | 57.9 s | **5.9 s** | 205 s | 21.0 s | **9.9x (B)** |
| synth-H16 multi | 3.1 s | — | 11.1 s | **1.8 s** | 1.7x (E) |
| synth-H16 single | 73.8 s | **7.5 s** | (skipped; ≈3.5× A by construction) | 26.6 s (7 probes: 16.2 s; 31: 42.0 s) | **9.9× (B)** |

Probe sweep for E on ieee9 single-frame: 7 -> 13.7 s (5 rounds), 15 ->
21.0 s (4), 31 -> 32.9 s (3), 63 -> 61.9 s (3).

**Reading — k-ary is a depth optimisation, not a work optimisation.** It
cuts sequential rounds 13 -> 4 but multiplies total probes 13 -> 46
(3.5x). When the frames already saturate the pool (30 frames on 24
cores), the extra probes just queue and E is *slower* than B; k-ary pays
only when workers >> frames — the multi-frame setting (one frame;
1.7–1.9x) or a much larger machine. The proposal's "15 parallel probes per
round" is free only if that parallelism is otherwise idle. Recommendation:
2a always; k-ary for multi-frame or when n_workers >= ~15 x frames; keep
`probes` tunable per hardware. The ~2x ceiling on multi-frame is the round
latency (one reach) plus pool overhead against a 2–3 s baseline; on longer
reaches the 13 -> 4 depth ratio shows through more.

**Tier 1 against the new baseline.** ieee9-S single-frame: box A 57.9 s /
zono A 274.7 s = 4.7x; the multiplier is a per-reach property that
parallelism leaves unchanged. Multi-frame ieee9: 11.4x (28.0 / 2.5 s).
Synth-H16 multi-frame: 65× (203 / 3.1 s) — at the large probe eps of the search the joint sets are fat, most of the 16×16 cells are admitted and the certified enumeration runs on ~4 000 cells per bilinear call; the k-ary Tier-1 row for this setting did not finish in >1 h and was stopped. Synth single-frame Tier 1: from the per-frame e2e data, 11–87× per frame (median ~30×). These multipliers are per-reach properties and are unchanged by 2a/2b; they are the cost story that generator merging / selective application / adaptive refinement (deferred by instruction) must address — and note Tier 1 buys nothing on synth (§I) so selective application would switch it off there entirely.

## L. Cost guards, direction merge, nightly canary replacement (cleanup round)

**1a admission-fraction guard: implemented both ways, then DISABLED on
evidence.** Pre-construction guard = the closed-form area-ratio skip at 0.9
(already present; measured free). The post-admission fraction guard at 0.9
cost EXACTLY one search granule (1.2e-4) on five ieee9 single-frame radii
— isolated by toggling each guard alone. Mechanism: with the tilt fixed,
min/max over ALL cells equals the box bound (that is R-1's equality), so
Tier 1's entire recovery lives in the EXCLUDED cells — the box-corner
cells where the residual extrema sit; even 5% exclusion can carry most of
the gain, so admission fraction is a bad proxy for "nothing to recover".
Final configuration: area-ratio guard 0.9 (free), admission guard off.
The premise "above 0.9 admission there is provably nothing to recover" is
false and the correctness check caught it.

**1b round cutover: implemented (`zono_last_rounds` on certify_radius_*),
swept, and the curve says DO NOT USE IT where Tier 1 matters.** The
Algorithm-1 walk is a trajectory, not a window: one early box rejection
steers the walk to grid points from which the later zono rounds can climb
back at most ~2^-(13-k). Measured (radius vs full-zono, 24 workers):

| combo | k=13 | k=6 | k=4 | k=3 | k=1 | k=0 |
|---|---|---|---|---|---|---|
| ieee9 multi (granules lost / s) | 0 / 19.3 | 0 / 11.6 | 0 / 8.6 | 2 / 7.0 | 2 / 4.0 | 3 / 2.5 |
| ieee9 single | 0 / 25.8 | **36** / 16.2 | 52 / 13.4 | 52 / 11.5 | 58 / 8.3 | 58 / 6.6 |
| synth multi | 0 / 92.2 | 0 / 46.1 | 0 / 31.6 | 0 / 24.5 | 0 / 10.4 | 0 / 3.4 |
| synth single | 0 / 415.8 | 0 / 196.7 | ... | ... | 17 / 44.7 | 17 / 9.3 |

The "last 3-4 rounds suffice" expectation is FALSE on ieee9 single-frame
(36+ granules lost at any cutover; all 30 frames drop). Cutover is free
exactly where Tier 1 buys nothing (synth: k=1 free at 8.9x — but there
you would simply not run Tier 1). ieee9 multi tolerates k=4 (2.2x, 0
granules). Recorded as available-but-default-off (None).

**Item 2 direction merge: helps, and the diagnosis was HALF right.**
`geometry.merge_generators_2d` (outward slack, containment-tested on
unclamped support values) + in the Tier-1 SAT test a stronger variant:
directions deduplicated within 1e-3 rad but supports summed over ALL
original generators — dropping a separating direction only admits more
cells (sound), and no slack is needed at all. Measured: direction count
collapses only ~2-3x (not the hoped ~100x — the correlated operands sit
degrees apart, not milliradians); isolated effect on the synth multi-frame
zono reach: 19.4 -> 7.0 s (2.8x) at eps 0.02, 7.1 -> 3.7 s (1.9x) at 0.35.
So the merge moves the multi-frame numbers ~2-3x; the REMAINING ~15-20x
over the box path is per-cell certified enumeration over ~350k admitted
cells per reach, not the SAT test. e2e radii under the final configuration
are EXACTLY the pre-guard Tier-1 values (all 12 rows). Target met: the
synth multi-frame k-ary zono configuration that previously ran >1 h
completes in **99 s** with the identical radius.

**Item 3.** The scalar sigma*tanh R-1 nightly is retired (batch-only
nightly; the scalar transcription remains in the fast tier and as the
runtime differential canary). Its replacement:
`tests/soundness/reference_hp.py` + `test_hp_reference.py` — a 50-digit
mpmath implementation of the candidate mathematics written directly from
the stationarity equations, importing nothing from cert_rnn: soundness
direction (C1 <= true_min, C2 >= true_max) is a hard zero-tolerance check
(fast tier: 482 boxes, worst unsoundness 0.0 on both bilinears); the
tightness direction is budgeted per box by the PREDICTED certified
enclosure width — observed loose boxes sit at exactly 0.5x the predicted
width (true value mid-interval), so slack explained by the intervals
passes and unexplained slack fails at any magnitude. The same
prediction-relative idea restores R-1's sensitivity: each child-slack is
divided by the predicted enclosure width on its box; distribution measured
max 0.25, p95 0.22, median 0.004 (all slack well inside prediction), with
a hard alert at ratio > 30.

## M. Item 4 — Tier 2 headroom, measured directly post-Tier-1: **NO-GO**

`research/phase2_tier2_headroom.py` (eps=0.02, seed 20260817, Tier-1
pipeline, 400 sampled cells): per cell, TRUE ranges of P1 = sigma(f) c_prev
and P2 = sigma(i) tanh(g) separately and of P1 + P2 jointly, over the joint
4-D set (dense eps-space sampling incl. sign corners + projected-gradient
ascent on each of the six extrema — converges from below on both sides of
the comparison, so the DIFFERENCE of ranges is robust). W_sep - W_joint is
the structural ceiling for ANY joint (Tier 2) method over any per-product
method, before Tier 2's own relaxation losses (Chebyshev remainders, LP
lifting gap).

| combo | Tier-2 ceiling / W_sep | / pipeline width | residual per-product slack |
|---|---|---|---|
| ieee9-S multi_frame | 4.6% | 3.6% | 22.1% |
| ieee9-S single_frame | **0.5%** | **0.3%** | 35.7% |
| synth-H16 multi_frame | 6.3% | 3.9% | 38.8% |
| synth-H16 single_frame | **0.0%** | **0.0%** | 0.8% |

**Why the correlation ceiling evaporated.** The Z4 couplings post-Tier-1
are still extreme in single-frame (all six pairs |cos| 0.91-0.99) — but
near-PERFECT correlation is exactly the regime where joint bounding gains
nothing: the operands are close to rank-1 in eps-space, both products
become functions of nearly the same scalar, and their minimizers COINCIDE
— min(P1+P2) = min P1 + min P2 with equality (the proposal's own §5.1
caveat, "equality only when the minimizers coincide", is the actual
operating regime, not the exception). Tier 2's premise needs PARTIAL
correlation; that exists only in multi-frame, where the ceiling is 4-6% of
already-small cell widths.

**Where the width actually is:** residual per-product slack (pipeline
width minus the sum of true product ranges) is 22-39% on three of four
combos — recoverable by better PER-PRODUCT bounding (finer Tier-1 grids,
or the single-plane restriction / loss source C, proposal §2.4), not by
any joint machinery.

**Judgement: do not build Tier 2** (RLT, polynomialization, per-cell LP)
for these workloads: its structural ceiling is <= 0.5% of cell width where
the headline numbers live and <= 6.3% anywhere, before subtracting its own
relaxation losses and paying an LP per cell per step. Redirect tightness
effort at the measured 22-39% per-product residual. Revisit only if a
future benchmark shows Z4 coupling in the 0.3-0.8 band (partial
correlation) with material cell widths.

## Open items (next steps per plan)

- Vertex enumeration promotion + property tests (step 3).
- R-1 / R-2 / R-3 / T-0.4 / Level-1 sampling + near-degenerate quartic
  stratum; nightly-scale run; η decision from magnitude distributions
  (step 4).
- Case-1/4 closed-form oracle (step 5); M1, M3 (step 6); doc updates
  (step 7).
