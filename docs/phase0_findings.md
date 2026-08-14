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

## Open items (next steps per plan)

- Vertex enumeration promotion + property tests (step 3).
- R-1 / R-2 / R-3 / T-0.4 / Level-1 sampling + near-degenerate quartic
  stratum; nightly-scale run; η decision from magnitude distributions
  (step 4).
- Case-1/4 closed-form oracle (step 5); M1, M3 (step 6); doc updates
  (step 7).
