# Soundness Verification Plan — Cert-RNN+

**Read this before writing any bounding code.**

## Why this document exists

Every change in this project makes bounds *tighter*. An unsound bound is also tighter. The two are indistinguishable from the headline metric.

This means the certified radius ε_c **cannot be used as a signal of correctness**. A bug that drops a valid constraint, flips a sign in a symmetric case, or rounds inward will show up as an improved result and will be written into the paper. Every soundness failure in this codebase is a silent failure that looks like success.

Consequence: soundness testing is not a phase. It is a gate on every commit that touches a transformer.

---

## 1. The soundness ledger

Every over-approximation in the pipeline must appear here with an explicit justification. Anything not on this list is an unaudited assumption.

| # | Step | Over-approximation | Direction | Justification |
|---|---|---|---|---|
| S1 | Input abstraction | ℓ∞ ball → zonotope | exact | A box is a zonotope |
| S2 | Affine layers | none | exact | Zonotopes closed under affine maps |
| S3 | σ / tanh transformer | curve → plane pair | outward | Existing Cert-RNN proof |
| S4 | Bilinear gate (Tier 1) | surface → plane pair over Z | outward | C₁ = min_Z F, C₂ = max_Z F; Z contains all reachable pairs |
| S5 | Joint zonotope extraction | none | exact | Generators copied from operand affine forms |
| S6 | Vertex enumeration | none | exact | Requires test — see T-2.3 |
| S7 | Chebyshev surrogate (Tier 2) | polynomial + certified remainder ρ | outward | Remainder must be *proved*, not measured — see T-4.1 |
| S8 | RLT lifting (Tier 2) | drop V_mn = v_m·v_n, keep linear consequences | outward | Relaxation: LP min ≤ true min, LP max ≥ true max |
| S9 | LP dual → affine form | none if duals exact | outward | Round dual-derived offsets outward |
| S10 | Fresh ε_new | discards error correlation | outward | Independent symbols assume worst-case alignment |
| S11 | Generator order reduction | boxes small generators | outward | Only if implemented |
| S12 | DJIS partition | must **cover** the ball | — | **Gaps are unsound.** Overlap is fine. See T-5.1 |
| S13 | DJIS combination | ε_c = min_j ε_c^(j) | outward | Property holds on the union iff it holds on every piece |
| S14 | Floating point | outward rounding by η on point-evaluated corner candidates; 2-ulp outward widening on every interval operation | outward | Implemented in `cert_rnn.certified` (η = 2e-15·(1+|A|max|x|+|B|max|y|)). Assumes libm exp/tanh/log/atanh within 2 ulps. |
| S16 | ZRLT Tier 1 (`cert_rnn.tier1`) | (C₁, C₂) = min/max over axis-aligned grid cells that intersect the joint 2-D zonotope Z (SAT test vs Z's generator perpendiculars, outward tol), each cell bounded by the S15 box enumeration with the box tilt | outward (cells ⊇ Z ⊇ reachable set); never looser than S15 (cells ⊆ box) | Covering property tested in eps-space with sign corners/edges (T-2.x, tests/soundness/test_tier1.py); inherits S15's F-1/F-2/F-3 unchanged, adds no new candidate machinery. See docs/phase0_findings.md §H. |
| S15 | Bilinear (C₁, C₂) | residual min/max over the box: corners (point, +η) ∪ CERTIFIED interval enclosures of g at every edge-stationary and interior critical point (Smith root-inclusion disks with rigorous residual bounds; admissibility decided on intervals; outward interval evaluation) | outward | `cert_rnn.certified`. Structurally complete AND certified in fp (up to the 2-ulp libm assumption). Former findings F-1 (near-double-root candidate drop, ≤1.7e-5), F-2 (sub-1e-12 width as point), F-3 (1e-12 gates) FIXED — docs/phase0_findings.md §J; R-1/R-2/R-3/T-0.4 zero violations both paths. Enclosure slack ≤ ~1e-7 only where roots are genuinely near-double. |

**Rule:** a PR that adds an over-approximation without adding a row here does not merge.

---

## 2. Test levels

### Level 0 — Per-transformer property tests (fast, every commit)

**T-0.1 Table 8 enclosure.** For each of the nine cases, generate boxes satisfying that case's sign conditions. Sample points, assert C₁ ≤ F(x,y) ≤ C₂ where F(x,y) = σ(x)tanh(y) − Ax − By.

**T-0.2 Case coverage.** Assert the box generator actually exercises all nine cases. A suite that silently never reaches case 9 is worse than no suite — case 9 (box contains the origin in both coordinates) is the hardest and most bug-prone.

**T-0.3 Symmetry consistency.** Table 8 states cases 3, 4, 6, 8 as symmetric to 2, 1, 5, 7. Assert each symmetric case equals the sign-flipped result of its partner, computed independently. This is the highest-yield test in the suite; transcription sign errors are the single most likely defect.

**T-0.4 Degenerate boxes.** Zero width (l = u); an endpoint exactly at 0; saturation regions (|x| > 10, where tanh′ → 0 and tangent-point solvers lose conditioning); very wide boxes ([−50, 50]).

**T-0.5 Slope reduction check.** For case 1, assert the derived (A, B) equal the average of the two diagonal secant slopes across the box — the direct analogue of A = (l_y+u_y)/2, B = (l_x+u_x)/2 for plain xy. A case-1 implementation that doesn't reduce to this pattern is wrong.

### Level 1 — Sampling methodology (the part that is easy to get wrong)

**T-1.1 Sample in ε-space, never in (x,y)-space.**

Draw ε ~ U[−1,1]^p and compute the induced (x,y). Do **not** sample the box and reject points outside Z — in even modest dimension the acceptance rate collapses to near zero, the test appears to pass on a handful of points, and it provides no coverage.

**T-1.2 Vertices and edges, not just interior.**

Extrema of F over a polytope usually sit at vertices. Uniform interior sampling almost never approaches one. The sampler must explicitly include:

- every vertex of Z (from the enumeration routine)
- points along each edge
- the ε sign corners ε ∈ {−1, +1}^p (these map to zonotope vertices)
- interior points

A suite that only samples the interior will pass on a bound that is wrong at every corner.

**T-1.3 Adversarial sampling.** Run a short local optimizer (or coordinate descent on ε) to *maximize* the violation max(C₁ − F, F − C₂). Random sampling finds gross errors; optimization finds the marginal ones that matter.

### Level 2 — New component soundness

**T-2.1 Tier 1 enclosure.** Sample per T-1.1 / T-1.2 in the joint zonotope; assert C₁ ≤ F ≤ C₂.

**T-2.2 Tier 1 tightness ordering.** Assert (C₂ − C₁)_Z ≤ (C₂ − C₁)_box for every gate, every time. Tier 1 strictly subsumes the box method, so a single violation is a bug — not a tuning issue.

**T-2.3 Vertex enumeration correctness.** For random generator sets, assert: (a) every enumerated vertex lies in Z; (b) the polygon is convex and centrally symmetric about the center; (c) its area matches an independent Monte-Carlo estimate; (d) vertex count ≤ 2p.

**T-2.4 Tier 2 enclosure.** Sample ε, compute the true P₁ + P₂, assert it lies within the LP-derived bounds.

**T-2.5 Tier 2 tightness ordering.** Assert the joint bound is no wider than the sum of the two independent bounds. Again, a violation is a bug.

**T-2.6 Dual extraction.** After recovering the affine form from LP duals, re-verify enclosure directly on samples. Do not trust the duals; check them.

### Level 3 — End-to-end

**T-3.1 Zero-perturbation collapse.** With ε = 0, the propagated zonotope must reduce to the exact forward pass. Assert ‖α₀ − F_exact(X₀)‖ < 1e−9 and that all generator magnitudes are zero. One assertion, enormous coverage of bookkeeping errors.

**T-3.2 Attack agreement — the strongest check.** For each test sample, run PGD (plus random restarts) at ε = ε_c against the *real* network. **Any success is a soundness violation.** This is the only test that exercises the whole pipeline against ground truth, and it catches errors no unit test will.

Also run PGD at 1.5·ε_c: it *should* usually succeed. If it rarely does, the bounds are very loose and there is more headroom than the ablations suggest — useful either way.

**T-3.3 Baseline regression.** For every sample, assert ε_c^new ≥ ε_c^Cert-RNN. Since every component strictly subsumes the baseline, any decrease is a bug.

**T-3.4 Monotonicity in T.** Certified radius must be non-increasing as sequence length grows on the same model.

### Level 4 — Tier 2 numerical guarantees

**T-4.1 The Chebyshev remainder must be a *bound*, not an estimate.** On a dense grid over the operand interval, plus both endpoints, assert |σ(u) − σ̃_d(u)| ≤ ρ_σ; repeat at higher precision (mpmath, 50 digits). If ρ comes from a truncation-error theorem, test the formula. If it was obtained by sampling the error, **it is not sound** and must be replaced.

**T-4.2 RLT cut validity.** Each generated cut h_k · h_ℓ ≥ 0 must hold at every sampled point of Z₄. Test cuts individually before testing the LP.

**T-4.3 LP relaxation direction.** Assert LP min ≤ sampled min and LP max ≥ sampled max. If the LP ever reports a *tighter* interval than the observed range, the lifting is wrong.

### Level 5 — DJIS

**T-5.1 The partition is a cover.** Sample points uniformly from the full ball; assert each lands in at least one sub-region. **Gaps are silently unsound and will not surface any other way.** Overlap is harmless — do not test for disjointness.

**T-5.2 Identity at k=1.** With one branch, output must match the baseline bit-for-bit.

**T-5.3 Split monotonicity.** Assert ε_c(k) ≥ ε_c(1) for all k, and that each branch's radius is ≥ the unsplit radius.

**T-5.4 Combination rule.** Assert the reported radius equals min_j ε_c^(j) exactly — not the mean, not the median. An off-by-one in the reduction is easy to write and impossible to notice.

---

## 3. Mutation testing

**A soundness suite that has never failed is unvalidated.**

Maintain a set of deliberate defects and a CI job asserting each is caught:

| Mutation | Must be caught by |
|---|---|
| Flip a sign in case 3's A | T-0.3 |
| Swap l_x ↔ u_x in case 7 | T-0.1, T-0.3 |
| Set outward rounding η = 0 | T-0.4 |
| Return box bounds from the Tier 1 path | T-2.2 |
| Drop one RLT cut from the LP | T-4.2 |
| Use `mean` instead of `min` in DJIS combination | T-5.4 |
| Shrink each sub-region by 1% (creating gaps) | T-5.1 |
| Skip the Chebyshev remainder term | T-2.4, T-4.1 |
| *Meta:* switch the sampler to interior-only | should cause the "flip a sign" mutation to **survive** |

That last row is the important one. If interior-only sampling still catches everything, the tests are not measuring what you think they are.

---

## 4. Floating point

**Round outward, always.** After computing C₁, C₂:

```
C1 -= eta
C2 += eta
```

with η scaled *relative* to the magnitude of the values involved. A fixed 1e−9 is meaningless when C₂ ~ 1e3.

Tangent points found by iterative solvers converge to *approximately* the tangency, and an approximate tangency can leave the plane a hair inside the surface. This fails on roughly one point in 1e6 — invisible to a 1000-sample test, fatal to a claimed guarantee.

**Test that η is doing work:** the degenerate-box tests should *fail* with η = 0. If they pass, either η is unnecessary (unlikely) or the tests aren't reaching the boundary.

For Tier 2, prefer an LP solver with exact rational arithmetic, or verify the returned bound independently by substituting it back into the constraints.

---

## 5. CI structure

| Job | Trigger | Contents | Budget |
|---|---|---|---|
| `fast` | every push | Levels 0, 1, 2 with N = 1e3 samples | < 2 min |
| `regression` | every PR | T-3.1, T-3.3, T-5.2 on a fixed 20-sample fixture | < 10 min |
| `nightly` | scheduled | All levels, N = 1e6, adversarial sampling, T-3.2 PGD on the full test set | hours |
| `mutation` | weekly | §3 mutation matrix | hours |

**Merge gate:** `fast` and `regression` must pass. A failing soundness test is never "flaky" — investigate every one.

**Seed and record.** Every sampling test logs its RNG seed. When nightly finds a violation, the failing configuration must be reproducible as a unit test.

---

## 6. What to do when a test fails

1. **Do not widen η to make it pass.** That masks the defect and inflates every bound in the system.
2. Minimize: shrink to the smallest box or generator set that still fails.
3. Add it as a permanent regression test *before* fixing.
4. Check whether the same error class exists in the symmetric cases.
5. Re-run the mutation job — a real defect usually reveals a coverage gap that let it through.

---

## 7. Reporting

The paper must state the soundness protocol explicitly: sampling method, sample counts, whether adversarial sampling was used, the PGD configuration for T-3.2, and the mutation-testing results.

Reviewers of verification papers ask about this, and "we sampled 1000 random points" is not a persuasive answer when the claim is a formal guarantee.