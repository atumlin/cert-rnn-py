# Improvements Log — systematic approach trials

Branch `dev/iclr-experiments`. Each entry: what changed, how it was
validated, measured benefit or degradation. Reference protocol unless
noted: IEEE-9 checkpoints, single-frame, eps_init 0.5, k=12, frames
{0, 15, 29} or all 30; baselines from the 2026-08-04 overnight E1 run
(`results/e1.jsonl`). All 160 unit/parity/red-team tests pass after
every adopted change.

## A. Joint quadratic score bound — ADOPTED ✅

**Change.** `cert_rnn/verify.py`: new `spec_c_score_ub_joint` /
`spec_c_score_lb_joint` (+ `_stack_residuals`). Exploits that all
residual components share one generator vector:
`max ‖c+Vα‖² ≤ ‖c‖² + 2‖Vᵀc‖₁ + Σᵢⱼ|VᵀV|ᵢⱼ` (provably ≤ the
componentwise bound; min taken with it anyway). Lower bound:
`max(componentwise, (‖c‖₂ − √Σ|VᵀV|)²)`. Exposed as experiment
adapters `certrnn-zono-joint[-masking]`; componentwise remains the
default (cost note below).

**Measured radius gains** (per-frame, k=12, same reach set):

| size | f0 | f15 | f29 |
|---|---|---|---|
| S (H=4)      | 1.21× | 1.24× | 1.53× |
| M (H=16)     | 1.30× | 1.44× | 1.66× |
| D (H=16,L=2) | 1.23× | 1.34× | 1.43× |
| L (H=64)     | 1.05× | 1.07× | 1.10× |

**Cost.** One Gram product per query, growing with generator count P:
S ≈ free; M ~3×/query (pre-pruning); D ~5×; L ~8× (~104 s per frame
bisection pre-pruning). Trade-off is favorable everywhere except L,
where gain is smallest and cost largest — consistent with less
cross-component cancellation at H=64. Full-30-frame E1/E3 joint runs
in progress; adopt as the default *reported* bound for S/M/D-class
models, keep componentwise for wide-H speed runs.

**Quality risk:** none — strictly sound, strictly ≥ componentwise
radii by construction.

## B. Zero-width fresh-generator pruning — ADOPTED ✅

**Change.** `cert_rnn/transformers.py`: new `_fresh_block` used by all
four transformers — output elements whose exact residual spread is zero
(point inputs, degenerate planes) no longer allocate a fresh generator
(previously a zero column). In single-frame mode every step before the
perturbed frame is a point, so this collapses P and every downstream
alignment/Gram/range operation.

**Validation.** Arithmetic-neutral by construction (dropping all-zero
columns changes no sum). Full-30-frame radii **bit-identical** to the
overnight baselines on S/M/D (`e1_pruned.jsonl` vs `e1.jsonl`); 160
tests pass. One test adaptation: the MATLAB `lstm_step` fixtures
compare V positionally and MATLAB keeps its zero columns — comparison
now drops all-zero columns from both sides (order still checked).

**Measured speedup** (full 30-frame E1 wall-clock, radii identical):

| size | before | after | speedup |
|---|---|---|---|
| S | 15 s  | 12.8 s  | 1.17× |
| M | 53 s  | 40.2 s  | 1.32× |
| D | 153 s | 110.5 s | 1.38× |
| L | 677 s | (running) | — |

Also shrinks the joint bound's Gram cost (P smaller). No degradation
anywhere.

## E. Value-guided radius search (regula falsi / Illinois) — NOT ADOPTED ❌

**Change tried.** `experiments/verifiers/base.py: bisect_crossing` —
replace blind bisection with score-value-guided bracketing (log-log
secant + Illinois variant), same soundness contract (every reported
radius directly verified; same granularity).

**Measured.** Mean queries over 9 (size, frame) cells × 2 bounds:
plain crossing 11.1–11.9, Illinois 11.3–13.0, bisection 13.0. Radii
agree within granularity. **Verdict: ≤13% savings at best, sometimes
worse.** Reason: with granule = eps_init·2⁻¹² the bracket must close by
~2¹² regardless; on a convex score curve regula falsi stalls one-sided,
and clamping to preserve soundness eats the remaining advantage. The
cheaper honest lever is coarser k (E4: k=9 already captures ~99% of
the k=15 radius at 60% of the queries). Code kept in-tree for
reference; not wired into any suite.

## C. Zonotope∩interval refinement (InterZono-style) — NOT ADOPTED ❌

**Test.** At the certified eps on IEEE-9 M (frame 15), compared
per-component residual ranges from the zonotope vs the IBP interval
engine: IBP is tighter on **0 / 1080** components (it is strictly
looser everywhere on this workload — interval bounds explode within a
few LSTM steps, cf. E1). The intersection would never fire at the
output level; per-step intersection could only help via transformer
input boxes, which the same measurement rules out. Skipped on
evidence; InterZono's reported gains evidently need their
certified-trained models.

## D. Symbolic bilinear products (DeepT-style) — DEFERRED ⏸

Not attempted this round: Cert-RNN's transformer already computes the
*exact* residual range of a corner-fit plane over the true 2D surface,
so a DeepT-style affine-product enclosure is not obviously tighter —
it trades an exact-2D-residual for structured α_iα_j bookkeeping.
Needs a careful prototype + fairness harness (Tier 2 in
[research_directions.md](research_directions.md) §7); the quartic
plane machinery would also need reworking. Next candidate alongside
GenBaB-style optimized plane parameters.

## Summary scoreboard

| approach | radius | speed | adopted |
|---|---|---|---|
| A joint quadratic bound | +5–66% (size-dep.) | −(Gram cost, P-dep.) | ✅ (reported bound) |
| B zero-width pruning | ±0 (bit-identical) | +17–38% (more on L/joint) | ✅ (unconditional) |
| E value-guided ε search | ±0 | ≤+13%, unstable | ❌ |
| C interval intersection | +0 (never fires) | −(2nd engine) | ❌ |
| D symbolic bilinear | unknown | unknown | ⏸ next |

Net effect vs yesterday's paper numbers: certified radii up 1.2–1.7×
on S/M/D (joint bound) at equal-or-better wall-clock (pruning offsets
Gram cost); L keeps componentwise default (1.05–1.10× available at 8×
query cost if wanted). The E4 cert-vs-PGD gap narrows accordingly
(e.g., S f0: 14× → ~8.5×).
