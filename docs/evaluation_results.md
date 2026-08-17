# Evaluation results — Cert-RNN+ (method frozen)

All numbers: ieee9-S (H=4, T=30, D=36, L=1; anchor train#index per
`examples/lstm_ae_ieee9/ae_loader.py`) and synth-H16 (SynthSpec H=16, T=30,
D=9, L=1, seed 0; anchor normal#0 unless stated). Certification is
deterministic (no RNG); Algorithm-1 search 12 rounds = 13 probes at
resolution 2^-13 (one granule = 1.22e-4). Wall-clock: single-thread BLAS,
28-core host, 24 workers where parallel. Sampling-based measurements carry
their seed. Nothing was tuned during this phase: n=16, area-ratio 0.9,
direction tolerance 1e-3, cutover off (None) — all fixed at their
pre-evaluation values.

## 1. Faithful Table 8 baseline

### 1.1 What "faithful" means here
Du et al.'s Table 8 prescribes, per sign case, a plane TILT (A, B) and a
LOCATION for the residual extrema (its C1/C2 formulas). Our transformer
computes the exact residual extrema for any tilt (`cert_rnn.certified`),
so the faithful baseline is Table 8's per-case tilts + exact offsets
(`cert_rnn.table8`, tilt mode `"table8"`), sound by construction. Cases
1 and 4 are the corner-fit tilt; case 2 lists five sub-solutions
(B.2.1–B.2.5, no selection rule stated; (5) is corner-fit) — we keep the
tightest; case 3 is case 2 under the exact y-reflection f(x,−y) = −f(x,y);
cases 5/6/9 A=0 with the min-secant B; cases 7/8 the two edge secants.
The MATLAB reference `CertRNN.m` never implemented cases 2–9 either (its
commented "7-candidate sweep" is corner-fit + tangent planes + IBP), so
this is the first executable Table 8. Full transcription notes are in the
module docstring; the table was read visually at 300 dpi.

### 1.2 Soundness suite on the new tilts (`tests/soundness/test_table8.py`, seed 20260819)
Coverage on the stratified generator: c1=59 c2=53 c3=53 c4=56 c5=52 c6=54
c7=55 c8=58 c9=80. Cases 1/4: (A,B,C1,C2) EXACTLY equal to the shipped
path (bit-for-bit). Cases 2/3: never looser than corner-fit (contain it).
R-3 grid oracle (325 boxes): 0 violations. Level-1 eps-space enclosure
(K=3 and 12): 0 violations. 50-digit reference (78 boxes): 0 unsound.

### 1.3 Findings about the PRINTED Table 8 (transcription trap)
Comparing the paper's stated extremum locations against the exact
extrema for its own tilts:
- Case 2: the table's "C1 = f(lx,uy) − Alx − Buy" is exactly the MAXIMUM
  (100% of boxes, 3e-15) — the label is swapped relative to the paper's
  own proof B.2.1, which calls that plane the upper one.
- Case 7: the table's "C2 = f(x*, ly) − ..." (bottom edge) is the
  MINIMUM (agrees to 1e-7); read literally, the upper plane would sit up
  to 0.60 BELOW the surface on 90% of sampled case-7 boxes.
- Case 5: the extremum sits on a vertical edge but on x=ux specifically
  only 19–34% of the time; the printed location is inside by ~1e-6.
- Case 1: "C1 at (lx,uy)" holds (98%, residual 5e-8).
A literal transcription of the printed offsets ships an unsound
transformer in cases 2 and 7. Using exact offsets makes the tilt choice
the only content of Table 8, which is what we compare.

### 1.4 Table 8 tilt vs universal corner-fit tilt (both exact offsets, box path)
Coverage on real gate boxes at eps=0.02 and per-element wins under
`"best"` (tightest per element):
| combo | c1 c2 c3 c4 c5 c6 c7 c8 c9 | element wins |
|---|---|---|
| ieee9-S multi | 1 0 8 9 10 29 0 5 418 | cornerfit 480/480 |
| ieee9-S single (frame 0) | 86 37 124 115 49 11 9 42 7 | cornerfit 467, table8 13 (all case-2/3 sub-sol. 4) |
| synth-H16 multi | 511 383 511 512 0 2 0 1 0 | cornerfit 1624, table8 296 |
| synth-H16 single | 511 385 511 512 0 0 0 1 0 | cornerfit 1630, table8 290 |

Certified radius and wall-clock (bisection, 24 workers):
| combo | cornerfit | table8 | best | table8 vs cornerfit |
|---|---|---|---|---|
| ieee9-S multi | 0.000732 / 2.6 s | 0.000732 / 4.1 s | 0.000732 / 4.1 s | 0 granules |
| ieee9-S single | 0.012817 / 5.6 s | **0.013062 / 8.9 s** | 0.012817 / 9.2 s | **+2 granules (+1.9%)** |
| synth-H16 multi | 0.014038 / 3.3 s | 0.014038 / 5.4 s | 0.014038 / 5.4 s | 0 |
| synth-H16 single | 0.176880 / 7.5 s | 0.176880 / 13.3 s | 0.176880 / 13.7 s | 0 |

Verdict: the code comment ("corner-fit beats the per-case sweep") is
true PER GATE — corner-fit has the smaller exact gap on 97–100% of
elements — and false END-TO-END on ieee9-S single-frame, where the Table 8
tilt yields a radius 1.9% higher. Per-element gap minimisation is greedy:
the tilt also shapes the affine part that propagates through later steps,
so "best" (per-element tightest) does NOT reproduce table8's end-to-end
edge. Consequence for reported gains: against the faithful baseline the
Tier-1 gain on ieee9-S single-frame is +52% (0.013062 → 0.019897) instead
of +55% against corner-fit; all other combos unchanged. Table 8 costs
1.6–1.8× (up to five tilts per case-2/3 element). Not tuned, not adopted:
the shipped baseline stays corner-fit and both are reported.

## 2. Ablation table

Certified radius / wall-clock (s) / M1 (share of the final bound width
carried by bilinear fresh symbols, measured at that row's certified
radius). Search: 12 rounds. Rows 1–4 and 7 serial (one core); rows 5–6
24 workers. Row 1 is the pre-everything code (git 24e4534) run out of a
worktree; its per-reach path is the old K<8 scalar loop on ieee9 (H=4).
† = wall-clock overlapped with a concurrent 24-worker sweep; re-measured
standalone: row 5 synth single-frame **415.3 s** (table shows 439), row 6
**1442.4 s** (table shows 1688). Radii unchanged (deterministic).

| row | ieee9-S multi: radius / s / M1 | ieee9-S single: radius / s / M1 | synth-H16 multi: radius / s / M1 | synth-H16 single: radius / s / M1 |
|---|---|---|---|---|
| 1 box baseline (pre-everything, 24e4534) | 0.000732 / 2 / 0.036 | 0.012817 / 56 / 0.017 | 0.014038 / 3 / 0.002 | 0.176880 / 75 / 0.015 |
| 2 + certified soundness machinery | 0.000732 / 2 / 0.036 | 0.012817 / 56 / 0.017 | 0.014038 / 3 / 0.002 | 0.176880 / 75 / 0.015 |
| 3 + Tier 1 | 0.001099 / 18 / 0.026 | 0.019897 / 264 / 0.012 | 0.014038 / 88 / 0.002 | 0.178955 / 1455 / 0.010 |
| 4 + Tier 1 + best tilt (table8/cornerfit) | 0.001099 / 19 / 0.026 | 0.019897 / 300 / 0.012 | 0.014038 / 90 / 0.002 | 0.178955 / 1503 / 0.010 |
| 5 + frame parallelism | 0.001099 / 18 / 0.026 | 0.019897 / 25 / 0.012 | 0.014038 / 88 / 0.002 | 0.178955 / 439† / 0.010 |
| 6 + k-ary search (15 probes) | 0.001099 / 15 / 0.026 | 0.019897 / 93 / 0.012 | 0.014038 / 99 / 0.002 | 0.178955 / 1688† / 0.010 |
| 7 faithful Table 8 tilt (box) | 0.000732 / 4 / 0.037 | 0.013062 / 93 / 0.019 | 0.014038 / 5 / 0.002 | 0.176880 / 135 / 0.016 |

Reading the table:
- **Rows 1→2 (soundness machinery): radius IDENTICAL in all four
  combos** — the certified candidate intervals, η, and F-2 slack change
  no reported number (as the propagation study predicted), and the
  end-to-end wall-clock is also unchanged (56.1 vs 55.9 s, 75.5 vs 74.8 s).
  This CONTRADICTS the earlier per-reach micro-benchmark (§J.2 of the
  findings: 2.4–3.3× at fixed eps=0.02); that benchmark measured a
  narrow-eps reach on the old scalar path, whereas the search is
  dominated by wide-eps probes where the batched certified path
  amortises. The earlier figure is superseded by this end-to-end one.
- **Rows 2→3 (Tier 1): the headline.** ieee9-S +50% (multi) and +55%
  (single, min over all 30 frames: 0.012817 → 0.019897); synth-H16 0%
  and +1.2%. M1 falls under Tier 1 (0.017 → 0.012; 0.015 → 0.010): the
  fresh symbols shrink, and the residual width is now dominated by the
  affine part.
- **Row 4 (best-of tilt): radius identical to row 3 everywhere** — the
  per-element tightest tilt buys nothing end-to-end (see §1.4); the +4–5
  pp tilt RE-OPTIMISATION evaluated in the findings (§E.3) was declined
  and is not in the shipped code, so this row is its implemented analogue.
- **Rows 5, 6 (throughput): radius bit-identical to row 3 in every
  combo** — the parallel walk and the k-ary walk are pure reorderings.
  These "unchanged" rows are the demonstration that the throughput and
  soundness work is orthogonal to the tightness work. Frame parallelism
  10.6× on ieee9 single-frame (264 → 25 s); k-ary is slower than plain
  parallel bisection on single-frame (93 vs 25 s: 3.5× the probes on an
  already-saturated pool) and slightly faster on multi-frame (15.4 vs
  18.3 s), exactly as §K predicted.
- **Row 7 (faithful Table 8, box): +1.9% over corner-fit on ieee9
  single-frame, identical elsewhere, 1.6–1.8× the cost.** Against this
  stronger baseline the ieee9 single-frame Tier-1 gain is +52%.


## 3. Sensitivity sweeps

### 3c. Perturbation magnitude ε — M1 and area-ratio headroom (the predictor's mechanism)
Instrumented reach at each ε (single-frame: frame 0). M1 = fresh-symbol
share of output width; headroom = median area(Z)/box (1 = box exact).

| eps | ieee9-S multi: M1 / headroom | ieee9-S single: M1 / headroom | synth-H16 multi: M1 / headroom | synth-H16 single: M1 / headroom |
|---|---|---|---|---|
| 0.001 | 0.0970 / 0.630 | 0.0007 / 0.139 | 0.0002 / 0.701 | 0.0001 / 0.248 |
| 0.003 | 0.3388 / 0.727 | 0.0023 / 0.187 | 0.0005 / 0.702 | 0.0003 / 0.250 |
| 0.01 | 0.4122 / 0.800 | 0.0105 / 0.293 | 0.0016 / 0.702 | 0.0009 / 0.256 |
| 0.02 | 0.4200 / 0.848 | 0.0711 / 0.424 | 0.0032 / 0.703 | 0.0017 / 0.265 |
| 0.05 | 0.4317 / 0.881 | 0.3636 / 0.680 | 0.0081 / 0.706 | 0.0042 / 0.288 |
| 0.1 | 0.4467 / 0.896 | 0.4081 / 0.791 | 0.0164 / 0.717 | 0.0084 / 0.327 |
| 0.2 | 0.4667 / 0.917 | 0.4135 / 0.842 | 0.0340 / 0.722 | 0.0166 / 0.385 |
| 0.5 | 0.4854 / 0.938 | 0.4159 / 0.866 | 0.5066 / 0.970 | 0.0423 / 0.514 |

Reading: on synth-H16 the bilinear error share stays ≤ 4% over two
decades of ε (0.001–0.2) — the network is effectively linear over its
perturbation range, which is why Tier 1 cannot move its radius; on
ieee9-S it rises from < 1% to ~40% by ε ≈ 0.05, i.e. exactly around its
certified radii (0.0007 multi, 0.013–0.02 single), which is where Tier 1
pays. Headroom (box over-coverage) shrinks toward 1 as ε grows in
multi-frame (fat joint sets) and stays small (0.14–0.42) for single-frame
ieee9 up to ε = 0.02.

### 3a. Grid resolution n (n=16 shipped; other n evaluated, not adopted)

Gate level (`research/phase1_tier1_recovery.py`, eps=0.02, seed 20260814):
recovered fraction of the exact box->Z improvement, gap-weighted, and
per-coordinate cost:

| n | ieee9 single | synth single | ieee9 multi | synth multi | cost (us/coord, synth single) |
|---|---|---|---|---|---|
| 4 | 0.33 | 0.36 | 0.08 | 0.08 | 5 800 |
| 8 | 0.62 | 0.69 | 0.19 | 0.30 | 6 200 |
| **16** | **0.80** | **0.85** | **0.33** | **0.45** | **8 000** |
| 32 | 0.89 | 0.93 | 0.51 | 0.49 | 14 100 |
| 64 | 0.94 | 0.96 | 0.63 | 0.53 | 43 300 |

End to end (`sweep_3a_e2e.csv`; Tier 1, k-ary 15 probes, 24 workers; the
first pass of this table was INVALID — the module default `n` was bound
at definition time so every row ran at n=16 — fixed in the parameter
plumbing without changing the default, and re-run):

| n | ieee9 multi radius / s | ieee9 single (frame 0) / s | synth multi / s | synth single (frame 0) / s |
|---|---|---|---|---|
| 4 | 0.000854 / 12 | 0.021973 / 3 | 0.014038 / 66 | 0.202148 / 76 |
| 8 | 0.000977 / 14 | 0.025757 / 4 | 0.014038 / 72 | 0.202148 / 81 |
| **16** | **0.001099 / 19** | **0.028198 / 5** | **0.014038 / 94** | **0.202148 / 91** |
| 32 | 0.001099 / 40 | 0.029419 / 13 | 0.014038 / 170 | 0.202148 / 159 |
| 64 | 0.001099 / 136 | 0.030151 / 53 | 0.014038 / 577 | 0.202148 / 525 |

Reading: on ieee9 single-frame the radius keeps rising past n=16
(0.0282 -> 0.0294 -> 0.0302 at n=32/64, i.e. +4% / +7%) at 2.6x / 10x the
cost; multi-frame saturates at n=16. Synth is n-independent (Tier 1 is
null there). This is the deployment operating-point table; n=16 sits at
the knee. Not tuned: reported, not adopted.


### 3b. Sequence length T (Tier 1 gain vs depth)

`sweep_3b_T.csv`, k-ary 15 probes, single-frame = frame 0. ieee9-S:
the same model on the anchor truncated to its first T frames; synth:
separately trained models of the cached family (H16 D9 L1, seed 0).

| T | ieee9 multi box -> zono (gain) | ieee9 single box -> zono (gain) | synth multi | synth single |
|---|---|---|---|---|
| 5 | 0.015991 -> 0.018677 (+16.8%) | 0.068970 -> 0.085083 (+23.4%) | — | — |
| 10 | 0.005127 -> 0.006104 (+19.0%) | 0.041504 -> 0.054199 (+30.6%) | 0.045044 -> 0.045776 (+1.6%) | 0.309570 (+0.0%) |
| 20 | 0.002075 -> 0.002930 (+41.2%) | 0.031372 -> 0.049438 (+57.6%) | +0.0% | 0.346191 (+0.0%) |
| 30 | 0.000732 -> 0.001099 (+50.0%) | 0.017334 -> 0.028198 (+62.7%) | 0.014038 (+0.0%) | 0.202148 (+0.0%) |
| 60 | — | — | 0.024658 (+0.0%) | 0.484375 (+0.0%) |
| 120 | — | — | skipped (>40 min per radius, no information) | skipped |

Reading: on ieee9 the Tier-1 gain GROWS monotonically with depth in both
settings — the consistency check between two independently measured
quantities (correlation grows with t, §C of the findings) PASSES; no
contradiction. On synth the gain is null at every T (M1 <= 4%).



## 4. PGD attack agreement

Anchors: ieee9-S train anchor (#0), synth-H16 normal#0/#1/#2 (all cached
certifiable anchors). Certified radius: Tier 1 (mode "zono"), Algorithm-1
12 rounds, 24 workers; multi-frame = one radius, single-frame = per-frame
radius for all 30 frames (each frame attacked at its own radius). PGD
(`experiments/verifiers/pgd.py`, unchanged from the ICLR suite): projected
sign-gradient ascent on the concrete reconstruction score, step = eps/5,
projection onto the L_inf ball each step, restarts uniform in the ball with
one restart from the anchor, torch seed 0; "std" = 10 restarts x 60 steps,
"strong" = 50 restarts x 300 steps. Success = concrete score > tau.

| anchor | setting | PGD | 1.0x | 1.25x | 1.5x | 2.0x |
|---|---|---|---|---|---|---|
| ieee9-S#0 | multi | std(10x60) | 0/1 | 0/1 | 0/1 | 0/1 |
| ieee9-S#0 | multi | strong(50x300) | 0/1 | 0/1 | 0/1 | 0/1 |
| ieee9-S#0 | single | std(10x60) | 0/30 | 0/30 | 0/30 | 0/30 |
| ieee9-S#0 | single | strong(50x300) | 0/30 | 0/30 | 0/30 | 0/30 |
| synth-H16#0 | multi | std(10x60) | 0/1 | 1/1 | 1/1 | 1/1 |
| synth-H16#0 | multi | strong(50x300) | 0/1 | 1/1 | 1/1 | 1/1 |
| synth-H16#0 | single | std(10x60) | 0/30 | 30/30 | 30/30 | 30/30 |
| synth-H16#0 | single | strong(50x300) | 0/30 | 30/30 | 30/30 | 30/30 |
| synth-H16#1 | multi | std(10x60) | 0/1 | 1/1 | 1/1 | 1/1 |
| synth-H16#1 | multi | strong(50x300) | 0/1 | 1/1 | 1/1 | 1/1 |
| synth-H16#1 | single | std(10x60) | 0/30 | 30/30 | 30/30 | 30/30 |
| synth-H16#1 | single | strong(50x300) | 0/30 | 30/30 | 30/30 | 30/30 |
| synth-H16#2 | multi | std(10x60) | 0/1 | 1/1 | 1/1 | 1/1 |
| synth-H16#2 | multi | strong(50x300) | 0/1 | 1/1 | 1/1 | 1/1 |
| synth-H16#2 | single | std(10x60) | 0/30 | 30/30 | 30/30 | 30/30 |
| synth-H16#2 | single | strong(50x300) | 0/30 | 30/30 | 30/30 | 30/30 |

**At 1.0x: 0/248 attack instances succeed — no soundness violation.**
Above the radius the two benchmarks separate sharply: on synth-H16 every
frame of every anchor is broken at 1.25x (and the multi-frame radius too),
so the certified radius is within 25% of the empirical robustness radius
there — near-tight; on ieee9-S nothing succeeds even at 2.0x with the strong
attack, so the certified bound is more than 2x below true robustness — the
looseness that remains there is not in the bilinear transformer (Tier 1
already removed the recoverable part) but in the score-bound step and
accumulated affine width, consistent with the earlier red-team analysis
(componentwise squared-error bound is the dominant slack on IEEE-9).


## 4.1 Contradictions with earlier findings (flagged, not smoothed)

1. Certified-machinery cost: findings §J.2 said 2.4–3.3× per reach (fixed
   eps micro-benchmark, old K<8 scalar path); the ablation shows end-to-end
   wall-clock UNCHANGED (rows 1 vs 2). The end-to-end number supersedes.
2. Table 8: the shipped comment "corner-fit beats the per-case sweep" is
   true per gate (97–100% of elements) and false end-to-end on ieee9-S
   single-frame (+1.9% for Table 8). Findings §H/§I gains vs corner-fit
   stand; gains vs the faithful baseline are 3 pp lower on that combo.
3. Sweep 3a's first end-to-end pass was invalid (parameter never reached
   the transformer); corrected and re-run — the corrected curve is the one
   above. Gate-level 3a and everything else in §H were unaffected.
No result contradicts the correlation-grows-with-t finding (3b confirms
it), the M1 predictor (3c confirms it), or the soundness claims (PGD 0/248).

## 5. Competitive head-to-head — scoping (nothing run)

Model facts that gate every tool: LSTM autoencoder (LSTMCell stacks,
enc→latent→dec, per-step linear head), Spec C = mean squared
reconstruction error ≤ τ (a QUADRATIC output spec), two threat models
(single-frame: one frame's D inputs perturbed, others fixed;
multi-frame: all T·D perturbed), float64 weights, T=30.

| tool | accepts our model | conversion needed | threat models | spec | effort / blocker |
|---|---|---|---|---|---|
| **α,β-CROWN / GenBaB** (VNN-COMP winner; BaB over general nonlinearities incl. LSTM `mul`, sigmoid, tanh) | Yes via ONNX or PyTorch nn.Module; our `experiments/verifiers/torch_ae.py::TorchLSTMAE` is already an explicit-op unrolled module (used by the auto_LiRPA adapter, fp32) | (i) ONNX export of the unrolled AE per T with the score head folded in (Sub, Mul, ReduceMean → single scalar output) so the spec is LINEAR: `y ≤ τ` in VNN-LIB; GenBaB supports pow/mul as general nonlinearities. (ii) VNN-LIB generator per (anchor, eps, frame): single-frame = box on frame t's D inputs, all others fixed to the anchor (per-input bounds — supported); multi-frame = full box. (iii) their conda env + a config yaml (`bab.branching.method: nonlinear`); GPU optional (CPU works, slower). (iv) They answer verified/unknown per instance; radius search wraps outside (13 probes per (anchor, frame); we would run frames {0,7,15,22,29} as in our e2e). | both | linear after folding the score head | **~3–5 days**; no licensing blocker (BSD). Lead-time risk: ONNX export of the fp32 unrolled graph and their ONNX→graph parser accepting the LSTMCell unroll (they support Sigmoid/Tanh/Mul/Add/Gemm; our unroll uses only these). Cost risk: T=30 unroll with 4·16 gates/step and BaB per instance is heavy — expect timeouts at larger eps; a timeout counts as "unknown", which biases their radius DOWN — must be reported as such. This is the bar and the only one worth running. |
| **auto_LiRPA (IBP / CROWN / α-CROWN)** — the linear-relaxation lineage of α,β-CROWN without BaB | Already wired (`experiments/verifiers/lirpa.py`, fp32) | none | both | score assembled outside from per-component bounds (holds the score step fixed) | Already runnable; earlier ICLR smoke: CROWN 0.0215 vs zono 0.0156 on IEEE9-S frame 0 at ~600× the cost. Should be re-run against Tier 1 (0.0282). ~half a day. |
| **POPQORN** (Ko et al., ICML'19) | LSTM yes, but classification-only (per-logit CROWN-style bounds); dead since 2020, torch-1.1 era, no license | rewrite of the output layer to a regression head + external quadratic score assembly (same trick as the auto_LiRPA adapter) inside dead code | single-frame natural (they perturb one frame); multi-frame requires their all-frame mode | not native | ~1–2 weeks of surgery in unmaintained code; the result would be a CROWN-family bound already covered by auto_LiRPA — low value. Recommend: cite, do not run. |
| **Prover / R2** (Ryou et al., CAV'21) | LSTM yes; argmax spec; needs Gurobi 9.1; dead since 2021 | has a `square` transformer so Spec C is in principle encodable, but the pipeline is classification-shaped end to end | both in principle | not native | ~1–2 weeks + Gurobi licence; polyhedral domain would be a genuinely different point of comparison, but the maintenance state makes it a poor bet. Recommend: cite; run only if a reviewer demands a polyhedral baseline. |
| RnnVerify (Jacoby et al.) | No — ReLU vanilla RNN only | — | — | — | infeasible; cite. |
| Cert-RNN original code | never released (confirmed earlier); our port + this Table 8 implementation is the only public executable | — | — | — | — |

Recommendation: run α,β-CROWN/GenBaB (start the ONNX export now — it is
the long-lead item) and re-run auto_LiRPA CROWN/α-CROWN against Tier 1;
cite the rest.
