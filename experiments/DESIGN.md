# Experiment Design — Certified LSTM-AE Anomaly Detection (ICLR 2027)

Target: ICLR 2027 (abstract **Sep 18, 2026** AoE, paper **Sep 25, 2026** AoE,
9 pages main text; style files:
https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).
This document is the contract between the code in `experiments/` and the
claims in `paper/`: every claim maps to a suite, every suite writes
`results/*.jsonl` records that `tabulate.py` / `plots.py` turn into the
paper's tables and figures with no by-hand numbers.

## 1. Claims the experiments must support

C1 **Verifiability.** The zonotope engine certifies meaningfully large
   L_inf radii on trained LSTM-AE anomaly detectors, where cheaper sound
   domains (interval/IBP) certify radii orders of magnitude smaller or
   nothing at the same bisection granularity. (Suite E1)

C2 **Scalability.** Certification cost scales to practically sized
   detectors (hidden width, window length, depth, feature count), with a
   speed/tightness trade-off against linear-relaxation baselines
   (CROWN): they are tighter per bound call but orders of magnitude
   slower, and fall off the time budget as models grow. (Suites E1, E2)

C3 **Detector-level guarantees.** Certification composes into the
   guarantees an operator wants from an anomaly detector: certified
   false-alarm robustness (score provably stays <= tau) AND certified
   masking robustness (an attacker provably cannot hide an anomaly,
   score stays >= tau) — yielding certified-robust FPR/TPR curves.
   To our knowledge no prior deterministic verifier targets the
   reconstruction-error spec of an LSTM autoencoder. (Suite E3)

C4 **Honest accounting.** The gap between certified radii and PGD
   empirical upper bounds is reported per tool (soundness cost), plus
   ablations on bisection depth and threat model. (Suite E4)

C5 **Reproducibility / benchmark contribution.** Besides the IEEE-9
   checkpoints, we release a *deterministic benchmark generator* (any
   (H, T, D, L) point regenerates byte-identically from a seed) and
   loaders for public TSAD datasets. VNN-COMP has never included a
   recurrent or anomaly-detection benchmark (checked 2021–2025 benchmark
   repos), so this doubles as a community benchmark proposal.

## 2. Tool landscape (verified 2026-08-04)

Runnable baselines (adapters in `experiments/verifiers/`):

| adapter | what it represents | status |
|---|---|---|
| `certrnn-zono` | this repo: Cert-RNN zonotope transformers (Du et al. CCS'21), pure-Python port, predicate-aliasing fix | the tool under study |
| `interval-ibp` | interval bound propagation, from scratch on the same model dicts | implemented, validated (converges to concrete score at eps→0) |
| `lirpa-ibp` | auto_LiRPA interval mode on the unrolled explicit-op graph | working |
| `lirpa-backward` | auto_LiRPA CROWN (backward linear relaxation) — the linear-relaxation SOTA family (alpha-beta-CROWN lineage) | working (fp32; ~600x slower per call than certrnn-zono on IEEE9-S, tighter radius) |
| `pgd` / `pgd-masking` | multi-restart PGD falsification, empirical UPPER bound | working |

Non-runnable / comparison-table-only (all checked on GitHub, none can
express the reconstruction-error spec without major surgery):

| tool | venue | LSTM? | AE spec? | repo state |
|---|---|---|---|---|
| POPQORN (Ko et al.) | ICML'19 | yes | no (classification margins) | dead since 2020, torch-1.1 era, no license |
| Prover/R2 (Ryou et al.) | CAV'21 | yes | no (argmax only; has a `square` transformer though) | dead since 2021, needs Gurobi 9.1 |
| RnnVerify (Jacoby et al.) | ATVA'20 | **no** (ReLU vanilla RNN only) | no | dead since 2020, 2019-Marabou build |
| Cert-RNN original (Du et al.) | CCS'21 | yes | no | **no public code release** — our port is the only public implementation (confirmed via GitHub/author page/paper page; email authors to double-confirm) |
| NNV star sets (Tran et al. HSCC'23; Pal et al. arXiv:2311.12130) | — | LSTM via stars | no | MATLAB; our MATLAB reference numbers already serve this comparison on IEEE-9 |

Positioning citations (not baselines): DTW-certified AD via randomized
smoothing (arXiv:2605.07690 — probabilistic certificates; we are
deterministic), RNN abstraction refinement (arXiv:2606.12490),
"Autoencoders for AD are unreliable" (arXiv:2501.13864 — motivation).

## 3. Benchmarks

| id | source | D | scale | role |
|---|---|---|---|---|
| `ieee9-{S,M,L,D}` | shipped checkpoints, IEEE-9 bus | 36 | H∈{4,16,64}, L∈{1,2}, T=30 | headline table; MATLAB reference exists (docs/lstm_ae_results.md) |
| `synth-H*-T*-D*-L*` | deterministic generator (`benchmarks/synthetic.py`) | any | lattice around (H16,T30,D9,L1): H∈{4..128}, T∈{10..120}, D∈{9..72}, L∈{1..3} | scalability lattice + E3 anomalies (spike/step/dropout) |
| `smd-machine-1-1` | Server Machine Dataset (OmniAnomaly repo, public) | 38 | H16 L1 T30 | cross-domain generality |
| planned: `morris` | Miss. State/ORNL power-attack PMU dataset (public direct download) | 128 | — | power-domain realism with real attack labels |
| planned: `skab` | SKAB testbed (public, GPL-3.0) | 8 | — | secondary cross-domain |
| optional: SWaT/WADI/EPIC | iTrust request form (~3 days) | — | — | only if time permits; EPIC is the power-relevant one |

Data-access notes: NASA SMAP/MSL now lives on Kaggle (original S3 link
dead); Yahoo S5 is request-gated; real utility PMU data is CEII-restricted,
which is why the field simulates IEEE-bus systems (pandapower/OPAL) — our
IEEE-9 pipeline is standard practice, and there is **no canonical public
IEEE-bus anomaly benchmark**, hence C5.

## 4. Suites

Fairness protocol (all suites): every sound tool answers the same
predicate "Spec provably holds at eps?" inside the SAME Algorithm-1
bisection (same `eps_init`, `n_iters`, frame loop — `verifiers/base.py`);
the quadratic score bound is assembled identically from componentwise
residual bounds for every tool, so comparisons isolate the abstract
domain. Single-thread BLAS everywhere. auto_LiRPA runs fp32 (its CROWN
path requires it) — recorded as a caveat; everything else fp64.

### E1 — Verifiability (`suites/e1_verifiability.py`)
- Grid: ieee9 S/M/L/D × {certrnn-zono, interval-ibp, lirpa-ibp,
  lirpa-backward} + pgd; single_frame, eps_init 0.5, 12 iters (identical
  to the existing MATLAB comparison, so that column is free).
- Outputs: per-frame radii, radius table (paper Table 1), radius bars.
- Cost estimate: certrnn ~25 min total (known); lirpa-backward is the
  budget risk — ~30 s/call on S ⇒ full 30-frame×13-eval runs need the
  3600 s/case budget; run L size last / possibly frames-subset.

### E2 — Scalability (`suites/e2_scalability.py`)
- Synthetic lattice, one axis at a time from center (H16,T30,D9,L1):
  H ∈ {4,8,16,32,64,128}; T ∈ {10,20,30,60,120}; D ∈ {9,18,36,72};
  L ∈ {1,2,3}. Fixed 3-frame subset, 8 iters, 900 s/cell timeout.
- Metric: seconds/bound-call vs axis (log-log slope = empirical
  complexity order), timeouts marked; cactus plot across the lattice.
- Expected: certrnn-zono polynomial in H (transformer loops), lirpa
  CROWN hits timeout early (its unrolled backward pass is
  O(T·(cost of graph))); IBP flat but never certifies past tiny eps.

### E3 — Certified detection (`suites/e3_detection.py`)
- Both properties: false-alarm (`score_ub <= tau`, normal windows) and
  masking (`score_lb >= tau`, anomalous windows — new lower-bound spec,
  `certrnn_zono.spec_c_score_lb` / interval analogue). multi_frame
  threat model (operator-relevant: attacker touches the whole window).
- Anchors filtered to certifiable subjects (correct side of tau at
  eps=0), same convention as certified accuracy on classifiers.
- Derived metric: certified-robust FPR/TPR at target eps = fraction of
  windows whose certified radius >= eps (`tabulate.py --certified-at`).
- Benchmarks: synthetic center + smd (+ morris when wired).

### E4 — Tightness & ablations (`suites/e4_tightness.py`)
- (a) certified radius vs PGD upper bound per frame (per tool) — the
  soundness gap. Smoke finding: on IEEE9-S PGD finds NO violation even
  at eps≈1.0 while certified radius is ~0.013 — the gap is dominated by
  the componentwise-squared score bound (~22% slack documented in
  red_team_report.md) plus zonotope growth; report honestly, motivates
  the "tighter quadratic bound" future-work/improvement phase.
- (b) bisection depth n_iters ∈ {6,9,12,15}: radius/time trade.
- (c) threat model single_frame vs multi_frame.

## 5. Results plumbing

- `common.Record` → JSONL in `results/` (env-stamped: git rev, versions,
  BLAS pinning, CPU); append-only, rerun-safe.
- `tabulate.py`: radius tables, certified-rate-at-eps, scaling tables.
- `plots.py`: cactus, scaling (log-log), radius bars → `figures/*.pdf`.
- `run_all.sh`: cheap→expensive battery order.

## 6. Preliminary smoke numbers (2026-08-04, 28-core Linux, 1-thread BLAS)

IEEE9-S frame 0, eps_init 0.5, 8 iters:

| tool | radius | s/bound-call | note |
|---|---|---|---|
| certrnn-zono | 0.0156 | 0.05 | |
| lirpa-backward (CROWN) | **0.0215** | ~28 | tighter, ~600× slower |
| lirpa-ibp | 0 | ~1.7 | can't certify at 2e-3 granularity |
| interval-ibp | 0 | 0.004 | ditto (validated: ub→concrete score as eps→0; blows up by 1e-4) |
| pgd | no violation ≤ eps≈1 | 0.3 | upper bound uninformative here |

Reading: the paper's story is a **Pareto frontier** (radius vs time),
not "we win everywhere": CROWN is tighter on small models but its cost
is already prohibitive at H=4/T=30 for full sweeps; IBP is fast but
useless; the zonotope engine holds the practical middle. E2 must show
where CROWN's cost curve crosses the budget vs where ours does.

## 7. Risks / open items

1. `lirpa-backward` cost on M/L sizes may force frames-subsets for E1-L;
   acceptable (record `frames`), or report CROWN as timeout — that IS
   the scalability result.
2. Synthetic detector quality: anomalies must be flagged by the clean
   model (loader now enforces certifiable subjects; raise epochs if the
   filter starves).
3. Morris/SKAB loaders are declared but `NotImplementedError` — wire
   before claiming three public datasets in the paper (SMD is the
   reference implementation).
4. alpha-CROWN (`lirpa-crown-optimized`) untested; try once on S — if it
   runs, it's a stronger baseline point at even higher cost.
5. Prover/R2 has a `square` transformer and could in principle encode
   the score; a pinned-env resurrection is a stretch goal, not a
   dependency (dead repo, Gurobi 9.1 pin).
6. fp32-vs-fp64 caveat for auto_LiRPA comparisons — one appendix
   paragraph.
7. Consider emailing Du et al. for the original Cert-RNN code to
   cross-validate the "only public implementation" claim.
