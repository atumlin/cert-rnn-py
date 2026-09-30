# Research Directions: Toward a Tighter, Faster Verification Algorithm

Status: literature synthesis in progress (2026-08-05). §1–§2 are
first-hand inventories of what our approach assumes and what is novel;
§3+ synthesize the external literature and rank improvement directions.

## 1. Current assumptions (what our results are conditioned on)

**A1 — Threat model.** Local robustness only: an $\ell_\infty$ ball
around a *specific* anchor window, on normalized sensor values;
`single_frame` (one timestep perturbed) or `multi_frame` (all steps
jointly). No semantic/temporal-warp perturbations (contrast: the DTW
smoothing line), no distributional guarantees.

**A2 — Abstract domain.** Plain (unconstrained) zonotopes. Every
bilinear transformer call adds one fresh generator per output element
(three bilinear ops per LSTM step per layer: $f\odot c$, $i\odot g$,
$o\odot\tanh c$ → $3H$ fresh generators per step/layer). **No order
reduction**: generator count grows as $O(THL)$, which is exactly why
cost scales $\sim H^{1.6} T^{1.7}$. Precision never degrades from
reduction, but cost is unbounded in depth.

**A3 — Fixed transformer parameters.** The Cert-RNN plane bounds for
$\sigma(a)\cdot b$ and $\sigma(a)\cdot\tanh(b)$ use fixed
interval-endpoint constructions; nothing is optimized per instance
(contrast: $\alpha$-CROWN optimizes relaxation slopes per query).

**A4 — Componentwise quadratic bound.** Spec C is judged via
$\mathrm{score\_ub} = \frac{1}{N}\sum_{t,d}(|c_{td}| + \sum_p |V_{td,p}|)^2$
— each residual component maximized *independently*, ignoring that all
components share the same generator vector $\alpha$. Measured slack
~22% at IEEE-9 scale; this slack dominates the 14–16× cert-vs-PGD gap
and single-handedly hides the MATLAB aliasing unsoundness from PGD.
The masking lower bound inherits the same componentwise structure.

**A5 — Search procedure.** Algorithm-1 bisection: 13 *independent*
sequential bound queries per frame; no information reuse across
epsilon values, frames, or anchors. Frames run sequentially; the
per-query monotone structure (score_ub is monotone in eps) is used
only for binary search, not for curve fitting or warm starts.

**A6 — Numerics & engineering.** fp64 numpy on CPU, round-to-nearest
(soundness holds up to fp rounding — no directed/outward rounding);
per-element Python loops in the transformers (the documented 2.5–4.9×
gap vs MATLAB is pure interpreter overhead); single-thread BLAS.
Nothing is batched or GPU-resident.

**A7 — Architecture scope.** Stacked `LSTMCell` encoder/decoder with
latent-broadcast decoding and a per-step linear head. No GRU,
bidirectional, attention, teacher-forced decoders, or probabilistic
scores (e.g., OmniAnomaly-style likelihoods).

**A8 — Detector model.** Fixed $\tau$ calibrated offline
(1.1×p99 of training scores); window-level mean-squared score;
threshold rule. Certificates are per-window; aggregate FPR/TPR claims
are over the tested window set, not the data distribution.

## 2. Novelty inventory (ours, as of 2026-08-05)

**N1 — First deterministic certification of recurrent-AE
reconstruction error**, in *both* directions: no-false-alarm
($\sup\,\mathrm{score} \le \tau$) and no-masking
($\inf\,\mathrm{score} \ge \tau$). The masking lower bound
(`spec_c_score_lb`) appears in no prior verifier. (Literature agents
tasked to confirm; see §6 threat list.)

**N2 — Detector-level certified FPR/TPR** framing: aggregating
per-window certified radii into operator-meaningful robust detection
rates — the certified-accuracy analogue for anomaly detection.

**N3 — Predicate-alignment soundness fix** (Minkowski-correct
generator identity), plus an in-the-wild demonstration that positional
padding silently aliases generators and inflates certified radii 6–9%
(MATLAB reference, two-layer model) *while remaining invisible to
PGD* — a cautionary result about attack-based validation of verifiers.

**N4 — Only public implementation** of the Cert-RNN transformer family
(original code never released; POPQORN/Prover/RnnVerify dead).

**N5 — Controlled cross-tool harness**: every sound tool answers the
same predicate inside the same bisection with the same quadratic-score
assembly — isolating the abstract domain as the experimental variable.

**N6 — Deterministic benchmark generator** (seed → byte-identical
trained detector at any $(H,T,D,L)$) + first proposed recurrent /
anomaly-detection benchmark for the VNN-COMP ecosystem.

## 3. Confirmed quick win: the joint quadratic bound (measured 2026-08-05)

The score is a quadratic in the *shared* generator vector:
$\mathrm{score} = \tfrac{1}{N}\lVert c + V\alpha\rVert^2$,
$\alpha \in [-1,1]^P$, where $(c, V)$ stack the per-step residual
zonotopes into the union predicate space. Then
$$\max_\alpha \lVert c+V\alpha\rVert^2 \le \lVert c\rVert^2
  + 2\lVert V^\top c\rVert_1 + \textstyle\sum_{ij} |(V^\top V)_{ij}|,$$
which is **provably $\le$ the componentwise bound** (the componentwise
form is recovered by pushing absolute values inside both inner
products) and strict whenever cancellation exists across components.
Cost: one $V^\top V$ matmul on the existing reach set.

Measured on IEEE-9 (per-frame certified radius, $k{=}12$, same reach):

| size | frame | componentwise | joint | gain |
|---|---|---|---|---|
| S | 0  | 0.01733 | 0.02100 | 1.21× |
| S | 15 | 0.02332 | 0.02893 | 1.24× |
| S | 29 | 0.02039 | 0.03113 | 1.53× |
| D | 0  | 0.03015 | 0.03711 | 1.23× |
| D | 15 | 0.02637 | 0.03540 | 1.34× |
| D | 29 | 0.04272 | 0.06116 | 1.43× |

Overhead: S ≈ negligible; D adds ~1.5 s/query (dense $P{\times}P$
Gram; optimizable). Next rungs on the same ladder: Shor/SDP relaxation
of $\max_\alpha \alpha^\top M \alpha$ (tighter than $\sum|M_{ij}|$,
cost $O(P^3)$-ish per query), and the masking lower bound has the
analogous joint form. Also note: a tighter score bound *shrinks the
slack that hid the MATLAB aliasing bug from PGD* — strengthening the
E4 story rather than weakening it.

## 4. Literature: tighter bilinear approximations (surveyed 2026-08-05)

### 4.1 Exact products via polynomial zonotopes — the domain upgrade
- **Kochdumper, Schilling, Althoff, Bak, NFM 2023** (arXiv:2207.02715) +
  **sparse PZs** (IEEE TAC 2021, arXiv:1901.01780): polynomial zonotopes
  are **closed under quadratic maps** — the product of two affine-in-α
  expressions becomes degree-2 monomial terms (α_iα_j) *exactly*, no
  fresh generator, no bilinear overapproximation. Only the scalar
  sigmoid/tanh polynomial fit carries a remainder. Code: CORA (MATLAB).
- **Ladner & Althoff, AAAI 2024** (exponent relaxation): the answer to
  PZ blow-up over T steps — restructure a PZ into a tight,
  low-complexity enclosure by relaxing dependent-factor exponents.
  This is the growth-control knob a PZ-based LSTM verifier needs.
- Assessment vs A2/A3: replacing our "plane bound + fresh generator per
  element" with quadMap keeps the σ(a)–b dependency exactly; the whole
  Cert-RNN bilinear transformer becomes *one* set operation. Biggest
  single precision lever; biggest implementation lift (new set type +
  range bounding + reduction policy).

### 4.2 Tighter *linear* relaxations (stay-zonotope alternatives)
- **GenBaB, TACAS 2025** (arXiv:2405.21063; in α,β-CROWN): optimizable
  McCormick-style planes for x·y / sigmoid / tanh (slopes tuned by
  gradient ascent on the verified bound) + branch-and-bound on
  nonlinearity input domains with pre-optimized branching points.
  Verified LSTMs (beat PROVER on MNIST-LSTM). The "optimize the
  relaxation against the end bound" recipe applies directly to our
  fixed Cert-RNN planes (A3), and BaB gives an anytime-tightening knob.
- **PROVER, CAV 2021**: volume-minimized planes for σ(x)·tanh(y),
  σ(x)·y fit by sampling + soundness repair — strictly tighter than
  Cert-RNN's closed-form planes; certification-aware selection.
- **DeepPrism, arXiv:2511.11699 (Nov 2025, fresh preprint, unverified)**:
  encloses z = σ(x)·tanh(y) with a volume/surface-minimized truncated
  rectangular prism directly in 3D — same bottleneck as ours, newest
  claim; read before building anything.
- **DeepT, PLDI 2021**: zonotope-native bilinear transformer for
  Transformer dot-products — multiply the affine forms symbolically,
  push only the α_iα_j residue into the fresh generator. A "poor man's
  PZ" implementable in our codebase with moderate effort. Likely the
  best precision-per-engineering-hour upgrade to the transformer.
- **RNN-Guard / InterZono, arXiv:2304.07980**: zonotope ∩ interval box,
  same asymptotics, up to 2.18× more precise RNN certification +
  certified training. Cheap add-on to our engine.
- Also: PRIMA (POPL 2022) multi-neuron hulls for S-shaped activations;
  α-sig (arXiv:2408.10491) tightest sigmoid tangent parameterizations;
  MILP S-shaped cuts (arXiv:2410.23362); Taylor-model lines
  (POLAR-Express TCAD 2023 — symbolic remainders, but **no LSTM
  support found**); hybrid zonotopes (functional decomposition,
  IEEE TAC 2025) — PWA-exact but MILP-cost per query, ReLU-oriented.

### 4.3 Tighter quadratic-score bounds (beyond §3's joint bound)
- Our §3 bound = the "exact diagonal + |off-diagonal|" split ≡ CORA's
  quadMap + PZ range bounding in set terms. Known machinery — but
  **no published work certifies a quadratic reconstruction-error score
  of an LSTM-AE end-to-end** (agent searched; absence flagged, not
  proven).
- **Shor/SDP relaxation** of the terminal box-QP
  max α^T(V^TV)α: Nesterov's π/2 approximation guarantee; one SDP of
  size P+1 per query at the very end, no upstream changes. The
  principled next rung; chordal sparsity (Automatica 2024) if P grows.
- **DeepSDP (Fazlyab et al., IEEE TAC 2022)**: quadratic output specs
  are *native* in the S-procedure spec matrix; and LSTM gate products
  are exact quadratic equalities — an SDP encoding of the whole
  unrolled LSTM-AE appears **unpublished** (gap/opportunity, though
  scalability is the obvious risk).
- **Zonotope norm maximization hardness** (arXiv:2509.22849, 2025):
  ℓ-norm maximization over zonotopes is NP-hard with known SDP
  approximation ratios — the complexity frame for our score problem.

## 5. Literature: speed — vectorization, batching, GPUs (surveyed 2026-08-05)

### 5.1 What the fast tools actually do
- **GPUPoly (MLSys 2021, arXiv:2007.10868; in ERAN/ELINA)**: DeepPoly
  backsubstitution as batched matmuls with a **custom CUTLASS kernel
  because cuBLAS can't do directed rounding**; fp-soundness via
  interval coefficients + round-toward-±∞ + Miné-style ulp bumps
  (~2× cost); early termination drops neurons whose interval bounds
  already suffice. ≥190× vs multithreaded CPU DeepPoly.
- **α,β-CROWN / auto_LiRPA**: batching on three axes — output neurons,
  BaB subdomains (10³–10⁵ per GPU pass, α warm-started from parents),
  and instances. VNN-COMP 2025 report: the top tools have converged on
  GPU bound propagation inside BaB; LP/MILP nearly abandoned.
  **Clip-and-Verify (NeurIPS 2025)**: solver-free GPU domain clipping,
  up to 96% fewer BaB subproblems.
- **jax_verify is archived (Oct 2024)** — but its design (bounds as a
  program transformation; `vmap` gives batching for free) is the model
  if we move the numpy engine to JAX. **BaVerLy (OOPSLA2 2025)**:
  verifies a *set* of ε-balls jointly by grouping similar instances —
  the precedent for batching our 30 frames.
- **FP soundness practice**: almost no competition tool is fp-sound
  (VNN-COMP reports); **SoundnessBench (TMLR 2025)** catches unsound
  verifiers with hidden counterexamples. Rigorous options: GPUPoly-style
  directed rounding (CUDA intrinsics `__dadd_ru/_rd` — NOT reachable
  from stock torch/JAX), or ulp-inflation in plain tensors. Defensible
  middle path: **fp32 GPU for the search, one fp64 sound pass to
  certify the final radius.** Our current fp64 round-to-nearest is
  "real-arithmetic sound" like nearly everyone else (assumption A6).

### 5.2 Speedups orthogonal to hardware
- **Incremental verification** (IVAN PLDI 2023, 2.4× geomean; FANC
  OOPSLA 2022) reuses proofs across *network* perturbations — nobody
  warm-starts across *ε values in a radius search*. **Gap: in our
  bisection, relaxation regimes and stable-gate classifications
  computed at ε_hi remain valid ∀ε<ε_hi. Unexploited, publishable.**
- **Spec-guided refinement** (DeepSRGR TACAS 2021, GRENA): tighten only
  the region that threatens the spec — for us: only tighten gate
  elements in the cone that feeds the score; intervals elsewhere.
- **Radius search without bisection**: for a fixed relaxation regime,
  bounds are ~affine in ε (CROWN closed-form radius lineage). Evaluate
  2–3 ε's, fit, solve the crossing, certify once at ε*−δ. **No
  published bisection-aware batched ε search for deterministic
  verifiers — second publishable gap.** Collapses 13 queries → ~3–5.

### 5.3 Porting priority for our engine (agent-assessed + our data)
1. **Vectorize the per-element transformer loops** (numpy matmuls over
   (4H × P) tensors). This is where the H^1.6 *constant* lives; DeepZ→
   GPUPoly evidence suggests 1–2 orders of magnitude before touching
   GPU. Also erases the documented 2.5–4.9× Python-vs-MATLAB gap.
2. **Batch ε × frames** as a leading tensor axis (13×30=390 queries/
   pass — tiny by α,β-CROWN standards).
3. **Affine-crossing ε solves** replacing most of the bisection.
4. **Warm-start/reuse across ε** (stable-gate caching).
5. **GPU last** (torch/JAX fp32-search + fp64-certify, or directed-
   rounding kernels), validated against SoundnessBench-style tests.

## 6. Literature: recurrent verification 2024–26 + novelty threats

### 6.1 What's new
- **GenBaB (TACAS 2025)** verified LSTMs: 84/100 on the PROVER
  MNIST-LSTM benchmark vs PROVER's 63; α,β-CROWN *without* BaB already
  beats PROVER's specialized relaxations. All classification-only,
  image-as-sequence toys. **Verdict: mandatory baseline** — a reviewer
  will ask whether GenBaB can encode our quadratic spec on the unrolled
  AE (it plausibly can, via a multiplication output layer); run it or
  document the encoding obstacle.
- **RNN-SDP (arXiv:2509.17898)**: SDP Lipschitz certification for
  *vanilla* RNNs only — evidence the SDP route hasn't reached LSTMs.
- **No Mamba/SSM verifier found; no certified-training-for-LSTM paper
  found (2023–26)** — two open-gap statements available (flagged as
  absence-of-evidence).
- **Contraction/δISS route** (RENs, IEEE TAC 2024; Bonassi et al.
  L4DC 2020 LSTM-ISS; R2DN 2025): per-step incremental gains compose
  into a cheap global certificate with no unrolling. **Adopt as the
  cheap-global strawman baseline** (an afternoon of work from trained
  weights); expected to be wildly conservative on unconstrained-trained
  LSTMs — which strengthens our local-certification case.
- **OCSDF (ICML 2023)**: 1-Lipschitz signed-distance one-class
  classifier with exact certificates and a **"certified AUROC"** metric
  — adopt the metric vocabulary for our certified FPR/TPR framing.
- **Attack literature to cite as motivation + baselines**: Erba et al.
  ACSAC 2020 (concealment attacks on ICS reconstruction detectors),
  LSTM-encoder-decoder evasion (Computers & Security 2024 — exactly
  our model class), 100%-evasion of AE detectors for protective relays
  (2024), TSAD adversarial-vulnerability protocol (arXiv:2208.11264).
- **VNN-COMP 2025 confirmed**: no recurrent/sequence benchmark in any
  year; closest are Collins-RUL (CNN) and Dist-Shift (FC). Our
  benchmark proposal fills an empty niche. PyRAT lists RNN support and
  is the fp-soundness-conscious tool to watch.

### 6.2 Novelty-threat list (for claim N1)
1. **Guidotti, Pandolfo, Pulina, IEA/AIE 2024** (LNCS 14748,
   "Verifying Autoencoders for Anomaly Detection in Predictive
   Maintenance") — **the one direct threat; content unverified
   (paywalled abstract), almost certainly feedforward ReLU AE
   (pyNeVer lineage). MUST read before submission.** Safe resolution:
   scope N1 to *recurrent/LSTM* AEs and both-direction certified radii,
   and cite them as feedforward precedent.
2. GenBaB — could plausibly encode the spec; neutralize as baseline.
3. CIVET (ICLR 2025, certified VAE training) — probabilistic,
   non-recurrent, training-time; cite and distinguish.
4. DTW smoothing (confirmed NeurIPS 2025 poster) — probabilistic;
   the deterministic/probabilistic foil.
5. Böing & Müller, DSAA 2022 (robust AE verification, feedforward;
   check their spec).
6. Star-set RNN reachability (HSCC 2023) / SHAP refinement
   (arXiv:2606.12490) — deterministic recurrent, classification-only.

**Claim wording that survives all of the above:** "first
*deterministic* certification of a *recurrent* (LSTM) autoencoder's
reconstruction-error anomaly score, in both the false-alarm and
masking directions."

## 7. Ranked improvement plan (toward a better algorithm)

Ordered by (evidence of gain) × (effort)⁻¹, with soundness risk noted.

**Tier 1 — do for this paper (weeks):**
1. **Joint quadratic score bound** (§3): measured 1.2–1.5× radius,
   provably sound, ~1 day to productionize (`spec_c_score_ub_joint` +
   masking analogue + E1/E3 rerun). Also shrinks the slack that
   currently hides aliasing bugs from PGD (improves E4's story).
2. **Vectorize the transformers** (§5.3): 1–2 orders of magnitude
   expected on CPU; erases the Python-vs-MATLAB gap; makes E2's
   frontier even more lopsided. No soundness change (fp64 kept).
3. **Batch ε×frames + affine-crossing radius search** (§5.2–5.3):
   13 queries → ~3–5; *publishable gap* (no bisection-aware ε search
   exists in the deterministic-verifier literature).
4. **GenBaB/α,β-CROWN as encoded baseline + δISS Lipschitz strawman**
   (§6.1): closes the two reviewer questions before they're asked.
5. **Read Guidotti et al. 2024; rescope N1 wording** (§6.2).

**Tier 2 — the "better algorithm" (the next paper or this one's
camera-ready stretch):**
6. **DeepT-style symbolic bilinear transformer** (§4.2): multiply
   affine forms symbolically, push only the α_iα_j residue into fresh
   generators — a "poor man's polynomial zonotope" inside our existing
   engine. Best precision-per-engineering-hour on the domain side.
7. **InterZono-style zonotope∩interval refinement** (§4.2): claimed up
   to 2.18× precision for RNNs at same asymptotics; cheap add-on.
8. **PROVER/GenBaB-style *optimized* plane parameters** (§4.2): tune
   our transformer planes per instance by gradient ascent on the final
   certified bound (soundness via the same repair step PROVER uses).
9. **Terminal Shor-SDP on the score QP** (§4.3): principled next rung
   above the joint bound (Nesterov π/2 guarantee); no upstream changes.

**Tier 3 — the domain rewrite (a separate research arc):**
10. **Sparse polynomial zonotopes with exact quadMap + AAAI-24
    exponent relaxation** for growth control: eliminates bilinear
    overapproximation entirely; biggest lever, biggest lift; also the
    natural home for the quadratic spec (score bounding = PZ range
    bounding).
11. **GPU with honest fp-soundness** (§5.1): fp32-search + fp64-certify
    split first; directed-rounding kernels only if the H≥256 regime
    matters; validate against SoundnessBench.

**Proposed algorithm sketch (Tiers 1–2 composed):** vectorized zonotope
forward with symbolic bilinear products and interval intersection;
per-instance optimized plane parameters where products can't stay
symbolic; residuals assembled in the union predicate space; score
bounded by the joint quadratic form (SDP-refined on demand); radius
found by batched affine-crossing search with stable-gate warm starts
across ε. Every component has a literature anchor, a soundness
argument, and a measured or cited gain estimate.

## 8. Sources
See the three literature-agent reports (2026-08-05) for full citation
lists; key anchors are inlined above. Unverified items are flagged
in-place (DeepPrism claims; Guidotti et al. content; DiffReach code;
absence-of-evidence statements).
