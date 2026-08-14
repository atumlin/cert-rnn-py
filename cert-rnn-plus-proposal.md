# Cert-RNN+ : Correlation-Aware, Refinement-Based Certification of Recurrent Networks

**A proposal extending Cert-RNN (Du et al., CCS 2021) with two orthogonal modules:**

- **ZRLT** — Zonotope-projected relaxation with Reformulation-Linearization cuts. Targets *tightness*.
- **DJIS** — Deferred-Join Input Splitting. Targets *throughput*, and converts spare compute into further tightness.

---

## 1. Thesis

Cert-RNN's certified radius is limited by two distinct, separable losses:

1. **Correlation loss at the gate.** Each bilinear gate operand pair is concretized to an independent interval box before relaxation, discarding the joint dependence that the zonotope already encodes. Worse, each gate introduces its own *fresh, independent* error symbol, so correlations *between* gates in the same cell are destroyed at birth.
2. **Compounding over time.** Whatever slack survives step $t$ widens the boxes at step $t+1$, so per-step error is amplified multiplicatively across $T$.

These call for different tools. ZRLT attacks (1) directly. DJIS attacks (2) by shrinking the input region so every downstream box is narrower, and it does so at near-zero wall-clock cost by exploiting parallelism Cert-RNN currently leaves on the table.

The two compose multiplicatively: DJIS narrows the boxes, and narrower boxes make ZRLT's cuts bind harder.

---

## 2. Background: where Cert-RNN loses precision

### 2.1 The existing pipeline

For an input sequence $X = [x^{(1)}, \dots, x^{(T)}]$ and perturbation radius $\epsilon$:

1. Abstract the $\ell_\infty$ ball as a zonotope $\hat z = \alpha_0 + \sum_i \alpha_i \varepsilon_i$, $\varepsilon_i \in [-1,1]$.
2. Propagate affine transformations exactly (zonotopes are closed under affine maps).
3. At each nonlinearity, apply an abstract transformer producing bounding planes $Z^L, Z^U$, and absorb the gap into a **fresh** noise symbol $\varepsilon_{\text{new}}$.
4. For the bilinear gate $f(x,y) = \sigma(x)\tanh(y)$, Theorem 4.2 / Table 8 supplies nine closed-form cases giving $Z^L = Ax + By + C_1$, $Z^U = Ax + By + C_2$.
5. Algorithm 1 binary-searches $\epsilon$ per time step (12 bisection rounds), then takes $\epsilon_c = \min_t \epsilon^{(t)}$.

### 2.2 Loss source A — the box discards joint structure

Table 8 requires the box $[l_x, u_x] \times [l_y, u_y]$, obtained by concretizing each operand independently: $l_x = \alpha_0 - \sum_i |\alpha_i|$, etc.

But $x$ and $y$ are pre-activations computed from the *same* hidden state and input. They share noise symbols. Their true joint reachable set is a **2-D zonotope** $Z$, and the box is merely its axis-aligned bounding rectangle. The corner regions of the box — precisely where Table 8 anchors $C_1$ and $C_2$ — are often unreachable.

Note that Theorem 4.2 itself states $(x,y) \in Z \subseteq [l_x,u_x] \times [l_y,u_y]$. The paper *names* $Z$ and then bounds over the box anyway. This is the gap.

Supporting result from the optimization literature: McCormick relaxation gives the exact convex hull of $z = xy$ over a **box**, but fails to give the hull when the domain is a general polytope, often producing poor bounds (Khademnia & Davarnia, 2023). The same failure mode applies here.

### 2.3 Loss source B — fresh error symbols destroy inter-gate correlation

This is the subtler and more damaging loss, and it is **not** fixable by tightening any single node.

An LSTM cell update is

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad f_t = \sigma(p),\ i_t = \sigma(q),\ g_t = \tanh(r)$$

Cert-RNN bounds $\sigma(p)\,c_{t-1}$ and $\sigma(q)\tanh(r)$ **separately**, each contributing its own independent $\varepsilon_{\text{new}}$, then sums the two zonotopes.

But $p$, $q$, $r$, and $c_{t-1}$ are all affine functions of the same underlying $\varepsilon$ vector. The two products' deviations are *correlated*. Summing independent bounds assumes worst-case alignment of two errors that in reality move together. The result is strictly conservative — and even a per-node-optimal transformer cannot recover it, because the loss occurs at the *sum*, not at either product.

**This is the target of ZRLT Tier 2, and it is where the novelty lives.**

### 2.4 Loss source C — the single-plane restriction

The zonotope form requires $Z^U - Z^L = C_2 - C_1$ to be constant, forcing both planes to share slopes $A, B$. The true convex hull of $\sigma(x)\tanh(y)$ over a box is piecewise-linear with several facets. Table 8 is near-optimal *within* the shared-slope restriction but structurally below the hull.

We do **not** attack this loss. Doing so requires abandoning the zonotope domain (DeepPoly-style back-substitution, or polynomial zonotopes) — a larger change, retained as a fallback in §11.

---

## 3. Architecture overview

```
                    ┌──────────────────────────────────────────┐
   input ball       │  DJIS: sensitivity-based partition       │
   B∞(X₀, ε)   ───► │  into k sub-regions (no join, ever)      │
                    └──────────────┬───────────────────────────┘
                                   │  k independent branches, batched
                    ┌──────────────▼───────────────────────────┐
                    │  for each sub-region j, for each t:      │
                    │                                          │
                    │   affine propagation  (exact)            │
                    │            │                             │
                    │            ▼                             │
                    │   ┌─────────────────────────────────┐    │
                    │   │ ZRLT cell transformer           │    │
                    │   │  T1: extract joint zonotope,    │    │
                    │   │      bound over Z not the box   │    │
                    │   │  T2: RLT cuts coupling the two  │    │
                    │   │      products; LP for the sum   │    │
                    │   └─────────────────────────────────┘    │
                    │            │                             │
                    │            ▼                             │
                    │   output zonotope for step t+1           │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │  Combine: ε_c = min over j of ε_c^(j)    │
                    │  (verification = AND over branches)      │
                    └──────────────────────────────────────────┘
```

Two additional throughput changes wrap the whole thing (§6): the $T$ loop runs in parallel, and the 12-round binary search on $\epsilon$ becomes a $k$-ary search.

---

## 4. Component A — ZRLT Tier 1: bound over the joint zonotope

### 4.1 Extracting the joint zonotope

At a gate, the two operands are affine forms over a shared noise vector:

$$x = a_0 + \sum_{i=1}^{p} a_i \varepsilon_i, \qquad y = b_0 + \sum_{i=1}^{p} b_i \varepsilon_i$$

Their joint reachable set is the 2-D zonotope with **center** $(a_0, b_0)$ and **generators** $g_i = (a_i, b_i) \in \mathbb{R}^2$.

Critically: **this is available in closed form at zero extra cost.** You already computed the coefficients. Prior MINLP work that tightens bilinear relaxations over non-rectangular domains must *recover* the projection by solving a sequence of LPs (Müller et al., 2019). Here the projection is handed to you by the representation. This changes the cost calculus and is a defensible part of the contribution.

### 4.2 Facet representation

For each generator $g_i = (a_i, b_i)$, the perpendicular direction is $n_i = (-b_i,\ a_i)$. The supporting slab in that direction is

$$|\langle n_i, (x,y)\rangle - d_i| \le w_i, \qquad d_i = \langle n_i, (a_0,b_0)\rangle, \qquad w_i = \sum_{j} |\langle n_i, g_j \rangle|$$

Two halfspaces per generator direction. The box is exactly the special case $n \in \{e_1, e_2\}$ — so Tier 1 strictly subsumes the current method.

**Vertices.** A 2-D zonotope with $p$ generators has at most $2p$ vertices, computable in $O(p \log p)$ by sorting generators by angle and walking them in order. Deduplicate near-parallel generators first (angular tolerance) — this typically collapses $p$ to a few dozen distinct directions.

### 4.3 Modified $C_1$, $C_2$ computation

Keep Table 8's case selection and its $(A, B)$ slopes unchanged. Replace only the offsets. With $F(x,y) = \sigma(x)\tanh(y) - Ax - By$:

$$C_1 = \min_{(x,y)\in Z} F, \qquad C_2 = \max_{(x,y)\in Z} F$$

Since $F$ is continuous and $Z$ is a compact polytope, both extrema lie either at a **vertex of $Z$** or at an **interior stationary point** of $F$ (where $\nabla F = 0$). The candidate-set search is structurally identical to the one already implemented — a hexagon or octagon replaces a rectangle.

Optionally, re-optimize $(A, B)$ over $Z$ too, via the cutting-plane LP of §5.4. Start without this; measure whether Table 8's slopes are still near-optimal on the smaller region.

### 4.4 Soundness

$Z \subseteq \text{box}$, so any bound valid on the box is valid on $Z$, and the new bounds are computed as exact extrema over $Z$ — which contains every reachable $(x,y)$. Soundness is immediate. **Round outward** ($C_1$ down, $C_2$ up) by a small $\eta$ to absorb floating-point error in vertex enumeration and stationary-point solves.

### 4.5 Cost

$O(p \log p)$ per gate for vertex enumeration, plus the same stationary-point solves already performed. With generator-direction deduplication and a cap on retained directions, this is a small constant factor over the current implementation.

---

## 5. Component B — ZRLT Tier 2: RLT cuts across coupled products

This is the novel core. Tier 1 makes each product tight; Tier 2 recovers what Tier 1 structurally cannot see.

### 5.1 The joint cell subproblem

For a single coordinate of the LSTM cell state:

$$c_t = \underbrace{\sigma(p)\,c_{t-1}}_{P_1} + \underbrace{\sigma(q)\tanh(r)}_{P_2}$$

with $p, q, r, c_{t-1}$ all affine in the shared $\varepsilon$. Instead of bounding $P_1$ and $P_2$ separately and summing, bound the **linear functional $P_1 + P_2$ jointly** over the 4-D joint zonotope $Z_4 \subseteq \mathbb{R}^4$ of $(p,q,r,c_{t-1})$.

Because $\min(P_1+P_2) \ge \min P_1 + \min P_2$ in general (with equality only when the minimizers coincide), joint bounding is never worse and typically strictly better.

### 5.2 Polynomialization

RLT operates on polynomial constraints, so replace the transcendentals with certified polynomial surrogates:

$$\sigma(u) \in \tilde\sigma_d(u) + [-\rho_\sigma, \rho_\sigma], \qquad \tanh(u) \in \tilde\tau_d(u) + [-\rho_\tau, \rho_\tau]$$

Use Chebyshev truncation of degree $d$ on each operand's interval; $\rho$ is the certified truncation remainder. These functions are analytic and saturating, so $\rho$ falls off rapidly — $d = 3$ or $4$ is typically sufficient over the intervals arising in practice. Precompute coefficients on a grid of interval endpoints and look them up.

The remainders enter additively and soundly:

$$P_1 \in \tilde\sigma_d(p)\,c_{t-1} + [-\rho_\sigma |c_{t-1}|_{\max},\ \rho_\sigma |c_{t-1}|_{\max}]$$

and analogously for $P_2$ (two remainder contributions, cross term negligible at $O(\rho^2)$).

### 5.3 RLT construction

**Reformulation.** Write $Z_4$'s facets as $h_k(v) \ge 0$, $v = (p,q,r,c_{t-1})$, each linear. For every retained pair $(k, \ell)$, the product

$$h_k(v)\cdot h_\ell(v) \ge 0$$

is a valid quadratic inequality — valid because both factors are non-negative on $Z_4$.

**Linearization.** Expand, then substitute a lifted variable $V_{mn}$ for each monomial $v_m v_n$. Every product constraint becomes **linear** in $(v, V)$. Add the standard McCormick bounds on each $V_{mn}$ from the interval bounds of $v_m, v_n$ (these are themselves the degree-2 RLT products of the box constraints — Tier 2 subsumes McCormick).

**Objective.** After polynomialization, $P_1 + P_2$ is a polynomial in $v$ of degree $\le d+1$. Lift its monomials to the same $V$ space (degree $> 2$ monomials require the higher-order RLT levels, or keep $d$ small and factor). Then

$$\min / \max \ \ \langle \text{lifted coefficients},\ (v, V)\rangle \quad \text{s.t. RLT cuts, facet constraints}$$

is a **linear program**. Its optimum is a valid (relaxed) bound on the true min/max of the cell update.

**Output.** Convert the resulting $[\text{lo}, \text{hi}]$ into the zonotope form: recover an affine part by taking the LP's dual multipliers on the facet constraints (these give the supporting-plane coefficients in $\varepsilon$-space), and absorb the residual gap into a **single** $\varepsilon_{\text{new}}$ for the whole cell update — instead of two.

Halving the number of fresh noise symbols is itself a compounding benefit over $T$ steps, independent of the tightness gain.

### 5.4 Solving efficiently: cutting-plane, not full LP

Do not generate all $O(m^2)$ cuts. Use constraint generation:

1. Solve with facet constraints + McCormick only.
2. Find the most-violated RLT product cut at the current LP optimum (separation is a cheap scan over pairs).
3. Add it, re-solve. Repeat to a fixed cut budget (start: 10–20 cuts).

This converges in few rounds because very few cuts are active at the optimum. The LP is small — 4 original variables, ~10 lifted variables, a few dozen constraints.

### 5.5 Cost control

- **Direction budget $m$**: retain only the $m$ largest-magnitude generator directions of $Z_4$; box the rest. $m = 8$ is a reasonable start.
- **Cut budget**: hard cap per cell.
- **Selective application**: apply Tier 2 only where it can pay. Cheap predicate — apply when the two products' operand generator vectors have cosine similarity above a threshold (high correlation ⇒ large potential gain) and skip otherwise, falling back to Tier 1. Expect this to fire on a minority of cells.

### 5.6 Soundness

Every RLT cut is a product of two constraints that are non-negative on $Z_4$, hence non-negative there — valid by construction. Lifting is a relaxation (we drop $V_{mn} = v_m v_n$ and keep only its linear consequences), so the LP optimum **over**-estimates the max and **under**-estimates the min. Both directions are conservative. Polynomial remainders are added outward. The composition is sound.

---

## 6. Component C — DJIS and throughput restructuring

### 6.1 Two free wall-clock wins

These fund everything else and carry zero tightness cost.

**Parallelize the $T$ loop.** Algorithm 1's outer loop computes $\epsilon^{(t)}$ independently per frame and takes the min at line 9. Nothing carries between iterations. Batch them.

**$k$-ary search for $\epsilon$.** The inner loop performs 12 sequential bisections for $10^{-4}$ precision. With 15 parallel probes per round the interval narrows $16\times$ instead of $2\times$: 3 rounds instead of 12. Same precision, one quarter the sequential depth.

Together these free roughly the budget needed for $k = 8$ input splitting.

### 6.2 Input partitioning

Partition the $\ell_\infty$ ball $B_\infty(X_0, \epsilon)$ into $k$ sub-regions and propagate each **independently through the entire network**. No intermediate join is ever performed.

Combination at the end:

- **Verification**: property holds on the full ball iff it holds on every sub-region (logical AND).
- **Certified radius**: $\epsilon_c = \min_j \epsilon_c^{(j)}$.

**Why deferral matters.** The union of zonotopes is not a zonotope. Splitting at an intermediate node forces a join that returns an enclosing zonotope, surrendering most of the gain — in the degenerate interval case, re-boxing two halves returns exactly the original box. Splitting at the input and joining only after all nonlinearity is the only placement that keeps the full benefit.

### 6.3 Why splitting pays: the $O(w^2)$ argument

The envelope gap is a second-order Taylor remainder, scaling as $O(w^2)$ in box width $w$. Halving a box quarters the gap. And the effect compounds: tighter bounds at step $t$ mean narrower boxes at $t+1$, so the per-step gain multiplies across $T$ — the same amplification Cert-RNN complains about, running in our favour.

### 6.4 Branching heuristic

Splitting all $T \times d$ input dimensions is impossible. Candidates, in increasing sophistication:

1. **Generator magnitude** (free): split the dimension with the largest coefficient in the final margin's affine form.
2. **Violation-guided** (from MINLP spatial branch-and-bound): split the variable whose relaxation is most violated at the relaxation optimum. This tells you where the relaxation is *lying*, which is a better signal than where the function is merely steep.
3. **Time-step targeting**: Algorithm 1 already identifies the binding frame $t^\star = \arg\min_t \epsilon^{(t)}$. Spend the split budget on that frame's input dimensions.

Start with (1)+(3); evaluate (2) as an ablation.

### 6.5 Adaptive depth

Splitting matters most when boxes are wide. Run the early $\epsilon$-search rounds unsplit (cheap, coarse) and enable splitting only for the final 3–4 refinement rounds, where the radius is actually decided.

### 6.6 Optional: adaptive branch-and-bound

The natural extension is a worklist: pop a region, attempt verification, discard if verified, split and re-push if not. This is *anytime* — stop at any budget and the radius so far remains a valid lower bound.

**Correctness requirement:** failure to verify a region is **not** a counterexample; it may be relaxation slack. Two distinct signals are required:

- *Bound inconclusive* → split further, or report unverified.
- *Concrete counterexample* (a point evaluated through the real network that misclassifies, found by a short PGD run inside the sub-box) → stop; the radius is below this $\epsilon$.

Conflating these silently under-reports the radius. Cap leaves (100–500) and report unverified on exhaustion — sound, merely conservative.

---

## 7. Why the components compose

| | Effect on box width | Effect on per-box tightness |
|---|---|---|
| DJIS | shrinks by $\sim k^{-1/d}$ per split dim | — |
| ZRLT T1 | — | removes correlation slack at each gate |
| ZRLT T2 | — | removes correlation slack *between* gates |

DJIS narrows the joint zonotope; a narrower zonotope makes RLT cuts bind harder relative to the box (the box-vs-zonotope discrepancy is scale-invariant, but the absolute recovered volume is what feeds forward). Conversely ZRLT reduces the per-step error that DJIS is fighting to contain.

**Honest caution on double-counting.** Both mechanisms reduce the *same* final quantity. Their gains will be sub-additive — if ZRLT already removes most of the error at a gate, DJIS's marginal contribution there shrinks, and vice versa. Budget expectations accordingly, and report the interaction explicitly in the ablation (§9.3) rather than claiming the product of individual gains.

---

## 8. Algorithm

```
CERT-RNN-PLUS(model F, input X₀, true label c, branch count k):

  # Phase 1 — partition (DJIS)
  dims  ← TOP-SENSITIVITY-DIMS(F, X₀, budget)
  R     ← PARTITION(B∞(X₀, ε_max), dims, k)      # k sub-regions

  # Phase 2 — per-region certification, fully parallel
  parallel for each region R_j in R:
     parallel for t in 1..T:
        ε ← K-ARY-SEARCH(λ e. VERIFY(F, R_j, e, t, c), rounds=3, probes=15)
        ε_j,t ← ε
     ε_j ← min_t ε_j,t

  # Phase 3 — combine (deferred join)
  return min_j ε_j


VERIFY(F, region, ε, t, c):
  ẑ ← ZONOTOPE(region, ε)
  for τ in 1..T:
     ẑ ← AFFINE(W, U, b, ẑ)                       # exact
     ẑ ← ZRLT-CELL(ẑ)                             # see below
  return MARGIN-HOLDS(ẑ, c)


ZRLT-CELL(ẑ):
  (p,q,r,c_prev) ← operands from ẑ
  Z₄ ← JOINT-ZONOTOPE(p,q,r,c_prev)               # closed form, free

  if CORRELATION(Z₄) > threshold:                 # Tier 2
     poly    ← CHEBYSHEV-SURROGATES(degree d, remainders ρ)
     cuts    ← RLT-CUTS(facets(Z₄), budget)       # cutting-plane
     lo, hi  ← LP-BOUND(P₁+P₂, cuts, poly)
     return ZONOTOPE-FROM-LP-DUAL(lo, hi)         # ONE ε_new
  else:                                            # Tier 1
     for each product:
        (A,B) ← TABLE-8-SLOPES(box(Z₄))
        C₁,C₂ ← EXTREMA(F, vertices(Z₂) ∪ stationary(F))
     return SUM-OF-PRODUCT-ZONOTOPES()             # two ε_new
```

---

## 9. Evaluation plan

### 9.1 Benchmarks

Reuse Cert-RNN's four tasks for direct comparability (MNIST sequence, Fashion-MNIST sequence, and the two NLP datasets), on both vanilla RNN and LSTM, across the sequence lengths and hidden widths reported in the original paper.

### 9.2 Baselines

| Baseline | Why |
|---|---|
| Cert-RNN | The foundation; the ablation floor |
| POPQORN | Historical reference used by Cert-RNN |
| Prover (Ryou et al., 2021) | Polyhedral RNN verification |
| GenBaB / α,β-CROWN | BaB on LSTM multiplication nodes; VNN-COMP winner. **The competitive bar.** |
| RNN abstraction-refinement (2026) | Node-level splitting at Sigmoid/Tanh gates. **The closest competitor to DJIS.** |

The last two are essential. Omitting them invites the reviewer question "isn't this just branch-and-bound, already done?"

### 9.3 Ablation matrix

| Configuration | Isolates |
|---|---|
| Cert-RNN (unmodified) | floor |
| + throughput only (§6.1) | wall-clock gain, zero tightness change (sanity check) |
| + ZRLT Tier 1 | intra-gate correlation slack |
| + ZRLT Tier 1 + 2 | inter-gate correlation slack |
| + DJIS only | refinement gain |
| Full (ZRLT + DJIS) | interaction / sub-additivity |

### 9.4 Metrics

- Certified radius $\epsilon_c$ (primary)
- Verified accuracy at fixed $\epsilon$ (comparability with prior work)
- Wall-clock per sample, and sequential depth (rounds), reported separately
- Bound-width decomposition by source (see §10)

---

## 10. Instrumentation — run this *before* implementing

Three measurements determine whether each component is worth building. All are cheap and can be taken on the existing codebase.

**M1 — Error attribution.** Per gate, log the width contributed by the product's $\varepsilon_{\text{new}}$ versus the $\sigma$ / $\tanh$ transformers. *If the product term does not dominate, ZRLT's ceiling is low.*

**M2 — Operand correlation over time.** Cosine similarity between the generator vectors of paired gate operands, as a function of $t$. *If this decays toward zero by late steps, the operands genuinely are near-independent and there is little correlation to recover — ZRLT Tier 2 would not pay.* This is the single highest-value measurement in the list.

**M3 — Wrapping diagnostic.** Ratio of enclosure width to a Monte-Carlo estimate of true reachable extent, versus $t$. *Geometric growth ⇒ compounding dominates ⇒ DJIS is the right lever. Flat growth with large per-step jumps ⇒ per-node tightness dominates ⇒ ZRLT is.*

---

## 11. Risks and fallbacks

| Risk | Detection | Fallback |
|---|---|---|
| Operands nearly independent; no correlation to recover | M2 | Drop ZRLT; pivot tightness effort to polynomial zonotopes (§2.4, loss source C) |
| Chebyshev remainders $\rho$ exceed the RLT gain | Compare $\rho$ against current $C_2 - C_1$ | Raise degree $d$; or restrict Tier 2 to gates with narrow operand intervals |
| LP overhead per cell dominates | Profile | Tighten selective-application threshold; reduce cut budget; precompute LP structure per case |
| DJIS gains subsumed by GenBaB-style node splitting | Head-to-head on §9.2 | Reposition DJIS as the *deferred-join* variant and measure join loss directly in the competitor |
| Single-plane restriction is the dominant slack | M1 with $(A,B)$ re-optimized vs. fixed | Domain change to polynomial zonotopes or DeepPoly back-substitution |

---

## 12. Novelty positioning

**Be precise about what is and isn't new.** Every ingredient exists somewhere; the contribution is the transfer and the RNN-specific construction.

*Not novel:* RLT itself (Sherali & Adams, ~1990). RLT in NN verification — Lan, Zheng & Lomuscio (AAAI'22) and Batten et al. (IJCAI'21) use RLT cuts, but for **ReLU networks in the SDP domain**. Tightening bilinear relaxations over non-rectangular domains — Müller et al. (2019) via 2-D projections, and subsequent MINLP work. Branch-and-bound on nonlinearities — GenBaB and the 2026 RNN refinement work.

*Novel, as far as the literature search found:*

1. RLT applied to the **$\sigma \odot \tanh$ gate** rather than ReLU.
2. RLT inside the **zonotope domain**, using generators that supply the joint projection in closed form — where prior MINLP work must recover it by solving a sequence of LPs.
3. Cuts that **couple the two products of an LSTM cell update**, recovering inter-gate correlation that no per-node transformer can reach, and halving the number of fresh noise symbols.
4. Applied to **recurrent** certification, where the correlation loss compounds over $T$.

Item 3 is the strongest claim. Lead with it.

**Suggested framing:** *Existing RNN certifiers concretize each gate's operands to independent intervals and assign each product an independent error symbol, discarding correlation the zonotope already encodes. We recover it with LP-cost cuts at no additional set-representation overhead.*

Do not lead with "we use RLT." Cite Müller et al. and Lan et al. yourself in related work — reviewers familiar with the MINLP literature will find them immediately.

---

## 13. Roadmap

| Phase | Work | Gate |
|---|---|---|
| 0 | Instrumentation M1–M3 on existing code | Go/no-go for each component |
| 1 | Throughput: parallel $T$ loop, $k$-ary search | Wall-clock gain, identical $\epsilon_c$ (regression test) |
| 2 | ZRLT Tier 1: joint zonotope, vertex enumeration, modified $C_1/C_2$ | $\epsilon_c$ improves; per-gate soundness fuzz test passes |
| 3 | DJIS: fixed $k$, sensitivity branching, batched | $\epsilon_c$ improves; cost within Phase-1 savings |
| 4 | ZRLT Tier 2: polynomialization, RLT cuts, cell LP | $\epsilon_c$ improves beyond Tier 1 on high-correlation cells |
| 5 | Adaptive depth, BaB worklist, head-to-head evaluation | Competitive with GenBaB on shared benchmarks |

**Soundness testing at every phase.** For each transformer, sample $10^4$ random points in the region and assert the bound holds. This catches sign errors in the symmetric Table 8 cases and orientation errors in vertex enumeration immediately — both are the most likely failure modes and both fail silently otherwise.

---

## References

- Du et al. *Cert-RNN: Towards Certifying the Robustness of Recurrent Neural Networks.* CCS 2021.
- Sherali & Adams. *A Reformulation-Linearization Technique for Solving Discrete and Continuous Nonconvex Problems.* 1990s.
- Lan, Zheng & Lomuscio. *Tight Neural Network Verification via Semidefinite Relaxations and Linear Reformulations.* AAAI 2022.
- Batten, Kouvaros, Lomuscio & Zheng. *Efficient Neural Network Verification via Layer-based Semidefinite Relaxations and Linear Cuts.* IJCAI 2021.
- Müller et al. *Using Two-Dimensional Projections for Stronger Separation and Propagation of Bilinear Terms.* arXiv:1903.05521.
- Khademnia & Davarnia. *Convexification of Bilinear Terms over Network Polytopes.* arXiv:2302.14151.
- Shi et al. *Neural Network Verification with Branch-and-Bound for General Nonlinearities (GenBaB).* arXiv:2405.21063.
- *Robustness Verification of Recurrent Neural Networks with Abstraction Refinement.* arXiv:2606.12490.
- Ko et al. *POPQORN: Quantifying Robustness of Recurrent Neural Networks.* ICML 2019.
- Ryou et al. *Scalable Polyhedral Verification of Recurrent Neural Networks.* CAV 2021.
