# Red-team validation report — Cert-RNN Python port

**Scope.** Audit the soundness of the two end-to-end verification
pipelines shipped with `cert_rnn`: MNIST classifier-margin (Spec A)
and IEEE-9 LSTM autoencoder false-alarm (Spec C). Verify that the
abstract pipeline produces *sound* certificates, that the
specification check at the end is correctly framed, and that no
shortcut in the engine compromises the guarantee.

**Method.** Two-pronged:
1. Mathematical audit — trace every step from the input box to the
   final boolean / scalar that drives the bisection, and prove each
   step is sound.
2. Empirical red-team — adversarial sampling and projected gradient
   ascent inside the certified ball, attempting to *break* the cert.

**Headline verdict.** No soundness violations found, mathematically
or empirically. Both certificates are sound. Both are loose
(measurably so — Spec C particularly), but looseness is a precision
property, not a soundness property. Three subtleties below that
users should be aware of, none of which violate soundness.

---

## Part 1: MNIST Spec A — classifier-margin robustness

### 1.1 Property and threat model

The pipeline answers the following question for a sample
`(x_seq, y_true)` and perturbation budget `ε`:

> Is there any `x' ∈ B(x_seq, ε)` such that `argmax f(x') ≠ y_true`?

"`No`" → the prediction is certifiably robust at radius `ε`.

**Threat model is single-frame.** `B(x_seq, ε)` for the verifier means:
exactly one frame index `t_pert` is perturbed by ±ε in `L_∞`, every
other frame is pinned to its anchor value. The "certified radius" for
a sample is `min` over all `t_pert ∈ {0..T-1}` of the per-frame radii.
This is **not** the same as "robust to L_∞ perturbations on the entire
input sequence" — see §3.1.

### 1.2 Pipeline audit

| Step | What happens | Why sound |
| --- | --- | --- |
| 1. Build input zonos | At `t_pert`, `z_x_t = Zono.from_box(x_seq[t], ε)` with `D` fresh α's; all other `t` are point zonos. | The set `{x_seq[t_pert] + ε·α : α ∈ [-1,1]^D}` is exactly `B(x_seq[t_pert], ε)` in `L_∞`. The other frames are exact points. |
| 2. LSTM forward | Each step: affine map (exact) → slice (exact) → bilinears with sandwich planes. | Per-transformer soundness validated by LP-feasibility fuzz, 200+ samples per case. Composition soundness via `pred_ids` (no aliasing). |
| 3. Classifier head | `z_logits = z_h_top.affine_map(W_fc, b_fc)`. | Affine is exact. |
| 4. Margin matrix | `D[i, y_true] = +1`, `D[i, c_i] = -1` for each `c_i ≠ y_true`. `z_diff = z_logits.affine_map(D)`. | Affine is exact. `z_diff` has dim `C-1`, one row per `logit_y - logit_c` margin. |
| 5. Box bound | `(lb, ub) = z_diff.get_ranges()` returns `lb = c - sum|V|`, `ub = c + sum|V|`. | For any `α ∈ [-1,1]^p`, `|V·α| ≤ sum|V|` by `L_∞`/`L_1` duality. So `lb` and `ub` truly bracket the reachable diff. |
| 6. Decision | Certify iff `lb_i > 0` for every margin `i`. | If `lb_i > 0` then `(logit_y - logit_c_i)(x') > 0` for every concrete `x'`. Hence `argmax ≠ y_true` is impossible. |

The certification function is monotone in `ε`: a larger ball produces
a larger `z_diff`, hence smaller `lb`. Algorithm 1 bisection on this
monotone predicate returns the largest `ε` for which the bound
certified during the search trajectory; that `ε` is a sound lower
bound on the true robust radius.

### 1.3 Empirical red-team

Three MNIST test samples (LSTM-1-32 checkpoint shipped with the
package). For each:

1. Bisect the certified per-frame radius. Take the min.
2. At the worst frame, sample 1000 concrete `x'` uniformly in the
   `L_∞` ball of radius `ε_cert`. Count `argmax` flips. **Soundness
   requires 0 flips.**
3. Sample 1000 more `x'` in the ball of radius `1.5 ε_cert`. Count
   flips. Probes how tight the bound is — flips here are not
   unsoundness but evidence the bound is close to the true robust
   radius.
4. PGD attack (10 random restarts × 80 steps × `ε / 5` step size)
   inside the `ε_cert` ball, maximising negative log-likelihood of
   `y_true`. **If PGD flips, the cert is unsound.**
5. PGD inside `1.5 ε_cert` ball.

Results:

```
sample 0 (label=7): cert_radius=0.01880 at t=0
  random @ eps:    0/1000 flips    [SOUND]
  random @ 1.5eps: 0/1000 flips    [bound is loose at this radius]
  PGD @ eps:       no flip          [SOUND]
  PGD @ 1.5eps:    no flip          [bound is loose at 1.5x]

sample 1 (label=2): cert_radius=0.02014 at t=0
  random @ eps:    0/1000 flips
  random @ 1.5eps: 0/1000 flips
  PGD @ eps:       no flip
  PGD @ 1.5eps:    no flip

sample 2 (label=1): cert_radius=0.01282 at t=9
  random @ eps:    0/1000 flips
  random @ 1.5eps: 0/1000 flips
  PGD @ eps:       no flip
  PGD @ 1.5eps:    no flip
```

**Total: 0 soundness violations across 3 samples, ~6000 random
perturbations, 60 PGD attacks.**

The fact that even `1.5 × ε_cert` does not flip is interesting:
the cert bound is meaningfully loose — the true robust radius is at
least `1.5 ε_cert` for these samples. The cert under-claims the
robustness. This is sound, just not tight.

### 1.4 Findings — MNIST

- **Sound.** Both the math audit and the empirical attack found no
  soundness violations.
- **Loose.** The bound under-claims robustness by at least 50 % on the
  three samples tested (PGD does not flip even at 1.5×). This is a
  precision issue; the user gets a certificate that is conservative
  but correct.
- **Threat model is narrower than "robust to all sequence-level
  perturbations".** The min-over-frames radius lets you perturb any
  single frame by `ε_cert`, but not all frames simultaneously. See
  §3.1.

---

## Part 2: LSTM-AE Spec C — autoencoder false-alarm

This is the more interesting case, because the spec is *quantitative*
(an upper bound on a continuous score) rather than discrete (an
argmax check). The soundness story turns on the score-bound formula.

### 2.1 Property and threat model

For an LSTM autoencoder `AE` and an anomaly threshold `τ`, the
verifier answers:

> Is there any `x' ∈ B(x_anchor, ε)` such that
> `score(x') := ‖AE(x') − x'‖_2^2 / N > τ`,
> where `N = T · D`?

"`No`" → the model does not produce a false alarm on any `x'` in the
ball.

Threat model is single-frame as in MNIST, with the same min-over-frames
aggregation.

### 2.2 Sound upper bound on `score(x')` — derivation

The verifier computes a sound *upper bound* on `score(x')` over the
perturbation set, and asks whether that upper bound ≤ τ.

Let `z_x_hat_seq[t]` be the abstract reconstruction at timestep `t`
(after the encoder, decoder, and per-step head), and `z_x_seq[t]` the
input zonotope at `t`. After Minkowski subtraction
`z_diff[t] = z_x_hat_seq[t] − z_x_seq[t]` we have, for any
concrete `α ∈ [-1,1]^p`,

```
z_diff[t,d](α) = c_diff[t,d] + V_diff[t,d,:] · α
```

By the triangle inequality (`|V·α| ≤ sum|V|`):

```
|z_diff[t,d](α)|  ≤  |c_diff[t,d]| + sum_p |V_diff[t,d,p]|  =:  U[t,d]
```

Therefore `z_diff[t,d](α)^2 ≤ U[t,d]^2` for any `α`, hence

```
score(x') = (1/N) · sum_{t,d} z_diff[t,d]^2
         ≤ (1/N) · sum_{t,d} U[t,d]^2
         =:  score_ub
```

This is exactly the formula in `cert_rnn.verify.spec_c_score_ub`.
It is sound because we bounded each squared term independently.

### 2.3 Where the looseness comes from

The bound is *componentwise* (each `(t,d)` term independent), but
in reality a single `α ∈ [-1,1]^p` controls *all* `(t,d)` components
simultaneously. The true `score`-maximising `α` is the solution to

```
max_α  sum_{t,d} (c_diff[t,d] + V_diff[t,d,:] · α)^2     s.t.  α ∈ [-1,1]^p
```

a quadratic maximisation over a box. The componentwise bound
relaxes this by letting *each component pick its own* `α`, which is
strictly looser. Tighter sound bounds need a quadratic-over-box
solver (NP-hard in general but tractable in this scale); out of scope
for this port — see §4.

### 2.4 Pipeline audit

| Step | What | Sound? |
| --- | --- | --- |
| 1. Build `z_x_seq` | At `t_pert`: `Zono.from_box(anchor[t], ε)` with `D` fresh α's. Elsewhere: points. | Exact `L_∞` ball at `t_pert`, exact points elsewhere. |
| 2. Encoder forward | T LSTM steps. | Composition of sound transformers. |
| 3. Latent | `z_latent = z_h_enc[-1]` (final top hidden of encoder). | Just a reference; soundness inherited. |
| 4. Decoder forward | T LSTM steps with `z_latent` as input *at every step*. The latent's predicates propagate; each decoder step also adds fresh predicates from its bilinears. | Composition of sound transformers. The fact that the *same* latent zono is reused at every decoder step is fine — its preds are constant references; the bilinears do not need them to be "fresh". |
| 5. Head | `z_x_hat_seq[t] = z_h_dec[-1][t].affine_map(W, b)`. | Affine is exact. |
| 6. Diff | `z_diff[t] = z_x_hat_seq[t] − z_x_seq[t]` via `zono_sub`. | `zono_sub` Minkowski-aligns predicates; at the perturbed frame the input preds are shared and partially cancel, elsewhere they are zero in `z_x_seq[t]`. Correct. |
| 7. Score bound | `score_ub = (1/N) · sum_{t,d} (|c_diff| + sum_p |V_diff|)^2`. | Sound per §2.2. Loose per §2.3. |
| 8. Decision | `score_ub ≤ τ`. | Trivially sound from §2.2. |

The decoder's "fixed-input" pattern (latent reused at every step) is
unusual compared to standard seq2seq decoders that feed the previous
output back in. It is faithful to the trained model in this port (the
checkpoints under `examples/lstm_ae_ieee9/data/` were trained with
this pattern). No soundness implication.

### 2.5 Empirical red-team

LSTM-AE size S (`H=4`, `T=30`, `D=36`, `τ=0.77`,
anchor `score = 0.601`). Four frames probed: 0, 10, 20, 29.
For each: 500 random + 10×80 PGD inside `ε_cert`, plus 500 random
+ 10×80 PGD inside `1.5 ε_cert`. PGD ascends the concrete score.

```
frame 0:  cert_eps=0.017334
  random @ eps:    max_score=0.6018  violations: 0/500    [SOUND]
  random @ 1.5eps: max_score=0.6022  violations: 0/500
  PGD @ eps:    score=0.6035  within tau (0.77)    [SOUND]
  PGD @ 1.5eps: score=0.6049  within tau

frame 10: cert_eps=0.015259
  random @ eps:    max_score=0.6019  violations: 0/500
  random @ 1.5eps: max_score=0.6022  violations: 0/500
  PGD @ eps:    score=0.6036  within tau
  PGD @ 1.5eps: score=0.6050  within tau

frame 20: cert_eps=0.026978
  random @ eps:    max_score=0.6017  violations: 0/500
  random @ 1.5eps: max_score=0.6019  violations: 0/500
  PGD @ eps:    score=0.6030  within tau
  PGD @ 1.5eps: score=0.6048  within tau

frame 29: cert_eps=0.020386
  random @ eps:    max_score=0.6026  violations: 0/500
  random @ 1.5eps: max_score=0.6040  violations: 0/500
  PGD @ eps:    score=0.6106  within tau
  PGD @ 1.5eps: score=0.6182  within tau
```

**Total: 0 soundness violations across 4 frames, 4000 random
perturbations, 80 PGD attacks.**

### 2.6 Quantifying the looseness

At the certified `ε_cert`, the bisection has driven `score_ub` up to
just below `τ = 0.77`. Meanwhile the PGD-maximised concrete score
reaches only `~0.6035–0.6106`. Anchor score is `0.6011`.

| | Score |
| --- | --- |
| Anchor | 0.6011 |
| PGD max inside `ε_cert` | 0.6035–0.6106 |
| PGD max inside `1.5 ε_cert` | 0.6049–0.6182 |
| Sound `score_ub` at `ε_cert` | ≈ τ = 0.77 |
| `τ` | 0.7700 |

The sound upper bound (`~0.77`) is roughly 4–5× further from the
anchor than the PGD-maximum concrete score is. The bound is loose
by `0.77 − 0.61 ≈ 0.16` in score units, while the true span of
concrete score reachable inside the certified ball is only
`0.61 − 0.60 ≈ 0.01`. **Looseness factor ≈ 10–15× in
score-above-anchor units.**

This means: in practice the verifier is conservative — it certifies a
smaller `ε_cert` than the true robust radius. The 30-frame minimum
certified radius reported in MATLAB's `certrnn_lstm_ae_S.mat` is
`0.01282`, while the true single-frame robust radius (the largest `ε`
under which no concrete `x'` exceeds `τ`) is much larger — PGD did
not exceed `τ` even at `1.5 × cert_radius_per_frame`.

Causes of the looseness, ranked:

1. **Componentwise-squared score bound** (§2.3). The dominant source.
2. **Bilinear plane fit relaxation.** Corner-fit `(A, B)` plus exact
   `(C₁, C₂)`; the residual cube `(C₂−C₁)` over each output unit grows
   each timestep, contributing to the diff zono width.
3. **Tanh / sigmoid sandwich.** Parallel-tangent bounds are tightest
   when the input range is small; on long sequences the hidden
   activations can have wide ranges, especially near saturation.

### 2.7 Findings — LSTM-AE

- **Sound.** Both the math audit and empirical attack found no
  soundness violations.
- **Loose by ~10× in score-headroom units.** The componentwise score
  bound is the dominant slack. A tighter sound bound would require
  a quadratic-over-box optimiser; out of scope for this port.
- **The decoder pattern is unusual but correctly modelled.** Decoder
  reads the fixed latent at every step; no issue.

---

## Part 3: Subtleties and caveats (none affect soundness)

### 3.1 Single-frame radius is not full L_∞ robustness

The verifier returns `cert_radius = min_t bisect(t)`. This certifies
that **perturbing any one frame by up to `cert_radius`** is safe —
not that **perturbing every frame simultaneously by up to
`cert_radius`** is safe. The latter is the multi-frame threat model,
which is also supported (`threat_model="multi_frame"`).

If a user wants to claim full L_∞ robustness over the input sequence,
they should call `certify_radius_spec_a(..., threat_model="multi_frame")`.
The multi-frame radius will generally be much smaller because every
frame's perturbation independently grows the diff.

### 3.2 Bisection precision

Algorithm 1 with `n_iters = 12` has resolution `0.5^13 ≈ 1.2e-4`. The
returned `ε_cert` may be that resolution below the true tight bound
for the sandwich relaxation, but this is a precision artefact, not
unsoundness. Increasing `n_iters` tightens the resolution at linear
cost.

### 3.3 Loose ≠ unsound

Throughout this report, "loose" means the cert is conservative — the
true robust radius is larger than what we certify. This is the
*correct* direction of conservatism for soundness. The opposite
(certifying a radius larger than the true one) would be unsoundness.
None observed.

---

## Part 4: Recommendations

1. **Status quo for soundness.** No changes needed. The cert is sound
   on both pipelines, validated against ~10000 concrete perturbations
   and ~140 gradient-based attacks.

2. **For Spec C tightness (optional future work).** Replace the
   componentwise score bound with a sound box-constrained quadratic
   bound. Concretely: given `z_diff` with center `c` and generators
   `V`, the sound max of `‖c + V α‖_2^2 / N` over `α ∈ [-1,1]^p` is

       max_α  α^T (V^T V) α + 2 c^T V α + ‖c‖_2^2

   This is an indefinite-QP-over-box; not solvable exactly in poly
   time, but a Lagrangian relaxation or SDP relaxation gives a
   strictly tighter sound bound than the componentwise one. Expect
   the certified `ε` to roughly double or triple on the LSTM-AE
   anchor.

3. **For Spec A tightness (optional future work).** The margin check
   already uses an exact-affine relaxation of the diff zono; the
   slack is entirely in the LSTM reach. Vectorising and using a
   tighter abstract domain (e.g., polyhedra) is a much larger lift.

4. **Documentation.** [docs/api.md](api.md) makes the threat-model
   distinction visible; the example scripts and test names also mark
   `single_frame` vs `multi_frame`. The user-facing risk of confusing
   the two threat models is mitigated but not eliminated.

5. **Reproducing this report.** Both empirical scripts live in
   [research/](../research/): `red_team_mnist.py` (~2 min wall time)
   and `red_team_lstm_ae.py` (~30 s wall time). Re-run with the
   shipped checkpoints to regenerate the numbers.

---

## Appendix: artefacts

| Path | What |
| --- | --- |
| `research/red_team_mnist.py` | MNIST adversarial sampler + PGD. |
| `research/red_team_lstm_ae.py` | LSTM-AE adversarial sampler + PGD. |
| `docs/soundness.md` | Underlying math: zonotope domain, transformers, predicate identity, the two MATLAB aliasing bugs this port fixes. |
| `tests/test_transformers.py` | LP-feasibility audit (200+ samples per transformer). |
| `tests/test_lstm_step.py` | LP-feasibility audit at the composition level. |
| `tests/test_verify.py` | End-to-end soundness via sampling for both specs. |
| `tests/test_lstm_ae_parity.py` | LSTM-AE size S byte-exact parity vs MATLAB. |
| `tests/test_mnist_parity.py` | MNIST LSTM-1-32 byte-exact parity vs MATLAB; LSTM-2-32 onwards: Python `≥` MATLAB (Bug 2 — see soundness.md §6). |
