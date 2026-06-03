# Cert-RNN soundness

This document explains what soundness means for the Cert-RNN abstract
interpretation, why predicate-identity tracking is required for
composition, the two latent bugs in the MATLAB reference implementation,
and how this Python port fixes them.

The math derivations of the individual abstract transformers (tanh,
sigmoid, σ(x)·tanh(y), x·σ(y)) follow Du et al., *Cert-RNN: Towards
Certifying the Robustness of Recurrent Neural Networks*, CCS 2021,
sections 4.2.2, 4.3.1, 4.3.2, and Appendices A–C. See the inline
docstrings in [src/cert_rnn/transformers.py](../src/cert_rnn/transformers.py)
for the exact formulas this port uses.

## 1. What we are proving

Given an LSTM `f` and an input ball `B(x₀, ε) = { x : ‖x − x₀‖_∞ ≤ ε }`,
the goal is to certify a property such as

> for every `x' ∈ B(x₀, ε)`, the predicted class of `f(x')` equals `y₀`

without enumerating concrete `x'`. We compute an over-approximation
`F̂(B) ⊇ { f(x') : x' ∈ B }` and check the property against `F̂(B)`.
The whole game is to keep `F̂(B)`

1. **sound** — never miss a reachable concrete output, and
2. **tight** enough that the property check succeeds.

## 2. The abstract domain — zonotopes with named noise

A zonotope is parameterised as

    Z = { c + V·α : α ∈ [-1, 1]^p }

| symbol | shape | meaning |
| --- | --- | --- |
| `c` | `(K,)` | center |
| `V` | `(K, p)` | generators; column k is the partial derivative wrt `α_k` |
| `α` | `(p,)` | independent noise / predicate variables, each in `[-1, 1]` |
| `K` | scalar | ambient dimension at this point in the network |
| `p` | scalar | accumulated predicate count |

The input perturbation ball `B(x₀, ε)` becomes
`Z_in = { x₀ + ε·α : α ∈ [-1,1]^D }` — `D` fresh `α`s, one per input dim.

**Each predicate variable has a stable identity throughout the forward
pass.** Column `k` of `V` always refers to the *same* `α_k`. This
identity is the central concept of the rest of this document.

## 3. The transformers

Cert-RNN composes three primitives:

### Affine `Wx + b`

Exact. `c → Wc + b`, `V → WV`, `α` preserved. No fresh preds.

### Elementwise nonlinearity (tanh, sigmoid)

The curved surface is not a zonotope. We sandwich it between two
parallel planes — for `tanh` on input `x_k ∈ [l_k, u_k]`:

    a · x_k + C₁ ≤ tanh(x_k) ≤ a · x_k + C₂

with slope `a` = secant `(tanh(u) − tanh(l)) / (u − l)`, and `(C₁, C₂)`
the parallel-tangent offsets. The output is

    a · c + (C₁+C₂)/2  +  a · V · α  +  diag((C₂−C₁)/2) · β

where `β` is a *new* vector of `K` independent noise variables (one per
output element). The `β`s are the abstraction error introduced here, and
they are independent of `α` because they do not derive from the input
perturbation — they represent the gap between the linear sandwich and
the true curve.

### Bilinear σ(x)·tanh(y) and x·σ(y)

Same idea in 2D. We sandwich:

    A · x + B · y + C₁ ≤ f(x, y) ≤ A · x + B · y + C₂

`(A, B)` is the corner-fit plane; `(C₁, C₂)` is the exact min/max of the
residual `g(x, y) = f(x, y) − A·x − B·y` over the input box, found by
enumerating corners, edge stationary points, and (for σ·tanh) interior
critical points via a quartic in `p = σ(x)`. See
[transformers.py](../src/cert_rnn/transformers.py) for the per-case code.

**Soundness of each primitive is local**: it only requires that
`(C₁, C₂)` truly bracket the residual over the input box. This is
proved in the paper and validated by LP-feasibility fuzz on every
transformer in [tests/test_transformers.py](../tests/test_transformers.py).

## 4. Composition and predicate identity

The bilinear transformer sees two zonotopes:

    Z_x = c_x + V_x · α_x        (with predicates α_x of count p_x)
    Z_y = c_y + V_y · α_y        (with predicates α_y of count p_y)

The output center is `A·c_x + B·c_y + (C₁+C₂)/2`. The output generators
are `A·V_x·α_x + B·V_y·α_y + diag((C₂−C₁)/2)·β`. The question is

> how are `α_x` and `α_y` related?

This determines the output's generator matrix `V_out`:

- **If they are the same alpha vector** (shared identity), they collapse.
  Column `k` of `V_out` is `A·V_x[:,k] + B·V_y[:,k]` — one shared `α_k`.
- **If they are disjoint** (different identities), they stay separate.
  Each gets its own column: `A·V_x[:,k]` and `B·V_y[:,k]`, with the
  associated alphas independent.

By the triangle inequality,

    |A·V_x[:,k] + B·V_y[:,k]|  ≤  |A·V_x[:,k]| + |B·V_y[:,k]|

So **aliasing two disjoint α's into the same column gives a strictly
tighter bound than treating them independently**. That tightening is
unsound when the α's really are independent: the true reachable set
covers all `(α_x, α_y) ∈ [-1,1]²` independently, while the aliased
bound only covers the diagonal `α_x = α_y`.

That is the entire bug pattern this document is about.

## 5. Worked example — where aliasing bites

Take a tiny bilinear `f(x, y) = x · y` (illustrative; the real LSTM
gates have σ and tanh wrappers but the aliasing mechanism is identical).
Two scalar zonotopes:

    Z_x = 0 + α   →   x ranges over [-1, +1]
    Z_y = 0 + β   →   y ranges over [-1, +1]

### Case A — disjoint preds (`α, β` independent)

True reachable: `{ x · y : (x, y) ∈ [-1, 1]² } = [-1, +1]`. The bilinear
transformer with `A = B = 0` and the exact residual gives `C₁ = -1`,
`C₂ = +1`, yielding output bounds `[-1, +1]`. Sound.

### Case B — aliased preds (treat `α = β`)

If we mistakenly alias them into a single shared predicate `α`:

    x · y  =  α · α  =  α²

ranges over `[0, +1]` because `α² ≥ 0`. The aliased "bound" claims the
reachable set is `[0, +1]` — missing the entire negative half. The
concrete point `x = +1, y = -1` (giving `x·y = -1`) is reachable but
*not in the aliased bound*. **Unsound.**

This is exactly what MATLAB's positional zero-padding does when it
combines two zonotopes whose column 0 refers to different α
variables: it implicitly forces `α = β`.

## 6. The two MATLAB bugs

MATLAB's `Zono` stores only `c` and `V` — no alpha identities. When
`CertRNN.lstmStep` combines zonotopes, it calls `pad_zono` to zero-pad
the shorter `V` to match the longer one's column count, then sums
column-by-column. The implicit assumption is that *column k of every
zonotope refers to the same α_k*.

That assumption holds in the simplest case — single-frame perturbation
on a single-layer LSTM — because all alphas trace back to one shared
set of `D` input preds and fresh `β`s are appended sequentially. It
breaks in two cases:

### Bug 1 — multi-frame perturbation

Documented in the original briefing and the `CertRNN.lstmStep` docstring.

    Step 2 input z_x:    D fresh α's (a brand-new set, not the step-1 α's)
    Step 2 prev hidden:  D + 3H preds (step-1 input + step-1 fresh β's)

MATLAB zero-pads `z_x` to `D + 3H` columns by appending zeros at the
end. But column 0 of `z_x` is "step-2 α₀", and column 0 of `z_h_prev`
is "step-1 α₀" — different variables. The bilinear's `A·V_x + B·V_h`
combines them as if they were the same, introducing a false
correlation.

This bug is observable: the original MATLAB red-team measured up to
**11 % relative violations** under fresh-per-timestep perturbation.

### Bug 2 — stacked single-frame perturbation (newly identified)

This was not documented in the original briefing; it surfaced while
porting the MNIST demo. Same family as Bug 1, on the inter-layer
dependency in stacked LSTMs.

    Step 2, layer 2 inputs:
      prev layer-2 state z_h²_₁:    cols [D+3H : D+6H] = layer-2 fresh β's from STEP 1
      current layer-1 output z_h¹_₂: cols [D+3H : D+6H] = layer-1 fresh β's from STEP 2

Both zonotopes have `D + 6H` columns, so positional padding aliases
column-`k` across them. But those columns reference different
allocations of fresh `β`s — variables introduced by different bilinear
calls, with no relation to each other.

On the MNIST LSTM-2-32 anchor at `eps = 0.005`, MATLAB's bounds are
tighter by 0.1 – 1.3 × 10⁻³ per frame (~4-5 % relative). On 500 random
concrete samples, both Python (correctly looser) and MATLAB
(aliasing-tightened) bounds happened to contain the reachable set —
the worst-correlation case is rare under random sampling, so the
MATLAB red-team did not catch it. The MATLAB bound is **unsound in the
adversarial worst case**.

The numerical demonstration is in the project's MNIST per-step-width
parity harness (`test_perstep_widths_multi_layer_python_at_least_as_loose`),
which asserts (a) Python bounds ≥ MATLAB bounds everywhere and (b) Python
bounds remain sound vs concrete sampling.

## 7. The Python fix

`Zono` carries a `pred_ids: tuple` — one hashable id per `V` column
([src/cert_rnn/zono.py](../src/cert_rnn/zono.py)).

`align_pred_space(z1, z2, …)` builds a unified column space:

- Predicate ids appearing in multiple inputs collapse to one column.
- Disjoint ids get disjoint columns.

Every transformer calls `align_pred_space` before computing combinations;
`zono_add` and `zono_sub` do too. Fresh `β`s from the transformers get
unique integer ids from a monotonic module-level `PredAllocator`.

There is no special-casing for any threat model. Alignment is uniform;
soundness falls out from preserving identity:

- **Single-layer single-frame**: all preds genuinely shared, alignment
  is a no-op, the Python output `(c, V)` is byte-exact to MATLAB's.
- **Multi-frame**: each step's input gets fresh ids → disjoint columns
  → no false correlation → sound.
- **Stacked single-frame**: layer-`i` and layer-`j` fresh `β`s get
  different ids → disjoint columns → sound but looser than MATLAB.

## 8. Cost

The pred count grows linearly with timesteps and depth. For an
MNIST-sequence LSTM-2-32 (T=28, L=2, H=32), the final top hidden
state carries roughly `D + 3·L·H·T ≈ 28 + 5376 ≈ 5400` predicates.

This does not affect correctness — LP-feasibility and box bounds
handle any pred count — but it makes the `V` matrices `K × p` large,
which is the dominant cost of the pure-Python forward (per-element
plane loops do `np.roots` on a degree-4 polynomial for each cell).

The Python port is currently ~3 × slower than MATLAB on the LSTM-AE
S example and ~40 × slower on the MNIST LSTM-2-32 example. Closing
that gap is a pure-performance work item — vectorising the
transformers across `K` — and is independent of the soundness story.

## 9. What is verified

Per-transformer soundness, LP-feasibility on 200+ random samples per
transformer:
[tests/test_transformers.py](../tests/test_transformers.py). Both
engines pass; this validates the sandwich-plane math.

Composition soundness, LP-feasibility:

- single-frame single-layer LSTM step: [tests/test_lstm_step.py](../tests/test_lstm_step.py)
- multi-frame ("Minkowski regression") LSTM step: same file
- stacked-LSTM single-frame: same file
- RNN step: same file
- 60-scenario red-team across both threat models:
  [tests/test_red_team.py](../tests/test_red_team.py)

Numerical parity vs MATLAB CertRNN:

- exact `(c, V)` match on 19 fixtures (transformers, lstm_step, rnn_step):
  [tests/test_cross_validation.py](../tests/test_cross_validation.py)
- per-frame certified ε for the LSTM-AE Spec C anchor, all 30 frames,
  diff = 0.000e+00 (LSTM-AE parity harness)
- per-sample certified radius on MNIST LSTM-1-32, diff = 0.000e+00
  (MNIST parity harness)
- per-timestep bound widths on MNIST LSTM-1-32 (single-layer), float
  noise match; on LSTM-2-32 / 2-64 / 4-32 / 7-32 (stacked), Python ≥
  MATLAB with Python sound on concrete sampling (the Bug 2
  demonstration): same file

PyTorch parity (cert engine on point inputs matches PyTorch forward to
1e-10 on `nn.LSTM`, `nn.LSTMCell`, `nn.RNN`, with and without head):
[tests/test_from_torch.py](../tests/test_from_torch.py).

## 10. The Du et al. paper

The paper's math is correct. The bugs above are not in the paper —
they are in the MATLAB reference implementation, specifically in the
predicate-tracking glue between transformer calls. The paper assumes
alpha identity tracking is handled; the MATLAB code uses positional
column indices as a shortcut that holds in the paper's headline
demonstration (single-frame single-layer MNIST) and breaks elsewhere.

This Python port makes the identity tracking explicit (`pred_ids`),
which removes both bugs without changing the underlying transformer
math.
