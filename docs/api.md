# Public API reference

Quick map of the `cert_rnn` package. For semantics and proofs see
[docs/soundness.md](soundness.md). For a complete end-to-end example
see [examples/demo_lstm_cell.py](../examples/demo_lstm_cell.py).

Every symbol below is importable from its module, and the full
user-facing API is also re-exported at the package top level, so the
common path is a single import:

```python
from cert_rnn import lstm_to_model_dict, certify_radius_spec_a
```

The only public symbols *not* re-exported at the top level are the
lower-level per-element transformers' unsound baseline
(`hadamard_affine_only`, kept as a regression target only).

## `cert_rnn` (top-level)

| Symbol | What |
| --- | --- |
| [`Zono`](../src/cert_rnn/zono.py) | Zonotope `{ c + V·α : α ∈ [-1,1]^p }`. Frozen dataclass; carries `c`, `V`, `pred_ids`. |
| [`zono_add(z1, z2)`](../src/cert_rnn/zono.py) | Sum two zonotopes. Shared `pred_ids` collapse; disjoint ones get separate columns. |
| [`zono_sub(z1, z2)`](../src/cert_rnn/zono.py) | Difference. Same alignment semantics. |
| [`align_pred_space(*zonos)`](../src/cert_rnn/zono.py) | Build a unified predicate space across multiple zonotopes; returns `(shared_ids, [V_emb_i, ...])`. The composition primitive that prevents the MATLAB aliasing bugs. |
| [`PredAllocator`](../src/cert_rnn/zono.py) | Monotonic integer allocator for fresh predicate ids. Module-level instance used by every transformer. |
| [`get_default_allocator()`](../src/cert_rnn/zono.py) | Current module-level allocator. |
| [`reset_pred_allocator(start=0)`](../src/cert_rnn/zono.py) | Reset the allocator. Used by test isolation. |

### `Zono` highlights

| | |
| --- | --- |
| `Zono(c, V, pred_ids)` | Construct. Validates shape and id uniqueness. |
| `Zono.point(c)` | Point zonotope (zero generators, no preds). |
| `Zono.from_box(c, radius, allocator=None)` | L_∞ ball with fresh preds, one per element. |
| `z.dim`, `z.n_pred` | Ambient dimension and predicate count. |
| `z.get_ranges()` | `(lb, ub)` axis-aligned bounding box. |
| `z.affine_map(W, b=None)` | Exact map `W z + b`. Preserves `pred_ids`. |
| `z.slice_rows(start, end)` | Half-open row slice. Preserves `pred_ids`. |

## `cert_rnn.transformers`

Each transformer adds K fresh predicates (one per output element) for
the abstraction error. Plane fit is corner-fit; `(C₁, C₂)` is exact
min/max of the residual over the input box.

| Symbol | Operation |
| --- | --- |
| [`tanh_zono(z, allocator=None)`](../src/cert_rnn/transformers.py) | Elementwise `tanh`. |
| [`sigmoid_zono(z, allocator=None)`](../src/cert_rnn/transformers.py) | Elementwise `σ` (sigmoid = rescaled tanh). |
| [`bilinear_sigmoid_tanh(z_x, z_y, allocator=None)`](../src/cert_rnn/transformers.py) | Elementwise `σ(x) · tanh(y)`. Pre-aligns `z_x` and `z_y`. |
| [`bilinear_sigmoid_identity(z_x, z_y, allocator=None)`](../src/cert_rnn/transformers.py) | Elementwise `x · σ(y)`. |
| [`hadamard_affine_only(z_x, z_y)`](../src/cert_rnn/transformers.py) | **Deliberately unsound** affine-only Hadamard baseline. Kept as a regression target documenting the bug the sound bilinears fix. |

## `cert_rnn.lstm`

Multi-step LSTM with Minkowski-correct predicate alignment.

| Symbol | What |
| --- | --- |
| [`lstm_state_init(H, L)`](../src/cert_rnn/lstm.py) | Build `L` lists of zero-point Zonos for `(h, c)`. |
| [`lstm_step(z_x, z_h_prev, z_c_prev, W_in, W_rec, b, allocator=None)`](../src/cert_rnn/lstm.py) | One LSTM cell step. Gate order `[i, f, g, o]`. Returns `(z_h, z_c)`. |
| [`lstm_step_stack(z_x, z_h_layers, z_c_layers, layers, allocator=None)`](../src/cert_rnn/lstm.py) | One time step of a stacked LSTM. `layers` is a list of `{"W_in", "W_rec", "b"}` dicts. |

## `cert_rnn.rnn`

| Symbol | What |
| --- | --- |
| [`rnn_step(z_x, z_h_prev, W_in, W_rec, b, allocator=None)`](../src/cert_rnn/rnn.py) | One Elman / tanh-RNN step. ReLU-RNN is out of scope. |

## `cert_rnn.audit`

| Symbol | What |
| --- | --- |
| [`lp_feasible(z, y_target, tol=1e-7)`](../src/cert_rnn/audit.py) | True iff `y_target ∈ z` via `scipy.optimize.linprog` (HiGHS). The rigorous (necessary + sufficient) zonotope-membership test used by the soundness suite. |

## `cert_rnn.from_torch`

PyTorch interop. Rejects bidirectional, dropout>0, and ReLU-RNN.

| Symbol | What |
| --- | --- |
| [`lstm_to_model_dict(rec, fc=None)`](../src/cert_rnn/from_torch.py) | Extract `nn.LSTM` (any `num_layers`) or `nn.LSTMCell`. Optional `nn.Linear` classifier head. |
| [`rnn_to_model_dict(rec, fc=None)`](../src/cert_rnn/from_torch.py) | Extract `nn.RNN(nonlinearity='tanh')`. |
| [`lstm_ae_to_model_dicts(encoder, decoder, head)`](../src/cert_rnn/from_torch.py) | Extract an LSTM autoencoder. `encoder`/`decoder` may each be an `nn.LSTM`, `nn.LSTMCell`, or a sequence of `nn.LSTMCell` (a stacked ModuleList); `head` is `nn.Linear` back to input space. Returns `{"encoder", "decoder", "head", "H", "D"}`. |

Both return a model dict directly consumable by `cert_rnn.lstm.lstm_step_stack` / `cert_rnn.rnn.rnn_step`:

```python
{
    "type":       "lstm" | "vanilla_rnn",
    "D": int, "H": int, "L": int,
    "layers":     [{"W_in", "W_rec", "b"}, ...],
    "head":       {"W", "b"},        # only if fc passed
    "gate_order": "ifgo",            # LSTM only
    "nonlinearity": "tanh",          # RNN only
}
```

## `cert_rnn.models`

Typed, ergonomic wrappers over the model dicts. Thin handles: they carry
the underlying dict(s) verbatim (`.as_dict()`, `.encoder`, ...) and expose
verification as methods, so a copying user never threads `encoder, decoder,
head` positionally or memorizes dict keys. Everything that takes a dict
keeps working — the wrappers are additive.

| Symbol | What |
| --- | --- |
| [`RNNModel.from_torch(rec, fc=None)`](../src/cert_rnn/models.py) | Wrap an `nn.LSTM`/`nn.LSTMCell`/`nn.RNN` (+ optional `nn.Linear` head). Properties `D, H, L, type, has_head`. |
| `RNNModel.reach(x_seq, eps, ...)` | Per-timestep top-layer hidden zono list (LSTM). |
| `RNNModel.certifies_margin(x_seq, eps, true_class, ...)` | Spec A boolean at a fixed `eps`. |
| `RNNModel.certify_radius(x_seq, true_class, ...)` | Algorithm 1 radius for Spec A. Returns `(min_over_frames, per_frame_array)`. |
| [`LSTMAutoencoder.from_torch(encoder, decoder, head)`](../src/cert_rnn/models.py) | Wrap an LSTM-AE via `lstm_ae_to_model_dicts`. Properties `H, D`. |
| `LSTMAutoencoder.score_ub(x_anchor, eps, ...)` | Sound recon-score upper bound over the eps-ball. |
| `LSTMAutoencoder.certifies(x_anchor, eps, tau, ...)` | Spec C boolean at a fixed `eps`. |
| `LSTMAutoencoder.certify_radius(x_anchor, tau, ...)` | Algorithm 1 radius for Spec C. |

```python
from cert_rnn import RNNModel, LSTMAutoencoder

clf = RNNModel.from_torch(my_lstm, my_fc)
radius, per_frame = clf.certify_radius(x_seq, true_class=3)

ae = LSTMAutoencoder.from_torch(encoder, decoder, head)
radius, per_frame = ae.certify_radius(x_anchor, tau=0.02)
```

## `cert_rnn.specs`

Property specs + the unified entry point. A spec answers "does the
property hold over this abstract output?"; `certify` bisects epsilon. See
[quickstart.md](quickstart.md).

| Symbol | What |
| --- | --- |
| [`certify(model, x, spec, *, threat_model, eps_init, n_iters)`](../src/cert_rnn/specs.py) | Algorithm 1 over any model wrapper + spec. Returns a `CertResult`. Also available as `model.certify(x, spec, ...)`. |
| [`MarginSpec(true_class)`](../src/cert_rnn/specs.py) | Classifier argmax preserved (on a logits `Zono`). |
| [`ThresholdSpec(upper=None, lower=None, indices=None)`](../src/cert_rnn/specs.py) | Output box bound; scalars or per-element arrays; optional dim subset. |
| [`ReconErrorSpec(tau)`](../src/cert_rnn/specs.py) | Autoencoder reconstruction score ≤ τ (on a `(z_x_hat_seq, z_x_seq)` tuple). |
| [`Spec`](../src/cert_rnn/specs.py) | Protocol: anything with a sound `holds(output) -> bool`. |
| [`CertResult`](../src/cert_rnn/specs.py) | `radius`, `per_frame`, `certified`, `threat_model`, `spec`. |

## `cert_rnn.runtime`

| Symbol | What |
| --- | --- |
| [`pin_blas_threads(n=1)`](../src/cert_rnn/runtime.py) | Set BLAS thread env vars (full effect requires calling before numpy import). |
| [`limit_blas_threads(n=1)`](../src/cert_rnn/runtime.py) | Context manager; clamps live pools via threadpoolctl when installed. |

## `cert-rnn` (CLI)

`cert-rnn {version,demo,info,verify}`. `info` lists a checkpoint's
tensors/shapes; `verify` certifies a pickled `nn.Module` classifier. See
[quickstart.md](quickstart.md#command-line).

## `cert_rnn.verify`

Algorithm 1 bisection, reach-set computation, and the two specs.

| Symbol | What |
| --- | --- |
| [`bisect_epsilon(certify_fn, eps_init=0.5, n_iters=12)`](../src/cert_rnn/verify.py) | Du et al. Algorithm 1. Returns the largest eps for which `certify_fn(eps)` returned True. |
| [`lstm_reach(model, x_seq, eps, threat_model, t_pert=None)`](../src/cert_rnn/verify.py) | Per-timestep top-layer hidden zono list. `threat_model ∈ {"single_frame", "multi_frame"}`. |
| [`lstm_ae_reach(encoder, decoder, head, x_anchor, eps, threat_model, t_pert=None)`](../src/cert_rnn/verify.py) | LSTM autoencoder forward; returns `(z_x_hat_seq, z_x_seq)`. |
| [`spec_a_margin(model, x_seq, eps, true_class, threat_model, t_pert=None)`](../src/cert_rnn/verify.py) | Classifier robustness: True iff `logit[true_class]` provably dominates every other logit over the perturbation set. |
| [`spec_c_score_ub(z_x_hat_seq, z_x_seq)`](../src/cert_rnn/verify.py) | Sound upper bound on the autoencoder reconstruction score over the perturbation set. |
| [`spec_c_holds(...)`](../src/cert_rnn/verify.py) | Spec C wrapper: True iff `spec_c_score_ub ≤ tau`. |
| [`certify_radius_spec_a(model, x_seq, true_class, ...)`](../src/cert_rnn/verify.py) | Bisect eps for Spec A. `single_frame`: returns `(min_over_frames, per_frame_array)`. `multi_frame`: `(eps, None)`. |
| [`certify_radius_spec_c(encoder, decoder, head, x_anchor, tau, ...)`](../src/cert_rnn/verify.py) | Same shape, for Spec C. |

## Threat models

`single_frame` (Algorithm 1, the paper's default): perturb one frame
`t_pert` of the input sequence by ±eps, pin every other frame to the
anchor. The certified radius for a sample is the min over `t_pert`.

`multi_frame`: every frame perturbed independently by ±eps with
disjoint predicate ids. Sound under this port's Minkowski-padded
`lstm_step`; **the MATLAB reference is unsound under this threat
model** (see [docs/soundness.md](soundness.md)).
