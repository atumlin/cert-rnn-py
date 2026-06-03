# Quickstart

Go from a trained PyTorch model to a certified L∞ robustness radius in a
few lines. For the full symbol map see [api.md](api.md); for the math and
soundness proofs see [soundness.md](soundness.md).

## Install

```bash
pip install -e .[dev]
pytest          # optional: 130+ soundness/parity tests
cert-rnn demo   # optional: tiny end-to-end certification, no data needed
```

## Certify a classifier (Spec A — argmax preserved)

```python
import numpy as np
from cert_rnn import RNNModel, MarginSpec

# Your trained recurrent classifier: an nn.LSTM / nn.LSTMCell / nn.RNN
# plus an optional nn.Linear head.
model = RNNModel.from_torch(my_lstm, my_fc)

x_seq = ...          # (T, D) input sequence, np.ndarray
true_class = 3

result = model.certify(x_seq, MarginSpec(true_class))
print(result)                 # CertResult(radius=..., certified=True, ...)
print(result.radius)          # largest single-frame eps that preserves argmax
print(result.per_frame)       # per-frame certified eps (min over frames = radius)
```

## Certify an autoencoder (Spec C — reconstruction error ≤ τ)

```python
from cert_rnn import LSTMAutoencoder, ReconErrorSpec

# encoder/decoder may each be an nn.LSTM, nn.LSTMCell, or a list of
# nn.LSTMCell (a stacked ModuleList); head is nn.Linear back to input space.
ae = LSTMAutoencoder.from_torch(encoder, decoder, head)

result = ae.certify(x_anchor, ReconErrorSpec(tau=0.02))
print(result.radius)
```

## Other built-in properties

```python
from cert_rnn import ThresholdSpec, certify

# Every output element stays within a box over the perturbation set:
certify(model, x_seq, ThresholdSpec(upper=1.0, lower=-1.0))

# Restrict to selected output dimensions:
certify(model, x_seq, ThresholdSpec(upper=1.0, indices=[0, 2]))
```

## Custom properties

Any object with a **sound** `holds(output) -> bool` is a spec. `output`
is whatever the model's `reach_output(...)` returns — a logits/hidden
`Zono` for `RNNModel`, a `(z_x_hat_seq, z_x_seq)` tuple for
`LSTMAutoencoder`. `holds` must return `True` only if the property holds
for *every* point in the set.

```python
class SecondClassNeverWins:
    """Class 1 never becomes the argmax over the perturbation set."""
    def holds(self, z_logits):
        # diff = logit[argmax_other] - logit[1]; certify it stays > 0 ... etc.
        lb, ub = z_logits.get_ranges()
        return bool(ub[1] < lb[0])   # toy example

certify(model, x_seq, SecondClassNeverWins())
```

## Threat models

- `single_frame` (default, the paper's Algorithm 1): perturb one frame by
  ±eps, pin the rest; the radius is the min over frames.
- `multi_frame`: every frame perturbed independently. Pass
  `threat_model="multi_frame"` to `certify` / `model.certify`.

## Command line

```bash
cert-rnn version
cert-rnn demo                                   # self-contained example
cert-rnn info  checkpoint.pt                     # list tensors + shapes
cert-rnn verify model.pt --input x.npy --spec margin --true-class 3
cert-rnn verify model.pt --input x.npy --spec threshold --upper 1 --lower -1
```

`verify` needs a pickled `nn.Module` (`torch.save(model, path)`) whose
class is importable, so it can recover the architecture. For anything
else, use the Python API above.

## Performance note

cert_rnn does many small numpy ops; multi-threaded BLAS adds contention.
Pin to one thread for best throughput. The most reliable way is to set the
env vars **before** importing numpy/cert_rnn:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import cert_rnn
```

`cert_rnn.runtime.pin_blas_threads()` / `limit_blas_threads()` wrap this;
see [examples/verify_my_model.py](../examples/verify_my_model.py).
