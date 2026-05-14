# cert-rnn

Python port of Cert-RNN — zonotope abstract-interpretation transformers
for certifying robustness of RNNs and LSTMs against L_inf input
perturbations.

Implements the sound bilinear transformers and tightened sigmoid/tanh
transformers from Du et al., *Cert-RNN: Towards Certifying the
Robustness of Recurrent Neural Networks*, CCS 2021, plus the
Algorithm 1 bisection on epsilon.

This is a standalone reimplementation, ported from a MATLAB reference
(NNV-based) into pure Python (numpy + scipy + torch). PyTorch
state-dicts are the supported input format — no .mat boundary.

## Status

Phase 0 skeleton. Engine, verification, examples, and the soundness
test suite are stubs to be filled in by phase. See `docs/soundness.md`
once it lands for the math + soundness proofs.

## Install

```bash
pip install -e .[dev]
pytest
```

## Layout

```
src/cert_rnn/    engine + verification
tests/           soundness contract (per-transformer fuzz, LP audit,
                 lstm step, red-team, MATLAB cross-validation)
examples/        MNIST-sequence (paper Table 2 repro), LSTM-AE Spec C
docs/            soundness.md, api.md
```
