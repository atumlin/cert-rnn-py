# cert-rnn

Python port of Cert-RNN — zonotope abstract-interpretation transformers
for certifying robustness of RNNs and LSTMs against L_inf input
perturbations.

Implements the sound bilinear transformers and tightened sigmoid/tanh
transformers from Du et al., *Cert-RNN: Towards Certifying the
Robustness of Recurrent Neural Networks*, CCS 2021, plus the
Algorithm 1 bisection on epsilon.

This is an independent reimplementation **with modifications**, not a
copy of the authors' original code. It was ported from a MATLAB
reference (NNV-based) into pure Python (numpy + scipy + torch), and
differs from the original work in that:

- the implementation is pure Python rather than the original codebase;
- the primary model-input path is PyTorch state-dicts (`.pt`) loaded
  via `cert_rnn.from_torch` — there is no `.mat` boundary in the
  engine. (The MNIST-sequence example keeps a `.mat` loader purely to
  cross-validate against the MATLAB reference numbers; it is not the
  intended path for new models.)
- soundness is independently validated here via a fuzz/LP-audit/
  red-team test suite and MATLAB cross-validation fixtures.

All credit for the underlying method (the sound bilinear and
sigmoid/tanh zonotope transformers and the Algorithm 1 bisection)
belongs to the original authors — see [Citation](#citation) below.

## Status

Implemented and tested. The zonotope engine, Algorithm 1 verification,
both end-to-end examples, and the soundness test suite are complete
(112 passing tests). A red-team audit (`docs/red_team_report.md`) found
no soundness violations on either shipped pipeline, mathematically or
empirically. See `docs/soundness.md` for the math + soundness proofs.

## Install

```bash
pip install -e .[dev]
pytest
cert-rnn demo     # tiny end-to-end certification, no data needed
```

## Quickstart

```python
from cert_rnn import RNNModel, MarginSpec

model = RNNModel.from_torch(my_lstm, my_fc)
result = model.certify(x_seq, MarginSpec(true_class=3))
print(result.radius)        # largest certified L_inf radius
```

See [docs/quickstart.md](docs/quickstart.md) for autoencoders, custom
properties, threat models, and the `cert-rnn` CLI.

## Layout

```
src/cert_rnn/    engine + verification + tool surface
                 engine:   zono, transformers, verify, lstm, rnn, audit
                 interop:  from_torch (incl. LSTM-AE extractor)
                 tool:     models (RNNModel/LSTMAutoencoder), specs
                           (certify + MarginSpec/ThresholdSpec/ReconErrorSpec),
                           cli, runtime
tests/           soundness contract (per-transformer fuzz, LP audit,
                 lstm step, red-team, MATLAB cross-validation, CLI/wrappers)
examples/        verify_my_model.py — copy-me template
                 mnist_sequence     — paper Table 2 repro
                 lstm_ae_ieee9      — IEEE-9 LSTM-AE false-alarm Spec C
research/        dev red-team / tabulation scripts (back the docs)
docs/            quickstart.md, api.md, soundness.md,
                 red_team_report.md, lstm_ae_results.md
```

## Citation

This repository reimplements the method introduced in:

> Tianyu Du, Shouling Ji, Lujia Shen, Yao Zhang, Jinfeng Li, Jie Shi,
> Chengfang Fang, Jianwei Yin, Raheem Beyah, and Ting Wang.
> **Cert-RNN: Towards Certifying the Robustness of Recurrent Neural
> Networks.** In *Proceedings of the 2021 ACM SIGSAC Conference on
> Computer and Communications Security (CCS '21)*, pp. 516–534.
> https://doi.org/10.1145/3460120.3484538

```bibtex
@inproceedings{du2021certrnn,
  title     = {Cert-RNN: Towards Certifying the Robustness of Recurrent Neural Networks},
  author    = {Du, Tianyu and Ji, Shouling and Shen, Lujia and Zhang, Yao and
               Li, Jinfeng and Shi, Jie and Fang, Chengfang and Yin, Jianwei and
               Beyah, Raheem and Wang, Ting},
  booktitle = {Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (CCS '21)},
  pages     = {516--534},
  year      = {2021},
  doi       = {10.1145/3460120.3484538},
}
```

If you use this Python port, please cite the original paper above.

## License

MIT — see [LICENSE](LICENSE). This license covers this reimplementation
only; the original Cert-RNN method and paper are the work of their
respective authors.
