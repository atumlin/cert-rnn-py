# research/

Dev/research scripts that back the analysis in [`docs/`](../docs). These
are **not** part of the `cert_rnn` tool surface — they are one-off
harnesses kept for reproducibility. For verifying your own models use the
library API (`docs/quickstart.md`) or the `cert-rnn` CLI.

Each resolves the repo root via `Path(__file__).resolve().parent.parent`
and adds the relevant `examples/` loader to `sys.path`.

| Script | What it does | Backs |
| --- | --- | --- |
| `red_team_mnist.py` | Adversarial sampler + PGD against the MNIST Spec-A certificates. | [docs/red_team_report.md](../docs/red_team_report.md) |
| `red_team_lstm_ae.py` | Adversarial sampler + PGD against the IEEE-9 LSTM-AE Spec-C certificates. | [docs/red_team_report.md](../docs/red_team_report.md) |
| `probe_D_cert_unsoundness.py` | Targeted probe of the deep (size D) LSTM-AE certificate. | [docs/lstm_ae_results.md](../docs/lstm_ae_results.md) |
| `tabulate_lstm_ae.py` | Tabulate the LSTM-AE Python-vs-MATLAB results (`--md` for markdown). | [docs/lstm_ae_results.md](../docs/lstm_ae_results.md) |
