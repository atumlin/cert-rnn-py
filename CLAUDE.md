# cert-rnn-py — Cert-RNN+ development

Python port of Cert-RNN (Du et al., CCS 2021): zonotope abstract-interpretation
certification of RNN/LSTM robustness, being extended with ZRLT (tightness) and
DJIS (throughput).

## Read before touching bounding code (open by path; do not @-import)
- Architecture spec: cert-rnn-plus-proposal.md
- Soundness plan:    soundness.md  (root file, not docs/soundness.md)
- Current findings:  docs/phase0_findings.md

## Non-negotiable invariants
- Every bug in this project makes certified bounds TIGHTER. The certified
  radius eps_c is NEVER evidence of correctness — only the soundness suite is.
- Sample in eps-space (alpha in [-1,1]^p). Never rejection-sample a box to get
  points in a zonotope.
- Sampling must include vertices, edge points, and the eps sign corners
  {-1,+1}^p — never interior-only.
- Round outward, relative to magnitude. Never inward.
- Never widen the rounding eta to make a failing test pass. (Whether eta should
  exist at all is decided by R-test violation magnitudes — see soundness.md §4.)
- Any new over-approximation requires a new row in the soundness ledger
  (soundness.md §1) before merge.
- A failing baseline is a FINDING: report it in docs/phase0_findings.md,
  minimize it, keep it as a regression test. Never fix silently.
- On any soundness-test failure: never widen tolerances; minimize; add the
  regression test BEFORE the fix; check symmetric cases for the same error.

## Tests
- Fast suite (every change):  pytest -q
- Soundness suite:            pytest tests/soundness -q
- Nightly-scale budgets:      pytest tests/soundness -m nightly
- Every sampling test logs its RNG seed; failures must be reproducible.

## Conventions
- tests/conftest.py pins BLAS threads to 1 and resets the predicate allocator
  to 10_000; hand-picked pred_ids in tests must stay below 10_000.
- Git: short commit messages; never push without explicit permission.
