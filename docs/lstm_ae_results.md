# LSTM-AE IEEE-9 Spec C — Python vs MATLAB

Comparison of the Python Cert-RNN port (this repo) against the MATLAB
reference (`nnv3-cert-rnn`) on all four shipped checkpoint sizes.

Both engines:

- run Algorithm 1 (Du et al., CCS 2021) per frame, `eps_init=0.5`, 12 bisection iterations;
- evaluate Spec C: `score(x') = ‖AE(x') − x'‖₂² / N ≤ τ` for every `x'` in the single-frame ε-ball around the anchor `x*`;
- report the certified ε per frame and the sample's certified radius (min over frames);
- load the **same** trained PyTorch checkpoint
  (`examples/lstm_ae_ieee9/data/lstm_ae_ieee9_{S,M,L,D}.pt`).

| size | description | shape |
| --- | --- | --- |
| S | small, single-layer enc/dec | H=4, L_enc=L_dec=1 |
| M | medium, single-layer enc/dec | H=16, L_enc=L_dec=1 |
| L | large, single-layer enc/dec | H=64, L_enc=L_dec=1 |
| D | deep, two-layer enc/dec | H=16, L_enc=L_dec=2 |

Hardware: Linux 6.8, 28 logical cores, single-thread BLAS
(`OMP_NUM_THREADS=1`). Numbers collected in a single session for
consistency.

## Summary table

| size | H | cert_eps py | cert_eps matlab | mean ε py | mean ε matlab | max\|Δε\| | frames py ≥ mat | s/frame py | s/frame matlab | py / mat |
|------|---|-------------|-----------------|-----------|---------------|----------|------------------|------------|----------------|----------|
| S | 4  | 0.01282 | 0.01282 | 0.02130 | 0.02130 | 0.00e+00 | 30/30 | 0.50  | 0.17  | 3.0× |
| M | 16 | 0.01331 | 0.01331 | 0.01715 | 0.01715 | 0.00e+00 | 30/30 | 2.15  | 0.71  | 3.1× |
| L | 64 | 0.02295 | 0.02295 | 0.02811 | 0.02811 | 0.00e+00 | 30/30 | 31.26 | 12.47 | 2.5× |
| D | 16 | 0.02283 | 0.02466 | 0.02711 | 0.02910 | 2.69e-03 | **0/30** | 6.35 | 1.29  | 4.9× |

**Headline.**

- **S, M, L** (single-layer enc + single-layer dec): per-frame ε
  **byte-exact** to MATLAB across all 30 frames.
- **D** (two-layer enc + two-layer dec): Python is **uniformly tighter
  than MATLAB** by 1.7 × 10⁻³ to 2.7 × 10⁻³ on every single frame.
  This is the predicate-aliasing signature from
  [soundness.md §6 Bug 2](soundness.md): MATLAB's positional padding
  produces a falsely-tight bound when distinct fresh α's coincide on
  the same column index between the previous-step state and the
  current-step layer-1 output.
- Python is 2.5–4.9× slower than MATLAB. Speed cost is the
  pure-Python per-element loop in the transformers; vectorisation is
  a future work item.

## Per-frame ε comparison

### S (H=4) — single-layer enc+dec

All 30 frames match MATLAB to 1e-10 or better (verified by
`tests/test_lstm_ae_parity.py::test_lstm_ae_S_certified_radius_parity`).

### M (H=16) — single-layer enc+dec

All 30 frames match MATLAB to 1e-10 or better
(`max |Δε| = 0.00e+00`, see summary table).

### L (H=64) — single-layer enc+dec

All 30 frames match MATLAB to 1e-10 or better (`max |Δε| = 0.00e+00`).

### D (H=16) — **two-layer enc+dec, the divergent case**

| t  | ε py     | ε matlab | matlab − py |
|----|----------|----------|-------------|
|  0 | 0.030151 | 0.032837 | +2.686e-03  |
|  1 | 0.029175 | 0.031738 | +2.563e-03  |
|  2 | 0.029541 | 0.032227 | +2.686e-03  |
|  3 | 0.028809 | 0.031372 | +2.563e-03  |
|  4 | 0.028076 | 0.030518 | +2.441e-03  |
|  5 | 0.027588 | 0.029785 | +2.197e-03  |
|  6 | 0.025391 | 0.027466 | +2.075e-03  |
|  7 | 0.025269 | 0.027344 | +2.075e-03  |
|  8 | 0.023926 | 0.025879 | +1.953e-03  |
|  9 | 0.024536 | 0.026489 | +1.953e-03  |
| 10 | 0.022827 | 0.024658 | +1.831e-03  |
| 11 | 0.023560 | 0.025391 | +1.831e-03  |
| 12 | 0.024292 | 0.026123 | +1.831e-03  |
| 13 | 0.024658 | 0.026367 | +1.709e-03  |
| 14 | 0.024414 | 0.026123 | +1.709e-03  |
| 15 | 0.026367 | 0.028198 | +1.831e-03  |
| 16 | 0.026855 | 0.028687 | +1.831e-03  |
| 17 | 0.025146 | 0.026978 | +1.831e-03  |
| 18 | 0.025146 | 0.027100 | +1.953e-03  |
| 19 | 0.025391 | 0.027344 | +1.953e-03  |
| 20 | 0.025024 | 0.026978 | +1.953e-03  |
| 21 | 0.027100 | 0.029053 | +1.953e-03  |
| 22 | 0.026489 | 0.028320 | +1.831e-03  |
| 23 | 0.025635 | 0.027466 | +1.831e-03  |
| 24 | 0.026001 | 0.027710 | +1.709e-03  |
| 25 | 0.027344 | 0.029053 | +1.709e-03  |
| 26 | 0.027588 | 0.029297 | +1.709e-03  |
| 27 | 0.030273 | 0.031982 | +1.709e-03  |
| 28 | 0.033936 | 0.035767 | +1.831e-03  |
| 29 | 0.042725 | 0.044678 | +1.953e-03  |

Stats:
- mean (matlab − py) = +1.990 × 10⁻³
- max (matlab − py)  = +2.686 × 10⁻³
- min (matlab − py)  = +1.709 × 10⁻³
- frames where MATLAB > Python: **30 / 30**
- frames where MATLAB = Python: 0 / 30

The bisection step at iteration 13 is `0.5^13 ≈ 1.22 × 10⁻⁴`. The
observed `matlab − py` differences are all an order of magnitude
larger than this. The divergence is not bisection-precision noise; it
is a systematic engine-level difference.

## Why D diverges and S/M/L do not

In the single-layer enc/dec configurations (S, M, L), every
`lstm_step_stack` call has `L=1`, so the alignment between previous-step
state and current-step layer output happens through input preds + the
3H fresh β's introduced *this* step. The previous-step state's pred
column ordering matches positionally because no inter-layer fresh β's
are involved.

In the two-layer configuration (D), the encoder at step `t` does

```
layer 1: bilinear(prev_h^1, layer-1 fresh β's from step t)
layer 2: bilinear(prev_h^2, layer-2 fresh β's from step t)
                                 ↑                  ↑
   previous-step layer-2 fresh β's    current-step layer-1 fresh β's
            (carried in prev_h^2)               (in layer-1 output)
            both share the same column index → MATLAB aliases them
```

Different α variables, same column index. MATLAB's positional
zero-padding collapses them into one column; by the triangle
inequality this makes MATLAB's `|V|`-sum strictly smaller than the
true sum, hence MATLAB's `score_ub` is strictly smaller, hence
MATLAB's bisection finds a larger certified ε. The same bug
mechanism appears in the MNIST multi-layer perstep-widths comparison
(see `tests/test_mnist_parity.py`); now it shows up directly in
*certified epsilons*, not just per-timestep bound widths.

## Is MATLAB's D cert empirically unsound?

We ran PGD on the concrete `score(x')` inside both bounds and looked
for a violator at MATLAB's certified ε. None of 7 frames (0, 5, 10,
15, 20, 25, 29) admitted a PGD-discoverable violation:

| frame | ε_py | ε_mat | PGD @ ε_py | PGD @ ε_mat | τ |
|-------|------|-------|------------|-------------|---|
| 0  | 0.03015 | 0.03284 | 0.66198 | 0.66212 | 0.770 |
| 5  | 0.02759 | 0.02979 | 0.66304 | 0.66324 | 0.770 |
| 10 | 0.02283 | 0.02466 | 0.66189 | 0.66202 | 0.770 |
| 15 | 0.02637 | 0.02820 | 0.66213 | 0.66225 | 0.770 |
| 20 | 0.02502 | 0.02698 | 0.66269 | 0.66289 | 0.770 |
| 25 | 0.02734 | 0.02905 | 0.66305 | 0.66323 | 0.770 |
| 29 | 0.04272 | 0.04468 | 0.66258 | 0.66268 | 0.770 |

PGD config: 20 random restarts, 120 ascent steps, step size `ε/5`.
Anchor score = 0.6606. PGD pushes only +0.0014 above anchor at MATLAB
ε, still ~0.108 below τ.

**Read.** MATLAB's D bound is theoretically risky (aliasing-tightened),
but the *score_ub formula itself is so loose* (componentwise-squared,
see [red_team_report.md §2.3](red_team_report.md)) that the
~7 % aliasing tightening sits inside the much larger ~22 % formula
slack. No empirical unsoundness witnessed on these 7 frames. MATLAB
*may* be unsound at adversarial worst-case configurations that PGD
does not find; we cannot rule that out from sampling alone.

## Soundness verdict per size

| size | Python sound (math + PGD) | MATLAB sound (PGD spot-check) | Python = MATLAB |
| --- | --- | --- | --- |
| S | ✓ | ✓ | ✓ (byte-exact) |
| M | ✓ | ✓ | ✓ (byte-exact) |
| L | ✓ | ✓ | ✓ (byte-exact) |
| D | ✓ | ✓ on tested frames; theoretically at-risk-of-unsoundness | ✗ (Python tighter on 30/30 frames) |

## Reproducibility

```bash
# Single-thread BLAS; ~25 min wall time on the reference hardware
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u examples/lstm_ae_ieee9/verify_all.py --sizes S M L D --atol 1e-3

# Plain text or markdown table comparing against
# examples/lstm_ae_ieee9/matlab_results/certrnn_lstm_ae_*.mat
python scripts/tabulate_lstm_ae.py            # plain
python scripts/tabulate_lstm_ae.py --md       # markdown

# Adversarial probe of MATLAB's D cert
python scripts/probe_D_cert_unsoundness.py
```

MATLAB reference results live in
`examples/lstm_ae_ieee9/matlab_results/certrnn_lstm_ae_{S,M,L,D}.mat`
and were produced by
`nnv3-cert-rnn/code/nnv/examples/Submission/CertRNN/scripts/verify_certrnn_lstm_ae.m`
on the same checkpoints.
