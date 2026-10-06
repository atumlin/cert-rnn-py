# Benchmark plan — Cert-RNN+ for LSTM-autoencoder verification (HSCC / ICCPS)

Approved 2026-10-06.

## 1. Goal

Show, on public grid benchmarks and on real Sandia Zone-A data, that **Cert-RNN+** proves false-alarm robustness of v3 LSTM-autoencoder anomaly detectors at least as tightly as the CROWN family (incl. α,β-CROWN), much faster, scaling where the others time out, across persistent / burst / transient FDI attacks, with soundness evidence throughout.

## 2. Benchmarks (all in the Zone-A file layout; v3 models trained from scratch, no legacy models or code)

| benchmark | data | features/s | window | layers | seeds | role |
|---|---|---|---|---|---|---|
| grid9 | IEEE 9-bus power-flow simulation: 3 h normal + 1 h with 24 events (6 types) | 36 | 10 s | 1 | 3 | main public benchmark |
| grid14 | IEEE 14-bus | 56 | 10 s | 1 | 1 | scaling: features |
| grid30 | IEEE 30-bus | 120 | 10 s | 1 | 1 | scaling: features |
| grid9-T20 / -T30 | grid9 | 36 | 20 / 30 s | 1 | 1 | scaling: window length |
| grid9-L2 | grid9 | 36 | 10 s | 2 | 1 | scaling: depth |
| zoneA | real Sandia PMU data (run at Sandia only) | 64 | 10 s | 1 | 3 | real-world case study |

Models (v3 recipe): time order kept; chronological 70/15/15 split; MinMax scaler fit on the train block; one code per second, H = round(0.8 x features); decoder step t reads code t; Adam lr 1e-3, 100 epochs, batch 16, best-validation weights; fixed seeds. tau = 99.9th percentile of validation scores. Detection metrics recorded per model.

Windows verified: 10 per model (normal test windows below tau, random with a fixed seed, non-overlapping).

## 3. Tools

| tool | role | hardware (recommended run) | budget per query |
|---|---|---|---|
| Cert-RNN (baseline: independent gates, per-value score bound) | verifier | CPU | 300 s |
| Cert-RNN+ (linked gates + linked score bound) | verifier (ours) | CPU | 300 s |
| IBP (auto_LiRPA) | verifier | GPU | 300 s |
| CROWN (auto_LiRPA) | verifier | GPU | 300 s |
| α-CROWN (auto_LiRPA CROWN-Optimized) | verifier | GPU | set after pilot (default 3600 s) |
| α,β-CROWN + GenBaB | verifier | GPU | 1800 s |
| PGD | attack, NOT a verifier: FALSIFIED labels, tightness gap, soundness checks | GPU | 300 s |

All tools verify byte-identical weights; CROWN-family tools get the network with the score folded in (their best native encoding).

## 4. Protocol

- Property: score <= tau for every input in the eps-box (no false alarm).
- Attacks (same for every tool): persistent (all seconds); burst (3 consecutive seconds: 4-6 for T=10, middle three otherwise); transient (single seconds: first, middle, last).
- Main: fixed-eps checks at eps in {0.0005, 0.001, 0.003, 0.01}; verdicts VERIFIED / FALSIFIED (PGD counterexample) / UNKNOWN / TIMEOUT / ERROR. Implied results are filled in and marked: verified at eps => verified at smaller eps; falsified at eps => falsified at larger eps.
- Secondary: certified radius, same bisection for every tool (eps_init 0.5, 12 rounds).
- Timing: wall-clock and CPU time per query; timeouts and errors counted.
- Soundness evidence: corner + PGD sampling inside every proved eps; cross-check verified verdicts against counterexamples from any tool.
- Statistics: medians/IQR, paired win/tie/loss on identical windows, per seed and pooled.

## 5. Restart / resume

Every script can be killed and rerun with the same command; finished work is skipped.

| stage | saved progress | on restart |
|---|---|---|
| data | one folder per benchmark, written to a temp name then renamed | skips complete benchmarks |
| training | checkpoint every epoch (weights, optimizer, history, RNG); final model.pt + meta.json = done | resumes mid-training |
| fixed-eps checks | one JSONL line per query, flushed immediately | skips finished queries |
| radius searches | bisection state after every probe | resumes mid-search |
| analysis | reads whatever exists | any time |

α,β-CROWN runs each query in its own process (a crash marks that query ERROR). `status.py` reports progress.

## 6. Scripts

0. `install.sh` + `check_install.py` (once per machine)
1. `01_make_data.sh`
2. `02_train.sh`
3. `03_check.sh` (fixed-eps, recommended hardware -> main table)
4. `04_radius.sh`
5. `05_fair.sh` (3-4 with every tool on one CPU process)
6. `06_analyze.sh`
- `reproduce.sh` runs 1-6 (`--quick` / `--full`); `status.py` shows progress. Settings: `config.yaml`.

## 7. Where things run

- This machine (28 cores + RTX 4060): grid benchmarks.
- Sandia server (GPUs): zoneA, via `git pull` + `install.sh` + scripts with `--data-dir`. Only result files leave Sandia; models, weights, data and counterexample inputs stay in the git-ignored `work/` folder.

## 8. Scope

Per model: 10 windows x 5 attacks x 4 eps = 200 queries per tool (fewer with implied skips). Cert-RNN, Cert-RNN+, IBP, CROWN, PGD on all models; α,β-CROWN (and α-CROWN if feasible) on the main benchmarks (grid9 x3, zoneA x3). Recommended-hardware run first; fair run afterwards.

## 9. Risks

α,β-CROWN single-second crashes (report with the error if they persist); α-CROWN may exceed any budget (report measured time); Sandia pip access (INSTALL.md covers proxy/mirror and a conda Python 3.11 fallback); 10 windows x 3 seeds is modest statistical power (paired comparisons; stated as a limitation).
