# Benchmark results handoff for paper writing

Entry point for writing up the Cert-RNN+ benchmark (HSCC/ICCPS). Final run: 8 Oct 2026, seed 0, validated 9 Oct.
All paths are relative to the repo root `cert-rnn-py/`. Read this file first; it supersedes
`paper_bench/results/recommended/seeds_0/RESULTS_FOR_SLIDES.md` (v4, numbers out of date).

## Where everything is

| What | Path |
|---|---|
| Approved benchmark plan | `paper_bench/PLAN.md` |
| How to run / reproduce (AE) | `paper_bench/README.md`, `paper_bench/INSTALL.md`, `paper_bench/reproduce.sh` |
| Configuration (models, eps, attacks, tools, budgets, devices) | `paper_bench/config.yaml`, `paper_bench/abcrown.yaml` |
| Tool adapters (how each verifier is called) | `paper_bench/tools.py`; score-graph encodings in `paper_bench/model.py` (`ScoreGraph`, `AttackScore`) |
| Raw per-query results (source of truth, one JSON line per query) | `paper_bench/results/recommended/check.jsonl` |
| Human-readable run logs (one line per query) | `paper_bench/results/recommended/logs/check_20261008-*.txt` |
| Tables and figures (regenerate: `python -m paper_bench.analyze --seeds 0`) | `paper_bench/results/recommended/seeds_0/` |
| Mathematical formulation with proofs and verified references (LaTeX) | `docs/formulation/cert_rnn_plus_formulation.tex` (+ `.pdf`) |
| Cert-RNN+ design / soundness notes | `cert-rnn-plus-proposal.md`, `soundness.md`, `docs/phase0_findings.md` (grid resolution: section H) |
| Cert-RNN+ implementation | `src/cert_rnn/` (grid cover: `tier1.py`; joint score bound: `verify.py:spec_c_score_ub_joint`) |

Files in `seeds_0/`: `verified_by_cutoff.csv` (main table), `check_main_{300,1800}s.{csv,tex}` (per attack family and eps),
`paired_check_{300,1800}s.csv` (Cert-RNN+ vs each tool, same queries), `certified_rate_{300,1800}s.pdf`,
`trajectory.{pdf,csv}` (% verified vs time per query), `scaling.{csv,pdf}`, `detection.{csv,tex}`, `soundness.csv`.
Ignore `certified_rate.pdf`, `check_main.csv`, `paired_check.csv` without a cutoff suffix (uncut duplicates), and do NOT use
the legacy figures in `paper/` (`fig_cactus_e2.pdf`, `fig_scaling_*.pdf`): they come from old experiments, not this benchmark.

## Protocol (what the numbers mean)

- **Detectors**: v3 LSTM autoencoders retrained from scratch (encoder LSTM, one code per second, code size 80% of features; decoder LSTM; linear head). Alarm iff window MSE > tau, tau = validation 99.9th percentile.
- **Benchmarks**: synthetic PMU-style data on IEEE 9/14/30-bus systems (pandapower). grid9 (main, 36 features), grid14 (56), grid30 (120), grid9_T20 / grid9_T30 (20 s / 30 s windows), grid9_L2 (2 LSTM layers). No Sandia data in any of these; Zone-A runs at Sandia (pending).
- **Property**: no false alarm for every input whose attacked readings move by at most eps (normalised units).
- **Query** = (model, window, attack, eps). 10 normal windows x 5 attacks x 4 eps = 200 per model.
  Attacks: persistent (all seconds), burst (seconds 5-7 of 10), transient first / middle (6th) / last second.
  eps in {0.0005, 0.001, 0.003, 0.01}.
- **Max verifiable** = 200 minus queries PGD falsified (real counterexamples, confirmed in float64; no sound tool can verify them).
- **Time limit**: one per-query limit of 1800 s for every verifier; results reported at 300 s and 1800 s from the same run
  (a query verified in t seconds counts at cutoff c iff t <= c). Every query was run and timed individually (no inferred verdicts).
- **Tools**: Cert-RNN (box gates, componentwise score bound); Cert-RNN+ (zonotope-linked gates, joint score bound); IBP, CROWN, alpha-CROWN (auto_LiRPA 0.7.2); alpha,beta-CROWN + GenBaB (commit e5c7e17). PGD (10 restarts x 100 steps) is an attack only.
- **Hardware / fairness**: 28-core CPU, no GPU used. Cert-RNN, Cert-RNN+, IBP, CROWN, alpha-CROWN: 1 thread per query. alpha,beta-CROWN: 8 threads per query (favours the baseline). All verifiers in float64. Baselines get the score folded into one graph with only attacked readings as inputs (verified equal to the Cert-RNN model to 1e-9). Early stops: alpha-CROWN stops once its bound is below tau (CROWN first); alpha,beta-CROWN stops when proved; PGD stops at a confirmed counterexample. alpha,beta-CROWN uses 20 initial alpha iterations (100 left no time for branch and bound).
- **Soundness check**: 0 of 3,249 verified verdicts contradicted by a PGD counterexample; every bound >= the best score PGD found; no tool errors.

## Final results (seed 0)

### Verified queries: 300 s cutoff / 1800 s limit

| Model | Max verifiable | Cert-RNN+ | alpha,beta-CROWN | alpha-CROWN | CROWN | Cert-RNN | IBP |
|---|---|---|---|---|---|---|---|
| grid9 | 182 | **172 / 172** | 154 / 159 | 142 / 142 | 141 / 141 | 136 / 136 | 0 / 0 |
| grid14 | 190 | **190 / 190** | - | - | 180 / 180 | 166 / 166 | 0 / 0 |
| grid30 | 189 | **169 / 176** | - | - | 140 / 140 | 131 / 131 | 0 / 0 |
| grid9_T20 | 190 | **180 / 180** | - | - | 170 / 170 | 161 / 161 | 0 / 0 |
| grid9_T30 | 180 | **173 / 173** | - | - | 162 / 163 | 139 / 139 | 0 / 0 |
| grid9_L2 | 193 | **190 / 190** | - | - | 177 / 177 | 163 / 163 | 0 / 0 |

alpha-CROWN and alpha,beta-CROWN were run on grid9 only.

### Median time per query (s)

| Model | Cert-RNN+ | alpha,beta-CROWN | alpha-CROWN | CROWN | Cert-RNN | IBP |
|---|---|---|---|---|---|---|
| grid9 | 3.4 | 18.5 | 3.2 | 4.9 | 0.09 | 0.07 |
| grid14 | 10.7 | - | - | 4.7 | 0.16 | 0.07 |
| grid30 | 72.0 | - | - | 5.5 | 0.61 | 0.06 |
| grid9_T20 | 16.9 | - | - | 16.6 | 0.27 | 0.12 |
| grid9_T30 | 43.3 | - | - | 35.3 | 0.46 | 0.19 |
| grid9_L2 | 16.0 | - | - | 13.3 | 0.25 | 0.12 |

### Mean time per query (s), 1800 s limit

| Model | Cert-RNN+ | alpha,beta-CROWN | alpha-CROWN | CROWN | Cert-RNN | IBP |
|---|---|---|---|---|---|---|
| grid9 | 4.1 | 259 (23 timeouts at 1800 s) | 29.2 | 15.2 | 0.10 | 12.2 |
| grid14 | 14.2 | - | - | 14.6 | 0.17 | 11.5 |
| grid30 | 103 | - | - | 16.3 | 0.74 | 11.7 |
| grid9_T20 | 23.4 | - | - | 38.2 | 0.28 | 23.2 |
| grid9_T30 | 63.8 | - | - | 63.9 | 0.53 | 36.6 |
| grid9_L2 | 21.9 | - | - | 34.1 | 0.27 | 22.7 |

At the 300 s cutoff only three means change: alpha,beta-CROWN grid9 65.1, Cert-RNN+ grid30 100, CROWN grid9_T30 63.7.
Medians are identical at both cutoffs. Times include all queries run (timeouts at the limit). IBP and CROWN means include a
one-time graph build (about 10-30 s on one thread) charged to the first query of each (window, attack) pair.

### Head to head (identical queries)

- 1800 s: on all six models, no baseline verifies a query that Cert-RNN+ misses ("only other" = 0 for every tool).
  grid9 extra queries verified by Cert-RNN+: +13 vs alpha,beta-CROWN, +30 vs alpha-CROWN, +31 vs CROWN, +36 vs Cert-RNN.
  vs CROWN on the other models: grid14 +10, grid30 +36, T20 +10, T30 +10, L2 +13.
- 300 s: the same holds on every model except grid30, where CROWN verifies 6 queries that Cert-RNN+ needs 300-463 s to verify.

### Per attack family, grid9 (counts verified out of 10 per eps for persistent/burst, 30 for transient)

See `seeds_0/check_main_300s.csv` and `check_main_1800s.csv`. Headline: on single-second attacks at eps = 0.01 Cert-RNN+
verifies 27/30, versus 14/30 at 300 s and 19/30 at 1800 s (alpha,beta-CROWN), 1/30 (CROWN), 2/30 (alpha-CROWN) and 0/30 (Cert-RNN); persistent at eps = 0.003 Cert-RNN+ verifies 5/10 against a
PGD limit of 7/10 and 0/10 for every baseline. Regenerate exact counts from the CSVs rather than copying from older notes.

### Detector quality (seed 0, held-out test data, no attack)

| Model | tau | ROC-AUC | F1 | Recall | False-alarm rate |
|---|---|---|---|---|---|
| grid9 | 7.6e-5 | 0.973 | 0.935 | 0.887 | 0.50% |
| grid14 | 1.9e-4 | 0.961 | 0.919 | 0.852 | 0.14% |
| grid30 | 3.5e-4 | 0.975 | 0.948 | 0.902 | 0.08% |
| grid9_T20 | 9.3e-5 | 0.979 | 0.941 | 0.888 | 0.03% |
| grid9_T30 | 8.4e-5 | 0.983 | 0.955 | 0.917 | 0.17% |
| grid9_L2 | 7.8e-4 | 0.934 | 0.888 | 0.806 | 0.44% |

## Claims that are and are not supported

Supported:
- Cert-RNN+ verifies the most queries on every model at both cutoffs (grid9: 172 of 182 verifiable).
- At 1800 s, no baseline verifies a query Cert-RNN+ misses, on any model.
- On grid9, Cert-RNN+, alpha-CROWN and CROWN take comparable median time (3-5 s); alpha,beta-CROWN needs far more (mean 259 s, 23 timeouts) and still verifies fewer (159).
- alpha-CROWN gains only 1 query over CROWN: the remaining looseness comes from box-based relaxations of gate products, which Cert-RNN+ replaces with joint-region relaxations (see the formulation PDF).

Do NOT claim:
- That Cert-RNN+ is faster than CROWN in general: CROWN is much faster on wide models (grid30: 5.5 s vs 72 s median).
- That Cert-RNN+ never loses a query to a baseline at 300 s: false on grid30 (6 queries).
- Anything about other seeds, certified radius, the single-CPU "fair" run, or Zone-A: not run yet.

## Not yet done

- Certified-radius runs; the single-process "fair hardware" run; Zone-A (Sandia server); seeds 1-2 for all tools.
- An ablation separating linked gates from the joint score bound (would support the "why" argument).
- Grid resolution sensitivity (n = 8 / 16 / 32) for Cert-RNN+.
