# paper_bench — benchmark suite for Cert-RNN+ (LSTM-autoencoder verification)

Reproduces the tool comparison in the paper: Cert-RNN and Cert-RNN+ against IBP, CROWN, alpha-CROWN and alpha-beta-CROWN (+ GenBaB) on time-windowed LSTM-autoencoder anomaly detectors for power-grid measurements. PGD is used as an attack (counterexamples / upper bounds), not as a verifier. Design and decisions: [PLAN.md](PLAN.md).

## What is verified

A detector reads a window of `T` seconds x `D` measurements, reconstructs it, and raises an alarm when the reconstruction score (mean squared error over the window) exceeds `tau`. For a normal window, the property is **no false alarm**: every input whose attacked measurements move by at most `eps` still scores `<= tau`. Attacks: persistent (all seconds), burst (3 consecutive seconds), transient (one second: first, middle, last).

## Install

See [INSTALL.md](INSTALL.md): `bash paper_bench/install.sh` (creates `paper_bench/.venv` with every tool; ends with `INSTALL OK`).

## Run

Each step is a separate, **resumable** script: if it stops for any reason (crash, reboot, Ctrl-C, lost SSH session), run the same command again and it continues where it left off. Running inside `tmux` is recommended.

| step | command | what it does |
|---|---|---|
| 1 | `paper_bench/01_make_data.sh` | generate the grid benchmarks (IEEE 9/14/30-bus simulations) |
| 2 | `paper_bench/02_train.sh` | train every model (3 seeds for grid9), record detection metrics, pick the 10 verified windows |
| 3 | `paper_bench/03_check.sh` | fixed-eps checks, every tool on its recommended hardware (main table) |
| 4 | `paper_bench/04_radius.sh` | certified radius (secondary table) |
| 5 | `paper_bench/05_fair.sh` | steps 3-4 again with every tool on one CPU process (fair comparison) |
| 6 | `paper_bench/06_analyze.sh` | tables (CSV + LaTeX) and figures (PDF) in `paper_bench/results/` |
| - | `paper_bench/reproduce.sh --quick` | the whole pipeline on a tiny configuration (about an hour) |
| - | `paper_bench/reproduce.sh --full` | the whole pipeline as in the paper (days) |

Progress at any time: `source paper_bench/.venv/bin/activate && python -m paper_bench.status`.

Useful arguments (all step scripts): `--bench grid9 grid14`, `--tools certrnn_plus crown`, `--quick`. Steps 3-4 can be split across machines by benchmark; results append to the same files without conflict as long as each machine runs different benchmarks.

## Zone-A (Sandia only)

The real Zone-A data never leaves Sandia; neither do models, weights or counterexample inputs. Only the result files are shared.

```bash
cd cert-rnn-py && git pull                      # get the latest suite
bash paper_bench/install.sh                     # once
tmux new -s bench
paper_bench/01_make_data.sh --bench zoneA --data-dir /path/to/zoneA   # checks the three CSVs
paper_bench/02_train.sh     --bench zoneA --data-dir /path/to/zoneA   # 3 seeds
paper_bench/03_check.sh     --bench zoneA
paper_bench/04_radius.sh    --bench zoneA
paper_bench/06_analyze.sh
```

The data folder must contain `SubFeatures_baseline3hr_zoneA.csv`, `SubFeatures_cyberphysical_0_1_zoneA.csv` and `CyPhyScenarioLabels_2Class_0_1_zoneA.csv`.

**What to bring back:** the folder `paper_bench/results/` (JSONL logs of verdicts, radii and timings; CSV/LaTeX tables; PDF figures). It contains numbers only. Do **not** copy `paper_bench/work/` (data, models, chosen windows, counterexample inputs).

## Where things go

| path | contents | share? |
|---|---|---|
| `paper_bench/work/data/` | generated benchmark data | public benchmarks only (regenerable) |
| `paper_bench/work/models/<bench>/seed<k>/` | `model.pt`, `anchors.npz`, `meta.json`, checkpoints | no (Zone-A); regenerable otherwise |
| `paper_bench/work/cex/` | PGD counterexample inputs | no (they contain data values) |
| `paper_bench/work/radius_state/` | resume state of radius searches | no |
| `paper_bench/results/<hardware>/` | `check.jsonl`, `radius.jsonl`, tables, figures | yes |

## Configuration

Everything (benchmarks, seeds, tools, budgets, eps values, attacks, number of windows) is in [config.yaml](config.yaml). Per-query budgets: 300 s for Cert-RNN, Cert-RNN+, IBP, CROWN and PGD; 1800 s for alpha-beta-CROWN; alpha-CROWN is set after the pilot (default 3600 s). alpha-CROWN and alpha-beta-CROWN run only on the benchmarks marked `slow_tools: true` (grid9 and zoneA).
