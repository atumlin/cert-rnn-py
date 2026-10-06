# Installing the benchmark suite (Sandia server or any Linux machine)

Everything runs from the cert-rnn-py repository. One script creates a self-contained Python environment with every tool.

## Requirements

- Linux, `git`, `bash`
- **Python 3.11** (alpha-beta-CROWN requires 3.11 exactly). If `python3.11` is not on the PATH, the script uses `conda` to create one.
- Internet access for `pip` and `git clone` (or a proxy / local mirror, see below)
- An NVIDIA GPU is recommended (the CROWN-family tools run on it); everything also works on CPU only.
- About 10 GB of disk for the environment.

## Install (once per machine)

```bash
cd cert-rnn-py
git pull
bash paper_bench/install.sh 2>&1 | tee install.log
```

The script is safe to rerun: finished steps are skipped. It ends with a smoke test that runs one tiny query with every tool and prints `INSTALL OK`.

What it installs, all inside `paper_bench/.venv` (nothing system-wide):

| component | version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.11.0 (CUDA 12.6 wheels by default) |
| alpha-beta-CROWN | commit `e5c7e17` (June 2026 release), cloned into `paper_bench/third_party/` |
| auto_LiRPA | the alpha-beta-CROWN submodule (commit `5a098e8`) |
| cert-rnn | this repository, plus `pandas pandapower numba matplotlib scikit-learn pyyaml` |

## Options (environment variables)

| variable | when to use |
|---|---|
| `PYTHON=/path/to/python3.11` | a specific Python 3.11 interpreter |
| `TORCH_INDEX=https://download.pytorch.org/whl/cpu` | no GPU / CPU-only wheels |
| `TORCH_INDEX=https://download.pytorch.org/whl/cu121` | an older CUDA driver (check `nvidia-smi`) |
| `ENV_DIR=/scratch/me/bench-env` | put the environment somewhere else (then export the same `ENV_DIR` before running the step scripts) |
| `ABCROWN_URL=/path/to/local/alpha-beta-CROWN` | no GitHub access: point at a copied checkout |
| `PIP_INDEX_URL=...`, `HTTPS_PROXY=...` | internal pip mirror or web proxy (standard pip / git variables) |

Example for a CPU-only machine behind a proxy:

```bash
HTTPS_PROXY=http://proxy.example:8080 TORCH_INDEX=https://download.pytorch.org/whl/cpu \
  bash paper_bench/install.sh
```

## Check

```bash
source paper_bench/.venv/bin/activate
python -m paper_bench.check_install
```

Expected: `[ok]` for every package and `tool ... -> verified` (or `no_violation` for PGD) for all seven tools, then `INSTALL OK`. If alpha-beta-CROWN fails, the last lines of its error are printed; send them back.

## Troubleshooting

- **`PYTHON must be 3.11`** -- install Python 3.11 or make `conda` available; or pass `PYTHON=`.
- **CUDA not available** (`CUDA available: False`) although the server has GPUs -- the PyTorch wheels do not match the driver: reinstall with a matching `TORCH_INDEX` (remove `paper_bench/.venv` first).
- **`git clone` fails** -- copy `alpha-beta-CROWN` from another machine (`git clone --recursive`, then `git checkout e5c7e17`) and use `ABCROWN_URL`.
- **Running out of GPU memory in alpha-beta-CROWN** -- lower `solver.batch_size` in `paper_bench/abcrown.yaml`.
