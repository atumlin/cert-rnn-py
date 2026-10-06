#!/usr/bin/env bash
# Install the benchmark environment: a Python 3.11 venv with PyTorch,
# auto_LiRPA + alpha-beta-CROWN (pinned commit), and cert-rnn (this repo).
# Re-runnable: every finished step is skipped. See INSTALL.md.
#
# Environment variables (all optional):
#   ENV_DIR        venv location            (default: paper_bench/.venv)
#   THIRD_PARTY    where alpha-beta-CROWN is cloned (default: paper_bench/third_party)
#   PYTHON         a Python 3.11 interpreter (default: python3.11, else conda)
#   TORCH_INDEX    PyTorch wheel index      (default: CUDA 12.6; CPU-only:
#                  https://download.pytorch.org/whl/cpu)
#   ABCROWN_URL    git URL or local path of alpha-beta-CROWN (for offline mirrors)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
ENV_DIR="${ENV_DIR:-$HERE/.venv}"
THIRD="${THIRD_PARTY:-$HERE/third_party}"
ABCROWN_URL="${ABCROWN_URL:-https://github.com/Verified-Intelligence/alpha-beta-CROWN.git}"
ABCROWN_COMMIT="e5c7e17bf0488843acb77b7519f59876717a49f4"   # June 2026 release
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
log() { echo "[install $(date +%T)] $*"; }

# 1. Python 3.11 ---------------------------------------------------------------
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  PY="${PYTHON:-$(command -v python3.11 || true)}"
  if [[ -z "$PY" ]]; then
    if command -v conda >/dev/null; then
      log "no python3.11 found; creating one with conda in $HERE/.py311"
      [[ -x "$HERE/.py311/bin/python" ]] || conda create -y -p "$HERE/.py311" python=3.11
      PY="$HERE/.py311/bin/python"
    else
      echo "Need Python 3.11 (alpha-beta-CROWN requires ~=3.11). Set PYTHON=/path/to/python3.11." >&2
      exit 1
    fi
  fi
  "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' \
    || { echo "PYTHON must be 3.11" >&2; exit 1; }
  log "creating venv $ENV_DIR with $PY"
  "$PY" -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel

# 2. PyTorch -------------------------------------------------------------------
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.11') else 1)" 2>/dev/null; then
  log "installing torch 2.11.0 from $TORCH_INDEX"
  python -m pip install "torch==2.11.0" torchvision --index-url "$TORCH_INDEX"
fi

# 3. alpha-beta-CROWN + auto_LiRPA (pinned) ------------------------------------
ABC="$THIRD/alpha-beta-CROWN"
if [[ ! -d "$ABC/.git" ]]; then
  log "cloning alpha-beta-CROWN"
  mkdir -p "$THIRD"
  git clone "$ABCROWN_URL" "$ABC"
fi
if [[ "$(git -C "$ABC" rev-parse HEAD)" != "$ABCROWN_COMMIT" ]]; then
  git -C "$ABC" fetch --quiet origin || true
  git -C "$ABC" checkout --quiet "$ABCROWN_COMMIT"
fi
git -C "$ABC" submodule update --init --recursive
if ! python -c "import auto_LiRPA" 2>/dev/null; then
  log "installing auto_LiRPA (alpha-beta-CROWN submodule)"
  python -m pip install "$ABC/auto_LiRPA"
fi
if ! python -c "import abcrown" 2>/dev/null; then
  log "installing alpha-beta-CROWN"
  python -m pip install "$ABC"
fi

# 4. cert-rnn + benchmark extras -------------------------------------------------
if ! python -c "import cert_rnn, pandapower, matplotlib, yaml, numba" 2>/dev/null; then
  log "installing cert-rnn with benchmark extras"
  python -m pip install -e "$REPO[bench]"
fi

# 5. smoke test ----------------------------------------------------------------
log "checking the installation"
ABCROWN_DIR="$ABC" python -m paper_bench.check_install ${CHECK_ARGS:-}
log "done. Activate with:  source $ENV_DIR/bin/activate"
