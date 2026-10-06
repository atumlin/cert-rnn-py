# Sourced by the step scripts: activate the benchmark environment.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
ENV_DIR="${ENV_DIR:-$HERE/.venv}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Environment not found at $ENV_DIR -- run paper_bench/install.sh first (see INSTALL.md)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export ABCROWN_DIR="${ABCROWN_DIR:-${THIRD_PARTY:-$HERE/third_party}/alpha-beta-CROWN}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
cd "$REPO"
