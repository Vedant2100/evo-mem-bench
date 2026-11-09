#!/usr/bin/env bash
# Reproducible ALFWorld text-server install with fail-fast dependency checks.
# Original setup.sh lacks set -e; it masks pip errors as missing download CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="$ROOT/vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/agentenv-alfworld"
[[ -d "$SERVER" ]] || { echo "Bootstrap pinned EvoMemBench first" >&2; exit 1; }
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
mkdir -p "$ALFWORLD_DATA"
python -m pip install "alfworld==0.3.3"
python -m pip uninstall -y opencv-python || true
python -m pip install -e "$SERVER"
if command -v alfworld-download >/dev/null 2>&1; then
  alfworld-download
else
  echo "Pip distribution lacks alfworld-download; using official release script"
  TMP_SOURCE="$(mktemp -d)"
  trap 'rm -rf "$TMP_SOURCE"' EXIT
  git clone --depth 1 --branch 0.3.3 https://github.com/alfworld/alfworld.git "$TMP_SOURCE/alfworld"
  python "$TMP_SOURCE/alfworld/scripts/alfworld-download"
fi
echo "Downloaded ALFWorld game files into $ALFWORLD_DATA"
find "$ALFWORLD_DATA/json_2.1.1" -name 'game.tw-pddl' | wc -l
