#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="3efc221fd8d5f4de7c2dc2022357e600d26325c7"
DEST="$ROOT/vendor/EvoMemBench"
mkdir -p "$ROOT/vendor"
if [[ ! -d "$DEST/.git" ]]; then
  [[ ! -e "$DEST" ]] || { echo "Not overwriting $DEST" >&2; exit 1; }
  git clone --filter=blob:none --no-checkout https://github.com/DSAIL-Memory/EvoMemBench.git "$DEST"
fi
if [[ -n "$(git -C "$DEST" status --porcelain)" && "$(git -C "$DEST" rev-parse HEAD)" != "$PIN" ]]; then
  echo "Dirty upstream checkout; aborting" >&2; exit 1
fi
git -C "$DEST" fetch origin "$PIN"
git -C "$DEST" sparse-checkout init --cone
git -C "$DEST" sparse-checkout set Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB EvoMemBench-Memory-Systems
git -C "$DEST" checkout --detach "$PIN"
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$PIN" ]]
python "$ROOT/scripts/patch_upstream.py" --upstream "$DEST" --adapter "$ROOT/src/evo_mem_bench/control_memory.py"
python -m compileall -q "$DEST/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/scripts"
echo "Pinned upstream prepared: $PIN"
