#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPONENT="$ROOT/vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB"
[[ -d "$COMPONENT/agentenv-alfworld" ]] || { echo "Run bootstrap first" >&2; exit 1; }
cd "$COMPONENT/agentenv-alfworld"
export ALFWORLD_PORT="${ALFWORLD_PORT:-36005}"
exec alfworld --host 0.0.0.0 --port "$ALFWORLD_PORT"
