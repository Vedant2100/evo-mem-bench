#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/vendor/EvoMemBench"
METHODS="\${*:-all}"

if [[ ! -d "$UPSTREAM/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB" ]]; then
  echo "Missing pinned upstream checkout. Run: bash scripts/bootstrap_upstream.sh" >&2
  exit 2
fi

install_amem() {
  python -m pip install -e "$UPSTREAM/EvoMemBench-Memory-Systems/A-mem"
}

install_agent_kb_awm() {
  python -m pip install "scikit-learn>=1.4" "python-dotenv>=1.0"
}

install_mem0() {
  python -m pip install "mem0ai>=0.1"
}

if [[ "$METHODS" == "all" ]]; then
  install_amem
  install_agent_kb_awm
  install_mem0
else
  for method in $METHODS; do
    case "$method" in
      amem|amem_evolving)
        install_amem
        ;;
      agent_kb|awm)
        install_agent_kb_awm
        ;;
      mem0)
        install_mem0
        ;;
      none|playbook|random|recent|full_history|bm25|success_bm25|outcome_bm25|bandit|contextual_bandit|supervised_writer|reflexion|expel|letta_consolidation|bayesian_state|evolutionary|ace|reasoning_bank|skillweaver|lightweight_memory)
        ;;
      *)
        echo "Unknown memory method: $method" >&2
        exit 2
        ;;
    esac
  done
fi

echo "Requested memory-method dependencies installed."
