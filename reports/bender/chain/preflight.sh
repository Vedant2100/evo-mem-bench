#!/usr/bin/env bash
# Readiness checks. Exit nonzero on first hard failure. No GPU use.
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; fail=0
ok(){ echo "OK   $*"; }; bad(){ echo "FAIL $*"; fail=1; }
cd $R
avail=$(df --output=avail -BG /var/tmp | tail -1 | tr -dc 0-9); [ "$avail" -ge 15 ] && ok "disk ${avail}G free" || bad "disk ${avail}G free (<15)"
W=$(ls -d $E/hf/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/* 2>/dev/null | head -1)
[ -n "$W" ] && [ "$(ls $W/*.safetensors 2>/dev/null | wc -l)" -ge 1 ] && ok "32B-AWQ weights $(du -shL $W | cut -f1)" || bad "32B-AWQ weights missing"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -le 1 ] && ok "GPU compute procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) (mine only)" || bad "foreign GPU processes"
[ -x $E/.venv-vllm/bin/vllm ] && ok "vllm binary" || bad "vllm binary"
. .venv-eval/bin/activate
python -m pytest -q 2>&1 | tail -1 | grep -q passed && ok "pytest" || bad "pytest"
python scripts/prepare_manifest.py --validate-only >/dev/null 2>&1 && ok "manifest valid" || bad "manifest"
python scripts/run_experiment.py --plan smoke --dry-run >/dev/null 2>&1 && ok "smoke dry-run" || bad "smoke dry-run"
[ "$(find $E/alfworld-data/json_2.1.1 -name game.tw-pddl | wc -l)" -ge 5000 ] && ok "ALFWorld games" || bad "ALFWorld games"
[ -f .env ] && ok ".env present" || bad ".env"
exit $fail
