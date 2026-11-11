#!/usr/bin/env bash
# Adapter/evolutionary smoke tests (no embedding endpoint needed), one at a time on own server 36119, each paired with none.
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
cd $R; . .venv-eval/bin/activate; set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ CUDA_VISIBLE_DEVICES=""
for M in evolutionary ace skillweaver lightweight_memory; do
  st "smoke(full-matrix) $M started"
  ALFWORLD_PORT=36119 python scripts/run_experiment.py --plan smoke --model Qwen/Qwen3-32B-AWQ --methods none,$M > $C/smoke2_$M.log 2>&1; RC=$?
  if [ $RC -eq 0 ]; then D=$(grep -o "runs/[0-9A-Za-z_]*" $C/smoke2_$M.log | tail -1); st "smoke $M OK: $D"; else st "smoke $M FAILED exit $RC: $(tail -2 $C/smoke2_$M.log | tr '\n' ' ' | cut -c1-250)"; fi
done
st "DONE: adapter smoke batch 1 (embedding-backed ones not yet tried)"
