#!/usr/bin/env bash
# Smoke tests for embedding-backed methods. CPU embedding service on :8001, own ALFWorld server 36119, one method at a time paired with none.
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
cd $R; . .venv-eval/bin/activate; set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ CUDA_VISIBLE_DEVICES=""
export DASHSCOPE_BASE_URL=http://127.0.0.1:8001/v1 DASHSCOPE_API_KEY=local EVO_MEM_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
curl -s -m 20 localhost:8001/v1/models | grep -q Embedding || { st "STOPPED(emb): embedding service not up"; exit 1; }
for M in amem amem_evolving reasoning_bank agent_kb awm mem0; do
  st "smoke(emb) $M started"
  ALFWORLD_PORT=36119 python scripts/run_experiment.py --plan smoke --model Qwen/Qwen3-32B-AWQ --methods none,$M > $C/smoke3_$M.log 2>&1; RC=$?
  if [ $RC -eq 0 ]; then D=$(grep -o "runs/[0-9A-Za-z_]*" $C/smoke3_$M.log | tail -1); st "smoke $M OK: $D"; else st "smoke $M FAILED exit $RC: $(tail -3 $C/smoke3_$M.log | tr '\n' ' ' | cut -c1-300)"; fi
done
st "DONE: embedding-backed smoke batch"
