#!/usr/bin/env bash
# Embedding-backed smoke tests, 3 streams in parallel (own ALFWorld servers), each method paired with none. No GPU use (embedding on CPU).
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
cd $R; . .venv-eval/bin/activate; set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ CUDA_VISIBLE_DEVICES=""
export DASHSCOPE_BASE_URL=http://127.0.0.1:8001/v1 DASHSCOPE_API_KEY=local EVO_MEM_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
curl -s -m 20 localhost:8001/v1/models | grep -q Embedding || { st "STOPPED(emb): embedding service not up"; exit 1; }
for P in 36105 36106 36118; do ss -ltn | grep -q ":$P " || { st "STOPPED(emb): ALFWorld $P not up"; exit 1; }; done
stream(){ port=$1; shift; for M in "$@"; do
  st "smoke(emb) $M started (port $port)"
  ALFWORLD_PORT=$port python scripts/run_experiment.py --plan smoke --model Qwen/Qwen3-32B-AWQ --methods none,$M > $C/smoke4_$M.log 2>&1; RC=$?
  if [ $RC -eq 0 ]; then D=$(grep -o "runs/[0-9A-Za-z_]*" $C/smoke4_$M.log | tail -1); st "smoke $M OK: $D"; else st "smoke $M FAILED exit $RC: $(tail -3 $C/smoke4_$M.log | tr '\n' ' ' | cut -c1-300)"; fi
done; }
stream 36105 amem &
stream 36106 amem_evolving &
stream 36118 mem0 &
wait
st "DONE: embedding-backed retry batch"
