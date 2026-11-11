#!/usr/bin/env bash
# FINAL: remaining 10 methods at full size (10 train + 10 test per category, seed 0, 40 rounds), 4 parallel streams, 'none' paired in each. No GPU use beyond the shared model server (embedding on CPU).
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
cd $R; . .venv-eval/bin/activate; set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ CUDA_VISIBLE_DEVICES=""
export DASHSCOPE_BASE_URL=http://127.0.0.1:8001/v1 DASHSCOPE_API_KEY=local EVO_MEM_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
curl -s -m 20 localhost:8100/v1/models | grep -q Qwen3-32B-AWQ || { st "STOPPED(final): model server not up"; exit 1; }
curl -s -m 20 localhost:8001/v1/models | grep -q Embedding || { st "STOPPED(final): embedding service not up"; exit 1; }
for P in 36105 36106 36118 36119; do ss -ltn | grep -q ":$P " || { st "STOPPED(final): ALFWorld $P not up"; exit 1; }; done
run(){ name=$1; port=$2; methods=$3
  ALFWORLD_PORT=$port python scripts/run_experiment.py --plan core --seeds 0 --train-per-category 10 --test-per-category 10 --model Qwen/Qwen3-32B-AWQ --methods $methods > $C/final_$name.log 2>&1; RC=$?
  D=$(grep -o "runs/[0-9A-Za-z_]*" $C/final_$name.log | tail -1)
  [ $RC -eq 0 ] && [ -n "$D" ] && python scripts/analyze_results.py $D >/dev/null 2>&1 && st "final $name analyzed: $D"
  st "DONE(final $name): exit $RC (missing episodes are NOT zero-success)"; }
st "FINAL full-size run of remaining 10 methods started (4 streams)"
run A 36105 none,ace,skillweaver,lightweight_memory &
run B 36106 none,reasoning_bank,agent_kb,awm &
run C 36118 none,amem,amem_evolving,mem0 &
run D 36119 none,evolutionary &
wait
st "DONE: FINAL full-size run finished"
