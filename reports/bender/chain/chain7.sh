#!/usr/bin/env bash
# Original design screen (stage 3): 10 methods, 10 train + 10 test per category, seed 0, 40 rounds. 3 streams, 'none' paired in each.
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
cd $R
. .venv-eval/bin/activate; set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ
curl -s -m 10 localhost:8100/v1/models | grep -q Qwen3-32B-AWQ || { st "STOPPED(full): model server not up"; exit 1; }
for P in 36105 36106 36118; do ss -ltn | grep -q ":$P " || { st "STOPPED(full): ALFWorld server $P not running"; exit 1; }; done
run(){ # name port methods
  ALFWORLD_PORT=$2 python scripts/run_experiment.py --plan core --seeds 0 --train-per-category 10 --test-per-category 10 --model Qwen/Qwen3-32B-AWQ --methods $3 > $C/full_$1.log 2>&1; RC=$?
  D=$(grep -o "runs/[0-9A-Za-z_]*" $C/full_$1.log | tail -1)
  [ -n "$D" ] && python scripts/analyze_results.py $D >/dev/null 2>&1 && st "full screen $1 analyzed: $D"
  st "DONE(full $1): exit $RC"; }
st "full original screen started (10 methods, 10+10 per category) in 3 streams"
run A 36105 none,playbook,recent,bm25 &
run B 36106 none,outcome_bm25,reflexion,expel &
run C 36118 none,letta_consolidation,supervised_writer,bayesian_state &
wait
st "DONE: full original screen finished (missing episodes are NOT zero-success)"
