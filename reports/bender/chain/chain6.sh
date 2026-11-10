#!/usr/bin/env bash
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
cd $R
grep -q "Uvicorn running on http://0.0.0.0:36118" $E/alfworld_36118.log || { st "STOPPED(C): own ALFWorld server 36118 not running"; exit 1; }
. .venv-eval/bin/activate; set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ
st "stream C (RL-style bandits, retry on port 36118) started"
ALFWORLD_PORT=36118 python scripts/run_experiment.py --plan core --seeds 0 --train-per-category 6 --test-per-category 6 --model Qwen/Qwen3-32B-AWQ --methods none,bandit,contextual_bandit > $C/r6c_screenC.log 2>&1; RC=$?
D=$(grep -o "runs/[0-9A-Za-z_]*" $C/r6c_screenC.log | tail -1)
[ -n "$D" ] && python scripts/analyze_results.py $D >/dev/null 2>&1 && st "stream C analyzed: $D"
st "DONE(C): exit $RC"
