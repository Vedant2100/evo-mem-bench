#!/usr/bin/env bash
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
cd $R
( . .venv-server/bin/activate; ALFWORLD_DATA=$E/alfworld-data ALFWORLD_PORT=36107 setsid nohup bash scripts/start_server.sh > $E/alfworld_36107.log 2>&1 & ) ; sleep 30
ss -ltn | grep -q ":36107 " || { st "STOPPED(C): ALFWorld server 36107 down"; exit 1; }
. .venv-eval/bin/activate; set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ
st "stream C (RL-style bandits) started"
ALFWORLD_PORT=36107 python scripts/run_experiment.py --plan core --seeds 0 --train-per-category 6 --test-per-category 6 --model Qwen/Qwen3-32B-AWQ --methods none,bandit,contextual_bandit > $C/r6_screenC.log 2>&1; RC=$?
D=$(grep -o "runs/[0-9A-Za-z_]*" $C/r6_screenC.log | tail -1)
[ -n "$D" ] && python scripts/analyze_results.py $D >/dev/null 2>&1 && st "stream C analyzed: $D"
st "DONE(C): exit $RC"
