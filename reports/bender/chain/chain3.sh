#!/usr/bin/env bash
# Waits for the running 40-round competence job, applies the Amendment-2 gate (40-95%), then runs the deadline-sized 2-stream screen.
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
stop(){ st "STOPPED: $*"; exit 1; }
cd $R
st "chain3 start: waiting for 40-round competence job (pid 281375)"
while kill -0 281375 2>/dev/null; do sleep 10; done
. .venv-eval/bin/activate; set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ
CR=$(grep -o "runs/[0-9A-Za-z_]*" $C/r40_competence.log | tail -1)
ls $R/$CR/seed_2/playbook >/dev/null 2>&1 || stop "competence run incomplete ($CR); not treated as a result"
python scripts/analyze_results.py $CR >/dev/null 2>&1
G=$(python $C/gate.py $CR 0.40 0.95); GR=$?
st "competence gate (40-round, floor 40% per Amendment 2): $G ($CR)"
[ $GR -eq 0 ] || stop "no-memory overall outside 40-95%; reduced screen NOT run"
a=$(df --output=avail -BG /var/tmp | tail -1 | tr -dc 0-9); [ "$a" -ge 4 ] || stop "disk only ${a}G free"
curl -s localhost:8100/v1/models | grep -q Qwen3-32B || stop "model server down"
for P in 36105 36106; do ss -ltn | grep -q ":$P " || stop "ALFWorld server $P down"; done
COMMON="--plan core --seeds 0 --train-per-category 6 --test-per-category 6 --model Qwen/Qwen3-32B-AWQ"
ALFWORLD_PORT=36105 python scripts/run_experiment.py $COMMON --methods none,playbook,bm25,reflexion > $C/r6_screenA.log 2>&1 & PA=$!
ALFWORLD_PORT=36106 python scripts/run_experiment.py $COMMON --methods none,letta_consolidation,supervised_writer > $C/r6_screenB.log 2>&1 & PB=$!
st "reduced screen started (A pid $PA, B pid $PB)"
wait $PA; RA=$?; wait $PB; RB=$?
for X in A B; do D=$(grep -o "runs/[0-9A-Za-z_]*" $C/r6_screen$X.log | tail -1); [ -n "$D" ] && python scripts/analyze_results.py $D >/dev/null 2>&1 && st "screen $X analyzed: $D"; done
st "DONE: reduced screen exit codes A=$RA B=$RB (missing episodes are NOT zero-success)"
