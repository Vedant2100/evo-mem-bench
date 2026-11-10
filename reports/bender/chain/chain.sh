#!/usr/bin/env bash
# Detached chain: swap to 32B-AWQ -> smoke -> competence gate -> 2-stream screen. Stops with a reason on any failed gate.
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; S=$C/STATUS
st(){ echo "$(date '+%F %T') $*" | tee -a $S; }
stop(){ st "STOPPED: $*"; exit 1; }
cd $R
st "chain start"
until [ -f $E/dl_32b.done ] || grep -q "snapshots" $E/dl_32b.log 2>/dev/null; do sleep 20; done
bash $C/preflight.sh >> $C/preflight.log 2>&1 || stop "preflight failed (see chain/preflight.log)"
st "preflight ok"
# swap vLLM: stop my 8B server, start 32B-AWQ across both A30s
pkill -f "vllm serve Qwen/Qwen3-8B"; sleep 20
export HF_HOME=$E/hf HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
setsid nohup $E/.venv-vllm/bin/vllm serve Qwen/Qwen3-32B-AWQ --host 127.0.0.1 --port 8100 --tensor-parallel-size 2 \
  --max-model-len 16384 --gpu-memory-utilization 0.88 --default-chat-template-kwargs '{"enable_thinking": false}' > $E/vllm32.log 2>&1 &
for i in $(seq 120); do curl -s localhost:8100/v1/models | grep -q Qwen3-32B && break; sleep 10; done
curl -s localhost:8100/v1/models | grep -q Qwen3-32B || stop "vLLM 32B-AWQ failed to start (see vllm32.log)"
OUT=$(curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3-32B-AWQ","messages":[{"role":"user","content":"Say hi."}],"max_tokens":30,"temperature":0}')
echo "$OUT" | grep -q "<think>" && stop "thinking not disabled"
st "vLLM 32B-AWQ ready (TP=2, thinking off)"
# ALFWorld servers: 36105 (existing) and 36106
. .venv-server/bin/activate
for P in 36105 36106; do
  ss -ltn | grep -q ":$P " || { ALFWORLD_DATA=$E/alfworld-data ALFWORLD_PORT=$P setsid nohup bash scripts/start_server.sh > $E/alfworld_$P.log 2>&1 & sleep 25; }
  ss -ltn | grep -q ":$P " || stop "ALFWorld server $P down"
done
deactivate
. .venv-eval/bin/activate
set -a; . ./.env; set +a
export WORKER_MODEL=Qwen/Qwen3-32B-AWQ EVO_MEM_LLM_MODEL=Qwen/Qwen3-32B-AWQ
# smoke
python scripts/run_experiment.py --plan smoke --model Qwen/Qwen3-32B-AWQ > $C/smoke.log 2>&1 || stop "smoke crashed"
SR=$(grep -o "runs/[^ ]*" $C/smoke.log | tail -1); ls $R/$SR/seed_0 >/dev/null 2>&1 || stop "smoke produced no results"
st "smoke ok: $SR"
# competence gate
python scripts/run_experiment.py --plan competence --model Qwen/Qwen3-32B-AWQ > $C/competence.log 2>&1 || stop "competence crashed"
CR=$(grep -o "runs/[0-9A-Za-z_]*" $C/competence.log | tail -1)
python scripts/analyze_results.py $CR >/dev/null 2>&1
G=$(python $C/gate.py $CR 0.55 0.95); GR=$?
st "competence gate: $G ($CR)"
[ $GR -eq 0 ] || stop "competence gate failed (no-memory overall outside 55-95%); screen NOT run"
# screen: two streams, separate ALFWorld servers
A=none,playbook,recent,bm25,outcome_bm25
B=none,reflexion,expel,letta_consolidation,supervised_writer,bayesian_state
COMMON="--plan core --seeds 0 --train-per-category 10 --test-per-category 10 --model Qwen/Qwen3-32B-AWQ"
ALFWORLD_PORT=36105 python scripts/run_experiment.py $COMMON --methods $A > $C/screenA.log 2>&1 & PA=$!
ALFWORLD_PORT=36106 python scripts/run_experiment.py $COMMON --methods $B > $C/screenB.log 2>&1 & PB=$!
st "screen started (A pid $PA, B pid $PB)"
wait $PA; RA=$?; wait $PB; RB=$?
for X in A B; do
  D=$(grep -o "runs/[0-9A-Za-z_]*" $C/screen$X.log | tail -1)
  [ -n "$D" ] && python scripts/analyze_results.py $D >/dev/null 2>&1 && st "screen $X analyzed: $D"
done
[ $RA -eq 0 ] && [ $RB -eq 0 ] && st "DONE: screen complete" || st "DONE WITH ERRORS: screen exit codes A=$RA B=$RB (missing episodes are NOT zero-success)"
