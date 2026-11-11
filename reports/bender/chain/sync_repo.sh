#!/usr/bin/env bash
# Every 20 min: write result summaries into the repo, commit locally, try to push. Never commits venvs/raw episodes/.env/keys.
E=/var/tmp/vbork001/evomem; R=$E/evo-mem-bench; C=$E/chain; L=$C/sync.log
export GIT_SSH_COMMAND="ssh -i $E/keys/evomem_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
while true; do
  cd $R; D=reports/bender; mkdir -p $D/chain
  cp $C/STATUS $D/STATUS.txt; cp $C/chain*.sh $C/gate.py $C/preflight.sh $D/chain/ 2>/dev/null
  for run in runs/*/; do
    id=$(basename $run); mkdir -p $D/runs/$id
    cp $run/manifest.json $D/runs/$id/ 2>/dev/null
    for f in REPORT.md category_and_overall.csv paired_effects.csv learning_curves.csv; do cp $run/analysis/$f $D/runs/$id/ 2>/dev/null; done
  done
  python3 - <<'PY'
import json,glob,os,csv
rows=[]
for p in glob.glob("runs/*/seed_*/*/*/*/alfworld_*.json"):
    parts=p.split("/")
    run,seed,method,phase,cat=parts[1],parts[2],parts[3],parts[4],parts[5]
    try: d=json.load(open(p))
    except Exception: continue
    rows.append((run,seed,method,phase,cat,1,1 if float(d.get("reward") or 0)>=1.0 else 0))
agg={}
for r in rows:
    k=r[:5]; a=agg.setdefault(k,[0,0]); a[0]+=r[5]; a[1]+=r[6]
with open("reports/bender/partial_progress.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["run","seed","method","phase","category","episodes_done","successes"])
    for k in sorted(agg): w.writerow(list(k)+agg[k])
PY
  git add reports/bender
  if ! git diff --cached --quiet; then
    git -c user.name=vbork001 -c user.email=vbork001@localhost commit -q -m "bender results update $(date '+%F %H:%M')"
    echo "$(date '+%F %T') committed" >> $L
  fi
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse -q origin/bender-screen-design 2>/dev/null)" ]; then
    git remote set-url origin git@github.com:Vedant2100/evo-mem-bench.git 2>/dev/null
    timeout 90 git push -q -u origin bender-screen-design >> $L 2>&1 && { echo "$(date '+%F %T') pushed" >> $L; git fetch -q origin bender-screen-design 2>/dev/null; } || echo "$(date '+%F %T') push failed (no access yet)" >> $L
  fi
  sleep 1200
done
