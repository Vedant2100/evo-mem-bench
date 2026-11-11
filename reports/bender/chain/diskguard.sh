#!/usr/bin/env bash
# Own processes only. Freeze (SIGSTOP) our experiment processes if /var/tmp free < 3G, resume (SIGCONT) when > 6G. Deletes nothing.
S=/var/tmp/vbork001/evomem/chain/STATUS; paused=0
pids(){ pgrep -u "$USER" -f '[r]un_experiment\.py|[e]val_alfworld'; }
while true; do
  a=$(df --output=avail -BG /var/tmp | tail -1 | tr -dc 0-9)
  if [ "$paused" = 0 ] && [ "$a" -lt 3 ]; then kill -STOP $(pids) 2>/dev/null; paused=1; echo "$(date '+%F %T') DISKGUARD: ${a}G free, experiment processes FROZEN (not failed; will resume above 6G)" >> $S
  elif [ "$paused" = 1 ] && [ "$a" -gt 6 ]; then kill -CONT $(pids) 2>/dev/null; paused=0; echo "$(date '+%F %T') DISKGUARD: ${a}G free, experiment processes RESUMED" >> $S; fi
  sleep 60
done
