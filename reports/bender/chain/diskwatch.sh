#!/usr/bin/env bash
# Detached: log free GB every 5 min to disk.log; write DISK WARNING to STATUS when <8G. Never deletes anything.
while true; do
  a=$(df --output=avail -BG /var/tmp | tail -1 | tr -dc 0-9)
  echo "$(date '+%F %T') ${a}G" >> /var/tmp/vbork001/evomem/chain/disk.log
  [ "$a" -lt 8 ] && echo "$(date '+%F %T') DISK WARNING: only ${a}G free on /var/tmp (runs may fail to write)" >> /var/tmp/vbork001/evomem/chain/STATUS
  sleep 300
done
