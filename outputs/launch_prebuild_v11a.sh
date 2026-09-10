#!/bin/bash
# Pre-build the V11a caches (ident v3 + charstats LOO) in ONE process.
# Two full corpus passes; must finish before any distributed launch so the
# DDP ranks do not race to write the same .pkl.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs
nohup ./.venv/bin/python outputs/prebuild_v11a_caches.py \
  > outputs/logs/prebuild_v11a.log 2>&1 < /dev/null &
echo "prebuild launched pid $!"
