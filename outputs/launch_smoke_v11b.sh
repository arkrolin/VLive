#!/bin/bash
# Smoke-test the V11b bank path (and warm outputs/motionbank_580_v9all.pkl).
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs
nohup ./.venv/bin/python outputs/smoke_v11b.py \
  > outputs/logs/smoke_v11b.log 2>&1 < /dev/null &
echo "smoke test launched pid $!"
