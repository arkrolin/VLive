#!/bin/bash
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs
nohup ./.venv/bin/python outputs/smoke_v11b_grad.py \
  > outputs/logs/smoke_v11b_grad.log 2>&1 < /dev/null &
echo "grad smoke launched pid $!"
