#!/bin/bash
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs
.venv/bin/python outputs/smoke_v9.py > outputs/logs/smoke_v9.log 2>&1
echo "EXIT=$?" >> outputs/logs/smoke_v9.log
