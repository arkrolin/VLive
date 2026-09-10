#!/bin/bash
# Measure the rig leak WITHOUT retraining.
#
# Arm R's 96d identity feature is aggregated over each character's motions
# INCLUDING the target motion, so it hands the model the target's amplitude on
# 29.2% of channels (5.2% exclusively). Re-scoring R's own checkpoint with
# rig_loo=True removes exactly that information and nothing else - the weights
# are untouched - so the abs_mae delta IS the size of the leak.
#
# Reference: R scores 0.9202 (best) / 0.9231 (latest) with the leaky rig.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs
nohup ./.venv/bin/python outputs/eval_ckpt.py abl_R_all_v9 \
  --ckpt best --device cuda:5 --rig_loo \
  --dump outputs/eval_dumps \
  > outputs/logs/eval_R_rigloo.log 2>&1 < /dev/null &
echo "launched R --rig_loo eval (pid $!) on GPU 5"
