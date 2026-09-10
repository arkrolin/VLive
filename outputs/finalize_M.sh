#!/bin/bash
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
LOG=outputs/logs/finalize_M.log
mkdir -p outputs/logs
echo "=== finalize start $(date) ===" >> $LOG
while pgrep -f 'train.py.*abl_M_rangefix' > /dev/null; do sleep 60; done
echo "=== training done $(date) ===" >> $LOG
for CK in best latest; do
  echo "--- ckpt_$CK ---" >> $LOG
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python outputs/eval_ckpt.py abl_M_rangefix --n_samples 0 --device cuda --ckpt $CK >> $LOG 2>&1
done
echo "--- sweep latest ---" >> $LOG
CUDA_VISIBLE_DEVICES=1 .venv/bin/python outputs/eval_ckpt.py abl_M_rangefix --n_samples 0 --device cuda --ckpt latest --sweep 0.5,0.75,0.9,1.0,1.1 >> $LOG 2>&1
echo "=== finalize done $(date) ===" >> $LOG
