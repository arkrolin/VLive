#!/bin/bash
# Finalize arm P (span_cond=log): offline full-val re-eval + residual sweep.
# Robust to being launched before P starts: first wait for P to START, then
# wait for it to FINISH. (This script's own cmdline has no 'train.py', so its
# pgrep never self-matches.)
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
LOG=outputs/logs/finalize_P.log
mkdir -p outputs/logs
echo "=== finalize start $(date) ===" >> $LOG
while ! pgrep -f 'train.py.*abl_P_spancond' > /dev/null; do sleep 60; done
echo "=== P training started $(date) ===" >> $LOG
while pgrep -f 'train.py.*abl_P_spancond' > /dev/null; do sleep 60; done
echo "=== training done $(date) ===" >> $LOG
for CK in best latest; do
  echo "--- ckpt_$CK ---" >> $LOG
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python outputs/eval_ckpt.py abl_P_spancond --n_samples 0 --device cuda --ckpt $CK >> $LOG 2>&1
done
echo "--- sweep latest ---" >> $LOG
CUDA_VISIBLE_DEVICES=5 .venv/bin/python outputs/eval_ckpt.py abl_P_spancond --n_samples 0 --device cuda --ckpt latest --sweep 0.5,0.75,0.9,1.0,1.1 >> $LOG 2>&1
echo "=== finalize done $(date) ===" >> $LOG
