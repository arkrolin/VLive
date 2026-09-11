#!/bin/bash
# Full 614-sample offline re-scoring of V11b (best then latest).
# Each ckpt gets its own process: shared-process state leaks and shifts abs_mae by up to 7%.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs

RUN=${RUN:-abl_V11b_bank}
DEV=${DEV:-cuda:5}

nohup bash -c "
  ./.venv/bin/python outputs/eval_ckpt.py $RUN --ckpt best \
      --device $DEV --dump outputs/eval_dumps > outputs/logs/eval_V11b_best.log 2>&1
  ./.venv/bin/python outputs/eval_ckpt.py $RUN --ckpt latest \
      --device $DEV --dump outputs/eval_dumps > outputs/logs/eval_V11b_latest.log 2>&1
" > /dev/null 2>&1 &

echo "launched eval driver pid $!"
