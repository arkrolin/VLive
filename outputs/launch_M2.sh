#!/bin/bash
# Arm M: identical to N (span_w=1, 70ep, select on abs) EXCEPT the range
# statistics are fixed (full-corpus scan + median fallback instead of n=400 +
# global range). N, already running, is the legacy-normalisation control.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
rm -rf outputs/train_runs/smoke_rangefix
mkdir -p outputs/logs
nohup bash -c "cd /root/work/nlp/xjzhao13/lijie_llama/VLive && \
CUDA_VISIBLE_DEVICES=5,6,7 .venv/bin/torchrun --nproc_per_node=3 --master_port=29672 \
src/live2d_vla/train.py --gen_mode regress --subset_models 285 --action_cond id \
--eval_every 1 --batch_size 8 --epochs 70 --span_w 1 --span_w_cap 8 \
--select_metric abs --residual_gate none \
--out_dir outputs/train_runs/abl_M_rangefix" \
> outputs/logs/abl_M.log 2>&1 &
echo "launched M (pid $!)"
