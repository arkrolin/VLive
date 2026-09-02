#!/bin/bash
# Launch the span-weighted-loss arms (K, L) in parallel, 3 GPUs each.
#
# Motivation (see outputs/_patch_span_weight.py)
# ----------------------------------------------
# The residual-scale sweep found, consistently across H/I/J:
#     abs_mae  (span-weighted, raw units)  -> best at w ~= 0.75
#     rel_f    (equal weight per param)    -> best at w == 1.0
#     w = 1.25                            -> catastrophic (worse than the prior)
# i.e. the residual overshoots on LARGE-range params, and the loss never charges
# the model for that because it weights every param-instance equally in
# normalised units. Weighting the loss by span**p makes the optimiser pay for
# raw-unit error, so it should learn the shrinkage itself.
#
#   K = span_w 1, cap 8   -> loss E[d^2 / span]   (moderate, max weight 8x)
#   L = span_w 2, cap 3   -> loss E[d^2]          (raw-unit MSE, max weight 9x)
#
# Both are otherwise IDENTICAL to H (the current best arm): regress mode,
# subset 285, action_cond id, 45 epochs, batch_size 8 x 3 GPUs = effective 24.
# H is therefore the span_w=0 control.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive

PY=".venv/bin/python"
TR=".venv/bin/torchrun"
[ -x "$TR" ] || { echo "torchrun missing at $TR"; exit 1; }

COMMON="--gen_mode regress --subset_models 285 --action_cond id --eval_every 1 --batch_size 8 --epochs 45"

# --- K : moderate span weighting --------------------------------------------
CUDA_VISIBLE_DEVICES=1,2,3 nohup "$TR" --nproc_per_node=3 --master_port=29660 \
  src/live2d_vla/train.py $COMMON \
  --span_w 1 --span_w_cap 8 \
  --out_dir outputs/train_runs/abl_K_span1 \
  > outputs/train_runs/abl_K.log 2>&1 < /dev/null &
echo "launched K (pid $!)"

# --- L : raw-unit MSE -------------------------------------------------------
CUDA_VISIBLE_DEVICES=5,6,7 nohup "$TR" --nproc_per_node=3 --master_port=29661 \
  src/live2d_vla/train.py $COMMON \
  --span_w 2 --span_w_cap 3 \
  --out_dir outputs/train_runs/abl_L_span2 \
  > outputs/train_runs/abl_L.log 2>&1 < /dev/null &
echo "launched L (pid $!)"

# Wait for epoch 1 + its reconstruction eval. NOTE: the in-training eval subset
# is now STRATIFIED (4 samples per held-out character, ~136 samples), so the
# exem baseline will NOT be the old 2.4405 any more - record whatever it prints.
sleep 360
echo "--- K epoch 1 ---"; grep -m1 '\[rank0\] ep 1 ' outputs/train_runs/abl_K.log
echo "--- L epoch 1 ---"; grep -m1 '\[rank0\] ep 1 ' outputs/train_runs/abl_L.log
echo "--- exem baselines ---"
grep -m1 'exem_abs' outputs/train_runs/abl_K.log
grep -m1 'exem_abs' outputs/train_runs/abl_L.log
echo "--- alive ---"; ps -eo pid,etime,cmd | grep '[t]orchrun' | head
