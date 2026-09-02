#!/bin/bash
# Launch two controlled ablation arms in parallel (3 GPUs each).
#
#   I = regularisation package : dropout 0.1->0.3, weight_decay 1e-2->5e-2
#       Everything else identical to H (45 epochs, same cosine schedule).
#       -> clean A/B against H's abs_mae = 1.1196, isolates the generalisation gap.
#   J = convergence headroom   : 45 -> 60 epochs, lr_min 2e-5 -> 5e-6
#       H's val_loss was still setting new minima at ep45, i.e. unconverged.
#
# Both keep the effective batch at 24 (batch_size 8 x 3 GPUs) so the comparison
# against H (batch_size 4 x 6 GPUs) is not confounded by batch size.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive

PY=".venv/bin/python"
TR=".venv/bin/torchrun"
[ -x "$TR" ] || { echo "torchrun missing at $TR"; exit 1; }

COMMON="--gen_mode regress --subset_models 285 --action_cond id --eval_every 1 --batch_size 8"

# --- I : regularisation ------------------------------------------------------
CUDA_VISIBLE_DEVICES=1,2,3 nohup "$TR" --nproc_per_node=3 --master_port=29650 \
  src/live2d_vla/train.py $COMMON \
  --epochs 45 --dropout 0.3 --weight_decay 0.05 \
  --out_dir outputs/train_runs/abl_I_reg45 \
  > outputs/train_runs/abl_I.log 2>&1 < /dev/null &
echo "launched I (pid $!)"

# --- J : longer training -----------------------------------------------------
CUDA_VISIBLE_DEVICES=5,6,7 nohup "$TR" --nproc_per_node=3 --master_port=29651 \
  src/live2d_vla/train.py $COMMON \
  --epochs 60 --lr_min 5e-6 \
  --out_dir outputs/train_runs/abl_J_ep60 \
  > outputs/train_runs/abl_J.log 2>&1 < /dev/null &
echo "launched J (pid $!)"

# wait for epoch 1 + its reconstruction eval, so we can verify up front that the
# exem baseline is 2.4405 (i.e. the eval subset is the same 64 val samples as in
# runs A..H). If it prints anything else, the batch-size/eval-cap confound is back.
sleep 300
echo "--- I epoch 1 ---"; grep -m1 '\[rank0\] ep 1 ' outputs/train_runs/abl_I.log
echo "--- J epoch 1 ---"; grep -m1 '\[rank0\] ep 1 ' outputs/train_runs/abl_J.log
echo "--- alive ---"; ps -eo pid,etime,cmd | grep '[t]orchrun' | head
