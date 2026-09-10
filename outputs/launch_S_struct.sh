#!/bin/bash
# V10 structured-head arm S: same config as R (abl_R_all_v9) but the dense
# PxT head is replaced by a structured latent head:
#   x0_hat = exem * (1 + alpha) + coef @ DCT_basis,  alpha in R^P, coef in R^(P,r)
# The ONLY variable vs R is head_mode=structured, residual_rank=2. Everything
# else (corpus, dedup, holdout, span_w, batch, epochs) is identical so abs_mae
# is directly comparable against R's 0.9202 / +39.4%.
#
# GPUs: 1,3,4,5 are free (0/2/6/7 taken). 4 GPUs x batch 6 = effective 24
# (matches R's 4 x 6).
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
TR=".venv/bin/torchrun"
[ -x "$TR" ] || { echo "torchrun missing"; exit 1; }

EPOCHS=70
COMMON="--gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
--span_w 1 --span_w_cap 8 --epochs $EPOCHS \
--action_map_path outputs/action_semantic_map.json \
--head_mode structured --residual_rank 2"

CUDA_VISIBLE_DEVICES=1,3,4,5 nohup "$TR" --nproc_per_node=4 --master_port=29721 \
  src/live2d_vla/train.py $COMMON \
  --batch_size 6 --subset_models 580 \
  --data_root data/all --whitelist_path outputs/model_whitelist_all.json \
  --gen_mask_path outputs/gen_mask_all.json \
  --val_holdout_path outputs/val_holdout_all.json \
  --dedup_skip_path outputs/dedup_skip_all.json --cache_tag v9all \
  --out_dir outputs/train_runs/abl_S_struct_r2 \
  > outputs/logs/abl_S_struct_r2.log 2>&1 < /dev/null &
echo "launched S (structured r=2) pid $! on GPU 1,3,4,5"
