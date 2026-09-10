#!/bin/bash
# V11a: character-conditioned features, leave-one-out.
#
# Identical to arm R (outputs/launch_R.sh) except TWO switches, both of which
# are data-side fixes, not architecture changes:
#   --rig_loo     : drop the TARGET motion from the character's own 96d identity
#                   feature. Measured leak: on 29.2% of channels the target
#                   contributes >=50% of the character's global per-param range
#                   (5.2% exclusively), so the rig was handing the model the
#                   answer for amplitude. That is why R improves point-wise MAE
#                   by +39.8% but shape (1st-difference) MAE by only +2.1%.
#   --char_stats mlp
#                 : per-token, leave-one-out statistics of THIS character's
#                   OTHER motions for the same param (rest value / typical
#                   amplitude / p0-p100 envelope / support count), fed through a
#                   ZERO-INIT Linear(5, d). Zero-init means the run starts
#                   bit-identical to the un-conditioned baseline and only uses
#                   the features to the extent they help.
#                   Measured headroom: the character's own bank is 3.5-6.7x more
#                   informative than any other character's curves (0.5031 vs
#                   3.3719 abs_mae, same-size candidate pool), and its rest value
#                   alone is worth 1.31 -> 0.03 on the 43.9% of channels whose
#                   target is a constant.
#
# R's reference number to beat: abs_mae 0.9202 (best, ep68) / 0.9231 (latest).
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
TR=".venv/bin/torchrun"
[ -x "$TR" ] || { echo "torchrun missing"; exit 1; }
# caches must exist: outputs/ident_cache_580_v3_v9all.pkl and
# outputs/charstats_loo_580_v9all.pkl (built by outputs/prebuild_v11a_caches.py)
[ -f outputs/ident_cache_580_v3_v9all.pkl ] || { echo "ident v3 cache missing"; exit 1; }
[ -f outputs/charstats_loo_580_v9all.pkl ] || { echo "charstats cache missing"; exit 1; }
echo "caches verified"

EPOCHS=70
COMMON="--gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
--span_w 1 --span_w_cap 8 --epochs $EPOCHS \
--action_map_path outputs/action_semantic_map.json \
--rig_loo --char_stats mlp"

# GPUs 1-4 are taken by another job; 5,6,7 are free. 3 ranks x batch 8 = 24,
# the same effective batch as R (4 x 6), so abs_mae stays comparable.
CUDA_VISIBLE_DEVICES=5,6,7 nohup "$TR" --nproc_per_node=3 --master_port=29731 \
  src/live2d_vla/train.py $COMMON \
  --batch_size 8 --subset_models 580 \
  --data_root data/all \
  --whitelist_path outputs/model_whitelist_all.json \
  --gen_mask_path outputs/gen_mask_all.json \
  --val_holdout_path outputs/val_holdout_all.json \
  --dedup_skip_path outputs/dedup_skip_all.json \
  --cache_tag v9all \
  --out_dir outputs/train_runs/abl_V11a_charloo \
  > outputs/logs/abl_V11a_charloo.log 2>&1 < /dev/null &
echo "launched V11a (pid $!) on GPU 5,6,7"
