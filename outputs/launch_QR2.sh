#!/bin/bash
# Restart BOTH V9 arms after the __getitem__ zero-fill fix
# (outputs/_patch_v9_safe_fallback.py). Q was at ep3, R at ep1 - cheap to redo,
# and cheaper than losing a run to a KeyError hours in.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive

TR=".venv/bin/torchrun"
grep -q "Zero-fill instead of KeyError" src/live2d_vla/dataset.py || {
  echo "safe-fallback patch missing"; exit 1; }
grep -q "n_empty" src/live2d_vla/dataset.py || { echo "empty-motion patch missing"; exit 1; }
echo "patches verified"

AMAP="outputs/action_semantic_map.json"
COMMON="--gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
--span_w 1 --span_w_cap 8 --epochs 70 --action_map_path $AMAP"

CUDA_VISIBLE_DEVICES=1,2,3 nohup "$TR" --nproc_per_node=3 --master_port=29714 \
  src/live2d_vla/train.py $COMMON \
  --batch_size 8 --subset_models 285 \
  --data_root standrad-live-2d \
  --whitelist_path outputs/model_whitelist.json \
  --gen_mask_path outputs/gen_mask.json \
  --val_holdout_path outputs/val_holdout_std.json \
  --dedup_skip_path outputs/dedup_skip_std.json \
  --cache_tag v9std \
  --out_dir outputs/train_runs/abl_Q_std_v9 \
  > outputs/logs/abl_Q.log 2>&1 < /dev/null &
echo "launched Q (pid $!) on GPU 1,2,3"

sleep 15

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup "$TR" --nproc_per_node=4 --master_port=29715 \
  src/live2d_vla/train.py $COMMON \
  --batch_size 6 --subset_models 580 \
  --data_root data/all \
  --whitelist_path outputs/model_whitelist_all.json \
  --gen_mask_path outputs/gen_mask_all.json \
  --val_holdout_path outputs/val_holdout_all.json \
  --dedup_skip_path outputs/dedup_skip_all.json \
  --cache_tag v9all \
  --out_dir outputs/train_runs/abl_R_all_v9 \
  > outputs/logs/abl_R.log 2>&1 < /dev/null &
echo "launched R (pid $!) on GPU 4,5,6,7"
