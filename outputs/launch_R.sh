#!/bin/bash
# Relaunch arm R only (Q is running on GPU 1,2,3).
# Reason for the restart: Live2d-model-master contains empty / zero-duration
# motion3.json; arm R crashed in __getitem__ with KeyError: 'Param1'.
# Fixed by outputs/_patch_v9_empty.py (drop them at corpus-build time).
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive

TR=".venv/bin/torchrun"
grep -q "n_empty" src/live2d_vla/dataset.py || { echo "empty-motion patch missing"; exit 1; }
echo "patch verified"

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup "$TR" --nproc_per_node=4 --master_port=29713 \
  src/live2d_vla/train.py \
  --gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
  --span_w 1 --span_w_cap 8 --epochs 70 \
  --action_map_path outputs/action_semantic_map.json \
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
