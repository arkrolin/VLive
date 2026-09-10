#!/bin/bash
# V9: two parallel arms.
#
#   arm Q = arm M's configuration + the two DATA-HYGIENE fixes, on standrad only
#           (a) sample-level (body, action) de-duplication      -740 samples
#           (b) canonical action naming + semantic consolidation
#           -> same 34 held-out characters as M, so abs_mae is directly
#              comparable against M's 0.9640 / +28.3%.
#   arm R = arm Q + the Live2d-model-master expansion (Cubism 3+ only)
#           -> same 34 held-out characters (pinned), train corpus ~2x bigger.
#
# Both arms keep effective batch = 24 (M used 6 GPUs x 4).
#   Q: 3 GPUs x batch 8 ;  R: 4 GPUs x batch 6
# (GPU 0 is taken by another user - never touch it.)
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive

PY=".venv/bin/python"
TR=".venv/bin/torchrun"
[ -x "$TR" ] || { echo "torchrun missing at $TR"; exit 1; }
mkdir -p outputs/logs

# --- 0. verify the V9 patches ----------------------------------------------
grep -q "cache_tag"        src/live2d_vla/config.py   || { echo "V9 data patch missing"; exit 1; }
grep -q "_canonical"       src/live2d_vla/io_motion.py || { echo "V9 io patch missing"; exit 1; }
grep -q "val_holdout_path" src/live2d_vla/train.py    || { echo "V9 cli patch missing"; exit 1; }
echo "V9 patches verified"

# --- 1. corpora -------------------------------------------------------------
Q_DATA="standrad-live-2d"
Q_WL="outputs/model_whitelist.json"
Q_GM="outputs/gen_mask.json"
Q_HO="outputs/val_holdout_std.json"
Q_DD="outputs/dedup_skip_std.json"
Q_SUB=285
Q_TAG="v9std"

R_DATA="data/all"
R_WL="outputs/model_whitelist_all.json"
R_GM="outputs/gen_mask_all.json"
R_HO="outputs/val_holdout_all.json"
R_DD="outputs/dedup_skip_all.json"
R_SUB=580
R_TAG="v9all"

AMAP="outputs/action_semantic_map.json"
EPOCHS=70

for f in "$Q_WL" "$Q_GM" "$Q_HO" "$Q_DD" "$R_WL" "$R_GM" "$R_HO" "$R_DD" "$AMAP"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done
echo "corpora present"

# --- 2. launch --------------------------------------------------------------
COMMON="--gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
--span_w 1 --span_w_cap 8 --epochs $EPOCHS --action_map_path $AMAP"

CUDA_VISIBLE_DEVICES=1,2,3 nohup "$TR" --nproc_per_node=3 --master_port=29711 \
  src/live2d_vla/train.py $COMMON \
  --batch_size 8 --subset_models $Q_SUB \
  --data_root "$Q_DATA" --whitelist_path "$Q_WL" --gen_mask_path "$Q_GM" \
  --val_holdout_path "$Q_HO" --dedup_skip_path "$Q_DD" --cache_tag "$Q_TAG" \
  --out_dir outputs/train_runs/abl_Q_std_v9 \
  > outputs/logs/abl_Q.log 2>&1 < /dev/null &
echo "launched Q (pid $!) on GPU 1,2,3"

sleep 15

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup "$TR" --nproc_per_node=4 --master_port=29712 \
  src/live2d_vla/train.py $COMMON \
  --batch_size 6 --subset_models $R_SUB \
  --data_root "$R_DATA" --whitelist_path "$R_WL" --gen_mask_path "$R_GM" \
  --val_holdout_path "$R_HO" --dedup_skip_path "$R_DD" --cache_tag "$R_TAG" \
  --out_dir outputs/train_runs/abl_R_all_v9 \
  > outputs/logs/abl_R.log 2>&1 < /dev/null &
echo "launched R (pid $!) on GPU 4,5,6,7"

echo "logs: outputs/logs/abl_Q.log  outputs/logs/abl_R.log"
