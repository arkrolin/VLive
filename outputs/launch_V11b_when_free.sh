#!/bin/bash
# Relaunch V11b as soon as enough GPUs are actually free.
#
# Why this exists: V11b was launched on GPU 1,2,3,4 at 19:35 while they were
# idle, but another 4-GPU job (~24 GB per card) took them over around 19:45.
# Our rank 3 was starved and NCCL killed the job:
#   Watchdog caught collective operation timeout: WorkNCCL(ALLREDUCE)
#   ran for 600006 ms before timing out  -> SIGABRT (exitcode -6)
# Nothing was wrong with the code; it was GPU contention on a shared box.
#
# This watcher polls every 60 s and launches on the first 3-4 cards that are
# genuinely idle (< 2 GB used), keeping the EFFECTIVE batch at 24 by scaling the
# per-rank batch (3 ranks x 8, or 4 ranks x 6) so the run stays comparable to
# V11a (3 x 8) and R (4 x 6).
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
LOCK=/tmp/v11b_watcher.lock
LOG=outputs/logs/v11b_watcher.log

if [ -e "$LOCK" ]; then
  echo "watcher already running ($LOCK)"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

mkdir -p outputs/logs
echo "[$(date '+%F %T')] watcher started (pid $$)" >> "$LOG"

while true; do
  # idle = < 2 GB used, and never GPU 0 (permanently shared with other work)
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
         | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0 < 2000 && $1+0 != 0) print $1}')
  N=$(printf '%s\n' "$FREE" | grep -c '[0-9]' || true)
  if [ "$N" -ge 3 ]; then
    DEV=$(printf '%s\n' "$FREE" | head -4 | paste -sd, -)
    NR=$(printf '%s' "$DEV" | awk -F, '{print NF}')
    BS=$(( 24 / NR ))
    TS=$(date '+%F %T')
    echo "[$TS] launching V11b on GPU $DEV ($NR ranks x batch $BS = 24)" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$DEV nohup .venv/bin/torchrun \
      --nproc_per_node="$NR" --master_port=29735 \
      src/live2d_vla/train.py \
      --gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
      --span_w 1 --span_w_cap 8 --epochs 70 \
      --action_map_path outputs/action_semantic_map.json \
      --rig_loo --char_stats mlp --bank_cond attn --bank_k 8 \
      --batch_size "$BS" --subset_models 580 \
      --data_root data/all \
      --whitelist_path outputs/model_whitelist_all.json \
      --gen_mask_path outputs/gen_mask_all.json \
      --val_holdout_path outputs/val_holdout_all.json \
      --dedup_skip_path outputs/dedup_skip_all.json \
      --cache_tag v9all \
      --out_dir outputs/train_runs/abl_V11b_bank \
      > outputs/logs/abl_V11b_bank.log 2>&1 < /dev/null &
    echo "[$(date '+%F %T')] launched pid $!" >> "$LOG"
    exit 0
  fi
  sleep 60
done
