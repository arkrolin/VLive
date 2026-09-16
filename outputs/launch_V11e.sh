#!/bin/bash
# V11e: the b/c arms designed in report §15.
#
# Every arm is V11b/C's config with ONE intended change, so effects are
# attributable. Baseline (all knobs default) reproduces the V11c-style run.
#
# Arms (ARM=...):
#   B1b  structured head, rank 8, shape_w=20            -> is amplitude/shape
#                                                          decoupling itself useful?
#   B1a  structured head, rank 2, shape_w=20            -> capacity LOWER bound
#                                                          (S-arm failed at rank 2)
#   B1c  structured head, rank 16, shape_w=20           -> capacity UPPER bound
#   B2   B1b + additive amplitude term                  -> fixes the multiplicative
#                                                          gain's blind spot
#   C1a  motion mask, shape_w=10                        -> re-weight shape gradient
#                                                          to moving channels,
#                                                          lambda rescaled (~x0.568)
#   C1b  motion mask, shape_w=20                        -> control: mask WITHOUT the
#                                                          lambda rescale (expected
#                                                          over-strong, per §15.3)
#   D1   B2 + C1a                                       -> combine both directions
#
# Usage:
#   ARM=B1b GPUS=1,2,4,5 RANKS=4 BS=6 PORT=29801 bash outputs/launch_V11e.sh
set -u

ARM=${ARM:-B1b}
GPUS=${GPUS:-1,2,4,5}
RANKS=${RANKS:-4}
BS=${BS:-6}
EPOCHS=${EPOCHS:-70}

# ---- per-arm switches (the ONLY things that differ between arms) ---------- #
case "$ARM" in
  B1a) HEAD_MODE="structured"; RANK=2;  ADD_AMP="";  SHAPE_W=20; MASK=0.0;   PORT=${PORT:-29801} ;;
  B1b) HEAD_MODE="structured"; RANK=8;  ADD_AMP="";  SHAPE_W=20; MASK=0.0;   PORT=${PORT:-29802} ;;
  B1c) HEAD_MODE="structured"; RANK=16; ADD_AMP="";  SHAPE_W=20; MASK=0.0;   PORT=${PORT:-29803} ;;
  B2)  HEAD_MODE="structured"; RANK=8;  ADD_AMP="--head_additive_amp"; SHAPE_W=20; MASK=0.0; PORT=${PORT:-29804} ;;
  C1a) HEAD_MODE="dense";      RANK=2;  ADD_AMP="";  SHAPE_W=10; MASK=1e-3;  PORT=${PORT:-29805} ;;
  C1b) HEAD_MODE="dense";      RANK=2;  ADD_AMP="";  SHAPE_W=20; MASK=1e-3;  PORT=${PORT:-29806} ;;
  D1)  HEAD_MODE="structured"; RANK=8;  ADD_AMP="--head_additive_amp"; SHAPE_W=10; MASK=1e-3; PORT=${PORT:-29807} ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

RUN=${RUN:-abl_V11e_${ARM}}
# minimum free MiB required per selected GPU before we agree to launch
MIN_FREE_MB=${MIN_FREE_MB:-20000}

cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs

# ---------------------------------------------------------------------------
# Idle-GPU precheck. The CSV below reports PHYSICAL indices, which is exactly
# what CUDA_VISIBLE_DEVICES consumes, so the check is direct (no remapping).
# A rank at bs=6 peaks around 13 GiB; if a GPU cannot spare MIN_FREE_MB the
# run would OOM mid-flight and waste a 70-epoch slot, so we refuse up front.
# ---------------------------------------------------------------------------
if [ "${SKIP_GPUCHECK:-0}" != "1" ]; then
  busy=""
  IFS=',' read -ra _want <<< "$GPUS"
  for g in "${_want[@]}"; do
    line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader \
            | tr -d ' ' | awk -F, -v want="$g" '$1==want{print $2}')
    if [ -z "$line" ]; then
      busy="$busy gpu$g(missing)"
    else
      # "73000MiB" -> 73000
      free_mb=${line%MiB}
      if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
        busy="$busy gpu$g(free=${free_mb}MiB)"
      fi
    fi
  done
  if [ -n "$busy" ]; then
    echo "REFUSING to launch: not enough free memory on$busy (need ${MIN_FREE_MB}MiB each)"
    echo "pick other GPUs (GPUS=...) or wait. Override with SKIP_GPUCHECK=1."
    exit 3
  fi
  echo "gpu precheck ok (need >=${MIN_FREE_MB}MiB on each of $GPUS)"
fi

TR=".venv/bin/torchrun"
[ -x "$TR" ] || { echo "torchrun missing"; exit 1; }
for c in outputs/ident_cache_580_v3_v9all.pkl outputs/charstats_loo_580_v9all.pkl \
         outputs/motionbank_580_v9all.pkl; do
  [ -f "$c" ] || { echo "missing cache $c"; exit 1; }
done

echo "launching $RUN: arm=$ARM head_mode=$HEAD_MODE rank=$RANK add_amp='$ADD_AMP' shape_w=$SHAPE_W mask=$MASK"
echo "  ranks=$RANKS bs=$BS (effective $((RANKS * BS))) GPUs=$GPUS port=$PORT epochs=$EPOCHS"

COMMON="--gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
--span_w 1 --span_w_cap 8 --epochs $EPOCHS \
--shape_w $SHAPE_W --shape_motion_mask $MASK \
--head_mode $HEAD_MODE --residual_rank $RANK $ADD_AMP \
--action_map_path outputs/action_semantic_map.json \
--rig_loo --char_stats mlp --bank_cond attn --bank_k 8"

# expandable_segments avoids the fragmentation blow-up the trainer hit before
# (it reported 30 MiB reserved-but-unallocated and then refused a 144 MiB ask).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES=$GPUS nohup "$TR" --nproc_per_node=$RANKS --master_port=$PORT \
  src/live2d_vla/train.py $COMMON \
  --batch_size $BS --subset_models 580 \
  --data_root data/all \
  --whitelist_path outputs/model_whitelist_all.json \
  --gen_mask_path outputs/gen_mask_all.json \
  --val_holdout_path outputs/val_holdout_all.json \
  --dedup_skip_path outputs/dedup_skip_all.json \
  --cache_tag v9all \
  --out_dir outputs/train_runs/$RUN \
  > outputs/logs/$RUN.log 2>&1 < /dev/null &
echo "launched $RUN (pid $!)"
