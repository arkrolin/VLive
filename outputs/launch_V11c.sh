#!/bin/bash
# V11c = V11b + explicit shape (first-difference) loss term.
#
# Rationale: the level loss is dominated by the ~43% of channels that are
# constant over time, so shape was never actually optimised. R -> V11a -> V11b
# took abs_mae from +39% to +45% while delta MAE never beat the exem prior.
# Everything except shape_w is identical to V11b, so the run is a clean
# single-variable change and the effect is attributable.
#
# Env knobs let one script launch a lambda sweep:
#   SHAPE_W=1.0 RUN=abl_V11c_s1 GPUS=1,2,3,4 RANKS=4 BS=6 PORT=29741 \
#     bash outputs/launch_V11c.sh
set -u

SHAPE_W=${SHAPE_W:-1.0}
RUN=${RUN:-abl_V11c_shape}
GPUS=${GPUS:-1,2,3,4}
RANKS=${RANKS:-4}
BS=${BS:-6}
PORT=${PORT:-29741}
EPOCHS=${EPOCHS:-70}

cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs

TR=".venv/bin/torchrun"
[ -x "$TR" ] || { echo "torchrun missing"; exit 1; }
for c in outputs/ident_cache_580_v3_v9all.pkl outputs/charstats_loo_580_v9all.pkl \
         outputs/motionbank_580_v9all.pkl; do
  [ -f "$c" ] || { echo "missing cache $c"; exit 1; }
done

echo "launching $RUN: shape_w=$SHAPE_W ranks=$RANKS bs=$BS (effective $((RANKS * BS))) on GPU $GPUS"

COMMON="--gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
--span_w 1 --span_w_cap 8 --shape_w $SHAPE_W --epochs $EPOCHS \
--action_map_path outputs/action_semantic_map.json \
--rig_loo --char_stats mlp --bank_cond attn --bank_k 8"

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
