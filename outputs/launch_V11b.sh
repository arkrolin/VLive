#!/bin/bash
# V11b: per-param cross-attention over the character's OWN motion bank.
#
# Everything from V11a (--rig_loo --char_stats mlp) plus:
#   --bank_cond attn --bank_k 8
#     Each sample carries K=8 of the SAME character's other motion curves
#     (target action strictly excluded, deterministic crc32 seeding so every
#     DDP rank sees the same bank). The param token queries them with per-param
#     cross-attention; the attended curve is fed to in_proj as a 3rd channel
#     whose weights are ZERO-INIT, so the run starts bit-identical to V11a.
#
# Why selection and not pooling - the decisive measurement (diag_ref_prior2.py):
#   cross-character same-action, per-param ORACLE over 90 characters : 1.7673
#   cross-character same-action, plain MEAN                          : 1.6242
#   THIS character's other motions, per-param ORACLE over ~22        : 0.5031
#   THIS character's other motions, plain MEAN                       : 1.8510
# Across characters averaging wins (candidates are noise); within a character
# selection wins by 3.7x (candidates are different actions). So the bank must
# be attended over, never pooled.
#
# Reference numbers to beat: R 0.9202 (leaky rig), V11a (see its log).
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
TR=".venv/bin/torchrun"
[ -x "$TR" ] || { echo "torchrun missing"; exit 1; }
for c in outputs/ident_cache_580_v3_v9all.pkl outputs/charstats_loo_580_v9all.pkl \
         outputs/motionbank_580_v9all.pkl; do
  [ -f "$c" ] || { echo "missing cache $c - run outputs/prebuild_v11a_caches.py"; exit 1; }
done
echo "caches verified"

EPOCHS=70
COMMON="--gen_mode regress --action_cond id --select_metric abs --eval_every 1 \
--span_w 1 --span_w_cap 8 --epochs $EPOCHS \
--action_map_path outputs/action_semantic_map.json \
--rig_loo --char_stats mlp --bank_cond attn --bank_k 8"

# Runs CONCURRENTLY with V11a (which owns GPU 5,6,7). GPU 1,2,3 are empty and
# 4 is effectively free (1154 MiB residual from a finished job), so V11b gets
# 4 ranks x batch 6 = 24 - the same effective batch as V11a's 3 x 8, and one
# extra rank to offset the extra cost of the bank branch.
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup "$TR" --nproc_per_node=4 --master_port=29733 \
  src/live2d_vla/train.py $COMMON \
  --batch_size 6 --subset_models 580 \
  --data_root data/all \
  --whitelist_path outputs/model_whitelist_all.json \
  --gen_mask_path outputs/gen_mask_all.json \
  --val_holdout_path outputs/val_holdout_all.json \
  --dedup_skip_path outputs/dedup_skip_all.json \
  --cache_tag v9all \
  --out_dir outputs/train_runs/abl_V11b_bank \
  > outputs/logs/abl_V11b_bank.log 2>&1 < /dev/null &
echo "launched V11b (pid $!) on GPU 1,2,3,4"
