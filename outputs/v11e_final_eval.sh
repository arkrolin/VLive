#!/bin/bash
# Phase 1 of the V11e final read: full-614 evaluation of all four arms.
#
# Discipline this script encodes:
#   * every ckpt gets its OWN process (reusing one process drifts up to 7%);
#   * every ckpt gets its OWN dump directory -- eval_ckpt names the npz after the
#     RUN only, so best/latest in one directory silently overwrite each other;
#   * baselines are re-evaluated in the SAME session as the arms, because
#     compare_dumps pairs per-instance errors and a stale npz drifts silently.
#
# 6 cards are idle (1,2,4,5,6,7), so we run 6 concurrently -- the documented
# concurrency ceiling for this box.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs
L=outputs/logs/v11e_final_eval.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
D=/tmp/v11e_final
mkdir -p "$D"
PY=./.venv/bin/python

run_one() {   # tag run ckpt gpu
  local tag="$1" run="$2" ckpt="$3" gpu="$4"
  mkdir -p "$D/$tag"
  CUDA_VISIBLE_DEVICES="$gpu" $PY outputs/eval_ckpt.py "$run" --ckpt "$ckpt" \
    --device cuda:0 --dump "$D/$tag" > "outputs/logs/eval_${tag}.log" 2>&1
  echo "$tag rc=$?" >> "$L"
}

say "=== phase 1 start: 11 full-614 evals ==="

# batch 1
for spec in \
  "B2_best:abl_V11e_B2:best:1" \
  "B1b_best:abl_V11e_B1b:best:2" \
  "C1a_best:abl_V11e_C1a:best:4" \
  "D1_best:abl_V11e_D1:best:5" \
  "V11c_best:abl_V11c_s20:best:6" \
  "V11b_best:abl_V11b_bank:best:7" ; do
  IFS=: read -r tag run ckpt gpu <<< "$spec"
  run_one "$tag" "$run" "$ckpt" "$gpu" &
done
wait
say "batch 1 done"

# batch 2 (latest of every arm, to answer whether select_metric picked well)
for spec in \
  "B2_latest:abl_V11e_B2:latest:1" \
  "B1b_latest:abl_V11e_B1b:latest:2" \
  "C1a_latest:abl_V11e_C1a:latest:4" \
  "D1_latest:abl_V11e_D1:latest:5" \
  "V11c_latest:abl_V11c_s20:latest:6" ; do
  IFS=: read -r tag run ckpt gpu <<< "$spec"
  run_one "$tag" "$run" "$ckpt" "$gpu" &
done
wait
say "batch 2 done"

say "--- summary (authoritative: eval_ckpt printed abs_mae) ---"
for tag in B2_best B1b_best C1a_best D1_best V11c_best V11b_best \
           B2_latest B1b_latest C1a_latest D1_latest V11c_latest; do
  line=$(grep -E "^abl_V11e|^abl_V11c|^abl_V11b" "outputs/logs/eval_${tag}.log" | tail -1)
  say "$tag | $line"
done
say "=== phase 1 done ==="
