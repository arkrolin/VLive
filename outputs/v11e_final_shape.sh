#!/bin/bash
# Phase 2 of the V11e final read: shape diagnosis for the four arms.
#
# Why this is the primary criterion and abs_mae is not: abs_mae is
# amplitude-dominated, and section 15.11 showed a lower delta-MAE can be bought
# by simply shrinking the predicted deltas (V11c: retention 0.762). So we also
# read |dpred|/|dtgt| from diag_shape_floor.
#
# Naming trap: gen_render_data writes "<run>_<ckpt>.pkl", with no epoch tag, so
# re-rendering an existing run silently overwrites the earlier artifact that a
# report section already quotes. Archive first.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs
L=outputs/logs/v11e_final_shape.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
PY=./.venv/bin/python
R=outputs/render_data

# the current abl_V11e_B2_best.pkl is the ep49 render quoted in section 15.11
if [ -f "$R/abl_V11e_B2_best.pkl" ] && [ ! -f "$R/abl_V11e_B2_ep49_best.pkl" ]; then
  cp "$R/abl_V11e_B2_best.pkl" "$R/abl_V11e_B2_ep49_best.pkl"
  say "archived ep49 B2 render -> abl_V11e_B2_ep49_best.pkl"
fi

say "=== phase 2 start: 4 renders (one per arm, one card each) ==="
i=0
for arm in B2 B1b C1a D1; do
  i=$((i + 1))
  gpu=$(echo "1 2 4 5" | cut -d' ' -f"$i")
  (
    CUDA_VISIBLE_DEVICES="$gpu" $PY outputs/gen_render_data.py \
      --run "abl_V11e_$arm" --ckpt best --device cuda:0 --out "$R" \
      > "outputs/logs/render_abl_V11e_$arm.log" 2>&1
    echo "render $arm rc=$?" >> "$L"
  ) &
done
wait
say "renders done"

say "--- diag_shape per arm (--ref abl_V11b_bank; exem column is self-scored, so
             each arm's absolute numbers need no common ref) ---"
for arm in B2 B1b C1a D1; do
  echo "" >> "$L"
  echo "### $arm" >> "$L"
  $PY outputs/diag_shape.py "$R/abl_V11e_${arm}_best.pkl" \
      --ref "$R/abl_V11b_bank_best.pkl" >> "$L" 2>&1
done

say "--- diag_shape_floor (positional only, no --ref) ---"
$PY outputs/diag_shape_floor.py \
  "$R/abl_V11e_B2_best.pkl" "$R/abl_V11e_B1b_best.pkl" \
  "$R/abl_V11e_C1a_best.pkl" "$R/abl_V11e_D1_best.pkl" \
  "$R/V11c_best.pkl" "$R/abl_V11b_bank_best.pkl" "$R/abl_V11a_charloo_best.pkl" \
  >> "$L" 2>&1

say "=== phase 2 done ==="
