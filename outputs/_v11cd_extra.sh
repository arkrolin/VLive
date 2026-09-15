#!/bin/bash
# Complete the LATEST dump set so latest can be paired-tested too.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
V=./.venv/bin/python
L=/tmp/v11cd_latest
$V outputs/eval_ckpt.py abl_V11b_bank --ckpt latest --device cuda:1 --dump "$L" \
   > /tmp/v11cd_logs/lat_v11b.log 2>&1 &
P1=$!
$V outputs/eval_ckpt.py abl_V11a_charloo --ckpt latest --device cuda:2 --dump "$L" \
   > /tmp/v11cd_logs/lat_v11a.log 2>&1 &
P2=$!
wait $P1; echo "v11b_latest rc=$?"
wait $P2; echo "v11a_latest rc=$?"
ls -la "$L"
echo "=== LATEST PAIRED ==="
$V outputs/compare_dumps.py "$L" 2>&1
echo "=== shape: V11c vs R (no-bank) ==="
$V outputs/diag_shape.py outputs/render_data/abl_V11c_s20_best.pkl \
   --ref outputs/render_data/abl_R_all_v9_best.pkl 2>&1 | tail -20
echo "=== shape: V11d vs R (no-bank) ==="
$V outputs/diag_shape.py outputs/render_data/abl_V11d_s50_best.pkl \
   --ref outputs/render_data/abl_R_all_v9_best.pkl 2>&1 | tail -20
echo "=== shape: V11b vs R (no-bank, sanity) ==="
$V outputs/diag_shape.py outputs/render_data/abl_V11b_bank_best.pkl \
   --ref outputs/render_data/abl_R_all_v9_best.pkl 2>&1 | tail -20
echo "EXTRA_DONE"
