#!/bin/bash
# Shape diagnostics for the two shape-loss arms (the PRIMARY criterion this round).
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
V=./.venv/bin/python
mkdir -p outputs/render_data /tmp/v11cd_logs
$V outputs/gen_render_data.py --run abl_V11c_s20 --ckpt best --device cuda:1 \
   --out outputs/render_data > /tmp/v11cd_logs/gen_v11c.log 2>&1 &
P1=$!
$V outputs/gen_render_data.py --run abl_V11d_s50 --ckpt best --device cuda:2 \
   --out outputs/render_data > /tmp/v11cd_logs/gen_v11d.log 2>&1 &
P2=$!
wait $P1; echo "gen v11c rc=$?"
wait $P2; echo "gen v11d rc=$?"
ls -la outputs/render_data/
echo "=== V11c vs V11b ==="
$V outputs/diag_shape.py outputs/render_data/abl_V11c_s20_best.pkl \
   --ref outputs/render_data/abl_V11b_bank_best.pkl 2>&1 | tee /tmp/v11cd_logs/diag_v11c.log
echo "=== V11d vs V11b ==="
$V outputs/diag_shape.py outputs/render_data/abl_V11d_s50_best.pkl \
   --ref outputs/render_data/abl_V11b_bank_best.pkl 2>&1 | tee /tmp/v11cd_logs/diag_v11d.log
