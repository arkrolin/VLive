#!/bin/bash
# Fix the dump-name collision: eval_ckpt --dump names files by RUN only, so
# ckpt_best and ckpt_latest of the same run race for the same .npz. Give each
# ckpt variant its OWN directory.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
V=./.venv/bin/python
B=/tmp/v11cd_best
L=/tmp/v11cd_latest
rm -rf "$B" "$L"; mkdir -p "$B" "$L" /tmp/v11cd_logs

# wave-1 dumps that had exactly ONE writer are trustworthy -> reuse for BEST
cp /tmp/v11cd_best_snapshot/abl_V11a_charloo.npz \
   "/tmp/v11cd_best_snapshot/abl_R_all_v9+rigloo.npz" \
   /tmp/v11cd_best_snapshot/abl_V11b_bank.npz \
   /tmp/v11cd_best_snapshot/abl_V11d_s50.npz "$B"/ || exit 1

# abl_V11c_s20.npz was raced by best+latest -> re-score best into its own dir
$V outputs/eval_ckpt.py abl_V11c_s20 --ckpt best \
   --device cuda:1 --dump "$B" > /tmp/v11cd_logs/fix_v11c_best.log 2>&1
echo "v11c_best rc=$?"

# latest arm dumps -> separate directory
$V outputs/eval_ckpt.py abl_V11c_s20 --ckpt latest \
   --device cuda:2 --dump "$L" > /tmp/v11cd_logs/lat_v11c.log 2>&1 &
P1=$!
$V outputs/eval_ckpt.py abl_V11d_s50 --ckpt latest \
   --device cuda:5 --dump "$L" > /tmp/v11cd_logs/lat_v11d.log 2>&1 &
P2=$!
wait $P1; echo "v11c_latest rc=$?"
wait $P2; echo "v11d_latest_fix rc=$?"
echo "=== BEST ==="; ls -la "$B"
echo "=== LATEST ==="; ls -la "$L"
