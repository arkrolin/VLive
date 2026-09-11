#!/bin/bash
# Re-export R with the same (fixed) gen_render_data.py so the V11a comparison
# uses identical code on both sides.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs outputs/render_data

nohup ./.venv/bin/python outputs/gen_render_data.py \
    --run abl_R_all_v9 \
    --ckpt best \
    --device cuda:6 \
    --out outputs/render_data \
    > outputs/logs/gen_render_R_fix.log 2>&1 &

echo "launched pid $!"
