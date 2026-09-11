#!/bin/bash
# Generate render payload (pred / target / exem curves) for the V11a best checkpoint.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs outputs/render_data

nohup ./.venv/bin/python outputs/gen_render_data.py \
    --run abl_V11a_charloo \
    --ckpt best \
    --device cuda:5 \
    --out outputs/render_data \
    > outputs/logs/gen_render_V11a.log 2>&1 &

echo "launched pid $!"
