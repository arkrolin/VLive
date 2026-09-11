#!/bin/bash
# Re-run the V11a render export now that gen_render_data.py forwards cstats/bank.
# The previous pickle was produced without them and therefore understated V11a.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs outputs/render_data

nohup ./.venv/bin/python outputs/gen_render_data.py \
    --run abl_V11a_charloo \
    --ckpt best \
    --device cuda:7 \
    --out outputs/render_data \
    > outputs/logs/gen_render_V11a_fix.log 2>&1 &

echo "launched pid $!"
