#!/bin/bash
# V8.2 arms M / N.
#   N = K config (span_w=1,cap=8) + 70 epochs + checkpoint selection on abs_mae.
#       vs K (45ep, val_loss selection) this isolates "train longer + select on
#       the metric the gate is written against".
#   M = N + learnable per-param-name residual gate (isolates the gate).
# Both inherit the best known config: regress @ subset=285, action_cond=id.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs

COMMON="--gen_mode regress --subset_models 285 --action_cond id \
--eval_every 1 --batch_size 8 --epochs 70 --span_w 1 --span_w_cap 8 \
--select_metric abs"

nohup bash -c "cd /root/work/nlp/xjzhao13/lijie_llama/VLive && \
CUDA_VISIBLE_DEVICES=1,2,3 .venv/bin/torchrun --nproc_per_node=3 --master_port=29670 \
src/live2d_vla/train.py $COMMON --residual_gate none \
--out_dir outputs/train_runs/abl_N_span1_ep70" \
> outputs/logs/abl_N.log 2>&1 &
echo "launched N (pid $!)"

nohup bash -c "cd /root/work/nlp/xjzhao13/lijie_llama/VLive && \
CUDA_VISIBLE_DEVICES=5,6,7 .venv/bin/torchrun --nproc_per_node=3 --master_port=29671 \
src/live2d_vla/train.py $COMMON --residual_gate name \
--out_dir outputs/train_runs/abl_M_span1_gate" \
> outputs/logs/abl_M.log 2>&1 &
echo "launched M (pid $!)"
