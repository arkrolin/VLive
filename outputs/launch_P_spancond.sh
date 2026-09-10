#!/bin/bash
# Arm P: M config (range-fix + span_w=1 + select abs, no gate) PLUS
#       span conditioning (span_cond=log). Injects log(span) as a per-token
#       feature so the model observes each param's absolute range directly,
#       targeting the residual-overshoot-vs-span defect (§20/§21).
#       span_enc is zero-init, so ep1 is bit-identical to M -> clean A/B.
#       Launched on the GPUs M vacates (5,6,7); waits for M to finish first.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs
echo "=== launch_P waiting for M to finish $(date) ===" >> outputs/logs/abl_P.log
while pgrep -f 'train.py.*abl_M_rangefix' > /dev/null; do sleep 60; done
echo "=== launch_P M done, launching $(date) ===" >> outputs/logs/abl_P.log
rm -rf outputs/train_runs/abl_P_spancond
nohup bash -c "cd /root/work/nlp/xjzhao13/lijie_llama/VLive && \
CUDA_VISIBLE_DEVICES=5,6,7 .venv/bin/torchrun --nproc_per_node=3 --master_port=29674 \
src/live2d_vla/train.py --gen_mode regress --subset_models 285 --action_cond id \
--eval_every 1 --batch_size 8 --epochs 70 --span_w 1 --span_w_cap 8 \
--select_metric abs --residual_gate none --span_cond log \
--out_dir outputs/train_runs/abl_P_spancond" \
>> outputs/logs/abl_P.log 2>&1 &
echo "launched P (pid $!)"
