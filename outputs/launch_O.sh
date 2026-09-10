#!/bin/bash
# Arm O: M config (range-fix + span_w=1 + select abs) + learnable residual gate.
#       Isolates the gate on TOP of the now-correct normalisation (M is the
#       no-gate control, still finishing on GPU 5/6/7).
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs
rm -rf outputs/train_runs/abl_O_gate
nohup bash -c "cd /root/work/nlp/xjzhao13/lijie_llama/VLive && CUDA_VISIBLE_DEVICES=1,2,3 .venv/bin/torchrun --nproc_per_node=3 --master_port=29673 src/live2d_vla/train.py --gen_mode regress --subset_models 285 --action_cond id --eval_every 1 --batch_size 8 --epochs 70 --span_w 1 --span_w_cap 8 --select_metric abs --residual_gate name --out_dir outputs/train_runs/abl_O_gate" > outputs/logs/abl_O.log 2>&1 &
echo "launched O (pid $!)"
