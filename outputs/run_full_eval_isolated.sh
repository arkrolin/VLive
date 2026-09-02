#!/bin/bash
# Score each checkpoint in its OWN process on the FULL shared val set.
#
# Why one process per checkpoint: eval_ckpt.py caches the train corpus + range
# stats across runs. Keying that cache on subset_models alone let arm D (which
# predates the ParamNameEncoder fix and is loaded with strict=False) contaminate
# later arms in the same process - the same H checkpoint scored 1.3543 alone but
# 1.2587 when D ran first (~6% shift), while two runs of H inside one process
# agreed to 0.0000. Separate processes reproduce to ~0.4%.
#
# Per-param-instance errors are dumped so compare_dumps.py can still run a
# PAIRED test across runs that never shared a process.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/eval_dumps

for r in abl_D_regress_285 abl_E_ddpm_285 abl_F_regress285_fixname \
         abl_G_regress285_actname abl_H_regress285_ep45; do
  echo "################ $r ################"
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python -u outputs/eval_ckpt.py "$r" \
      --n_samples 0 --device cuda:0 --dump outputs/eval_dumps 2>&1 \
      | grep -Ev 'Warning|warnings.warn'
done
echo "################ DONE ################"
