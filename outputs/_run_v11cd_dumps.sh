#!/bin/bash
# Independent per-ckpt dumps for V11c / V11d full-614 re-eval.
# Each ckpt is scored in its OWN process (mandatory) and dumped to an
# INDEPENDENT directory so no later eval can silently overwrite the baseline.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
DUMP=/tmp/v11cd_dump
mkdir -p "$DUMP" /tmp/v11cd_logs
V=./.venv/bin/python

job() {
  local tag="$1"; shift
  local gpu="$1"; shift
  echo ">>> START $tag on cuda:$gpu"
  $V outputs/eval_ckpt.py "$@" --device "cuda:$gpu" --dump "$DUMP" \
     > "/tmp/v11cd_logs/$tag.log" 2>&1
  echo "<<< END $tag rc=$?"
}

# wave 1 (6 free-ish GPUs; GPU0 and GPU3 are occupied by other users)
job v11c_best   1 abl_V11c_s20            --ckpt best   &
job v11d_best   2 abl_V11d_s50            --ckpt best   &
job v11b_best   4 abl_V11b_bank           --ckpt best   &
job v11a_best   5 abl_V11a_charloo        --ckpt best   &
job rigloo_best 6 abl_R_all_v9 --rig_loo  --ckpt best   &
job v11c_latest 7 abl_V11c_s20            --ckpt latest &
wait
echo "=== wave 1 done ==="

job v11d_latest 4 abl_V11d_s50            --ckpt latest &
wait
echo "=== wave 2 done ==="
ls -la "$DUMP"
