#!/bin/bash
# One-shot snapshot of the STOPPED B2 run (killed by hand at ep51/70, 09-17).
#
# Why: the training val metric cannot predict full-614 (see report §15.10), so
# the only trustworthy numbers for this run are the eval_ckpt ones. B2 was
# stopped short of 70ep, so we must lock in what we do have --
# ckpt_best(ep49) and ckpt_latest(ep50) -- otherwise the ~22h already spent
# are not usable evidence.
#
# GPU policy: everything runs on GPU 6 only. GPU 1/2/4/5 were deliberately
# freed for an urgent task and must stay untouched.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs
D_BEST=/tmp/v11eB2_ep49_best
D_LATEST=/tmp/v11eB2_ep50_latest
mkdir -p "$D_BEST" "$D_LATEST"
L=outputs/logs/post_B2_ep50.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
PY=./.venv/bin/python
GPU=6

say "=== start: B2 killed at ep51; ckpt_best=ep49 ckpt_latest=ep50; all on GPU $GPU ==="

# keep the ep38 render that report 15.10 quotes, before we regenerate into the
# same filename (gen_render_data names its output <run>_best.pkl, no epoch tag)
if [ -f outputs/render_data/abl_V11e_B2_best.pkl ] &&
   [ ! -f outputs/render_data/abl_V11e_B2_ep38_best.pkl ]; then
  cp outputs/render_data/abl_V11e_B2_best.pkl outputs/render_data/abl_V11e_B2_ep38_best.pkl
  say "archived previous render -> abl_V11e_B2_ep38_best.pkl"
fi

# each ckpt gets its OWN process and its OWN dump dir (npz name carries only
# the run name, so best/latest in one dir would silently overwrite)
CUDA_VISIBLE_DEVICES=$GPU $PY outputs/eval_ckpt.py abl_V11e_B2 --ckpt best \
  --device cuda:0 --dump "$D_BEST" >> "$L" 2>&1
say "eval B2 best (ep49) rc=$?"

CUDA_VISIBLE_DEVICES=$GPU $PY outputs/eval_ckpt.py abl_V11e_B2 --ckpt latest \
  --device cuda:0 --dump "$D_LATEST" >> "$L" 2>&1
say "eval B2 latest (ep50) rc=$?"

say "--- shape diagnosis (the real criterion; abs_mae is only amplitude) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY outputs/gen_render_data.py --run abl_V11e_B2 \
  --ckpt best --device cuda:0 --out outputs/render_data >> "$L" 2>&1
say "render rc=$?"

$PY outputs/diag_shape.py outputs/render_data/abl_V11e_B2_best.pkl \
  --ref outputs/render_data/abl_V11b_bank_best.pkl >> "$L" 2>&1
say "diag_shape rc=$?"

$PY outputs/diag_shape_floor.py outputs/render_data/abl_V11e_B2_best.pkl \
  --ref outputs/render_data/abl_V11b_bank_best.pkl >> "$L" 2>&1
say "diag_shape_floor rc=$?"

say "=== done ==="
