#!/bin/bash
# Chain the V11e b/c ablation arms on ONE 4-GPU set, serially.
#
# Why serial: the span-weight reference inside noise_loss is a PER-RANK batch
# mean (`ref = span_t[token_mask>0.5].mean()`), so BS must stay 6 and the
# global batch must stay 4x6=24 to remain comparable with V11b/c/d. That means
# one arm occupies all 4 GPUs, and arms must queue.
#
# Arms (in run order) and what each one isolates, always against V11c
# (dense head, shape_w=20, no mask) which is already on disk:
#   B2   structured/rank8 + additive amp, lambda 20
#        -> isolates the ADDITIVE amplitude term (vs B1b). Target: delta MAE
#           on motion channels, because the multiplicative gain exem*(1+a)
#           cannot move the 28.4% of motion channels where exem under-scales.
#   B1b  structured/rank8, lambda 20
#        -> isolates the STRUCTURED HEAD itself (vs V11c dense).
#   C1a  dense + motion mask(1e-3), lambda 10
#        -> isolates the CHANNEL RE-WEIGHTING (vs V11c). lambda is divided by
#           ~1.76 because masking concentrates the mean on 56.8% of channels,
#           so mask+lambda20 would over-strengthen the shape term (§15.3).
#
# Idempotent: an arm whose log already shows the final epoch is skipped, so the
# chain can be re-run after an interruption.
#
# Usage:
#   GPUS=1,2,4,5 RANKS=4 BS=6 nohup bash outputs/chain_V11e.sh &
set -u

GPUS=${GPUS:-1,2,4,5}
RANKS=${RANKS:-4}
BS=${BS:-6}
EPOCHS=${EPOCHS:-70}
ARMS=${ARMS:-B2 B1b C1a}

cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs
CHAIN_LOG=outputs/logs/_v11e_chain.log

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }

say "chain start  arms='$ARMS' gpus=$GPUS ranks=$RANKS bs=$BS epochs=$EPOCHS"

for ARM in $ARMS; do
  RUN=abl_V11e_${ARM}
  LOG=outputs/logs/${RUN}.log

  # --- idempotency: has this arm already finished the full epoch budget? ---
  if [ -f "$LOG" ] && grep -q "ep ${EPOCHS}/${EPOCHS}\]" "$LOG"; then
    say "SKIP $ARM (already reached ${EPOCHS}/${EPOCHS})"
    continue
  fi

  # --- port must differ per arm so a stale rendezvous cannot collide ---
  case "$ARM" in
    B1a) PORT=29801 ;; B1b) PORT=29802 ;; B1c) PORT=29803 ;;
    B2)  PORT=29804 ;; C1a) PORT=29805 ;; C1b) PORT=29806 ;;
    D1)  PORT=29807 ;; *)   PORT=$((29900 + RANDOM % 90)) ;;
  esac

  # --- wait for the GPUs to actually be free before committing an arm -----
  # (another user may have grabbed a card; we must not OOM 10h in)
  waited=0
  while :; do
    busy=""
    IFS=',' read -ra _want <<< "$GPUS"
    for g in "${_want[@]}"; do
      free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader \
                | tr -d ' ' | awk -F, -v want="$g" '$1==want{print $2}' | sed 's/MiB//')
      [ -n "$free_mb" ] || free_mb=0
      [ "$free_mb" -ge 40000 ] || busy="$busy gpu$g(${free_mb}MiB)"
    done
    [ -z "$busy" ] && break
    if [ "$waited" -ge 21600 ]; then
      say "GIVE UP $ARM: after 6h still short on$busy"
      break
    fi
    say "wait $ARM: not enough free memory on$busy"
    sleep 300; waited=$((waited + 300))
  done
  [ -n "${busy:-}" ] && continue

  say "START $ARM -> $RUN (port=$PORT)"
  ARM="$ARM" RUN="$RUN" GPUS="$GPUS" RANKS="$RANKS" BS="$BS" \
    EPOCHS="$EPOCHS" PORT="$PORT" bash outputs/launch_V11e.sh \
    >> "$CHAIN_LOG" 2>&1
  rc=$?
  say "launched $ARM (launcher rc=$rc)"

  # --- wait for this arm's trainer to exit before starting the next --------
  sleep 60
  for _ in $(seq 1 2880); do   # up to 24h
    if ! pgrep -f "train_runs/abl_V11e_${ARM}" > /dev/null 2>&1; then
      break
    fi
    sleep 30
  done

  if [ -f "$LOG" ] && grep -q "ep ${EPOCHS}/${EPOCHS}\]" "$LOG"; then
    tail -3 "$LOG" | sed 's/^/    /' | tee -a "$CHAIN_LOG"
    say "DONE $ARM"
  else
    say "ARM $ARM did NOT reach ${EPOCHS}/${EPOCHS} -- see $LOG"
  fi
done

say "chain finished"
