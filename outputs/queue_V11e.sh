#!/bin/bash
# Queue for the remaining V11e arms, with a threshold that matches reality.
#
# Replaces chain_V11e.sh + supervise_V11e.sh, which together had two problems:
#   1. the chain demands >= 40000 MiB free per GPU before starting an arm, but
#      the shared box leaves only ~39.6 GB on GPU 1/2/4/5, so it waits 6h and
#      then gives up -- for EACH remaining arm (up to 12h wasted);
#   2. the chain's 6h give-up and the supervisor's handover are sequential, so
#      the waste stacks instead of being avoided.
# A single rank only needs ~22-30 GB, so 30000 MiB is the honest threshold.
#
# The wait below includes B2 explicitly: launching B1b on the same four GPUs
# while B2 is still training would put 8 ranks on 4 cards and roughly halve
# both arms' throughput, which is worse than waiting.
#
# Usage:
#   nohup bash outputs/queue_V11e.sh > /dev/null 2>&1 &
set -u

GPUS=${GPUS:-1,2,4,5}
RANKS=${RANKS:-4}
BS=${BS:-6}
EPOCHS=${EPOCHS:-70}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
# B2 first on purpose: we must not start a second arm while it trains.
WAIT_FOR=${WAIT_FOR:-B2}
ARMS=${ARMS:-B1b C1a}

cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs
QLOG=outputs/logs/_v11e_queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$QLOG"; }

say "queue start wait_for='$WAIT_FOR' arms='$ARMS' min_free=${MIN_FREE_MB}MiB"

for ARM in $WAIT_FOR; do
  # a trainer for WAIT_FOR must be gone, and its run must be finished
  for _ in $(seq 1 11520); do   # up to 96h
    pgrep -f "train_runs/abl_V11e_${ARM}" > /dev/null 2>&1 || break
    sleep 30
  done
  say "$ARM trainer gone"
done
sleep 90   # let the CUDA context and its memory actually release

for ARM in $ARMS; do
  RUN=abl_V11e_${ARM}
  LOG=outputs/logs/${RUN}.log

  if [ -f "$LOG" ] && grep -q "ep ${EPOCHS}/${EPOCHS}\]" "$LOG"; then
    say "OK $ARM already reached ${EPOCHS}/${EPOCHS}"
    continue
  fi

  case "$ARM" in
    B1a) PORT=29801 ;; B1b) PORT=29802 ;; B1c) PORT=29803 ;;
    B2)  PORT=29804 ;; C1a) PORT=29805 ;; C1b) PORT=29806 ;;
    D1)  PORT=29807 ;; *)   PORT=$((29900 + RANDOM % 90)) ;;
  esac

  waited=0
  while :; do
    busy=""
    IFS=',' read -ra _want <<< "$GPUS"
    for g in "${_want[@]}"; do
      free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader \
                | tr -d ' ' | awk -F, -v want="$g" '$1==want{print $2}' | sed 's/MiB//')
      [ -n "$free_mb" ] || free_mb=0
      [ "$free_mb" -ge "$MIN_FREE_MB" ] || busy="$busy gpu$g(${free_mb}MiB)"
    done
    [ -z "$busy" ] && break
    if [ "$waited" -ge 86400 ]; then
      say "GIVE UP $ARM after 24h, short on$busy"
      break
    fi
    say "wait $ARM: short on$busy"
    sleep 300; waited=$((waited + 300))
  done
  [ -n "${busy:-}" ] && continue

  say "START $ARM -> $RUN (port=$PORT)"
  ARM="$ARM" RUN="$RUN" GPUS="$GPUS" RANKS="$RANKS" BS="$BS" \
    EPOCHS="$EPOCHS" PORT="$PORT" MIN_FREE_MB="$MIN_FREE_MB" \
    bash outputs/launch_V11e.sh >> "$QLOG" 2>&1
  say "launched $ARM"

  sleep 60
  for _ in $(seq 1 5760); do     # up to 48h
    pgrep -f "train_runs/abl_V11e_${ARM}" > /dev/null 2>&1 || break
    sleep 30
  done

  if [ -f "$LOG" ] && grep -q "ep ${EPOCHS}/${EPOCHS}\]" "$LOG"; then
    say "DONE $ARM"
  else
    say "ARM $ARM did NOT reach ${EPOCHS}/${EPOCHS}"
  fi
done

say "queue finished"
