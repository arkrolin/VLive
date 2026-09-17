#!/bin/bash
# Supervisor for the V11e chain: make sure the queued arms actually run even if
# chain_V11e.sh gives up.
#
# Why this exists: chain_V11e.sh refuses to start an arm unless every target GPU
# has >= 40000 MiB free, and it only waits 6h before skipping the arm. The box
# is heavily shared (all 8 GPUs at ~100% util, ~40GB free on 1/2/4/5), so that
# threshold can trip through no fault of ours. A single rank only needs ~22-30GB,
# so 40000 MiB was over-conservative.
#
# Race safety: this script does NOTHING until the chain process is gone. That is
# what makes it safe -- the chain can never be starting an arm at the same time.
# It then looks at each arm's log and launches only the ones that never reached
# the full epoch budget.
#
# Usage (background):
#   nohup bash outputs/supervise_V11e.sh > /dev/null 2>&1 &
set -u

GPUS=${GPUS:-1,2,4,5}
RANKS=${RANKS:-4}
BS=${BS:-6}
EPOCHS=${EPOCHS:-70}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
ARMS=${ARMS:-B1b C1a}
CHAIN_PID_FILE=${CHAIN_PID_FILE:-}

cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs
SLOG=outputs/logs/_v11e_supervisor.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$SLOG"; }

say "supervisor start arms='$ARMS' min_free=${MIN_FREE_MB}MiB"

# --- 1. wait until no chain and no trainer for these arms is alive ----------
while :; do
  alive=0
  pgrep -f 'chain_V11e.sh' > /dev/null 2>&1 && alive=1
  for a in $ARMS; do
    pgrep -f "train_runs/abl_V11e_${a}" > /dev/null 2>&1 && alive=1
  done
  [ "$alive" -eq 0 ] && break
  sleep 120
done
say "chain/trainers all gone; checking which arms still need to run"

# --- 2. run whatever never finished ----------------------------------------
for ARM in $ARMS; do
  RUN=abl_V11e_${ARM}
  LOG=outputs/logs/${RUN}.log

  if [ -f "$LOG" ] && grep -q "ep ${EPOCHS}/${EPOCHS}\]" "$LOG"; then
    say "OK $ARM already reached ${EPOCHS}/${EPOCHS}"
    continue
  fi

  # same port table as the chain, so a stale rendezvous cannot collide
  case "$ARM" in
    B1a) PORT=29801 ;; B1b) PORT=29802 ;; B1c) PORT=29803 ;;
    B2)  PORT=29804 ;; C1a) PORT=29805 ;; C1b) PORT=29806 ;;
    D1)  PORT=29807 ;; *)   PORT=$((29900 + RANDOM % 90)) ;;
  esac

  say "WAIT $ARM for ${MIN_FREE_MB}MiB on each of $GPUS"
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
      say "GIVE UP $ARM after 24h, still short on$busy"
      break
    fi
    sleep 300; waited=$((waited + 300))
  done
  [ -n "${busy:-}" ] && continue

  say "START $ARM -> $RUN (port=$PORT)"
  ARM="$ARM" RUN="$RUN" GPUS="$GPUS" RANKS="$RANKS" BS="$BS" \
    EPOCHS="$EPOCHS" PORT="$PORT" MIN_FREE_MB="$MIN_FREE_MB" \
    bash outputs/launch_V11e.sh >> "$SLOG" 2>&1
  say "launched $ARM"

  # wait for this arm to finish before moving on
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

say "supervisor finished"
