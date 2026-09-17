#!/bin/bash
# Push to GitHub with retries. The server has no proxy and github.com:443
# intermittently times out, so a single attempt is not enough.
# Token arrives on stdin (never as argv, which would show up in `ps`).
#
# NOTE: this script used to background the retry loop with `nohup ... &`.
# That returned immediately and left the caller reading a STALE push.log from a
# previous run -> false "PUSH OK" confirmation. It now runs in the FOREGROUND
# and ends by comparing `git rev-parse HEAD` with `origin/main`, which is the
# only trustworthy confirmation. Truncate the log first so it can never be stale.
#
# 09-17 hardening -- the token is now removed even if we are killed:
#   A caller-side timeout (the local ssh had a 5 min cap, while git's own
#   network timeout is 300s) sent SIGTERM in the middle of the retry loop, so
#   the script never reached its cleanup and left `x-access-token:ghp_...`
#   inside .git/config. An EXIT/INT/TERM/HUP trap now restores the URL
#   unconditionally, and the saved ORIG_URL is scrubbed of credentials that a
#   previous aborted run may have left behind.
#   Corollary for callers: give this script a budget LONGER than 6*300s+5*25s,
#   or the trap is the only thing standing between you and a leaked token.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs

TOK=$(cat)
if [ -z "$TOK" ]; then
    echo "ERROR: empty token on stdin" >&2
    exit 1
fi

LOG=outputs/logs/push.log
: > "$LOG"

# never re-apply credentials that a previous aborted run left in the config
ORIG_URL=$(git remote get-url origin | sed -E 's#://[^@/]*@#://#')
LOCAL_SHA=$(git rev-parse HEAD)

restore_remote() { git remote set-url origin "$ORIG_URL"; }
trap restore_remote EXIT INT TERM HUP

git remote set-url origin "https://x-access-token:${TOK}@github.com/arkrolin/VLive.git"

RC=1
for i in 1 2 3 4 5 6; do
    if git push origin main >> "$LOG" 2>&1; then
        echo "PUSH OK on attempt $i"
        RC=0
        break
    fi
    echo "attempt $i failed, retrying in 25s (the branch is unchanged in the log if it is empty)"
    sleep 25
done

# Never leave the token in .git/config (the trap also covers this; kept explicit).
restore_remote
LEFT=$(grep -c x-access-token .git/config)

git fetch origin -q 2>/dev/null
REMOTE_SHA=$(git rev-parse origin/main)
echo "LOCAL_HEAD=$LOCAL_SHA"
echo "ORIGIN_MAIN=$REMOTE_SHA"
if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
    echo "VERIFIED: HEAD == origin/main"
else
    echo "MISMATCH: push not confirmed"
    RC=1
fi
echo "TOKEN_LEFT_IN_CONFIG=$LEFT"
[ "$LEFT" = "0" ] || RC=1
exit $RC
