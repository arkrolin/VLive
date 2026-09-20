#!/bin/bash
# Single-shot push with TIGHT network timeouts, so the token sits in
# .git/config for a few seconds instead of up to ~32 minutes.
#
# Why this exists (09-20, second occurrence):
#   push_retry.sh retries 6x with git's default network timeout of 300s each.
#   When the local ssh caller is killed mid-loop -- which happens, because the
#   caller's own timeout can be far shorter than 6*300s -- the remote script
#   KEEPS RUNNING with the token already written into origin's URL, and the
#   token stays there until that orphan finishes. Twice now this left
#   `x-access-token:ghp_...` in .git/config.
#
#   Failing fast shrinks the exposure window to seconds AND makes the whole
#   operation fit inside a short caller budget. For retries, call this in a
#   loop from the caller instead of relying on the in-script loop.
#
# Same success criterion as push_retry.sh:
#   VERIFIED: HEAD == origin/main   and   TOKEN_LEFT_IN_CONFIG=0
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1

TOK=$(cat)
if [ -z "$TOK" ]; then
    echo "ERROR: empty token on stdin" >&2
    exit 1
fi

# never re-apply credentials a previous aborted run may have left behind
ORIG_URL=$(git remote get-url origin | sed -E 's#://[^@/]*@#://#')
LOCAL_SHA=$(git rev-parse HEAD)

restore_remote() { git remote set-url origin "$ORIG_URL"; }
trap restore_remote EXIT INT TERM HUP

git remote set-url origin "https://x-access-token:${TOK}@github.com/arkrolin/VLive.git"

# abort if the transfer drops below 1 KB/s for 25s (instead of hanging for 300s)
git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=25 push origin main
RC=$?

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
