#!/bin/bash
# Push to GitHub using a token supplied on stdin (never as an argv, which would
# show up in `ps`). The credential is written into the remote URL only for the
# duration of the push and stripped immediately afterwards.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1
mkdir -p outputs/logs

TOK=$(cat)
if [ -z "$TOK" ]; then
    echo "ERROR: empty token on stdin" >&2
    exit 1
fi

git remote set-url origin "https://x-access-token:${TOK}@github.com/arkrolin/VLive.git"

nohup bash -c '
  cd /root/work/nlp/xjzhao13/lijie_llama/VLive
  git push origin main > outputs/logs/push.log 2>&1
  echo "EXIT=$?" >> outputs/logs/push.log
  git remote set-url origin https://github.com/arkrolin/VLive.git
  echo "TOKEN_LEFT_IN_CONFIG=$(grep -c x-access-token .git/config)" >> outputs/logs/push.log
' > /dev/null 2>&1 &

echo "push started in background; see outputs/logs/push.log"
