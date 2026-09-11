#!/bin/bash
# Push to GitHub with retries. The server has no proxy and github.com:443
# intermittently times out, so a single attempt is not enough.
# Token arrives on stdin (never as argv, which would show up in `ps`).
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
  for i in 1 2 3 4 5 6; do
    if git push origin main > outputs/logs/push.log 2>&1; then
      echo "PUSH OK on attempt $i" >> outputs/logs/push.log
      break
    fi
    echo "attempt $i failed, retrying in 25s" >> outputs/logs/push.log
    sleep 25
  done
  echo "EXIT=$?" >> outputs/logs/push.log
  git remote set-url origin https://github.com/arkrolin/VLive.git
  echo "TOKEN_LEFT_IN_CONFIG=$(grep -c x-access-token .git/config)" >> outputs/logs/push.log
' > /dev/null 2>&1 &

echo "push with retries started; see outputs/logs/push.log"
