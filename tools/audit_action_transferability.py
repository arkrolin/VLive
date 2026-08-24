"""Which specific actions transfer across characters, and which are arbitrary?

The aggregate cross-character prior is 2.6x over chance. That average is only
useful if it is spread evenly. If instead a handful of actions are strongly
stereotyped ("wave" always means a big right-arm sweep) while the rest are
artist-arbitrary ("shy" can be anything), then the aggregate is misleading and
the right move is to scope v1 to the stereotyped set rather than train on all
130 labels.

Per action, on a leave-one-character-out split, this reports

  consistency  cosine between the action's coarse plan on one character and the
               same action's mean plan built from the OTHER characters, minus
               the same quantity against all other actions. Positive means the
               label predicts something character-independent.
  recall       does that action's own centroid win the nearest-centroid vote

Usage:
    uv run python tools/audit_action_transferability.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audit_generative_prior import CACHE, NS, SLOTS, dedup  # noqa: E402

GLOSS = {
    "dazhaohu": "打招呼", "haixiu": "害羞", "shengqi": "生气", "diantou": "点头",
    "motouweixiao": "摸头微笑", "weixiao": "微笑", "kaixin": "开心",
    "nanguo": "难过", "kunhuo": "困惑", "jingya": "惊讶", "sikao": "思考",
    "tiaowu": "跳舞", "zhuanquan": "转圈", "bishi": "鄙视", "wunai": "无奈",
    "shuijiao": "睡觉", "chifan": "吃饭", "hejiu": "喝酒", "kanshu": "看书",
    "zhaoshou": "招手", "gulizhang": "鼓掌", "baoquan": "抱拳", "kutou": "哭",
    "shengri": "生日", "aixin": "爱心", "bizui": "闭嘴", "yaotou": "摇头",
    "juesexuanze": "角色选择", "denglu": "登录", "daiji": "待机",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--top", type=int, default=22)
    args = ap.parse_args()

    z = np.load(CACHE, allow_pickle=True)
    amp, shape, pres = z["amp"], z["shape"], z["pres"]
    dur, ya, ym, yc = z["dur"], z["act"], z["pack"], z["char"]
    k = dedup(amp, shape, pres, dur, ya, ym, yc)
    amp, shape, pres, dur = amp[k], shape[k], pres[k], dur[k]
    ya, ym, yc = ya[k], ym[k], yc[k]

    scale = np.array([np.median(amp[pres[:, i], i]) if pres[:, i].any() else 1.0
                      for i in range(NS)], np.float32)
    F = np.hstack([
        np.log1p(np.abs(amp) / (np.abs(amp).mean(0, keepdims=True) + 1e-6)),
        (amp > 0.15 * np.maximum(scale, 1e-6)).astype(np.float32),
        np.log(np.maximum(dur, 1e-3))[:, None].astype(np.float32),
    ])
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-6)

    per_chars = defaultdict(set)
    for a, c in zip(ya, yc):
        per_chars[a].add(c)
    keep = np.array([len(per_chars[a]) >= args.min_chars for a in ya])
    F, ya, yc = F[keep], ya[keep], yc[keep]
    print(f"=== {len(ya)} assets | {len(set(ya))} actions "
          f"(>= {args.min_chars} characters) | {len(set(yc))} characters ===\n")

    stats: dict[str, list] = defaultdict(lambda: [0, 0, [], []])
    for c in sorted(set(yc)):
        te, tr = np.where(yc == c)[0], np.where(yc != c)[0]
        if len(te) < 5 or len(tr) < 200:
            continue
        mu = F[tr].mean(0)
        rig = F[te].mean(0) - mu
        names, cents = [], []
        for a in sorted(set(ya[te])):
            m = tr[ya[tr] == a]
            if len(m) >= 3:
                names.append(a)
                cents.append(F[m].mean(0) - mu)
        if len(names) < 4:
            continue
        C = np.stack(cents)
        Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-6)
        for i in te:
            a = ya[i]
            if a not in names:
                continue
            r = F[i] - mu - rig
            rn = r / max(np.linalg.norm(r), 1e-6)
            sims = Cn @ rn
            j = names.index(a)
            others = np.delete(sims, j)
            st = stats[a]
            st[0] += int(np.argmax(sims) == j)
            st[1] += 1
            st[2].append(float(sims[j]))
            st[3].append(float(others.mean()))

    rows = []
    for a, (hit, n, own, oth) in stats.items():
        if n < 6:
            continue
        rows.append((a, n, hit / n, float(np.mean(own) - np.mean(oth))))
    rows.sort(key=lambda r: -r[3])
    overall = sum(r[1] * r[2] for r in rows) / max(sum(r[1] for r in rows), 1)
    chance = 1.0 / max(len(rows), 1)

    def show(rs, title):
        print(title)
        print(f"    {'action':20s} {'gloss':10s} {'n':>4s} {'recall':>7s} "
              f"{'consistency':>12s}")
        for a, n, rec, con in rs:
            print(f"    {a[:20]:20s} {GLOSS.get(a, ''):10s} {n:4d} "
                  f"{100*rec:6.1f}% {con:+12.3f}")

    show(rows[:args.top], "TOP -- the label predicts a character-independent plan")
    print()
    show(rows[-10:], "BOTTOM -- artist-arbitrary, the label predicts nothing")
    print(f"\n  overall recall {100*overall:.1f}%  (chance {100*chance:.1f}%)")

    good = [r for r in rows if r[3] > 0.05]
    cov = sum(r[1] for r in good) / max(sum(r[1] for r in rows), 1)
    grec = sum(r[1] * r[2] for r in good) / max(sum(r[1] for r in good), 1)
    print(f"  actions with consistency > 0.05: {len(good)}/{len(rows)}  "
          f"covering {100*cov:.1f}% of assets  recall {100*grec:.1f}%")


if __name__ == "__main__":
    main()
