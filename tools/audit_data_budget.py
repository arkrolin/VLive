"""What is actually left after the character-leakage audit?

audit_leakage.py proved that cross-CHARACTER action retrieval is ~8.7%, not the
85.8% we first measured. That kills the "133k free cross-model pairs" plan, but
it does NOT by itself tell us whether text conditioning is learnable. Two very
different worlds are consistent with r@1=8.7%:

  WORLD 1 (fatal)   Trajectories carry no character-agnostic action semantics.
                    Text conditioning can only ever be learned per rig.

  WORLD 2 (fine)    The signal is real but coarse. 1319 exact action names is a
                    brutal 1319-way retrieval task; 'haixiu' vs 'xiuse' vs
                    'motouhaixiu' are near synonyms that get counted as errors.
                    At the level a user actually types ("wave hello", "be shy")
                    the signal survives.

This script separates the two, and sizes the real data budget:

  1. per-CHARACTER budget      -- 39 characters is the true N, how big is each?
  2. coarse action categories  -- collapse 1319 pinyin names into ~14 intents
  3. cross-character retrieval at coarse level vs exact level
  4. leave-one-character-out nearest-centroid classification
     (the actual "can a model trained on other characters predict this one"
      question, which retrieval only approximates)
  5. the deployment-shaped test: few-shot from the SAME rig, k=1,3,5

Usage:
    uv run python tools/audit_data_budget.py
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from audit_leakage import char_of  # noqa: E402
from survey_motion_names import norm_action  # noqa: E402
from verify_action_alignment import encode_motion  # noqa: E402

# ---------------------------------------------------------------------------
# Coarse intent lexicon. Ordered: first match wins, so put compounds first.
# These are the granularities a user would actually type in a prompt.
# ---------------------------------------------------------------------------
INTENT = [
    ("greet",    r"dazhaohu|zhaohu|huishou|baibai|wenhou|jingli|xingli"),
    ("nod",      r"diantou|dianou|kending|tongyi"),
    ("shake",    r"yaotou|yaotouhuang|fouding|bu$"),
    ("shy",      r"haixiu|xiuse|xiunu|hongl?ian|paixiu"),
    ("angry",    r"shengqi|baonu|^nu$|fennu|nuhuo|qifen|zeguai|jiasheng"),
    ("smile",    r"weixiao|xiao$|kaixin|tianxiao|gaoxing|kuaile|daxiao|touxiao"),
    ("surprise", r"jingxi|jingya|jingxia|chijing|yiwai"),
    ("confused", r"yihuo|kunhuo|buji|xunwen|wenhao|sikao|sikaozhong"),
    ("sad",      r"tanqi|nanguo|beishang|kuqi|liulei|shiluo|youshang|touteng"),
    ("embarrass",r"ganga|huangzhang|jinzhang|wunai|wuyu"),
    ("sleepy",   r"dahaqian|haqian|kunle|shuijiao|pijuan|biyan"),
    ("cute",     r"keai|sajiao|maimeng|qiaopi"),
    ("pose",     r"zhanshi|chuchang|zhanli|xianqunzi|zhuanquan|xuanzhuan|zaoxing"),
    ("interact", r"motou|touchmo|dianxiong|baoxiong|duishouzhi|songli|ganxie|"
                 r"zhamao|shihao|xianqi|dianji|chumo"),
    ("serious",  r"yansu|wushi|leng|mianwubiaoqing|pingjing"),
]
INTENT_RE = [(name, re.compile(pat, re.I)) for name, pat in INTENT]


def intent_of(action: str) -> str | None:
    for name, rx in INTENT_RE:
        if rx.search(action):
            return name
    return None


def retrieval(X, y, block, tag, verbose=True):
    """recall@1/@5, forbidding pairs where block[i]==block[j]."""
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    S = Z @ Z.T
    S = np.where(block[:, None] == block[None, :], -np.inf, S)
    np.fill_diagonal(S, -np.inf)
    valid = np.isfinite(S).any(axis=1)
    if valid.sum() == 0:
        return 0.0, 0.0
    order = np.argsort(-S[valid], axis=1)[:, :5]
    tgt, lbl = y[valid], y[order]
    r1 = float((lbl[:, 0] == tgt).mean())
    r5 = float((lbl == tgt[:, None]).any(1).mean())
    cnt = Counter(y[valid])
    n = len(y[valid])
    rand = sum(c * (c - 1) for c in cnt.values()) / max(n * (n - 1), 1)
    if verbose:
        print(f"  {tag:44s} r@1={100*r1:5.1f}%  r@5={100*r5:5.1f}%  "
              f"chance={100*rand:4.1f}%  lift={r1/max(rand,1e-9):4.1f}x  "
              f"(n={n})")
    return r1, r5


def loco_centroid(X, y, chars, tag):
    """Leave-one-character-out nearest-centroid classification.

    This is the honest version of 'train on other characters, predict this one'.
    Retrieval can be gamed by one lucky near-duplicate; a centroid cannot.
    """
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    labels = sorted(set(y))
    correct = total = 0
    per_class_hit = Counter()
    per_class_n = Counter()
    for c in sorted(set(chars)):
        te = chars == c
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        cents, names = [], []
        for lab in labels:
            m = tr & (y == lab)
            if m.sum() < 2:
                continue
            v = Z[m].mean(0)
            cents.append(v / max(np.linalg.norm(v), 1e-6))
            names.append(lab)
        if not cents:
            continue
        C = np.stack(cents)
        keep = np.isin(y[te], names)
        if keep.sum() == 0:
            continue
        pred = np.array(names)[np.argmax(Z[te][keep] @ C.T, axis=1)]
        gold = y[te][keep]
        correct += int((pred == gold).sum())
        total += int(keep.sum())
        for g, p in zip(gold, pred):
            per_class_n[g] += 1
            per_class_hit[g] += int(g == p)
    acc = correct / max(total, 1)
    prior = max(per_class_n.values()) / max(total, 1) if per_class_n else 0
    print(f"  {tag:44s} acc={100*acc:5.1f}%  majority={100*prior:5.1f}%  "
          f"(n={total})")
    if per_class_n:
        rows = sorted(per_class_n, key=lambda k: -per_class_n[k])[:8]
        print("      per-intent: " + "  ".join(
            f"{k}={100*per_class_hit[k]/per_class_n[k]:.0f}%" for k in rows))
    return acc


def fewshot_same_rig(X, y, models, ks=(1, 3, 5)):
    """Deployment shape: k labelled motions from the TARGET rig, classify the rest.

    This is what the user actually has -- a reference video / existing motions
    from the same character. Reports accuracy over coarse intents.
    """
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    rng = np.random.default_rng(0)
    for k in ks:
        correct = total = 0
        chance_num = 0.0          # sum over queries of 1/|classes in that rig|
        nways: list[int] = []
        for m in sorted(set(models)):
            idx = np.where(models == m)[0]
            if len(idx) < 4:
                continue
            labs = y[idx]
            support, query = [], []
            for lab in set(labs):
                pool = idx[labs == lab]
                if len(pool) <= k:
                    continue
                pick = rng.choice(pool, k, replace=False)
                support += list(pick)
                query += [i for i in pool if i not in set(pick)]
            if len(set(y[support])) < 2 or not query:
                continue
            names = sorted(set(y[support]))
            C = np.stack([Z[[i for i in support if y[i] == lab]].mean(0)
                          for lab in names])
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-6)
            pred = np.array(names)[np.argmax(Z[query] @ C.T, axis=1)]
            correct += int((pred == y[query]).sum())
            total += len(query)
            # honest baseline: the support set only has len(names) classes, so
            # a coin flip is 1/len(names), not 1/15.
            chance_num += len(query) / len(names)
            nways.append(len(names))
        acc = correct / max(total, 1)
        ch = chance_num / max(total, 1)
        print(f"  few-shot k={k} on the SAME rig               "
              f"acc={100*acc:5.1f}%  chance={100*ch:5.1f}%  "
              f"lift={acc/max(ch,1e-9):4.1f}x  "
              f"(n={total}, mean {np.mean(nways):.1f}-way)"
              if nways else f"  few-shot k={k}: no data")


def head_to_head(X, y, models, chars, k=1):
    """The money comparison, on IDENTICAL queries and IDENTICAL label sets.

    For every rig we hold out k examples per intent as 'the few-shot support'
    and score the remaining queries twice:

      OTHER-CHARS : centroids built from every character except this one
                    (= what a big pretrained cross-character prior gives you)
      SAME-RIG-k  : centroids built from just the k held-out examples
                    (= what one reference clip from the user's own model gives)
      BOTH        : average of the two centroid sets

    If SAME-RIG-1 beats OTHER-CHARS, the entire cross-character corpus is worth
    less than a single example from the target rig, and M2 must be built around
    few-shot adaptation rather than a shared text->motion prior.
    """
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    rng = np.random.default_rng(0)
    hit_o = hit_s = hit_b = total = 0
    for m in sorted(set(models)):
        idx = np.where(models == m)[0]
        if len(idx) < 4:
            continue
        c = chars[idx[0]]
        labs = y[idx]
        support, query = [], []
        for lab in set(labs):
            pool = idx[labs == lab]
            if len(pool) <= k:
                continue
            pick = rng.choice(pool, k, replace=False)
            support += list(pick)
            query += [i for i in pool if i not in set(pick)]
        names = sorted(set(y[support]))
        if len(names) < 2 or not query:
            continue
        # same-rig centroids
        Cs = np.stack([Z[[i for i in support if y[i] == lab]].mean(0)
                       for lab in names])
        # other-character centroids over the SAME label set
        oth = chars != c
        Co, ok = [], []
        for lab in names:
            sel = oth & (y == lab)
            if sel.sum() >= 2:
                Co.append(Z[sel].mean(0))
                ok.append(lab)
        if len(ok) < 2:
            continue
        Cs /= np.maximum(np.linalg.norm(Cs, axis=1, keepdims=True), 1e-6)
        Co = np.stack(Co)
        Co /= np.maximum(np.linalg.norm(Co, axis=1, keepdims=True), 1e-6)
        # restrict everything to the intersection so the task is identical
        common = [lab for lab in names if lab in ok]
        if len(common) < 2:
            continue
        si = [names.index(lab) for lab in common]
        oi = [ok.index(lab) for lab in common]
        Cs, Co = Cs[si], Co[oi]
        q = [i for i in query if y[i] in set(common)]
        if not q:
            continue
        gold = y[q]
        arr = np.array(common)
        hit_s += int((arr[np.argmax(Z[q] @ Cs.T, 1)] == gold).sum())
        hit_o += int((arr[np.argmax(Z[q] @ Co.T, 1)] == gold).sum())
        Cb = Cs + Co
        Cb /= np.maximum(np.linalg.norm(Cb, axis=1, keepdims=True), 1e-6)
        hit_b += int((arr[np.argmax(Z[q] @ Cb.T, 1)] == gold).sum())
        total += len(q)
    if total:
        print(f"  OTHER-CHARACTERS prior (35 chars)            "
              f"acc={100*hit_o/total:5.1f}%")
        print(f"  SAME-RIG few-shot, k={k}                       "
              f"acc={100*hit_s/total:5.1f}%")
        print(f"  BOTH combined                                "
              f"acc={100*hit_b/total:5.1f}%   (n={total}, identical queries)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    args = ap.parse_args()

    root = Path(args.root)
    wl = Path(args.whitelist)
    models = (sorted(orjson.loads(wl.read_bytes())["kept"]) if wl.exists()
              else sorted(d.name for d in root.iterdir() if d.is_dir()))

    # ---------------- 1. the true data budget ------------------------------
    by_char: dict[str, list[str]] = defaultdict(list)
    for m in models:
        by_char[char_of(m)].append(m)
    mot_per_char: Counter = Counter()
    for c, ms in by_char.items():
        seen: set[str] = set()
        for m in ms:
            d = root / m / "motions"
            if d.is_dir():
                seen |= {p.stem for p in d.glob("*.json")}
        mot_per_char[c] = len(seen)

    print("=== TRUE data budget (whitelist, deduped by character) ===")
    print(f"  {len(models)} model dirs -> {len(by_char)} characters")
    tot = sum(mot_per_char.values())
    vals = sorted(mot_per_char.values(), reverse=True)
    print(f"  distinct motion NAMES per character: total={tot}  "
          f"median={int(np.median(vals))}  max={vals[0]}  min={vals[-1]}")
    print(f"  characters with >=20 distinct motions: "
          f"{sum(1 for v in vals if v >= 20)}")
    print("  top: " + ", ".join(f"{c}({n})" for c, n in mot_per_char.most_common(8)))

    # ---------------- encode ------------------------------------------------
    X, y_exact, y_int, y_model, y_char = [], [], [], [], []
    for name in models:
        mdir = root / name / "motions"
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("*.json")):
            act, _ = norm_action(f.stem)
            if act in ("stand", "idle"):
                continue
            it = intent_of(act)
            if it is None:
                continue
            v = encode_motion(f)
            if v is None:
                continue
            X.append(v.reshape(-1))
            y_exact.append(act)
            y_int.append(it)
            y_model.append(name)
            y_char.append(char_of(name))

    X = np.stack(X)
    y_exact, y_int = np.array(y_exact), np.array(y_int)
    y_model, y_char = np.array(y_model), np.array(y_char)
    print(f"\n=== encoded {len(X)} motions | {len(set(y_model))} models | "
          f"{len(set(y_char))} chars | {len(set(y_int))} intents ===")
    print("  intent counts: " + ", ".join(
        f"{k}={v}" for k, v in Counter(y_int).most_common()))

    # ---------------- 2/3. exact vs coarse, cross-character ----------------
    print("\n=== retrieval: does coarsening rescue the signal? ===")
    retrieval(X, y_exact, y_model, "exact action | block MODEL   (leaky)")
    retrieval(X, y_exact, y_char, "exact action | block CHARACTER")
    retrieval(X, y_int, y_model, "coarse intent| block MODEL   (leaky)")
    retrieval(X, y_int, y_char, "coarse intent| block CHARACTER")

    # per-rig bias removal on top of coarsening
    Xc = X.copy().astype(np.float64)
    for m in set(y_model):
        sel = y_model == m
        Xc[sel] -= Xc[sel].mean(0)
    retrieval(Xc, y_int, y_char, "coarse + per-rig centering | block CHAR")

    # ---------------- 4. leave-one-character-out ---------------------------
    print("\n=== leave-one-CHARACTER-out nearest-centroid ===")
    loco_centroid(X, y_int, y_char, "coarse intent, raw")
    loco_centroid(Xc, y_int, y_char, "coarse intent, per-rig centered")

    # ---------------- 5. deployment shape ----------------------------------
    print("\n=== deployment shape: few-shot from the target rig ===")
    fewshot_same_rig(X, y_int, y_model)

    print("\n=== head-to-head: cross-character prior vs one own-rig example ===")
    for k in (1, 2):
        print(f"  --- k={k} ---")
        head_to_head(X, y_int, y_model, y_char, k=k)


if __name__ == "__main__":
    main()
