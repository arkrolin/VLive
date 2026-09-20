"""Corpus inventory for the V9 (all580) training set + the pinned val split.

Prints every number the overview document needs, defensively (each item in its
own try/except) so an unexpected JSON shape does not kill the whole run.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
O = BASE / "outputs"


def show(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] ERROR {type(e).__name__}: {e}")
    print("-" * 70)


def j(name):
    return json.loads((O / name).read_text())


def _wl():
    wl = j("model_whitelist_all.json")
    print(f"[whitelist_all] top-level keys = {list(wl)}")
    print(f"[whitelist_all] _meta = {wl.get('_meta')}")
    for k in ("kept", "models"):
        v = wl.get(k)
        if isinstance(v, (list, dict)):
            print(f"[whitelist_all] {k}: type={type(v).__name__} len={len(v)}")
            if isinstance(v, list) and v:
                print(f"               head={v[:3]}")


def _split():
    """Train/val split + sample-weighted long-tail share."""
    bank = pickle.load(open(O / "motionbank_580_v9all.pkl", "rb"))
    holdout = set(j("val_holdout_all.json"))
    act2char, samples = {}, []
    for k in bank:
        c, a = eval(k) if isinstance(k, str) else k
        act2char.setdefault(a, set()).add(c)
        samples.append((c, a))
    val_all = [s for s in samples if s[0] in holdout]
    val_ge5 = [s for s in val_all if len(act2char[s[1]]) >= 5]
    train = [s for s in samples if s[0] not in holdout]
    print(f"[split] total = {len(samples)}")
    print(f"[split] val characters({len(holdout)}) instances: unfiltered={len(val_all)} "
          f"-> after >=5 filter = {len(val_ge5)}  ({len(val_ge5)/max(len(val_all),1)*100:.1f}% kept)")
    print(f"[split] train instances = {len(train)}")
    n1 = sum(1 for _, a in train if len(act2char[a]) == 1)
    nlt5 = sum(1 for _, a in train if len(act2char[a]) < 5)
    print(f"[split] TRAIN samples whose action is character-EXCLUSIVE = {n1} "
          f"({n1/max(len(train),1)*100:.1f}%)")
    print(f"[split] TRAIN samples whose action is in <5 characters   = {nlt5} "
          f"({nlt5/max(len(train),1)*100:.1f}%)")
    print("[split] => those samples train the model but are NEVER evaluated")


def _vh():
    for f in ("val_holdout_all.json", "val_holdout_std.json"):
        v = j(f)
        print(f"[{f}] type={type(v).__name__} len={len(v)} head={list(v)[:3]}")


def _dedup():
    d = j("dedup_skip_all.json")
    n = sum(len(v) for v in d.values()) if isinstance(d, dict) else "?"
    print(f"[dedup_skip_all] models={len(d)} dropped_samples={n}")


def _amap():
    d = j("action_semantic_map.json")
    g = d.get("groups", {})
    names = sum(len(v) for v in g.values())
    print(f"[action_semantic_map] groups={len(g)} names_covered={names}")


def _bank():
    b = pickle.load(open(O / "motionbank_580_v9all.pkl", "rb"))
    char2act, act2char = {}, {}
    P = []
    for k, v in b.items():
        c, a = eval(k) if isinstance(k, str) else k
        char2act.setdefault(c, set()).add(a)
        act2char.setdefault(a, set()).add(c)
        try:
            P.append(v[1].shape[0])
        except Exception:
            pass
    nA = np.array([len(v) for v in char2act.values()])
    nC = np.array([len(v) for v in act2char.values()])
    print(f"[motionbank] (char,action) entries = {len(b)}")
    print(f"[motionbank] characters = {len(char2act)}   distinct actions = {len(act2char)}")
    print(f"[motionbank] actions-per-character : min={nA.min()} p25={np.percentile(nA,25):.0f} "
          f"med={np.median(nA):.0f} p75={np.percentile(nA,75):.0f} max={nA.max()} mean={nA.mean():.1f}")
    print(f"[motionbank] characters-per-action : 1-only={int((nC==1).sum())} "
          f"({(nC==1).mean()*100:.1f}%)  2={int((nC==2).sum())}  3-4={int(((nC>=3)&(nC<=4)).sum())} "
          f"5-19={int(((nC>=5)&(nC<20)).sum())}  20+={int((nC>=20).sum())}")
    print(f"[motionbank] actions appearing in >=5 characters = {int((nC>=5).sum())} "
          f"({(nC>=5).mean()*100:.1f}% of actions)")
    if P:
        P = np.array(P)
        print(f"[motionbank] params per (char,action) : min={P.min()} med={np.median(P):.0f} "
              f"p90={np.percentile(P,90):.0f} max={P.max()} mean={P.mean():.1f}")


def _exem():
    e = pickle.load(open(O / "exem_cache_580_v9all.pkl", "rb"))
    acts = {k[0] for k in e}
    print(f"[exem_cache] (action,param) entries = {len(e)}  distinct actions = {len(acts)}")


def _range():
    r = pickle.load(open(O / "range_cache_580_all_v9all_12083.pkl", "rb"))
    print(f"[range_cache_580_all_v9all_12083] type={type(r).__name__} len={len(r)}")


def _val():
    ren = pickle.load(open(O / "render_data/abl_V11e_B1b_best.pkl", "rb"))
    it = ren["items"]
    print(f"[render B1b] n={ren['n']} epoch={ren['epoch']}")
    chars = Counter(x["model"] for x in it)
    acts = Counter(x["action"] for x in it)
    print(f"[render B1b] val characters={len(chars)}  val actions={len(acts)}")
    print(f"[render B1b] instances per character: min={min(chars.values())} "
          f"med={int(np.median(list(chars.values())))} max={max(chars.values())}")
    P = np.array([x["pred"].shape[0] for x in it])
    print(f"[render B1b] P per instance: min={P.min()} med={int(np.median(P))} max={P.max()}")
    # constant-channel fraction + moving-channel count
    nconst = nmov = nslot = 0
    for x in it:
        t = np.asarray(x["target"])
        ptp = t.max(-1) - t.min(-1)
        nslot += ptp.size
        nmov += int((ptp >= 1e-6).sum())
    nconst = nslot - nmov
    print(f"[render B1b] slots={nslot}  const={nconst} ({nconst/nslot*100:.1f}%)  "
          f"moving={nmov} ({nmov/nslot*100:.1f}%)")


show("whitelist", _wl)
show("split", _split)
show("val_holdout", _vh)
show("dedup", _dedup)
show("action_map", _amap)
show("bank", _bank)
show("exem", _exem)
show("range", _range)
show("val", _val)
