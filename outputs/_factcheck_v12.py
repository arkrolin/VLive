"""Fact-check take 4: model3.json parameter bounds + real motion-format counts."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

BASE = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
O = BASE / "outputs"
root = BASE / "data" / "all"
kept = json.loads((O / "model_whitelist_all.json").read_text())["kept"]

print("=" * 78)
print("A. .model3.json content: does it carry per-parameter Min/Max/Default?")
m0 = kept[0]
d0 = root / m0
m3 = list(d0.glob("*.model3.json"))
print(f"   probe {m0}: model3 files = {[p.name for p in m3]}")
if m3:
    blob = json.loads(m3[0].read_text())
    print(f"   top-level keys: {list(blob)}")
    print(f"   Version: {blob.get('Version')}")
    params = blob.get("FileReferences", {})  # not here
    ps = blob.get("Parameters")
    if ps is None:
        # Cubism 3 stores them under FileReferences? print structure
        for k, v in blob.items():
            if isinstance(v, (dict, list)):
                print(f"   {k}: {type(v).__name__} len={len(v)}")
    else:
        print(f"   Parameters: {len(ps)}")
        for p in ps[:3]:
            print(f"     {json.dumps(p, ensure_ascii=False)[:220]}")

print("=" * 78)
print("B. parameter count in model3.json vs the motions")
if m3:
    ps = json.loads(m3[0].read_text()).get("Parameters", [])
    withdef = [p for p in ps if "Min" in p and "Max" in p and "Default" in p]
    print(f"   params={len(ps)}  with Min/Max/Default={len(withdef)}")
    # how many have a non-trivial range
    nz = [p for p in withdef if abs(float(p["Max"]) - float(p["Min"])) > 1e-9]
    print(f"   with non-zero range={len(nz)}")
    print(f"   examples: {[ (p['Id'], p['Min'], p['Max'], p['Default']) for p in withdef[:4] ]}")
    # parameter ids: semantic or not?
    ids = [str(p.get("Id", "")) for p in ps]
    import re
    sem = sum(1 for i in ids if re.search(r"[A-Za-z]{3,}", i))
    print(f"   ids containing a >=3-letter alpha run: {sem}/{len(ids)}")

print("=" * 78)
print("C. how many models expose model3.json with parameters? (over kept 580)")
n_ok = n_par = 0
tot_par = []
for i, m in enumerate(kept):
    d = root / m
    if not d.exists():
        continue
    f = list(d.glob("*.model3.json"))
    if not f:
        continue
    try:
        b = json.loads(f[0].read_text())
    except Exception:
        continue
    ps = b.get("Parameters") or []
    if ps:
        n_ok += 1
        tot_par.append(len(ps))
    n_par += 1
    if i > 120:      # sample the first 120 to keep it fast
        break
import numpy as np
if tot_par:
    a = np.array(tot_par)
    print(f"   models checked={n_par}  with Parameters list={n_ok}")
    print(f"   params per model: min={a.min()} med={np.median(a):.0f} max={a.max()}")

print("=" * 78)
print("D. motion formats (fixed glob)")
c = Counter()
for p in BASE.rglob("*.motion3.json"):
    c["motion3"] += 1
for p in BASE.rglob("*.mtn"):
    c["mtn"] += 1
print(f"   *motion3.json = {c['motion3']}    *.mtn = {c['mtn']}")

# how many model dirs have ONLY mtn (i.e. Cubism 2, unusable today)
n_only_mtn = n_has_m3 = 0
for m in kept:
    d = root / m
    if not d.exists():
        continue
    has3 = any(d.rglob("*.motion3.json"))
    has2 = any(d.rglob("*.mtn"))
    if has3:
        n_has_m3 += 1
    elif has2:
        n_only_mtn += 1
print(f"   kept models with motion3.json = {n_has_m3}")
print(f"   kept models with ONLY .mtn    = {n_only_mtn}")
