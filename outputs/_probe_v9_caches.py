"""One-off structure probe for the V9 motion-bank / exem caches + a render pkl.

Why: to test whether "the bank fails on long-tail characters" is really the
cause of the implementation gap (0.143 corr), we need the bank's actual
coverage per validation instance. This probe finds out what fields are
available before writing the real stratified analysis.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

BASE = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive/outputs")


def probe(path: Path, name: str):
    if not path.exists():
        print(f"[{name}] MISSING {path}")
        return
    d = pickle.load(open(path, "rb"))
    size = len(d) if hasattr(d, "__len__") else "?"
    print(f"[{name}] type={type(d).__name__} len={size} bytes={path.stat().st_size}")

    if isinstance(d, dict):
        ks = list(d.keys())
        print(f"  keys[:5] = {[repr(k)[:90] for k in ks[:5]]}")
        k0 = ks[0]
        v0 = d[k0]
        print(f"  val[type]={type(v0).__name__}")
        if isinstance(v0, dict):
            vk = list(v0.keys())
            print(f"  val.keys[:10] = {[repr(k)[:40] for k in vk[:10]]}")
            for kk in vk[:5]:
                vv = v0[kk]
                shp = getattr(vv, "shape", None)
                print(f"    .{kk}: {type(vv).__name__} shape={shp} len={getattr(vv,'__len__',None)}")
        else:
            print(f"  val repr[:400] = {repr(v0)[:400]}")
    elif isinstance(d, (list, tuple)):
        print(f"  elem0 type={type(d[0]).__name__} repr[:400]={repr(d[0])[:400]}")
        print(f"  elem1 type={type(d[1]).__name__} repr[:200]={repr(d[1])[:200]}")


for fn, nm in [
    ("motionbank_580_v9all.pkl", "motionbank"),
    ("exem_cache_580_v9all.pkl", "exem"),
    ("charstats_loo_580_v9all.pkl", "charstats_loo"),
    ("render_data/abl_V11e_B1b_best.pkl", "render_B1b"),
]:
    try:
        probe(BASE / fn, nm)
    except Exception as e:  # noqa: BLE001
        print(f"[{nm}] ERROR {type(e).__name__}: {e}")
    print("-" * 70)

v = json.load(open(BASE / "val_holdout_all.json"))
print(f"[val_holdout] type={type(v).__name__} len={len(v)}")
print("  repr[:400] =", repr(v)[:400])

m = json.load(open(BASE / "all_manifest.json"))
print(f"[all_manifest] type={type(m).__name__} len={len(m)}")
print("  repr[:400] =", repr(m)[:400])
