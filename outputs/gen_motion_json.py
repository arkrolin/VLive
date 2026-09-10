"""Export val predictions as standard Live2D motion3.json files.

Rendering happens separately on the Windows box (live-2d venv + live2d-py);
this side only has to emit plain Cubism motion files, so anything that can
load a .model3.json can play them.

For every selected sample three motions are written:
    GEN_<action>__pred    - our model's output
    GEN_<action>__target  - ground truth
    GEN_<action>__exem    - the action-mean prior (the baseline to beat)
so the three can be played side by side.

Also writes manifest.json listing what was produced and where each character
lives, for the render step to consume.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "tools"))
from motion_io import save_motion  # noqa: E402

TAGS = ("pred", "target", "exem")


def pick(manifest, n: int, per_char: int = 1):
    """Largest-motion samples, at most `per_char` per character."""
    seen: dict[str, int] = {}
    out = []
    for c in sorted(manifest, key=lambda x: -x["span_sum"]):
        if seen.get(c["model"], 0) >= per_char:
            continue
        seen[c["model"]] = seen.get(c["model"], 0) + 1
        out.append(c)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(ROOT / "outputs/render_data/abl_R_all_v9_best.pkl"))
    ap.add_argument("--out", default=str(ROOT / "outputs/gen_motions"))
    ap.add_argument("--idx", default="", help="explicit sample indices (overrides --auto)")
    ap.add_argument("--auto", type=int, default=8)
    ap.add_argument("--per-char", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    blob = pickle.load(open(args.pkl, "rb"))
    items = {it["idx"]: it for it in blob["items"]}
    man = {m["idx"]: m for m in blob["manifest"]}

    if args.idx:
        sel = [int(x) for x in args.idx.split(",") if x.strip()]
    else:
        sel = [c["idx"] for c in pick(blob["manifest"], args.auto, args.per_char)]

    out = Path(args.out)
    entries = []
    for i in sel:
        it = items.get(i)
        if it is None:
            print(f"  idx {i}: missing")
            continue
        model = it["model"]
        real = model[len("std__"):] if model.startswith("std__") else model
        action = it["action"].replace("~", "_")
        mdir = out / real / "motions"
        mdir.mkdir(parents=True, exist_ok=True)

        T = it["pred"].shape[1]
        rec = dict(idx=i, model=real, action=it["action"], T=int(T),
                   n_params=len(it["names"]), files={},
                   err_pred=man[i]["err_pred"], err_exem=man[i]["err_exem"])
        for tag in TAGS:
            curves = {n: it[tag][j] for j, n in enumerate(it["names"])}
            fp = mdir / f"GEN_{action}__{tag}.motion3.json"
            meta = save_motion(curves, fp, fps=args.fps,
                               duration=T / args.fps, kind="linear")
            rec["files"][tag] = fp.name
        entries.append(rec)
        print(f"  idx {i:4d} {real:22s} {it['action'][:24]:24s} P={len(it['names']):3d} "
              f"T={T}  err_model={man[i]['err_pred']:.3f} err_exem={man[i]['err_exem']:.3f}")

    mf = out / "manifest.json"
    mf.write_text(json.dumps(dict(
        run=blob["run"], ckpt=blob["ckpt"], epoch=blob.get("epoch"),
        fps=args.fps, tags=list(TAGS), entries=entries), indent=2, ensure_ascii=False))
    print(f"\n{len(entries)} samples -> {out}\nmanifest: {mf}")


if __name__ == "__main__":
    main()
