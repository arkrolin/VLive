"""Export a few pred / target / exem curve triplets as small JSON for plotting.

Picks the two samples where the model beats the exem prior by the largest
margin and the two where it loses, so the plot shows both the win and the
failure mode. Only the four largest-motion parameters per sample are kept.
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np


def main() -> int:
    pkl, out = sys.argv[1], sys.argv[2]
    blob = pickle.load(open(pkl, "rb"))
    items = {it["idx"]: it for it in blob["items"]}
    man = {m["idx"]: m for m in blob["manifest"]}

    scored = sorted((m["err_pred"] - m["err_exem"], m["idx"])
                    for m in blob["manifest"])
    sel = [i for _, i in scored[:2]] + [i for _, i in scored[-2:]]

    res = []
    for idx in sel:
        it = items[idx]
        tgt, pred, exm = it["target"], it["pred"], it["exem"]
        ptp = tgt.max(axis=1) - tgt.min(axis=1)
        top = np.argsort(-ptp)[:4]
        res.append(dict(
            idx=int(idx), model=it["model"], action=it["action"],
            err_pred=round(man[idx]["err_pred"], 3),
            err_exem=round(man[idx]["err_exem"], 3),
            params=[it["names"][j] for j in top],
            target=[np.round(tgt[j], 3).tolist() for j in top],
            pred=[np.round(pred[j], 3).tolist() for j in top],
            exem=[np.round(exm[j], 3).tolist() for j in top],
        ))
    json.dump(res, open(out, "w"), ensure_ascii=False)
    print(f"wrote {out}: {len(res)} samples")
    for r in res:
        verdict = "MODEL WINS" if r["err_pred"] < r["err_exem"] else "model loses"
        print(f"  idx={r['idx']:4d} {r['model']:26s} {r['action'][:20]:20s} "
              f"pred {r['err_pred']:7.3f}  exem {r['err_exem']:7.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
