"""Error-decomposition diagnostic (v2) for the Live2D VLA model.

v2 fixes over v1:
  * uses the EXACT rel_mae definition from train.recon_metrics
    (mean|err| over T -> divide by RAW per-param range (floor 1e-3) -> clip 2.0).
    NOTE: this differs from the *training loss*, which floors the span at 2% of
    the global range. The metric is therefore dominated by near-constant params.
  * character priors are LEAVE-ONE-OUT (exclude the target action) so they are
    not trivially leaking the answer.
  * all priors cover the SAME (sample, param) set (grand-mean fallback).
  * results are split by how many models share the action (>=5 = "shared",
    else "rare"), because 72% of actions occur in only one model.

This quantifies the headroom for:
  (1) richer action conditioning   -> value of A vs AC vs KNN
  (2) more / diverse data          -> compare subset=60 vs subset=285
  (3) predictive character identity-> value of C / AC over A
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, "src/live2d_vla")
sys.path.insert(0, ".")

from schema import load_whitelist, load_gen_mask          # noqa: E402
from io_motion import list_motions, load_target_curves    # noqa: E402

T, FPS = 48, 30.0
SEED = 1234


# --------------------------------------------------------------------------- #
def load_corpus(subset, targets):
    cache, per_model_acts = {}, {}
    for m in subset:
        if m not in targets:
            continue
        acts = list_motions(ROOT / "standrad-live-2d" / m)
        per_model_acts[m] = sorted(acts)
        for action in acts:
            c = load_target_curves(ROOT / "standrad-live-2d" / m, action,
                                   targets[m], fps=FPS, T=T)
            if c:
                cache[(m, action)] = {p: np.asarray(v, np.float32).ravel()
                                      for p, v in c.items() if len(v)}
    return cache, per_model_acts


def range_stats(samples, cache):
    per_lo, per_hi = {}, {}
    for s in samples:
        for p, a in cache[s].items():
            lo, hi = float(a.min()), float(a.max())
            if p not in per_lo:
                per_lo[p], per_hi[p] = lo, hi
            else:
                per_lo[p] = min(per_lo[p], lo)
                per_hi[p] = max(per_hi[p], hi)
    g_lo = min(per_lo.values()) if per_lo else 0.0
    g_hi = max(per_hi.values()) if per_hi else 1.0
    return g_lo, g_hi, per_lo, per_hi


def evaluate(val_s, cache, predict_fn, per_lo, per_hi, g_lo, g_hi, grand):
    """rel_mae matching train.recon_metrics exactly."""
    tot, cnt = 0.0, 0
    for s in val_s:
        curves = cache[s]
        pred = predict_fn(s, curves) or {}
        for p, tgt in curves.items():
            pp = pred.get(p)
            if pp is None:
                pp = grand.get(p)
            if pp is None:
                continue
            e = float(np.abs(np.asarray(pp, np.float32) - tgt).mean())
            lo = per_lo.get(p, g_lo)
            hi = per_hi.get(p, g_hi)
            span = max(hi - lo, 1e-3)
            tot += min(e / span, 2.0)
            cnt += 1
    return (tot / cnt if cnt else float("nan")), cnt


def _mean_dict(acc):
    return {k: np.mean(np.stack(v, 0), 0) for k, v in acc.items()}


def build_action_mean(samples, cache):
    acc = {}
    for (m, a) in samples:
        for p, c in cache[(m, a)].items():
            acc.setdefault((a, p), []).append(c)
    return _mean_dict(acc)


def build_model_mean_loo(cache):
    """(model, action, param) -> mean over the model's OTHER actions."""
    acc = {}
    for (m, a) in cache:
        for p, c in cache[(m, a)].items():
            acc.setdefault((m, a, p), []).append(c)
    # accumulate per (m,p) all actions, then subtract current action
    per_mp = {}
    for (m, a, p), lst in acc.items():
        per_mp.setdefault((m, p), []).extend(lst)
    out = {}
    for (m, a, p), lst in acc.items():
        all_lst = per_mp[(m, p)]
        if len(all_lst) > len(lst):          # other actions exist
            s = np.sum(np.stack(all_lst, 0), 0) - np.sum(np.stack(lst, 0), 0)
            out[(m, a, p)] = s / (len(all_lst) - len(lst))
        else:
            out[(m, a, p)] = np.mean(np.stack(lst, 0), 0)
    return out


def run(subset_n: int, val_frac: float, tag: str, do_knn: bool = True):
    wl = json.loads((ROOT / "outputs" / "model_whitelist.json").read_bytes()).get("kept", [])
    gm = json.loads((ROOT / "outputs" / "gen_mask.json").read_bytes()).get("models", {})
    subset = wl[:subset_n]
    targets = {m: gm[m]["target"] for m in subset if m in gm}

    cache, per_model_acts = load_corpus(subset, targets)
    act_nmodels = Counter()
    for m, acts in per_model_acts.items():
        for a in acts:
            act_nmodels[a] += 1

    rng = random.Random(SEED)
    n_val = max(1, int(round(len(subset) * val_frac)))
    holdout = set(rng.sample(subset, n_val))
    train_s = [s for s in cache if s[0] not in holdout]
    val_s = [s for s in cache if s[0] in holdout]

    n_uniq = sum(1 for v in act_nmodels.values() if v == 1)
    shared_s = [s for s in val_s if act_nmodels[s[1]] >= 5]
    rare_s = [s for s in val_s if act_nmodels[s[1]] < 5]

    print(f"\n{'='*74}\n[{tag}] subset={subset_n} models={len(targets)} "
          f"samples={len(cache)}  train={len(train_s)} val={len(val_s)} "
          f"(val chars={len({s[0] for s in val_s})})")
    print(f"  actions={len(act_nmodels)} (unique-to-one-model={n_uniq}, "
          f"{n_uniq/max(len(act_nmodels),1):.0%})   "
          f"val: shared(>=5 models)={len(shared_s)}  rare(<5)={len(rare_s)}")
    print("=" * 74)

    g_lo, g_hi, per_lo, per_hi = range_stats(train_s, cache)

    am_train = build_action_mean(train_s, cache)
    am_all = build_action_mean(list(cache), cache)
    mm_loo = build_model_mean_loo(cache)

    acc = {}
    for s in train_s:
        for p, c in cache[s].items():
            acc.setdefault(p, []).append(c)
    grand = _mean_dict(acc)

    def mk_action(am):
        return lambda s, curves: {p: am[(s[1], p)] for p in curves if (s[1], p) in am}

    def mk_char(s, curves):
        m, a = s
        return {p: mm_loo[(m, a, p)] for p in curves if (m, a, p) in mm_loo}

    def mk_additive(s, curves):
        m, a = s
        out = {}
        for p in curves:
            amv, gmv = am_train.get((a, p)), grand.get(p)
            mmv = mm_loo.get((m, a, p))
            if amv is None or gmv is None or mmv is None:
                continue
            out[p] = amv + mmv - gmv
        return out

    priors = {
        "P_A_train  (action, clean)": mk_action(am_train),
        "P_A_all    (current exem)": mk_action(am_all),
        "P_C_loo    (character LOO)": mk_char,
        "P_AC_loo   (additive)": mk_additive,
    }

    if do_knn:
        all_params = sorted({p for s in train_s for p in cache[s]})
        train_models = sorted({s[0] for s in train_s})

        def _uvec(m, a):
            parts = []
            for p in all_params:
                v = mm_loo.get((m, a, p))
                if v is None:
                    continue
                n = np.linalg.norm(v)
                parts.append(v / n if n > 1e-8 else v)
            return np.concatenate(parts) if parts else None

        def mk_knn(s, curves):
            m, a = s
            q = _uvec(m, a)
            if q is None:
                return {}
            best, bs = None, -2.0
            for tm in train_models:
                acts = [x for x in per_model_acts.get(tm, []) if (tm, x, all_params[0]) in mm_loo]
                if not acts:
                    continue
                c = _uvec(tm, acts[0])
                if c is None:
                    continue
                n = min(len(c), len(q))
                if n == 0:
                    continue
                cv, qv = c[:n], q[:n]
                sim = float(np.dot(cv, qv) / (np.linalg.norm(cv) * np.linalg.norm(qv) + 1e-8))
                if sim > bs:
                    bs, best = sim, tm
            if best is None or (best, a) not in cache:
                return {}
            src = cache[(best, a)]
            return {p: src[p] for p in curves if p in src}

        priors["P_KNN_loo  (nearest char)"] = mk_knn

    groups = [("ALL val", val_s)]
    if shared_s and rare_s:
        groups += [(f"shared(action>={5} models)", shared_s),
                   (f"rare(action<{5} models)", rare_s)]

    print(f"  {'prior':32s} " + " ".join(f"{g[0][:22]:>24s}" for g in groups))
    for name, fn in priors.items():
        cells = []
        for gname, gs in groups:
            v, c = evaluate(gs, cache, fn, per_lo, per_hi, g_lo, g_hi, grand)
            cells.append(f"{v:.4f} (n={c})".rjust(24))
        print(f"  {name:32s} " + " ".join(cells))
    return priors


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    vf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.12
    do_full = len(sys.argv) > 3 and sys.argv[3] == "full"
    run(n, vf, f"current setup (subset={n})")
    if do_full:
        run(285, 0.12, "full corpus (subset=285)")
