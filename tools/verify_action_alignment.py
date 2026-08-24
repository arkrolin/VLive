"""The decisive experiment: do motion FILE NAMES actually predict the motion?

survey_motion_names.py showed 88.4% of motions carry a cross-model shared action
key (dazhaohu / haixiu / diantou ...). That is only useful if two models labelled
'dazhaohu' actually move alike. If they do, we have 133k pairs of
"same semantic action, different rigging" -- direct supervision for cross-model
generalisation, which the Live2D literature has none of.

Test: take parameters that are standard across models, time-normalise each motion
to a fixed length, then ask whether same-action pairs are closer than
different-action pairs, and whether nearest-neighbour retrieval recovers the
action label ACROSS models (never comparing a model to itself).

Usage:
    uv run python tools/verify_action_alignment.py
    uv run python tools/verify_action_alignment.py --n-models 120 --verbose
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
from motion_curves import sample_curve  # noqa: E402
from survey_motion_names import norm_action  # noqa: E402

# canonical channels present in most models; the alias groups absorb the
# Cubism2 (PARAM_ANGLE_X) vs Cubism4 (ParamAngleX) naming split
CANON: dict[str, re.Pattern] = {
    "head_x": re.compile(r"^param_?angle_?x$", re.I),
    "head_y": re.compile(r"^param_?angle_?y$", re.I),
    "head_z": re.compile(r"^param_?angle_?z$", re.I),
    "body_x": re.compile(r"^param_?body_?angle_?x$", re.I),
    "body_y": re.compile(r"^param_?body_?angle_?y$", re.I),
    "body_z": re.compile(r"^param_?body_?angle_?z$", re.I),
    "eye_l": re.compile(r"^param_?eye_?l_?open$", re.I),
    "eye_r": re.compile(r"^param_?eye_?r_?open$", re.I),
    "mouth_open": re.compile(r"^param_?mouth_?open_?y$", re.I),
    "mouth_form": re.compile(r"^param_?mouth_?form$", re.I),
    "brow_l": re.compile(r"^param_?brow_?l_?y$", re.I),
    "brow_r": re.compile(r"^param_?brow_?r_?y$", re.I),
    "eyeball_x": re.compile(r"^param_?eye_?ball_?x$", re.I),
    "eyeball_y": re.compile(r"^param_?eye_?ball_?y$", re.I),
}
T_NORM = 48          # every motion is resampled to this many steps
MIN_PER_ACTION = 6   # an action needs this many models to enter the test


def load_json(p: Path):
    try:
        return orjson.loads(p.read_bytes())
    except Exception:
        return None


def encode_motion(path: Path) -> np.ndarray | None:
    """-> (T_NORM, len(CANON)) time-normalised, per-channel z-scored trajectory."""
    j = load_json(path)
    if not j:
        return None
    dur = float((j.get("Meta") or {}).get("Duration") or 0.0)
    if dur <= 0.3:
        return None
    curves = {c.get("Id"): c.get("Segments") or []
              for c in (j.get("Curves") or []) if c.get("Target") == "Parameter"}
    if not curves:
        return None
    # time-normalise: sample T_NORM points spread over the motion's own duration
    times = np.linspace(0.0, dur, T_NORM, dtype=np.float32)
    cols = []
    for _name, pat in CANON.items():
        hit = next((cid for cid in curves if pat.match(cid)), None)
        cols.append(sample_curve(curves[hit], times) if hit
                    else np.zeros(T_NORM, dtype=np.float32))
    x = np.stack(cols, axis=1)
    # z-score per channel WITHIN the motion: removes the model's rest pose and
    # its parameter scale, keeping only the shape of the movement
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    x = (x - mu) / np.maximum(sd, 1e-3)
    x[:, (sd[0] < 1e-3)] = 0.0
    return x.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--n-models", type=int, default=140)
    ap.add_argument("--exclude-stand", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    wl = Path(args.whitelist)
    models = (sorted(orjson.loads(wl.read_bytes())["kept"]) if wl.exists()
              else sorted(d.name for d in root.iterdir() if d.is_dir()))
    rng = np.random.default_rng(args.seed)
    if len(models) > args.n_models:
        models = list(rng.choice(models, args.n_models, replace=False))

    X, y_action, y_model = [], [], []
    for name in models:
        mdir = root / name / "motions"
        if not mdir.is_dir():
            continue
        for f in sorted({p.name: p for p in mdir.glob("*.json")}.values()):
            act, _ = norm_action(f.stem)
            if args.exclude_stand and act in ("stand", "idle"):
                continue
            v = encode_motion(f)
            if v is None:
                continue
            X.append(v.reshape(-1))
            y_action.append(act)
            y_model.append(name)
    if not X:
        raise SystemExit("nothing encoded")

    X = np.stack(X)
    y_action = np.array(y_action)
    y_model = np.array(y_model)

    # keep actions that span enough DISTINCT models
    per_action_models = defaultdict(set)
    for a, m in zip(y_action, y_model):
        per_action_models[a].add(m)
    good = {a for a, ms in per_action_models.items() if len(ms) >= MIN_PER_ACTION}
    keep = np.array([a in good for a in y_action])
    X, y_action, y_model = X[keep], y_action[keep], y_model[keep]
    print(f"=== {len(X)} motions | {len(set(y_model))} models | "
          f"{len(good)} actions (>= {MIN_PER_ACTION} models each) ===")
    print(f"    feature = {T_NORM} steps x {len(CANON)} canonical channels, "
          f"per-motion z-scored\n")

    # cosine similarity on L2-normalised vectors
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)

    same_model = y_model[:, None] == y_model[None, :]
    same_act = y_action[:, None] == y_action[None, :]
    cross = ~same_model & np.isfinite(S)

    d_same = S[cross & same_act]
    d_diff = S[cross & ~same_act]
    pooled = np.sqrt(0.5 * (d_same.var() + d_diff.var()))
    print("-- cross-model similarity (higher = more alike) --")
    print(f"  same action      : mean={d_same.mean():+.4f} sd={d_same.std():.4f} "
          f"n={len(d_same)}")
    print(f"  different action : mean={d_diff.mean():+.4f} sd={d_diff.std():.4f} "
          f"n={len(d_diff)}")
    print(f"  effect size (Cohen's d) = {(d_same.mean()-d_diff.mean())/pooled:.3f}")

    # cross-model retrieval: query motion -> nearest motions in OTHER models
    S_cross = np.where(same_model, -np.inf, S)
    order = np.argsort(-S_cross, axis=1)
    n = len(X)
    hit1 = hit5 = hit10 = 0
    per_action_hit = Counter()
    per_action_tot = Counter()
    for i in range(n):
        top = order[i, :10]
        lbl = y_action[top]
        ok1 = lbl[0] == y_action[i]
        hit1 += ok1
        hit5 += (lbl[:5] == y_action[i]).any()
        hit10 += (lbl == y_action[i]).any()
        per_action_tot[y_action[i]] += 1
        per_action_hit[y_action[i]] += ok1

    # random baseline: chance that a random other-model motion shares the label
    counts = Counter(y_action)
    rand = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    print(f"\n-- cross-model action retrieval ({n} queries, never same model) --")
    print(f"  recall@1  = {100*hit1/n:5.1f}%")
    print(f"  recall@5  = {100*hit5/n:5.1f}%")
    print(f"  recall@10 = {100*hit10/n:5.1f}%")
    print(f"  random    = {100*rand:5.2f}%   -> {hit1/n/max(rand,1e-9):.1f}x chance")

    print(f"\n-- best-recovered actions (r@1, >= 12 queries) --")
    rows = [(per_action_hit[a] / per_action_tot[a], per_action_tot[a], a)
            for a in per_action_tot if per_action_tot[a] >= 12]
    rows.sort(reverse=True)
    for r, t, a in rows[:14]:
        print(f"  {100*r:5.1f}%  (n={t:3d})  {a}")
    print(f"\n-- worst-recovered --")
    for r, t, a in rows[-10:]:
        print(f"  {100*r:5.1f}%  (n={t:3d})  {a}")

    if args.verbose:
        print(f"\n-- channel ablation: r@1 using one channel group at a time --")
        names = list(CANON)
        groups = {"head(3)": [0, 1, 2], "body(3)": [3, 4, 5],
                  "eyes(2)": [6, 7], "mouth(2)": [8, 9],
                  "brow(2)": [10, 11], "gaze(2)": [12, 13]}
        Xr = X.reshape(len(X), T_NORM, len(names))
        for gname, idxs in groups.items():
            sub = Xr[:, :, idxs].reshape(len(X), -1)
            Zs = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-6)
            Ss = np.where(same_model, -np.inf, Zs @ Zs.T)
            np.fill_diagonal(Ss, -np.inf)
            nn = np.argmax(Ss, axis=1)
            print(f"    {gname:9s} r@1 = {100*(y_action[nn]==y_action).mean():5.1f}%")


if __name__ == "__main__":
    main()
