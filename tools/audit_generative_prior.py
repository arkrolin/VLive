"""Is the cross-character prior useless, or is retrieval just the wrong probe?

Context. The user's requirement is now pinned down: the whole point of the
generative model is the case where NO existing version of the action exists for
the target character. So "retrieve the same action from an older pack of the
same character and retarget" -- which scores 84% -- is explicitly out of scope.
That makes every same-character number irrelevant and puts all the weight on
the cross-character signal, which retrieval put at a dismal ~4% r@1.

But r@1 asks "is the nearest neighbour in 960-dim trajectory space the same
action". That demands point-wise alignment of two different artists' curves.
A generative model does not need that. It needs, in decreasing order of
importance:

    L1  WHICH channels move        (wave -> right arm, shy -> head tilt + hands)
    L2  HOW MUCH each one moves    (amplitude signature)
    L3  WHAT RHYTHM                (energy envelope over normalised time)
    L4  the exact trajectory       (<- the only thing r@1 measured)

If L1-L3 transfer across characters while L4 does not, a shared prior is worth
training and the architecture is "prior predicts the coarse plan, few-shot rig
adaptation fills in the exact curves". If none of them transfer, the corpus is
not a text->motion pretraining set at all and M1 must be pure per-rig few-shot.

The decisive test is an additive factorisation on held-out characters:

    f(char, action) ~= mu + rig(char) + act(action)

where rig(char) is estimated ONLY from that character's OTHER actions (exactly
the few-shot support the deployment scenario provides) and act(action) is
estimated ONLY from OTHER characters (exactly the shared prior). If adding the
act term on top of the rig term reduces error, the prior carries real action
information that few-shot alone cannot supply.

Usage:
    uv run python tools/audit_generative_prior.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from audit_arm_channels import ARM_HINT, ARM_SLOTS, canon_arm  # noqa: E402
from audit_leakage import char_of  # noqa: E402
from motion_curves import sample_curve  # noqa: E402
from survey_motion_names import norm_action  # noqa: E402
from verify_action_alignment import CANON, T_NORM  # noqa: E402

SLOTS = list(CANON) + ARM_SLOTS
NS = len(SLOTS)
POS = {s: i for i, s in enumerate(SLOTS)}
CACHE = Path("outputs/_genprior_cache.npz")


def slot_of(pid: str) -> str | None:
    for name, rx in CANON.items():
        if rx.match(pid):
            return name
    if ARM_HINT.search(pid):
        k = canon_arm(pid)
        if k in POS:
            return k
    return None


def encode(path: Path):
    """Return (amp[NS], shape[NS,T], present[NS], duration) or None."""
    try:
        j = orjson.loads(path.read_bytes())
    except Exception:
        return None
    dur = float(j.get("Meta", {}).get("Duration", 0) or 0)
    if not (0.2 < dur < 60):
        return None
    t = np.linspace(0.0, dur, T_NORM)
    amp = np.zeros(NS, np.float32)
    shape = np.zeros((NS, T_NORM), np.float32)
    present = np.zeros(NS, bool)
    for c in j.get("Curves", []):
        if c.get("Target") != "Parameter":
            continue
        s = slot_of(str(c.get("Id", "")))
        if s is None or present[POS[s]]:
            continue
        try:
            v = sample_curve(c.get("Segments") or [], t)
        except Exception:
            continue
        i = POS[s]
        present[i] = True
        sd = float(np.std(v))
        amp[i] = sd
        if sd > 1e-6:
            shape[i] = (v - float(np.mean(v))) / sd
    if present.sum() < 4:
        return None
    return amp, shape, present, dur


def build(root: Path, models: list[str], refresh: bool):
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        return (z["amp"], z["shape"], z["pres"], z["dur"],
                z["act"], z["pack"], z["char"])
    A, S, P, D, ya, ym, yc = [], [], [], [], [], [], []
    for n, name in enumerate(models):
        d = root / name / "motions"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            a, _ = norm_action(f.stem)
            if a in ("stand", "idle"):
                continue
            r = encode(f)
            if r is None:
                continue
            A.append(r[0]); S.append(r[1]); P.append(r[2]); D.append(r[3])
            ya.append(a); ym.append(name); yc.append(char_of(name))
        if n % 60 == 0:
            print(f"  ...{n}/{len(models)} packs, {len(A)} motions", flush=True)
    out = (np.stack(A), np.stack(S), np.stack(P), np.array(D, np.float32),
           np.array(ya), np.array(ym), np.array(yc))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, amp=out[0], shape=out[1], pres=out[2],
                        dur=out[3], act=out[4], pack=out[5], char=out[6])
    return out


def dedup(amp, shape, pres, dur, ya, ym, yc, thr=0.95):
    """Collapse retarget copies: same character + same action + cos>=thr."""
    keep = np.ones(len(ya), bool)
    flat = shape.reshape(len(ya), -1)
    nrm = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-6)
    groups = defaultdict(list)
    for i, (c, a) in enumerate(zip(yc, ya)):
        groups[(c, a)].append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        idx = np.array(idx)
        sim = nrm[idx] @ nrm[idx].T
        taken: list[int] = []
        for j, gi in enumerate(idx):
            if any(sim[j, np.where(idx == t)[0][0]] >= thr for t in taken):
                keep[gi] = False
            else:
                taken.append(gi)
    return keep


LAMBDAS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def factorise(F, ya, yc, tag):
    """f(char,action) ~= mu + rig(char) + lam * act(action), leave-one-CHAR-out.

    rig(char) uses ONLY the held-out character's other actions (the few-shot
    support). act(action) uses ONLY the training characters (the shared prior).

    lam is swept rather than pinned at 1.0: the action term is estimated from
    other rigs, so its raw scale is wrong even when its DIRECTION is right.
    Reporting only lam=1 would confuse "badly scaled" with "no information".
    """
    chars = sorted(set(yc))
    err_mu, err_rig, err_act = [], [], []
    err_both = {lam: [] for lam in LAMBDAS}
    n_eval = 0
    for c in chars:
        te = np.where(yc == c)[0]
        tr = np.where(yc != c)[0]
        if len(te) < 6 or len(tr) < 200:
            continue
        mu = F[tr].mean(0)
        act_tbl = {}
        for a in set(ya[te]):
            m = tr[ya[tr] == a]
            if len(m) >= 3:
                act_tbl[a] = F[m].mean(0) - mu
        for i in te:
            a = ya[i]
            if a not in act_tbl:
                continue
            sup = te[(ya[te] != a)]            # same rig, DIFFERENT actions
            if len(sup) < 3:
                continue
            rig = F[sup].mean(0) - mu
            act = act_tbl[a]
            truth = F[i]
            err_mu.append(float(np.mean((truth - mu) ** 2)))
            err_rig.append(float(np.mean((truth - mu - rig) ** 2)))
            err_act.append(float(np.mean((truth - mu - act) ** 2)))
            for lam in LAMBDAS:
                err_both[lam].append(
                    float(np.mean((truth - mu - rig - lam * act) ** 2)))
            n_eval += 1
    if n_eval < 30:
        print(f"  {tag}: too few eval points ({n_eval})")
        return
    base = np.mean(err_mu)
    e_rig = np.mean(err_rig)
    print(f"  {tag}  (n={n_eval} held-out, leave-one-character-out)")
    print(f"      global mean only                       "
          f"MSE={base:8.4f}   R2={0.0:+6.3f}")
    print(f"      + rig term only (few-shot same rig)    "
          f"MSE={e_rig:8.4f}   R2={1 - e_rig / base:+6.3f}")
    print(f"      + action term only (cross-char prior)  "
          f"MSE={np.mean(err_act):8.4f}   "
          f"R2={1 - np.mean(err_act) / base:+6.3f}")
    curve = {lam: np.mean(v) for lam, v in err_both.items()}
    best = min(curve, key=curve.get)
    sweep = "  ".join(f"{lam:.2f}:{100*(1 - curve[lam] / e_rig):+5.1f}%"
                      for lam in LAMBDAS)
    print(f"      rig + lam*action, gain over rig-only:  {sweep}")
    print(f"      -> best lam={best:.2f}, action prior adds "
          f"{100*(1 - curve[best] / e_rig):+.1f}% on top of few-shot "
          f"(total R2={1 - curve[best] / base:+.3f})")


def per_channel_gain(F, ya, yc, lam=0.35):
    """Same factorisation, but keep the error per channel instead of averaging.

    The headline number hides the thing that matters most here: the project has
    to generate ARMS. If the action prior only helps the mouth, it is useless
    for this use case even if the average looks positive.
    """
    chars = sorted(set(yc))
    e_rig, e_both, e_mu = [], [], []
    for c in chars:
        te = np.where(yc == c)[0]
        tr = np.where(yc != c)[0]
        if len(te) < 6 or len(tr) < 200:
            continue
        mu = F[tr].mean(0)
        act_tbl = {a: F[tr[ya[tr] == a]].mean(0) - mu
                   for a in set(ya[te]) if (ya[tr] == a).sum() >= 3}
        for i in te:
            if ya[i] not in act_tbl:
                continue
            sup = te[ya[te] != ya[i]]
            if len(sup) < 3:
                continue
            rig = F[sup].mean(0) - mu
            d = F[i]
            e_mu.append((d - mu) ** 2)
            e_rig.append((d - mu - rig) ** 2)
            e_both.append((d - mu - rig - lam * act_tbl[ya[i]]) ** 2)
    if len(e_rig) < 30:
        print("  (too few)")
        return
    E0, E1, E2 = (np.mean(np.stack(x), 0) for x in (e_mu, e_rig, e_both))
    order = np.argsort(-(E1 - E2) / np.maximum(E1, 1e-9))
    print(f"  lam={lam}, sorted by extra gain from the action label")
    print(f"    {'channel':12s} {'R2 rig':>8s} {'R2 rig+act':>11s} "
          f"{'action adds':>12s}")
    for i in order:
        print(f"    {SLOTS[i]:12s} {1 - E1[i]/E0[i]:+8.3f} "
              f"{1 - E2[i]/E0[i]:+11.3f} {100*(1 - E2[i]/E1[i]):+11.1f}%")


def oracle_gap(F, ya, yc, tag, lam=0.5):
    """Upper bound check.

    deployable : mu + rig(char, other actions) + lam * act(OTHER characters)
    oracle     : mu + rig(char, other actions) + act(SAME character, same
                 action, other packs)  <- needs an older version to retarget

    Everything between the two is what a generative model has to synthesise
    rather than copy. Only motions that HAVE a same-character sibling can be
    scored, so this runs on a subset.
    """
    chars = sorted(set(yc))
    d_rig, d_dep, d_ora, base = [], [], [], []
    for c in chars:
        te = np.where(yc == c)[0]
        tr = np.where(yc != c)[0]
        if len(te) < 6 or len(tr) < 200:
            continue
        mu = F[tr].mean(0)
        for i in te:
            a = ya[i]
            m = tr[ya[tr] == a]
            sib = te[(ya[te] == a) & (np.arange(len(yc))[te] != i)]
            sup = te[ya[te] != a]
            if len(m) < 3 or len(sib) < 1 or len(sup) < 3:
                continue
            rig = F[sup].mean(0) - mu
            act = F[m].mean(0) - mu
            ora = F[sib].mean(0) - mu
            t = F[i]
            base.append(float(np.mean((t - mu) ** 2)))
            d_rig.append(float(np.mean((t - mu - rig) ** 2)))
            d_dep.append(float(np.mean((t - mu - rig - lam * act) ** 2)))
            d_ora.append(float(np.mean((t - mu - ora) ** 2)))
    if len(base) < 30:
        print(f"  {tag}: too few paired samples ({len(base)})")
        return
    b = np.mean(base)
    print(f"  {tag}  (n={len(base)} motions that DO have a same-char sibling)")
    for lbl, v in (("few-shot rig only", d_rig),
                   ("rig + cross-char action prior (deployable)", d_dep),
                   ("same-char sibling (oracle / retarget)", d_ora)):
        print(f"      {lbl:44s} R2={1 - np.mean(v) / b:+6.3f}")
    gap = (np.mean(d_dep) - np.mean(d_ora)) / b
    print(f"      -> generation must close a gap of {gap:.3f} R2")


def classify_by_prior(F, ya, yc, tag):
    """Given an observed motion + its rig, pick the action using ONLY a
    cross-character action table. This is the honest 'does the word carry
    signal' test -- no same-character information about the target action."""
    chars = sorted(set(yc))
    hit = hit5 = tot = 0
    chance_num = 0.0
    for c in chars:
        te = np.where(yc == c)[0]
        tr = np.where(yc != c)[0]
        if len(te) < 6 or len(tr) < 200:
            continue
        mu = F[tr].mean(0)
        names, cents = [], []
        for a in sorted(set(ya[te])):
            m = tr[ya[tr] == a]
            if len(m) >= 3:
                names.append(a)
                cents.append(F[m].mean(0) - mu)
        if len(names) < 4:
            continue
        cents = np.stack(cents)
        cnt = Counter(ya[te])
        for i in te:
            if ya[i] not in names:
                continue
            sup = te[ya[te] != ya[i]]
            if len(sup) < 3:
                continue
            rig = F[sup].mean(0) - mu
            d = np.sum((F[i] - (mu + rig + cents)) ** 2, axis=1)
            order = np.argsort(d)
            hit += int(names[order[0]] == ya[i])
            hit5 += int(ya[i] in [names[j] for j in order[:5]])
            tot += 1
            chance_num += cnt[ya[i]] / max(sum(cnt[a] for a in names), 1)
    if tot < 30:
        print(f"  {tag}: too few ({tot})")
        return
    print(f"  {tag:34s} top1={100*hit/tot:5.1f}%  top5={100*hit5/tot:5.1f}%  "
          f"chance={100*chance_num/tot:4.1f}%  "
          f"lift={(hit/tot)/max(chance_num/tot,1e-9):4.1f}x  (n={tot})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    models = sorted(orjson.loads(Path(args.whitelist).read_bytes())["kept"])

    print("=== encoding (20 channels: 14 face/body + 6 arm) ===")
    amp, shape, pres, dur, ya, ym, yc = build(root, models, args.refresh)
    print(f"  {len(ya)} motions | {len(set(ym))} packs | {len(set(yc))} chars")

    k = dedup(amp, shape, pres, dur, ya, ym, yc)
    amp, shape, pres, dur = amp[k], shape[k], pres[k], dur[k]
    ya, ym, yc = ya[k], ym[k], yc[k]
    print(f"  after retarget-dedup: {len(ya)} distinct assets "
          f"({100*(1-k.mean()):.1f}% removed)")

    per_act_chars = defaultdict(set)
    for a, c in zip(ya, yc):
        per_act_chars[a].add(c)
    for th in (2, 3, 5, 8):
        n = sum(1 for v in per_act_chars.values() if len(v) >= th)
        cov = sum(1 for a in ya if len(per_act_chars[a]) >= th)
        print(f"  actions seen in >={th} characters: {n:4d}   "
              f"covering {100*cov/len(ya):4.1f}% of assets")

    keep = np.array([len(per_act_chars[a]) >= 3 for a in ya])
    amp, shape, pres = amp[keep], shape[keep], pres[keep]
    dur, ya, ym, yc = dur[keep], ya[keep], ym[keep], yc[keep]
    print(f"  eval subset (action in >=3 chars): {len(ya)} assets, "
          f"{len(set(ya))} actions, {len(set(yc))} chars")

    # ---------- L1/L2: channel activation + amplitude signature ----------
    F_amp = np.log1p(np.abs(amp) / (np.abs(amp).mean(0, keepdims=True) + 1e-6))
    # "active" must mean the channel actually MOVES. Using `pres` (the curve
    # exists in the file) measures the exporter's habit, not the choreography:
    # Cubism writes every rigged parameter whether or not it is animated.
    scale = np.array([np.median(amp[pres[:, i], i]) if pres[:, i].any() else 1.0
                      for i in range(NS)], np.float32)
    F_act = (amp > 0.15 * np.maximum(scale, 1e-6)).astype(np.float32)
    print(f"  channels present/file: {pres.sum(1).mean():.1f}   "
          f"channels actually moving: {F_act.sum(1).mean():.1f}")
    # ---------- L3: rhythm, energy envelope over normalised time ----------
    ener = np.abs(np.diff(shape, axis=2)).sum(1)
    F_env = ener / np.maximum(ener.sum(1, keepdims=True), 1e-6)
    # ---------- L4: full trajectory ----------
    F_traj = shape.reshape(len(ya), -1)

    print("\n=== additive factorisation: does the action prior add "
          "anything on top of few-shot? ===")
    factorise(F_amp, ya, yc, "L2 amplitude signature (20d)")
    factorise(F_act, ya, yc, "L1 channel activation (20d)")
    factorise(F_env, ya, yc, "L3 rhythm envelope (47d)")
    factorise(F_traj, ya, yc, "L4 full trajectory (960d)")

    print("\n=== per-channel L2 amplitude: where does the label help? ===")
    per_channel_gain(F_amp, ya, yc)
    print("\n=== per-channel L1 activation: does the label at least say "
          "WHICH channels move? ===")
    per_channel_gain(F_act, ya, yc, lam=0.75)

    print("\n=== ceiling: what would a same-character oracle buy? ===")
    print("   (the oracle needs an older version of the SAME action on the SAME")
    print("    character -- i.e. exactly the retrieval+retarget path that is")
    print("    out of scope. The gap is what generation has to invent.)")
    oracle_gap(F_amp, ya, yc, "L2 amplitude")
    oracle_gap(F_act, ya, yc, "L1 activation")

    print("\n=== action recognition from a CROSS-CHARACTER prior only ===")
    classify_by_prior(F_amp, ya, yc, "L2 amplitude")
    classify_by_prior(F_act, ya, yc, "L1 activation")
    classify_by_prior(F_env, ya, yc, "L3 rhythm")
    classify_by_prior(F_traj, ya, yc, "L4 trajectory")
    classify_by_prior(np.hstack([F_amp, F_act, F_env]), ya, yc,
                      "L1+L2+L3 combined")

    print("\n=== how much does duration alone say? ===")
    classify_by_prior(np.log(dur)[:, None].astype(np.float32), ya, yc,
                      "duration only (1d)")


if __name__ == "__main__":
    main()
